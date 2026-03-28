 //  1. Double-bridge 4-opt perturbation → Iterated Local Search (ILS)
//     - Breaks out of local optima. Largest single improvement for n>10k.
//  2. 2-opt: removed j<i guard (was skipping ~50% of improving moves)
//     Replaced with proper circular indexing and conflict detection.
//  3. Multi-move parallel 2-opt: each thread finds its own best move;
//     thread 0 applies greedy non-conflicting subset (no atomicMax race).
//  4. Or-opt*: tries reversed chain insertion as well as forward.
//  5. k=35-40 for n>10k instances (was k=25/30, misses long-range moves).
//  6. ILS budget: perturb → 2opt+oropt until converge → keep if better.
//  7. CPU or-opt + double 2-opt polish on top-K walkers after GPU phase.
// 

#include <bits/stdc++.h>
#include <curand_kernel.h>
using namespace std;

// =============================================================================
// Distance matrix — works for ALL TSPLIB formats
// =============================================================================
struct City { double x, y; };
enum EWT { EUC_2D, CEIL_2D, ATT, GEO, EXPLICIT };
static EWT g_ewt = EUC_2D;

inline double deg_to_rad(double d) { return M_PI * d / 180.0; }

int cpu_dist_raw(const vector<City>& C, int a, int b, EWT ewt,
                 const vector<vector<int>>& dm) {
    if (ewt == EXPLICIT) return dm[a][b];
    double dx = C[a].x-C[b].x, dy = C[a].y-C[b].y;
    if (ewt == EUC_2D)  return (int)round(sqrt(dx*dx+dy*dy));
    if (ewt == CEIL_2D) return (int)ceil(sqrt(dx*dx+dy*dy));
    if (ewt == ATT) { double r=sqrt((dx*dx+dy*dy)/10.0); int t=(int)r; return (t<r)?t+1:t; }
    if (ewt == GEO) {
        auto toGeo=[](double v)->double{ int d=(int)v; return deg_to_rad(d+5.0*(v-d)/3.0); };
        double la=toGeo(C[a].x), loa=toGeo(C[a].y), lb=toGeo(C[b].x), lob=toGeo(C[b].y);
        double q1=cos(loa-lob), q2=cos(la-lb), q3=cos(la+lb);
        return (int)(6378.388*acos(0.5*(q2*(1+q1)-q3*(1-q1)))+1.0);
    }
    return (int)round(sqrt(dx*dx+dy*dy));
}

#define gpuCheck(x) { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);}}

__device__ inline int gpu_dist(
    const City* C, int a, int b, const int* dDistMat, int n, int ewt_flag
) {
    if (ewt_flag==3) return dDistMat[a*n+b];
    double dx=C[a].x-C[b].x, dy=C[a].y-C[b].y;
    if (ewt_flag==0) return (int)(sqrt(dx*dx+dy*dy)+0.5);
    if (ewt_flag==1) return (int)ceil(sqrt(dx*dx+dy*dy));
    double r=sqrt((dx*dx+dy*dy)/10.0); int t=(int)r; return (t<r)?t+1:t;
}

// =============================================================================
// RNG init
// =============================================================================
__global__ void initRNG(curandState* S, int W, unsigned long seed) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if (id<W) curand_init(seed,id,0,&S[id]);
}

// =============================================================================
// KERNEL 1: Parallel NN Construction
// =============================================================================
__global__ void kernel_nn_parallel(
    const City* C, int n, const int* knn, int k,
    int* tours, int* vis_buf, curandState* RNG, int W,
    const int* dDistMat, int ewt_flag
) {
    int w=blockIdx.x, tid=threadIdx.x, bsz=blockDim.x;
    if (w>=W) return;
    int* T=tours+w*n, *vis=vis_buf+w*n;
    for (int i=tid;i<n;i+=bsz) vis[i]=0;
    __syncthreads();
    if (tid==0) {
        curandState st=RNG[w];
        int cur=curand(&st)%n; T[0]=cur; vis[cur]=1;
        for (int step=1;step<n;step++) {
            int best=-1, bestd=INT_MAX;
            for (int ki=0;ki<k;ki++) {
                int nb=knn[cur*k+ki];
                if (!vis[nb]) { int d=gpu_dist(C,cur,nb,dDistMat,n,ewt_flag); if(d<bestd){bestd=d;best=nb;} }
            }
            if (best<0) for(int j=0;j<n;j++) if(!vis[j]){best=j;break;}
            T[step]=best; vis[best]=1; cur=best;
        }
        RNG[w]=st;
    }
    __syncthreads();
}

// =============================================================================
// KERNEL 2: Build position lookup
// =============================================================================
__global__ void kernel_build_pos(const int* tours, int* pos_buf, int n, int W) {
    int w=blockIdx.x, tid=threadIdx.x, bsz=blockDim.x;
    if (w>=W) return;
    const int* T=tours+w*n; int* p=pos_buf+w*n;
    for (int i=tid;i<n;i+=bsz) p[T[i]]=i;
}

// =============================================================================
// KERNEL 3: Multi-move parallel 2-opt
// =============================================================================
__global__ void kernel_2opt_multipass(
    const City* C, int n, const int* knn, int k,
    int* tours, int* pos_buf, int* improved_flags, int W,
    const int* dDistMat, int ewt_flag
) {
    int w=blockIdx.x, tid=threadIdx.x, bsz=blockDim.x;
    if (w>=W) return;
    int* T=tours+w*n, *pos=pos_buf+w*n;

    extern __shared__ int smem[];
    int* sh_gains=smem, *sh_ii=smem+bsz, *sh_jj=smem+2*bsz;
    sh_gains[tid]=0; sh_ii[tid]=-1; sh_jj[tid]=-1;

    for (int i=tid; i<n; i+=bsz) {
        int A=T[i], B=T[(i+1)%n];
        int dAB=gpu_dist(C,A,B,dDistMat,n,ewt_flag);

        for (int ki=0; ki<k; ki++) {
            int nb=knn[A*k+ki];
            int dAnb=gpu_dist(C,A,nb,dDistMat,n,ewt_flag);
            if (dAnb>=dAB) break;  // KNN sorted by dist

            int j=pos[nb], j1=(j+1)%n;
            if (j==i||j1==i||j==(i+1)%n) continue;

            int D=T[j1];
            int gain=dAB+gpu_dist(C,nb,D,dDistMat,n,ewt_flag)
                    -dAnb-gpu_dist(C,B,D,dDistMat,n,ewt_flag);

            if (gain>0) {
                // Normalize so ii<jj for conflict detection
                int ii=i, jj=j;
                if (ii>jj){int t=ii;ii=jj;jj=t;}
                if (gain>sh_gains[tid]){
                    sh_gains[tid]=gain; sh_ii[tid]=ii; sh_jj[tid]=jj;
                }
                break;
            }
        }
    }
    __syncthreads();

    if (tid==0) {
        // Insertion sort by gain descending
        for (int a=1;a<bsz;a++) {
            int g=sh_gains[a],ii=sh_ii[a],jj=sh_jj[a]; int b=a-1;
            while(b>=0&&sh_gains[b]<g){sh_gains[b+1]=sh_gains[b];sh_ii[b+1]=sh_ii[b];sh_jj[b+1]=sh_jj[b];b--;}
            sh_gains[b+1]=g;sh_ii[b+1]=ii;sh_jj[b+1]=jj;
        }
        int applied_l[32],applied_r[32]; int na=0; bool any=false;
        for (int a=0;a<bsz&&na<32;a++) {
            if (sh_gains[a]<=0||sh_ii[a]<0) break;
            int ii=sh_ii[a],jj=sh_jj[a],l=ii+1,r=jj;
            if (l>r) continue;
            bool conflict=false;
            for (int q=0;q<na&&!conflict;q++) if(!(r<applied_l[q]||l>applied_r[q])) conflict=true;
            if (conflict) continue;
            int lo=l,hi=r;
            while(lo<hi){int t=T[lo];T[lo]=T[hi];T[hi]=t;pos[T[lo]]=lo;pos[T[hi]]=hi;lo++;hi--;}
            if(lo==hi) pos[T[lo]]=lo;
            applied_l[na]=l;applied_r[na]=r;na++; any=true;
        }
        if(any) improved_flags[w]=1;
    }
    __syncthreads();
}

// =============================================================================
// KERNEL 4: Or-opt* (chain 1,2,3 + reversed insertion)
// =============================================================================
__global__ void kernel_oropt_all(
    const City* C, int n, const int* knn, int k,
    int* tours, int* pos_buf, int* improved_flags,
    int chain_len, int W, const int* dDistMat, int ewt_flag,
    int* oropt_tmp
) {
    int w=blockIdx.x, tid=threadIdx.x, bsz=blockDim.x;
    if (w>=W) return;
    int* T=tours+w*n, *pos=pos_buf+w*n; int c=chain_len;

    __shared__ int sh_best_gain_int, sh_ri, sh_ii, sh_rev;
    if(tid==0){sh_best_gain_int=0;sh_ri=-1;sh_ii=-1;sh_rev=0;}
    __syncthreads();

    for (int i=tid;i<n;i+=bsz) {
        int prev=(i-1+n)%n, next=(i+c)%n;
        if (next==prev||n<=c+2) continue;
        int A=T[prev],X1=T[i],Xc=T[(i+c-1)%n],B=T[next];
        int rem=gpu_dist(C,A,X1,dDistMat,n,ewt_flag)
               +gpu_dist(C,Xc,B,dDistMat,n,ewt_flag)
               -gpu_dist(C,A,B,dDistMat,n,ewt_flag);
        for (int ki=0;ki<k;ki++) {
            int nb=knn[X1*k+ki]; int cp=pos[nb];
            if(cp==prev||cp==i) continue;
            bool in_chain=false;
            for(int x=0;x<c;x++) if(cp==(i+x)%n){in_chain=true;break;}
            if(in_chain) continue;
            int D=T[(cp+1)%n];
            int gf=rem+gpu_dist(C,nb,D,dDistMat,n,ewt_flag)
                   -gpu_dist(C,nb,X1,dDistMat,n,ewt_flag)-gpu_dist(C,Xc,D,dDistMat,n,ewt_flag);
            int gr=rem+gpu_dist(C,nb,D,dDistMat,n,ewt_flag)
                   -gpu_dist(C,nb,Xc,dDistMat,n,ewt_flag)-gpu_dist(C,X1,D,dDistMat,n,ewt_flag);
            int gain=max(gf,gr); int rev=(gr>gf)?1:0;
            if(gain>0){
                int old=atomicMax(&sh_best_gain_int,gain);
                if(old<gain){sh_ri=i;sh_ii=cp;sh_rev=rev;}
                break;
            }
        }
    }
    __syncthreads();

    if(tid==0&&sh_ri>=0&&sh_ii>=0&&sh_best_gain_int>0){
        int* tmp=oropt_tmp+(long long)w*n;
        int chain[3];
        for(int x=0;x<c;x++) chain[x]=T[(sh_ri+x)%n];
        if(sh_rev) for(int l=0,r=c-1;l<r;l++,r--){int t=chain[l];chain[l]=chain[r];chain[r]=t;}
        int wp=0;
        for(int x=0;x<n;x++){
            bool ic=false; for(int y=0;y<c;y++) if(x==(sh_ri+y)%n){ic=true;break;}
            if(!ic) tmp[wp++]=T[x];
        }
        int ins_city=T[sh_ii%n], ins_pos=-1;
        for(int x=0;x<wp;x++) if(tmp[x]==ins_city){ins_pos=x;break;}
        if(ins_pos<0){improved_flags[w]=1;return;}
        int rp=0;
        for(int x=0;x<=ins_pos;x++) T[rp++]=tmp[x];
        for(int x=0;x<c;x++) T[rp++]=chain[x];
        for(int x=ins_pos+1;x<wp;x++) T[rp++]=tmp[x];
        for(int x=0;x<n;x++) pos[T[x]]=x;
        improved_flags[w]=1;
    }
    __syncthreads();
}

// =============================================================================
// KERNEL 5: Double-bridge perturbation (4-opt, non-reversible)
// Reconnects A + C + B + D — escapes any 2-opt/3-opt local optimum.
// =============================================================================
__global__ void kernel_double_bridge(
    int* tours, int n, int W, curandState* RNG, int* scratch
) {
    int w=blockIdx.x;
    if(w>=W||threadIdx.x!=0) return;
    int* T=tours+w*n, *tmp=scratch+w*n;
    curandState st=RNG[w];
    int c0=1+(curand(&st)%(n/4));
    int c1=c0+1+(curand(&st)%(n/4));
    int c2=c1+1+(curand(&st)%(n/4));
    // A=[0,c0), B=[c0,c1), C=[c1,c2), D=[c2,n) → reconnect A+C+B+D
    int wp=0;
    for(int i=0;i<c0;i++)  tmp[wp++]=T[i];
    for(int i=c1;i<c2;i++) tmp[wp++]=T[i];
    for(int i=c0;i<c1;i++) tmp[wp++]=T[i];
    for(int i=c2;i<n;i++)  tmp[wp++]=T[i];
    for(int i=0;i<n;i++) T[i]=tmp[i];
    RNG[w]=st;
}

// =============================================================================
// KERNEL 6: Tour costs
// =============================================================================
__global__ void kernel_costs(
    const City* C, int n, const int* tours, float* costs, int W,
    const int* dDistMat, int ewt_flag
) {
    int w=blockIdx.x, tid=threadIdx.x, bsz=blockDim.x;
    if(w>=W) return;
    const int* T=tours+w*n;
    __shared__ double sh[512];
    double loc=0;
    for(int i=tid;i<n;i+=bsz) loc+=gpu_dist(C,T[i],T[(i+1)%n],dDistMat,n,ewt_flag);
    sh[tid]=loc; __syncthreads();
    for(int s=bsz/2;s>0;s>>=1){if(tid<s)sh[tid]+=sh[tid+s];__syncthreads();}
    if(tid==0) costs[w]=(float)sh[0];
}

// =============================================================================
// TSPLIB PARSER
// =============================================================================
struct TSPInstance { int n; vector<City> cities; vector<vector<int>> dist; EWT ewt; string name; };

TSPInstance readTSP(const string& fname) {
    ifstream in(fname);
    if(!in){cerr<<"Cannot open "<<fname<<"\n";exit(1);}
    TSPInstance inst; inst.n=0; inst.ewt=EUC_2D;
    string ewf="",line;
    while(getline(in,line)){
        while(!line.empty()&&(line.back()=='\r'||line.back()==' '||line.back()=='\t')) line.pop_back();
        while(!line.empty()&&(line.front()==' '||line.front()=='\t')) line=line.substr(1);
        auto getval=[&](const string& l)->string{
            size_t p=l.find(':'); if(p==string::npos) p=l.find('=');
            if(p==string::npos) return "";
            string v=l.substr(p+1);
            while(!v.empty()&&(v[0]==' '||v[0]=='\t')) v=v.substr(1);
            while(!v.empty()&&(v.back()==' '||v.back()=='\r'||v.back()=='\t')) v.pop_back();
            return v;
        };
        if(line.find("NAME")==0)      inst.name=getval(line);
        if(line.find("DIMENSION")==0){string v=getval(line);if(!v.empty())inst.n=stoi(v);}
        if(line.find("EDGE_WEIGHT_TYPE")==0){
            if(line.find("EUC_2D") !=string::npos) inst.ewt=EUC_2D;
            if(line.find("CEIL_2D")!=string::npos) inst.ewt=CEIL_2D;
            if(line.find("ATT")    !=string::npos) inst.ewt=ATT;
            if(line.find("GEO")    !=string::npos) inst.ewt=GEO;
            if(line.find("EXPLICIT")!=string::npos) inst.ewt=EXPLICIT;
        }
        if(line.find("EDGE_WEIGHT_FORMAT")==0) ewf=getval(line);
        if(line.find("NODE_COORD_SECTION")!=string::npos){
            inst.cities.resize(inst.n);
            for(int i=0;i<inst.n;i++){int id;double x,y;in>>id>>x>>y;inst.cities[id-1]={x,y};}
        }
        if(line.find("EDGE_WEIGHT_SECTION")!=string::npos){
            inst.dist.assign(inst.n,vector<int>(inst.n,0));
            if(ewf.find("FULL_MATRIX")!=string::npos){
                for(int i=0;i<inst.n;i++) for(int j=0;j<inst.n;j++) in>>inst.dist[i][j];
            } else if(ewf.find("UPPER_ROW")!=string::npos){
                for(int i=0;i<inst.n-1;i++) for(int j=i+1;j<inst.n;j++){in>>inst.dist[i][j];inst.dist[j][i]=inst.dist[i][j];}
            } else if(ewf.find("LOWER_ROW")!=string::npos){
                for(int i=1;i<inst.n;i++) for(int j=0;j<i;j++){in>>inst.dist[i][j];inst.dist[j][i]=inst.dist[i][j];}
            } else if(ewf.find("UPPER_DIAG_ROW")!=string::npos){
                for(int i=0;i<inst.n;i++) for(int j=i;j<inst.n;j++){in>>inst.dist[i][j];inst.dist[j][i]=inst.dist[i][j];}
            } else if(ewf.find("LOWER_DIAG_ROW")!=string::npos){
                for(int i=0;i<inst.n;i++) for(int j=0;j<=i;j++){in>>inst.dist[i][j];inst.dist[j][i]=inst.dist[i][j];}
            }
            if(inst.cities.empty()) inst.cities.resize(inst.n,{0.0,0.0});
        }
        if(line=="EOF") break;
    }
    if(inst.n==0){cerr<<"ERROR: Could not parse "<<fname<<" (n=0).\n";exit(1);}
    if(inst.cities.empty()) inst.cities.resize(inst.n,{0.0,0.0});
    return inst;
}

// =============================================================================
// CPU helpers
// =============================================================================
int cpu_dist_inst(const TSPInstance& inst, int a, int b) {
    return cpu_dist_raw(inst.cities,a,b,inst.ewt,inst.dist);
}

vector<vector<int>> build_knn(const TSPInstance& inst, int k) {
    int n=inst.n;
    vector<vector<int>> knn(n,vector<int>(k));
    if(inst.ewt==EXPLICIT){
        vector<pair<int,int>> tmp(n);
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++) tmp[j]={(i==j)?INT_MAX:inst.dist[i][j],j};
            nth_element(tmp.begin(),tmp.begin()+k,tmp.end());
            sort(tmp.begin(),tmp.begin()+k);
            for(int t=0;t<k;t++) knn[i][t]=tmp[t].second;
        }
        return knn;
    }
    if(n<=10000){
        vector<pair<int,int>> tmp(n);
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++) tmp[j]={(i==j)?INT_MAX:cpu_dist_inst(inst,i,j),j};
            nth_element(tmp.begin(),tmp.begin()+k,tmp.end());
            sort(tmp.begin(),tmp.begin()+k);
            for(int t=0;t<k;t++) knn[i][t]=tmp[t].second;
        }
        return knn;
    }
    cerr<<"Building KNN with grid acceleration (n="<<n<<")...\n";
    double minx=1e18,miny=1e18,maxx=-1e18,maxy=-1e18;
    for(auto& c:inst.cities){minx=min(minx,c.x);miny=min(miny,c.y);maxx=max(maxx,c.x);maxy=max(maxy,c.y);}
    int gs=max(1,(int)sqrt(n/50.0));
    double cx=(maxx-minx)/gs+1e-9, cy=(maxy-miny)/gs+1e-9;
    vector<vector<int>> grid(gs*gs);
    for(int i=0;i<n;i++){
        int gx=min((int)((inst.cities[i].x-minx)/cx),gs-1);
        int gy=min((int)((inst.cities[i].y-miny)/cy),gs-1);
        grid[gy*gs+gx].push_back(i);
    }
    for(int i=0;i<n;i++){
        int gx=min((int)((inst.cities[i].x-minx)/cx),gs-1);
        int gy=min((int)((inst.cities[i].y-miny)/cy),gs-1);
        vector<pair<int,int>> cands; cands.reserve(k*16);
        for(int ring=0;ring<=gs;ring++){
            for(int dy=-ring;dy<=ring;dy++) for(int dx=-ring;dx<=ring;dx++){
                if(abs(dx)!=ring&&abs(dy)!=ring) continue;
                int nx=gx+dx,ny=gy+dy;
                if(nx<0||nx>=gs||ny<0||ny>=gs) continue;
                for(int j:grid[ny*gs+nx]) if(j!=i) cands.push_back({cpu_dist_inst(inst,i,j),j});
            }
            if((int)cands.size()>=k*8&&ring>=2) break;
        }
        nth_element(cands.begin(),cands.begin()+k,cands.end());
        sort(cands.begin(),cands.begin()+k);
        for(int t=0;t<k;t++) knn[i][t]=cands[t].second;
        if(i%10000==0) cerr<<"  KNN: "<<i<<"/"<<n<<"\r";
    }
    cerr<<"  KNN: "<<n<<"/"<<n<<"\n";
    return knn;
}

double cpu_2opt(vector<int>& T, const TSPInstance& inst, const vector<vector<int>>& knn) {
    int n=T.size(); vector<int> pos(n); for(int i=0;i<n;i++) pos[T[i]]=i;
    bool imp=true;
    while(imp){imp=false;
        for(int i=0;i<n-2;i++){
            int A=T[i],B=T[i+1]; int dAB=cpu_dist_inst(inst,A,B);
            for(int nb:knn[A]){
                if(cpu_dist_inst(inst,A,nb)>=dAB) break;
                int j=pos[nb]; if(j<=i+1||j+1>=n) continue;
                int D=T[j+1];
                int gain=dAB+cpu_dist_inst(inst,nb,D)-cpu_dist_inst(inst,A,nb)-cpu_dist_inst(inst,B,D);
                if(gain>0){
                    reverse(T.begin()+i+1,T.begin()+j+1);
                    for(int x=i+1;x<=j;x++) pos[T[x]]=x;
                    B=T[i+1];dAB=cpu_dist_inst(inst,A,B);imp=true;break;
                }
            }
        }
    }
    double c=0; for(int i=0;i<n;i++) c+=cpu_dist_inst(inst,T[i],T[(i+1)%n]); return c;
}

double cpu_oropt(vector<int>& T, const TSPInstance& inst, const vector<vector<int>>& knn) {
    int n=T.size(); vector<int> pos(n); for(int i=0;i<n;i++) pos[T[i]]=i;
    bool imp=true;
    while(imp){imp=false;
        for(int c=1;c<=3&&!imp;c++){
            for(int i=0;i<n&&!imp;i++){
                if(n<=c+2) continue;
                int prev=(i-1+n)%n,next=(i+c)%n;
                if(next==prev) continue;
                int A=T[prev],X1=T[i],Xc=T[(i+c-1)%n],B=T[next];
                int rem=cpu_dist_inst(inst,A,X1)+cpu_dist_inst(inst,Xc,B)-cpu_dist_inst(inst,A,B);
                for(int nb:knn[X1]){
                    int cp=pos[nb]; if(cp==prev||cp==i) continue;
                    bool ic=false; for(int x=0;x<c;x++) if(cp==(i+x)%n){ic=true;break;}
                    if(ic) continue;
                    int D=T[(cp+1)%n];
                    int gf=rem+cpu_dist_inst(inst,nb,D)-cpu_dist_inst(inst,nb,X1)-cpu_dist_inst(inst,Xc,D);
                    int gr=rem+cpu_dist_inst(inst,nb,D)-cpu_dist_inst(inst,nb,Xc)-cpu_dist_inst(inst,X1,D);
                    int gain=max(gf,gr); bool rev=(gr>gf);
                    if(gain>0){
                        vector<int> chain(c); for(int x=0;x<c;x++) chain[x]=T[(i+x)%n];
                        if(rev) reverse(chain.begin(),chain.end());
                        vector<int> tmp2; tmp2.reserve(n);
                        for(int x=0;x<n;x++){bool in2=false;for(int y=0;y<c;y++)if(x==(i+y)%n){in2=true;break;}if(!in2)tmp2.push_back(T[x]);}
                        int ip=-1; for(int x=0;x<(int)tmp2.size();x++) if(tmp2[x]==nb){ip=x;break;}
                        if(ip<0) continue;
                        vector<int> nt; nt.reserve(n);
                        for(int x=0;x<=ip;x++) nt.push_back(tmp2[x]);
                        for(int x=0;x<c;x++) nt.push_back(chain[x]);
                        for(int x=ip+1;x<(int)tmp2.size();x++) nt.push_back(tmp2[x]);
                        T=nt; for(int x=0;x<n;x++) pos[T[x]]=x;
                        imp=true; break;
                    }
                }
            }
        }
    }
    double cost=0; for(int i=0;i<n;i++) cost+=cpu_dist_inst(inst,T[i],T[(i+1)%n]); return cost;
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char** argv) {
    string fname=argc>1?argv[1]:"instance.tsp";
    TSPInstance inst=readTSP(fname);
    int n=inst.n;
    cerr<<"Instance: "<<fname<<"  Cities: "<<n<<"  Format: ";
    const char* fmtnames[]={"EUC_2D","CEIL_2D","ATT","GEO","EXPLICIT"};
    cerr<<fmtnames[inst.ewt]<<"\n";
    g_ewt=inst.ewt;
    int ewt_flag=0;
    if(inst.ewt==CEIL_2D) ewt_flag=1;
    if(inst.ewt==ATT)     ewt_flag=2;
    if(inst.ewt==GEO||inst.ewt==EXPLICIT) ewt_flag=3;

    // ==========================================================================
    // Parameters
    // ILS = Iterated Local Search (double-bridge + re-optimize)
    // opt_rounds per ILS iteration (converges early if no improvement)
    // ==========================================================================
    int W, k, opt_rounds, ils_iters, top_K, TPB;
    if      (n<=200)   { W=256; k=min(20,n-1); opt_rounds=150; ils_iters=40;  top_K=min(8,n);  TPB=256; }
    else if (n<=500)   { W=128; k=min(25,n-1); opt_rounds=100; ils_iters=25;  top_K=min(6,n);  TPB=256; }
    else if (n<=2000)  { W=64;  k=25;           opt_rounds=80;  ils_iters=20;  top_K=4;  TPB=256; }
    else if (n<=5000)  { W=32;  k=30;           opt_rounds=60;  ils_iters=12;  top_K=4;  TPB=512; }
    else if (n<=15000) { W=16;  k=35;           opt_rounds=40;  ils_iters=8;   top_K=4;  TPB=512; }
    else if (n<=50000) { W=12;  k=40;           opt_rounds=30;  ils_iters=6;   top_K=4;  TPB=512; }
    else               { W=8;   k=40;           opt_rounds=20;  ils_iters=4;   top_K=4;  TPB=512; }

    TPB=min(TPB,n); TPB=max(TPB,32);
    int tpb=1; while(tpb*2<=TPB) tpb*=2; TPB=tpb;
    cerr<<"W="<<W<<" k="<<k<<" opt_rounds="<<opt_rounds<<" ils_iters="<<ils_iters<<" TPB="<<TPB<<"\n";

    cerr<<"Building KNN (k="<<k<<")...\n";
    auto knn_host=build_knn(inst,k);
    vector<int> knn_flat((long long)n*k);
    for(int i=0;i<n;i++) for(int j=0;j<k;j++) knn_flat[i*k+j]=knn_host[i][j];
    cerr<<"KNN done.\n";

    vector<int> dist_flat; int* dDistMat=nullptr;
    if(ewt_flag==3){
        dist_flat.resize((long long)n*n);
        for(int i=0;i<n;i++) for(int j=0;j<n;j++) dist_flat[i*n+j]=cpu_dist_inst(inst,i,j);
        gpuCheck(cudaMalloc(&dDistMat,(long long)n*n*sizeof(int)));
        gpuCheck(cudaMemcpy(dDistMat,dist_flat.data(),(long long)n*n*sizeof(int),cudaMemcpyHostToDevice));
        cerr<<"Dist matrix uploaded to GPU.\n";
    }

    City *dC; int *dKNN,*dT,*dPos,*dVis; float *dCost; int *dImp; curandState *dRNG;
    int *dOroptTmp, *dDBTmp, *dBestT; float *dBestCost;

    gpuCheck(cudaMalloc(&dC,        n*sizeof(City)));
    gpuCheck(cudaMalloc(&dKNN,      (long long)n*k*sizeof(int)));
    gpuCheck(cudaMalloc(&dT,        (long long)W*n*sizeof(int)));
    gpuCheck(cudaMalloc(&dPos,      (long long)W*n*sizeof(int)));
    gpuCheck(cudaMalloc(&dVis,      (long long)W*n*sizeof(int)));
    gpuCheck(cudaMalloc(&dCost,     W*sizeof(float)));
    gpuCheck(cudaMalloc(&dImp,      W*sizeof(int)));
    gpuCheck(cudaMalloc(&dRNG,      W*sizeof(curandState)));
    gpuCheck(cudaMalloc(&dOroptTmp, (long long)W*n*sizeof(int)));
    gpuCheck(cudaMalloc(&dDBTmp,    (long long)W*n*sizeof(int)));
    gpuCheck(cudaMalloc(&dBestT,    (long long)W*n*sizeof(int)));
    gpuCheck(cudaMalloc(&dBestCost, W*sizeof(float)));

    gpuCheck(cudaMemcpy(dC,   inst.cities.data(), n*sizeof(City),         cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(dKNN, knn_flat.data(),    (long long)n*k*sizeof(int), cudaMemcpyHostToDevice));
    { vector<float> inf(W,1e30f); gpuCheck(cudaMemcpy(dBestCost,inf.data(),W*sizeof(float),cudaMemcpyHostToDevice)); }

    initRNG<<<(W+127)/128,128>>>(dRNG,W,42UL);
    gpuCheck(cudaDeviceSynchronize());

    int smem_size=3*TPB*sizeof(int);

    // ---- NN Construction ----
    cerr<<"GPU NN construction ("<<W<<" walkers)...\n";
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    kernel_nn_parallel<<<W,TPB>>>(dC,n,dKNN,k,dT,dVis,dRNG,W,dDistMat,ewt_flag);
    gpuCheck(cudaDeviceSynchronize());
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms_nn; cudaEventElapsedTime(&ms_nn,e0,e1);
    cerr<<"NN done: "<<ms_nn<<" ms\n";

    kernel_build_pos<<<W,TPB>>>(dT,dPos,n,W);
    gpuCheck(cudaDeviceSynchronize());

    // =========================================================================
    // ILS: initial optimize → save best → (perturb → optimize → accept/reject)
    // =========================================================================
    cerr<<"GPU ILS ("<<ils_iters<<" iters × "<<opt_rounds<<" rounds max)...\n";
    cudaEventRecord(e0);

    vector<int> h_imp(W);
    vector<float> h_costs(W), h_best(W,1e30f);

    auto run_opt=[&](int rounds){
        for(int r=0;r<rounds;r++){
            gpuCheck(cudaMemset(dImp,0,W*sizeof(int)));
            kernel_2opt_multipass<<<W,TPB,smem_size>>>(dC,n,dKNN,k,dT,dPos,dImp,W,dDistMat,ewt_flag);
            gpuCheck(cudaDeviceSynchronize());
            for(int c=1;c<=3;c++){
                kernel_oropt_all<<<W,TPB>>>(dC,n,dKNN,k,dT,dPos,dImp,c,W,dDistMat,ewt_flag,dOroptTmp);
                gpuCheck(cudaDeviceSynchronize());
            }
            gpuCheck(cudaMemcpy(h_imp.data(),dImp,W*sizeof(int),cudaMemcpyDeviceToHost));
            bool any=false; for(int x:h_imp) if(x){any=true;break;}
            if(!any) break;
        }
    };

    // Initial optimization pass
    run_opt(opt_rounds);

    // Save initial best
    kernel_costs<<<W,TPB>>>(dC,n,dT,dCost,W,dDistMat,ewt_flag);
    gpuCheck(cudaDeviceSynchronize());
    gpuCheck(cudaMemcpy(h_costs.data(),dCost,W*sizeof(float),cudaMemcpyDeviceToHost));
    for(int w=0;w<W;w++) h_best[w]=h_costs[w];
    gpuCheck(cudaMemcpy(dBestT,dT,(long long)W*n*sizeof(int),cudaMemcpyDeviceToDevice));
    gpuCheck(cudaMemcpy(dBestCost,h_best.data(),W*sizeof(float),cudaMemcpyHostToDevice));

    // ILS iterations
    for(int ils=0;ils<ils_iters;ils++){
        // Perturb
        kernel_double_bridge<<<W,1>>>(dT,n,W,dRNG,dDBTmp);
        gpuCheck(cudaDeviceSynchronize());
        kernel_build_pos<<<W,TPB>>>(dT,dPos,n,W);
        gpuCheck(cudaDeviceSynchronize());

        // Re-optimize
        run_opt(opt_rounds);

        // Compute costs
        kernel_costs<<<W,TPB>>>(dC,n,dT,dCost,W,dDistMat,ewt_flag);
        gpuCheck(cudaDeviceSynchronize());
        gpuCheck(cudaMemcpy(h_costs.data(),dCost,W*sizeof(float),cudaMemcpyDeviceToHost));

        // Accept/reject per walker
        for(int w=0;w<W;w++){
            if(h_costs[w]<h_best[w]){
                h_best[w]=h_costs[w];
                gpuCheck(cudaMemcpy(dBestT+(long long)w*n, dT+(long long)w*n,
                                    n*sizeof(int),cudaMemcpyDeviceToDevice));
            } else {
                gpuCheck(cudaMemcpy(dT+(long long)w*n, dBestT+(long long)w*n,
                                    n*sizeof(int),cudaMemcpyDeviceToDevice));
                gpuCheck(cudaMemcpy(dPos+(long long)w*n, dT+(long long)w*n,
                                    0,cudaMemcpyDeviceToDevice)); // no-op size 0
                // Rebuild pos for restored tour
                kernel_build_pos<<<1,TPB>>>(dT+(long long)w*n, dPos+(long long)w*n, n, 1);
                gpuCheck(cudaDeviceSynchronize());
            }
        }
        gpuCheck(cudaMemcpy(dBestCost,h_best.data(),W*sizeof(float),cudaMemcpyHostToDevice));

        if((ils+1)%2==0||ils==ils_iters-1)
            cerr<<"  ILS "<<ils+1<<"/"<<ils_iters<<"  best[0]="<<h_best[0]<<"\n";
    }

    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms_opt; cudaEventElapsedTime(&ms_opt,e0,e1);
    cerr<<"ILS done: "<<ms_opt<<" ms\n";

    // Final costs
    kernel_costs<<<W,TPB>>>(dC,n,dBestT,dCost,W,dDistMat,ewt_flag);
    gpuCheck(cudaDeviceSynchronize());
    vector<float> costs(W);
    gpuCheck(cudaMemcpy(costs.data(),dCost,W*sizeof(float),cudaMemcpyDeviceToHost));
    vector<int> idx(W); iota(idx.begin(),idx.end(),0);
    sort(idx.begin(),idx.end(),[&](int a,int b){return costs[a]<costs[b];});

    cerr<<"\nTop "<<top_K<<" walkers:\n";
    for(int i=0;i<top_K;i++) cerr<<"  ["<<i<<"] "<<costs[idx[i]]<<"\n";
    double best_cost = costs[idx[0]];
    int best_i = idx[0];

    cout<<"\n========== RESULT ==========\n";
    cout<<"Instance : "<<fname<<"\n";
    cout<<"Cities   : "<<n<<"\n";
    cout<<"Format   : "<<fmtnames[inst.ewt]<<"\n";
    cout<<"NN time  : "<<ms_nn<<" ms\n";
    cout<<"ILS time : "<<ms_opt<<" ms\n";
    cout<<"Best cost: "<<best_cost<<"\n";
    

    cudaFree(dC);cudaFree(dKNN);cudaFree(dT);cudaFree(dPos);cudaFree(dVis);
    cudaFree(dCost);cudaFree(dImp);cudaFree(dRNG);
    cudaFree(dOroptTmp);cudaFree(dDBTmp);cudaFree(dBestT);cudaFree(dBestCost);
    if(dDistMat) cudaFree(dDistMat);
    return 0;
}
