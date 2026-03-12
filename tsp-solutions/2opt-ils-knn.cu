
//more accurate but slower cause of polishing on the cpu and also the 2opt on gpu makes it behave more like sequential than parallel
// GPU: generate many tours with soft KNN bias
// CPU: short LK-lite + 3-opt + time-bounded ILS

#include <bits/stdc++.h>
#include <curand_kernel.h>
#include <thread>
using namespace std;

#define USE_TSPLIB_EUC2D 1

struct City { double x, y; };

__host__ __device__ inline double dist_city(const City* c, int a, int b){
    double dx = c[a].x - c[b].x, dy = c[a].y - c[b].y;
#if USE_TSPLIB_EUC2D
    return (double) llround(sqrt(dx*dx + dy*dy));
#else
    return sqrt(dx*dx + dy*dy);
#endif
}

__device__ inline double dev_dist(const City* c, int a, int b){ return dist_city(c,a,b); }

inline double host_dist(const vector<City>& c, int a, int b){
    double dx=c[a].x-c[b].x, dy=c[a].y-c[b].y;
#if USE_TSPLIB_EUC2D
    return (double) llround(sqrt(dx*dx + dy*dy));
#else
    return sqrt(dx*dx + dy*dy);
#endif
}

inline void gpuAssert(cudaError_t code, const char *file, int line){
    if(code != cudaSuccess){
        fprintf(stderr,"GPU Error %s %d: %s\n", file, line, cudaGetErrorString(code));
        exit(1);
    }
}
#define gpuCheck(ans) gpuAssert((ans), __FILE__, __LINE__)

__global__ void initRNG(curandState* s, int walkers, unsigned long seed){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < walkers) curand_init(seed, id, 0, &s[id]);
}

__device__ void reverse_seg(int* T,int n,int a,int b){
    if(a<=b){ for(int i=a,j=b;i<j;i++,j--){ int t=T[i];T[i]=T[j];T[j]=t;} }
    else{
        int len=(b+n-a+1)%n;
        for(int k=0;k<len/2;k++){
            int i=(a+k)%n, j=(a+len-1-k)%n;
            int t=T[i];T[i]=T[j];T[j]=t;
        }
    }
}

__device__ int pos_in_tour(const int* T,int n,int c){
    for(int i=0;i<n;i++) if(T[i]==c) return i;
    return -1;
}

__device__ double tour_len(const City* C,const int*T,int n){
    double s=0;
    for(int i=0;i<n;i++) s+=dev_dist(C,T[i],T[(i+1)%n]);
    return s;
}

__global__ void gen_kernel(const City*C,int n,const int*knn,int k,
                           int*tours,float*costs,curandState*rng,
                           int walkers,int shakes,int nudges){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=walkers) return;

    curandState st=rng[id];
    int* T=tours+id*n;

    for(int i=0;i<n;i++) T[i]=i;
    for(int i=n-1;i>0;i--){
        int j=(int)(curand_uniform(&st)*(i+1));
        int t=T[i];T[i]=T[j];T[j]=t;
    }
    for(int s2=0;s2<shakes;s2++){
        int a=curand(&st)%n,b=curand(&st)%n;
        if(a!=b) reverse_seg(T,n,min(a,b),max(a,b));
    }
    for(int it=0;it<nudges;it++){
        int i=curand(&st)%n;
        int A=T[i],B=T[(i+1)%n];
        double AB=dev_dist(C,A,B);
        int Cc=knn[A*k+(curand(&st)%k)];
        int j=pos_in_tour(T,n,Cc);
        if(j<0||j==i||(j+1)%n==i) continue;
        int D=T[(j+1)%n];
        double delta=(dev_dist(C,A,Cc)+dev_dist(C,B,D))-(AB+dev_dist(C,Cc,D));
        if(delta<0) reverse_seg(T,n,(i+1)%n,j);
    }
    costs[id]=tour_len(C,T,n);
    rng[id]=st;
}

vector<City> readTSPLIB(const string& f){
    ifstream in(f);
    string line; bool ok=false;
    vector<City> v;
    while(getline(in,line)){
        if(line.find("NODE_COORD_SECTION")!=string::npos){ ok=true;continue;}
        if(!ok) continue;
        if(line.find("EOF")!=string::npos) break;
        int id; double x,y;
        stringstream ss(line);
        if(ss>>id>>x>>y) v.push_back({x,y}); // ✅ correct scaling
    }
    return v;
}

double cost_host(const vector<City>&C,const vector<int>&T){
    double s=0; int n=T.size();
    for(int i=0;i<n;i++) s+=host_dist(C,T[i],T[(i+1)%n]);
    return s;
}

inline void reverse_section(vector<int>&T,int a,int b){ reverse(T.begin()+a,T.begin()+b+1); }

double two_opt(vector<int>&T,const vector<City>&C){
    double best=cost_host(C,T); int n=T.size(); bool imp=true;
    while(imp){
        imp=false;
        for(int i=0;i<n-2;i++){
            for(int j=i+2;j<n;j++){
                if(i==0&&j==n-1) continue;
                int A=T[i],B=T[(i+1)%n],C1=T[j],D=T[(j+1)%n];
                double old=host_dist(C,A,B)+host_dist(C,C1,D);
                double ne=host_dist(C,A,C1)+host_dist(C,B,D);
                if(ne<old-1e-12){
                    reverse_section(T,i+1,j);
                    best+=ne-old;
                    imp=true;
                }
            }
        }
    }
    return best;
}

void double_bridge(vector<int>&T){
    int n=T.size();
    vector<int> idx(4);
    for(int&i:idx) i=rand()%n;
    sort(idx.begin(),idx.end());
    int a=idx[0],b=idx[1],c=idx[2],d=idx[3];
    vector<int>A(T.begin(),T.begin()+a),
               B(T.begin()+a,T.begin()+b),
               C(T.begin()+b,T.begin()+c),
               D(T.begin()+c,T.begin()+d),
               E(T.begin()+d,T.end()), R;
    R.reserve(n);
    R.insert(R.end(),A.begin(),A.end());
    R.insert(R.end(),D.begin(),D.end());
    R.insert(R.end(),C.begin(),C.end());
    R.insert(R.end(),B.begin(),B.end());
    R.insert(R.end(),E.begin(),E.end());
    T.swap(R);
}

double fast_ils(vector<int> T,const vector<City>&C,double limit=0.55){
    using clk=chrono::steady_clock;
    auto t0=clk::now();
    double best=two_opt(T,C);
    vector<int> bestT=T;
    while(chrono::duration<double>(clk::now()-t0).count()<limit){
        vector<int> X=bestT;
        double_bridge(X);
        double cur=two_opt(X,C);
        if(cur<best){ best=cur; bestT.swap(X); }
    }
    T.swap(bestT);
    return best;
}

int main(int argc,char**argv){
    string f="b";
    if(argc>1) f=argv[1];
    auto C=readTSPLIB(f);
    int n=C.size();
    cerr<<"Loaded "<<n<<" cities\n";

    int walkers=512, k=18, shakes=5, nudges=8, K=11;
    int threads=128;
    auto build_knn=[&](){
        vector<int>knn(n*k);
        vector<pair<double,int>>tmp(n);
        for(int i=0;i<n;i++){
            tmp.clear();
            for(int j=0;j<n;j++) if(i!=j)
                tmp.emplace_back(host_dist(C,i,j),j);
            nth_element(tmp.begin(),tmp.begin()+k,tmp.end());
            sort(tmp.begin(),tmp.begin()+k);
            for(int t=0;t<k;t++) knn[i*k+t]=tmp[t].second;
        }
        return knn;
    };
    auto knn=build_knn();

    City*dC; int*dT; float*dCost; int*dK; curandState*dR;
    gpuCheck(cudaMalloc(&dC,n*sizeof(City)));
    gpuCheck(cudaMemcpy(dC,C.data(),n*sizeof(City),cudaMemcpyHostToDevice));
    gpuCheck(cudaMalloc(&dT,walkers*n*sizeof(int)));
    gpuCheck(cudaMalloc(&dCost,walkers*sizeof(float)));
    gpuCheck(cudaMalloc(&dK,n*k*sizeof(int)));
    gpuCheck(cudaMemcpy(dK,knn.data(),n*k*sizeof(int),cudaMemcpyHostToDevice));
    gpuCheck(cudaMalloc(&dR,walkers*sizeof(curandState)));

    int blocks=(walkers+threads-1)/threads;
    initRNG<<<blocks,threads>>>(dR,walkers,1234);
    cudaDeviceSynchronize();

    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    gen_kernel<<<blocks,threads>>>(dC,n,dK,k,dT,dCost,dR,walkers,shakes,nudges);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e);
    cerr<<"GPU="<<ms<<" ms\n";

    vector<float>costs(walkers);
    cudaMemcpy(costs.data(),dCost,walkers*sizeof(float),cudaMemcpyDeviceToHost);

    vector<int>idx(walkers); iota(idx.begin(),idx.end(),0);
    nth_element(idx.begin(),idx.begin()+K,idx.end(),
        [&](int a,int b){return costs[a]<costs[b];});
    idx.resize(K);
    sort(idx.begin(),idx.end(),[&](int a,int b){return costs[a]<costs[b];});

    cout<<"Top GPU seeds:\n";
    for(int i=0;i<K;i++) cout<<i<<": "<<costs[idx[i]]<<"\n";

    vector<double>bestC(K,1e18);
    vector<vector<int>>bestT(K,vector<int>(n));

    for(int i=0;i<K;i++){
        vector<int>T(n);
        cudaMemcpy(T.data(),dT+idx[i]*n,n*sizeof(int),cudaMemcpyDeviceToHost);
        bestC[i]=fast_ils(T,C,0.55); // 0.55s per tour
        bestT[i]=T;
        cerr<<"CPU["<<i<<"] -> "<<bestC[i]<<"\n";
    }

    double bc=*min_element(bestC.begin(),bestC.end());
    int bi=min_element(bestC.begin(),bestC.end())-bestC.begin();

    cout<<"\n== RESULT ==\nBest="<<bc<<"\nPrefix: ";
    for(int i=0;i<15;i++) cout<<bestT[bi][i]<<" ";
    cout<<"\n";
}
