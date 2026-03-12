
// PIPELINE:
//   1. CPU : Build KNN candidate list
//   2. GPU : NN construction  (W walkers, each from different start city)
//   3. GPU : TPN 2-opt rounds (parallel, one thread per pair)
//   4. GPU : Or-opt-1/2/3     (parallel, one thread per city)
//      Steps 3+4 alternate until no improvement
//   5. CPU : KNN 2-opt polish on top-K tours (fast, no wrap-around)
//
// WHY OR-OPT HELPS:
//   2-opt fixes "crossing" edges. Or-opt fixes "misplaced" cities/chains.
//   Together they escape local minima that either alone cannot.
//   Or-opt-1: relocate 1 city   → O(n*k) per pass
//   Or-opt-2: relocate 2 cities → O(n*k) per pass
//   Or-opt-3: relocate 3 cities → O(n*k) per pass


#include <bits/stdc++.h>
#include <curand_kernel.h>
using namespace std;

// --------------------------------------------------------------------------
// City + EUC_2D distance
// --------------------------------------------------------------------------
struct City { double x, y; };

__host__ __device__ inline double euc2d(const City* C, int a, int b) {
    double dx = C[a].x-C[b].x, dy = C[a].y-C[b].y;
    return (double)llround(sqrt(dx*dx+dy*dy));
}
inline double heuc(const vector<City>& C, int a, int b) {
    double dx = C[a].x-C[b].x, dy = C[a].y-C[b].y;
    return (double)llround(sqrt(dx*dx+dy*dy));
}

#define gpuCheck(x) { cudaError_t _e=(x); \
    if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n", \
    __FILE__,__LINE__,cudaGetErrorString(_e));exit(1);}}

// --------------------------------------------------------------------------
// RNG init
// --------------------------------------------------------------------------
__global__ void initRNG(curandState* S, int W, unsigned long seed) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < W) curand_init(seed, id, 0, &S[id]);
}

// --------------------------------------------------------------------------
// KERNEL 1: NN Construction
// Each walker builds its own greedy nearest-neighbor tour from a random start.
// vis[] stored in global memory (vis_buf) — supports any n, no stack overflow.
// vis_buf layout: [W * n] bytes, walker w uses vis_buf + w*n
// --------------------------------------------------------------------------
__global__ void kernel_nn_construct(
    const City* C, int n,
    const int* knn, int k,
    int* tours,
    int* vis_buf,      // [W * n] global scratch for visited flags
    curandState* RNG, int W
) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id >= W) return;
    curandState st = RNG[id];
    int* T   = tours   + id*n;
    int* vis = vis_buf + id*n;  // each walker gets its own vis[] slice

    for (int i = 0; i < n; i++) vis[i] = 0;

    int cur = curand(&st) % n;
    T[0] = cur; vis[cur] = 1;

    for (int step = 1; step < n; step++) {
        int best = -1; double bestd = 1e18;
        // Fast path: check KNN candidates first
        for (int ki = 0; ki < k; ki++) {
            int nb = knn[cur*k+ki];
            if (!vis[nb]) {
                double d = euc2d(C, cur, nb);
                if (d < bestd) { bestd=d; best=nb; }
            }
        }
        // Fallback: full scan (only when all KNN are visited — rare)
        if (best < 0) {
            for (int j = 0; j < n; j++) {
                if (!vis[j]) {
                    double d = euc2d(C, cur, j);
                    if (d < bestd) { bestd=d; best=j; }
                }
            }
        }
        T[step]=best; vis[best]=1; cur=best;
    }
    RNG[id] = st;
}

// --------------------------------------------------------------------------
// KERNEL 2: TPN 2-opt (one thread per pair, for ONE walker)
// Same as main_pihc.cu — finds best improving 2-opt move atomically.
// --------------------------------------------------------------------------
__device__ void linear_to_ij(int id, int n, int* oi, int* oj) {
    int i = n-2-(int)floor((sqrt((double)(8LL*(n*(n-1)/2-id-1)+1))-1.0)/2.0);
    int j = id - i*(n-1) + i*(i+1)/2 + 1;
    *oi=i; *oj=j;
}

__global__ void kernel_2opt_round(
    const City* C, int n,
    int* tour,
    int* best_i, int* best_j,
    double* best_gain,
    int pairs
) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id >= pairs) return;
    int i, j; linear_to_ij(id, n, &i, &j);
    if (i<0||j>=n||i>=j) return;

    int A=tour[i], B=tour[i+1], Ci=tour[j], D=tour[(j+1)%n];
    double gain = euc2d(C,A,B)+euc2d(C,Ci,D)
                - euc2d(C,A,Ci)-euc2d(C,B,D);

    if (gain > 1e-10) {
        unsigned long long* bg = (unsigned long long*)best_gain;
        unsigned long long newv = __double_as_longlong(gain);
        unsigned long long old  = atomicMax(bg, newv);
        if (old < newv) { *best_i=i; *best_j=j; }
    }
}

// --------------------------------------------------------------------------
// KERNEL 3: Or-opt (parallel, one thread per city, for ONE walker)
//
// Thread i checks if moving the chain starting at position i
// (of length chain_len = 1, 2, or 3) to a better position improves the tour.
//
// Uses position lookup array pos[] for O(1) city lookups.
//
// For chain [X1, X2, ..., Xc] between predecessor A and successor B:
//   Removal gain  = d(A,X1) + d(Xc,B) - d(A,B)
//   Insertion gain at edge (C,D): d(C,D) - d(C,X1) - d(Xc,D)
//   Total gain = removal_gain + insertion_gain
//
// Reports best (pos_remove, pos_insert, gain) via atomics.
// --------------------------------------------------------------------------
__global__ void kernel_oropt_round(
    const City* C, int n,
    const int* knn, int k,
    const int* tour,
    const int* pos,       // pos[city] = index in tour
    int chain_len,        // 1, 2, or 3
    int* best_ri,         // best removal index
    int* best_ii,         // best insertion index
    double* best_gain     // best gain found
) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;

    // The chain to potentially relocate: tour[i], tour[i+1], ..., tour[i+c-1]
    int c = chain_len;

    // Predecessor and successor of the chain
    int prev  = (i - 1 + n) % n;
    int next  = (i + c) % n;

    int A  = tour[prev];   // city before chain
    int X1 = tour[i];      // first city in chain
    int Xc = tour[(i+c-1)%n]; // last city in chain
    int B  = tour[next];   // city after chain

    // Skip if wraps awkwardly for short tours
    if (n <= c + 2) return;

    // Gain from removing the chain from its current position
    double remove_gain = euc2d(C,A,X1) + euc2d(C,Xc,B) - euc2d(C,A,B);

    // Now try inserting the chain after each KNN neighbor of X1
    // This is the key: we only check K candidate insertion positions
    // instead of all n positions — makes it O(n*k) not O(n^2)
    for (int ki = 0; ki < k; ki++) {
        int C_city = knn[X1*k + ki];

        // Don't insert back into the same position
        int cp = pos[C_city];
        if (cp == prev || cp == i ||
            cp == (i+1)%n || cp == (i+c-1)%n) continue;

        int D_city = tour[(cp+1)%n];
        if (D_city == X1) continue;

        // Gain from inserting chain between C_city and D_city
        double insert_gain = euc2d(C,C_city,D_city)
                           - euc2d(C,C_city,X1)
                           - euc2d(C,Xc,D_city);

        double total_gain = remove_gain + insert_gain;

        if (total_gain > 1e-10) {
            // Atomically record best gain
            unsigned long long* bg  = (unsigned long long*)best_gain;
            unsigned long long  newv = __double_as_longlong(total_gain);
            unsigned long long  old  = atomicMax(bg, newv);
            if (old < newv) {
                *best_ri = i;   // where to remove from
                *best_ii = cp;  // where to insert after
            }
            break; // KNN is sorted by distance, so first improvement is good
        }
    }
}

// --------------------------------------------------------------------------
// CPU: apply 2-opt reversal (i+1 .. j), safe, no wrap
// --------------------------------------------------------------------------
void apply_2opt(vector<int>& T, int i, int j) {
    reverse(T.begin()+i+1, T.begin()+j+1);
}

// --------------------------------------------------------------------------
// CPU: apply Or-opt relocation
// Remove chain of length c starting at position ri,
// insert it after position ii in the (modified) tour.
// --------------------------------------------------------------------------
void apply_oropt(vector<int>& T, int ri, int ii, int c) {
    int n = T.size();
    // Extract the chain
    vector<int> chain;
    for (int x = 0; x < c; x++)
        chain.push_back(T[(ri+x)%n]);

    // Remove chain from tour
    vector<int> rest;
    rest.reserve(n-c);
    for (int x = 0; x < n; x++) {
        bool in_chain = false;
        for (int y = 0; y < c; y++)
            if (x == (ri+y)%n) { in_chain=true; break; }
        if (!in_chain) rest.push_back(T[x]);
    }

    // Find insertion point in rest[]
    // ii was an index in T before removal — find where that city is now in rest
    int ins_city = T[ii % n];
    int ins_pos  = -1;
    for (int x = 0; x < (int)rest.size(); x++)
        if (rest[x] == ins_city) { ins_pos=x; break; }
    if (ins_pos < 0) { T = rest; return; } // safety fallback

    // Insert chain after ins_pos
    vector<int> result;
    result.reserve(n);
    for (int x = 0; x <= ins_pos; x++) result.push_back(rest[x]);
    for (int x : chain)              result.push_back(x);
    for (int x = ins_pos+1; x < (int)rest.size(); x++) result.push_back(rest[x]);
    T = result;
}

// --------------------------------------------------------------------------
// CPU: KNN 2-opt polish (clean, no wrap-around)
// --------------------------------------------------------------------------
double cpu_2opt(vector<int>& T, const vector<City>& C,
                const vector<vector<int>>& knn) {
    int n = T.size();
    vector<int> pos(n);
    for (int i = 0; i < n; i++) pos[T[i]] = i;
    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 0; i < n-2; i++) {
            int A=T[i], B=T[i+1];
            double dAB = heuc(C,A,B);
            for (int nb : knn[A]) {
                if (heuc(C,A,nb) >= dAB) break;
                int j = pos[nb];
                if (j <= i+1 || j+1 >= n) continue;
                int D = T[j+1];
                double gain = dAB+heuc(C,nb,D)-heuc(C,A,nb)-heuc(C,B,D);
                if (gain > 1e-10) {
                    apply_2opt(T, i, j);
                    for (int x=i+1; x<=j; x++) pos[T[x]]=x;
                    B=T[i+1]; dAB=heuc(C,A,B);
                    improved=true; break;
                }
            }
        }
    }
    double cost=0;
    for (int i=0;i<n;i++) cost+=heuc(C,T[i],T[(i+1)%n]);
    return cost;
}

// --------------------------------------------------------------------------
// CPU: Or-opt polish (chain 1,2,3) — clean CPU version for final polish
// --------------------------------------------------------------------------
double cpu_oropt(vector<int>& T, const vector<City>& C,
                 const vector<vector<int>>& knn, int chain_len) {
    int n = T.size();
    vector<int> pos(n);
    for (int i = 0; i < n; i++) pos[T[i]] = i;
    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 0; i < n; i++) {
            int c  = chain_len;
            int prev = (i-1+n)%n;
            int next = (i+c)%n;
            if (next == prev) continue;
            int A  = T[prev], X1 = T[i];
            int Xc = T[(i+c-1)%n], B = T[next];
            double rem = heuc(C,A,X1)+heuc(C,Xc,B)-heuc(C,A,B);
            for (int nb : knn[X1]) {
                int cp = pos[nb];
                if (cp==prev||cp==i) continue;
                bool in_chain=false;
                for (int x=0;x<c;x++) if(cp==(i+x)%n){in_chain=true;break;}
                if (in_chain) continue;
                int D = T[(cp+1)%n];
                double ins = heuc(C,nb,D)-heuc(C,nb,X1)-heuc(C,Xc,D);
                if (rem+ins > 1e-10) {
                    apply_oropt(T, i, cp, c);
                    for (int x=0;x<n;x++) pos[T[x]]=x;
                    improved=true; break;
                }
            }
        }
    }
    double cost=0;
    for (int i=0;i<n;i++) cost+=heuc(C,T[i],T[(i+1)%n]);
    return cost;
}

// --------------------------------------------------------------------------
// TSPLIB parser
// --------------------------------------------------------------------------
vector<City> readTSP(const string& f) {
    ifstream in(f);
    if (!in){cerr<<"Cannot open: "<<f<<"\n";exit(1);}
    string line; bool go=false; vector<City> v;
    while (getline(in,line)) {
        if (line.find("NODE_COORD_SECTION")!=string::npos){go=true;continue;}
        if (!go) continue;
        if (line.find("EOF")!=string::npos) break;
        istringstream ss(line); int id; double x,y;
        if (ss>>id>>x>>y) v.push_back({x,y});
    }
    return v;
}

vector<vector<int>> build_knn(const vector<City>& C, int k) {
    int n=C.size();
    vector<vector<int>> knn(n, vector<int>(k));
    vector<pair<double,int>> tmp(n);
    for (int i=0;i<n;i++) {
        for (int j=0;j<n;j++) tmp[j]={heuc(C,i,j),j};
        tmp[i].first=1e18;
        nth_element(tmp.begin(),tmp.begin()+k,tmp.end());
        sort(tmp.begin(),tmp.begin()+k);
        for (int t=0;t<k;t++) knn[i][t]=tmp[t].second;
    }
    return knn;
}

// --------------------------------------------------------------------------
// MAIN
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
    string fname = argc>1 ? argv[1] : "instance.tsp";
    auto C = readTSP(fname);
    int n = C.size();
    cerr << "Instance: " << fname << "  Cities: " << n << "\n";

    // ---- Auto-tune parameters ----
    int W, k, max_rounds, top_K;
    if      (n <= 200)  { W=512; k=15; max_rounds=300; top_K=8; }
    else if (n <= 500)  { W=256; k=20; max_rounds=200; top_K=6; }
    else if (n <= 1000) { W=128; k=20; max_rounds=150; top_K=4; }
    else if (n <= 2000) { W=64;  k=25; max_rounds=100; top_K=4; }
    else                { W=32;  k=25; max_rounds=80;  top_K=4; }

    long long pairs_ll = (long long)n*(n-1)/2;
    int pairs = (int)pairs_ll;  // safe up to ~65k cities
    int TPB   = 256;
    int wBlocks = (W+TPB-1)/TPB;
    int pBlocks = (pairs+TPB-1)/TPB;
    int nBlocks = (n+TPB-1)/TPB;

    cerr << "W="<<W<<" k="<<k<<" max_rounds="<<max_rounds<<"\n";

    // ---- Build KNN ----
    cerr << "Building KNN (k="<<k<<")...\n";
    auto knn_host = build_knn(C, k);
    vector<int> knn_flat(n*k);
    for (int i=0;i<n;i++)
        for (int j=0;j<k;j++)
            knn_flat[i*k+j]=knn_host[i][j];

    // ---- GPU alloc ----
    City*        dC;   gpuCheck(cudaMalloc(&dC,   n*sizeof(City)));
    int*         dKNN; gpuCheck(cudaMalloc(&dKNN, n*k*sizeof(int)));
    int*         dT;   gpuCheck(cudaMalloc(&dT,   W*n*sizeof(int)));
    int*         dPos; gpuCheck(cudaMalloc(&dPos, n*sizeof(int)));    // pos[] for oropt
    int*         dVis; gpuCheck(cudaMalloc(&dVis, W*n*sizeof(int)));  // vis[] for NN (global, any n)
    curandState* dRNG; gpuCheck(cudaMalloc(&dRNG, W*sizeof(curandState)));
    int*         d_bi; gpuCheck(cudaMalloc(&d_bi, sizeof(int)));
    int*         d_bj; gpuCheck(cudaMalloc(&d_bj, sizeof(int)));
    double*      d_bg; gpuCheck(cudaMalloc(&d_bg, sizeof(double)));

    gpuCheck(cudaMemcpy(dC,   C.data(),        n*sizeof(City),  cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(dKNN, knn_flat.data(), n*k*sizeof(int), cudaMemcpyHostToDevice));

    initRNG<<<wBlocks,TPB>>>(dRNG, W, 42UL);
    gpuCheck(cudaDeviceSynchronize());

    // ---- NN Construction ----
    cerr << "GPU NN construction...\n";
    cudaEvent_t e0,e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    kernel_nn_construct<<<wBlocks,TPB>>>(dC,n,dKNN,k,dT,dVis,dRNG,W);
    gpuCheck(cudaDeviceSynchronize());
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms_nn; cudaEventElapsedTime(&ms_nn,e0,e1);
    cerr << "NN done: " << ms_nn << " ms\n";

    // ---- 2-opt + Or-opt hill climbing per walker ----
    cerr << "GPU 2-opt + Or-opt per walker...\n";
    vector<int>    h_tour(n);
    vector<double> wcosts(W, 1e18);
    vector<vector<int>> wtours(W, vector<int>(n));

    cudaEventRecord(e0);

    for (int w = 0; w < W; w++) {
        int* tourPtr = dT + w*n;

        // Build initial pos[] on CPU from NN tour
        gpuCheck(cudaMemcpy(h_tour.data(), tourPtr, n*sizeof(int), cudaMemcpyDeviceToHost));
        vector<int> h_pos(n);
        for (int i=0;i<n;i++) h_pos[h_tour[i]]=i;
        gpuCheck(cudaMemcpy(dPos, h_pos.data(), n*sizeof(int), cudaMemcpyHostToDevice));

        int rounds = 0;
        bool any_improvement = true;

        while (any_improvement && rounds < max_rounds) {
            any_improvement = false;

            // --- 2-opt round ---
            {
                double zero=0.0; int neg=-1;
                gpuCheck(cudaMemcpy(d_bg,&zero,sizeof(double),cudaMemcpyHostToDevice));
                gpuCheck(cudaMemcpy(d_bi,&neg, sizeof(int),   cudaMemcpyHostToDevice));
                gpuCheck(cudaMemcpy(d_bj,&neg, sizeof(int),   cudaMemcpyHostToDevice));

                kernel_2opt_round<<<pBlocks,TPB>>>(dC,n,tourPtr,d_bi,d_bj,d_bg,pairs);
                gpuCheck(cudaDeviceSynchronize());

                int bi,bj; double bg;
                gpuCheck(cudaMemcpy(&bi,d_bi,sizeof(int),   cudaMemcpyDeviceToHost));
                gpuCheck(cudaMemcpy(&bj,d_bj,sizeof(int),   cudaMemcpyDeviceToHost));
                gpuCheck(cudaMemcpy(&bg,d_bg,sizeof(double),cudaMemcpyDeviceToHost));

                if (bi>=0 && bj>=0 && bg>1e-10) {
                    gpuCheck(cudaMemcpy(h_tour.data(),tourPtr,n*sizeof(int),cudaMemcpyDeviceToHost));
                    apply_2opt(h_tour, bi, bj);
                    gpuCheck(cudaMemcpy(tourPtr,h_tour.data(),n*sizeof(int),cudaMemcpyHostToDevice));
                    // Update pos
                    for (int x=bi+1;x<=bj;x++) h_pos[h_tour[x]]=x;
                    gpuCheck(cudaMemcpy(dPos,h_pos.data(),n*sizeof(int),cudaMemcpyHostToDevice));
                    any_improvement = true;
                }
            }

            // --- Or-opt rounds (chain 1, 2, 3) ---
            for (int chain = 1; chain <= 3; chain++) {
                double zero=0.0; int neg=-1;
                gpuCheck(cudaMemcpy(d_bg,&zero,sizeof(double),cudaMemcpyHostToDevice));
                gpuCheck(cudaMemcpy(d_bi,&neg, sizeof(int),   cudaMemcpyHostToDevice));
                gpuCheck(cudaMemcpy(d_bj,&neg, sizeof(int),   cudaMemcpyHostToDevice));

                kernel_oropt_round<<<nBlocks,TPB>>>(
                    dC,n,dKNN,k,tourPtr,dPos,chain,d_bi,d_bj,d_bg
                );
                gpuCheck(cudaDeviceSynchronize());

                int ri,ii; double bg;
                gpuCheck(cudaMemcpy(&ri,d_bi,sizeof(int),   cudaMemcpyDeviceToHost));
                gpuCheck(cudaMemcpy(&ii,d_bj,sizeof(int),   cudaMemcpyDeviceToHost));
                gpuCheck(cudaMemcpy(&bg,d_bg,sizeof(double),cudaMemcpyDeviceToHost));

                if (ri>=0 && ii>=0 && bg>1e-10) {
                    gpuCheck(cudaMemcpy(h_tour.data(),tourPtr,n*sizeof(int),cudaMemcpyDeviceToHost));
                    apply_oropt(h_tour, ri, ii, chain);
                    gpuCheck(cudaMemcpy(tourPtr,h_tour.data(),n*sizeof(int),cudaMemcpyHostToDevice));
                    // Rebuild pos fully after or-opt (indices shift)
                    for (int x=0;x<n;x++) h_pos[h_tour[x]]=x;
                    gpuCheck(cudaMemcpy(dPos,h_pos.data(),n*sizeof(int),cudaMemcpyHostToDevice));
                    any_improvement = true;
                }
            }
            rounds++;
        }

        gpuCheck(cudaMemcpy(wtours[w].data(),tourPtr,n*sizeof(int),cudaMemcpyDeviceToHost));
        double c=0;
        for (int i=0;i<n;i++) c+=heuc(C,wtours[w][i],wtours[w][(i+1)%n]);
        wcosts[w]=c;
        if (w%16==0 || w==W-1)
            cerr<<"  Walker "<<w<<"/"<<W<<" cost="<<c<<" rounds="<<rounds<<"\n";
    }

    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms_opt; cudaEventElapsedTime(&ms_opt,e0,e1);
    cerr << "2opt+oropt total: " << ms_opt << " ms\n";

    // ---- Pick top-K ----
    vector<int> idx(W); iota(idx.begin(),idx.end(),0);
    sort(idx.begin(),idx.end(),[&](int a,int b){return wcosts[a]<wcosts[b];});

    cerr << "\nTop "<<top_K<<" walkers (GPU):\n";
    for (int i=0;i<top_K;i++)
        cerr<<"  ["<<i<<"] "<<wcosts[idx[i]]<<"\n";

    // ---- CPU polish: 2-opt then Or-opt 1,2,3 ----
    cerr << "\nCPU polish (2-opt + Or-opt)...\n";
    double best_cost=1e18; int best_i=0;
    vector<vector<int>> polished(top_K);

    for (int i=0;i<top_K;i++) {
        polished[i] = wtours[idx[i]];
        double c = cpu_2opt(polished[i], C, knn_host);
        c = cpu_oropt(polished[i], C, knn_host, 1);
        c = cpu_oropt(polished[i], C, knn_host, 2);
        c = cpu_oropt(polished[i], C, knn_host, 3);
        c = cpu_2opt(polished[i], C, knn_host); // final 2-opt after or-opt
        cerr<<"  Tour "<<i<<": "<<wcosts[idx[i]]<<" -> "<<c<<"\n";
        if (c < best_cost){ best_cost=c; best_i=i; }
    }

    // ---- Result ----
    cout<<"\n========== RESULT ==========\n";
    cout<<"Instance : "<<fname<<"\n";
    cout<<"Cities   : "<<n<<"\n";
    cout<<"NN time  : "<<ms_nn<<" ms\n";
    cout<<"Opt time : "<<ms_opt<<" ms\n";
    cout<<"Best cost: "<<best_cost<<"\n";
    cout<<"First 20 : ";
    for (int i=0;i<min(20,n);i++) cout<<polished[best_i][i]<<" ";
    cout<<"...\n\nFull tour:\n";
    for (int i=0;i<n;i++) cout<<polished[best_i][i]<<(i<n-1?" ":"\n");

    cudaFree(dC); cudaFree(dKNN); cudaFree(dT);
    cudaFree(dPos); cudaFree(dVis); cudaFree(dRNG);
    cudaFree(d_bi); cudaFree(d_bj); cudaFree(d_bg);
    return 0;
}
