
// Parallel Iterative Hill Climbing for TSP (GPU)
// Based on: Pramod Yelmewad sir & Talawar sir's "Parallel Iterative Hill Climbing on GPU"
// Faster but still the error margin for this was 0.72%-8%
// DESIGN:
//   - Multiple independent walkers, each with their own tour
//   - Each walker: NN construction → repeated 2-opt hill climbing
//   - Thread mapping: TPN (one thread per 2-opt neighbor pair)
//   - CPU: pick best walker, quick final 2-opt polish (no wrap-around bugs)

#include <bits/stdc++.h>
#include <curand_kernel.h>
using namespace std;

// --------------------------------------------------------------------------
// City + EUC_2D distance (TSPLIB standard)
// --------------------------------------------------------------------------
struct City { double x, y; };

__host__ __device__ inline double euc2d(const City* C, int a, int b) {
    double dx = C[a].x - C[b].x, dy = C[a].y - C[b].y;
    return (double)llround(sqrt(dx*dx + dy*dy));
}
inline double heuc(const vector<City>& C, int a, int b) {
    double dx = C[a].x-C[b].x, dy = C[a].y-C[b].y;
    return (double)llround(sqrt(dx*dx+dy*dy));
}

#define gpuCheck(x) { cudaError_t _e=(x); \
    if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);}}

// --------------------------------------------------------------------------
// RNG init — one state per walker
// --------------------------------------------------------------------------
__global__ void initRNG(curandState* S, int W, unsigned long seed) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < W) curand_init(seed, id, 0, &S[id]);
}

// --------------------------------------------------------------------------
// KERNEL 1: NN Construction
// Each walker (thread) builds its own tour using nearest-neighbor heuristic
// starting from a different random city.
// This gives much better starting tours than random shuffle.
// --------------------------------------------------------------------------
__global__ void kernel_nn_construct(
    const City* C, int n,
    const int* knn, int k,   // precomputed KNN for fast NN lookup
    int* tours,              // [W * n] output
    curandState* RNG, int W
) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id >= W) return;

    curandState st = RNG[id];
    int* T = tours + id*n;

    // visited[] stored in T temporarily (we'll overwrite with tour)
    // Use a separate bool approach: mark visited by negating
    // Actually: build directly into T, track visited with a small trick
    bool vis[8192]; // max n supported = 8192
    for (int i = 0; i < n; i++) vis[i] = false;

    // Random start city (different per walker)
    int cur = curand(&st) % n;
    T[0] = cur;
    vis[cur] = true;

    for (int step = 1; step < n; step++) {
        int best = -1;
        double bestd = 1e18;
        // Check KNN first (fast path — covers most cases)
        for (int ki = 0; ki < k; ki++) {
            int nb = knn[cur*k + ki];
            if (!vis[nb]) {
                double d = euc2d(C, cur, nb);
                if (d < bestd) { bestd = d; best = nb; }
            }
        }
        // Fallback: full scan (only when all KNN neighbors are visited)
        if (best < 0) {
            for (int j = 0; j < n; j++) {
                if (!vis[j]) {
                    double d = euc2d(C, cur, j);
                    if (d < bestd) { bestd = d; best = j; }
                }
            }
        }
        T[step] = best;
        vis[best] = true;
        cur = best;
    }
    RNG[id] = st;
}

// --------------------------------------------------------------------------
// KERNEL 2: TPN 2-opt improvement (one thread per neighbor pair)
// This is the core of PIHC from the paper.
// Each thread checks one specific (i,j) pair for a 2-opt improvement.
// If ANY thread finds an improvement, we record the best one and apply it.
// Repeat until no improvement found.
//
// For walker `w`, thread covers pair given by global linear index `id`.
// --------------------------------------------------------------------------

// Convert linear index to (i,j) pair using the paper's formula (Eq 6,7)
__device__ void linear_to_ij(int id, int n, int* out_i, int* out_j) {
    // Use double precision sqrt as the paper specifies (avoid precision issues)
    int i = n - 2 - (int)floor(
        (sqrt((double)(8*(long long)(n*(n-1)/2 - id - 1) + 1)) - 1.0) / 2.0
    );
    int j = id - i*(n-1) + i*(i+1)/2 + 1;
    *out_i = i;
    *out_j = j;
}

// One round of 2-opt: each thread checks its (i,j) pair for ONE walker
// Returns improvement found via atomicMin on shared best gain
__global__ void kernel_2opt_round(
    const City* C, int n,
    int* tour,          // one walker's tour [n]
    double tour_cost,   // current cost
    int* best_i,        // output: best swap i
    int* best_j,        // output: best swap j
    double* best_gain,  // output: best gain found (negative = improvement)
    int pairs           // = n*(n-1)/2
) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id >= pairs) return;

    int i, j;
    linear_to_ij(id, n, &i, &j);
    if (i < 0 || j >= n || i >= j) return;

    int A = tour[i],   B = tour[i+1];
    int Ci= tour[j],   D = tour[(j+1)%n];

    double gain = euc2d(C,A,B) + euc2d(C,Ci,D)
                - euc2d(C,A,Ci) - euc2d(C,B,D);

    // gain > 0 means improvement (we REMOVE gain from cost)
    if (gain > 1e-10) {
        // Atomically record if this is the best improvement seen
        // We use integer atomic on scaled gain
        // Store as negative (so atomicMax finds best improvement)
        unsigned long long* bg = (unsigned long long*)best_gain;
        unsigned long long old_val = *bg;
        unsigned long long new_val = __double_as_longlong(gain);
        // Only update if our gain is larger
        while (new_val > old_val) {
            unsigned long long assumed = old_val;
            old_val = atomicCAS(bg, assumed, new_val);
        }
        // If we set the gain, also record i,j (best effort — slight race is ok,
        // we just need A valid improving move, not necessarily THE best)
        if (__longlong_as_double(atomicAdd((unsigned long long*)best_gain, 0)) == gain) {
            *best_i = i;
            *best_j = j;
        }
    }
}

// --------------------------------------------------------------------------
// CPU: apply a 2-opt reversal (no wrap-around, safe)
// --------------------------------------------------------------------------
void apply_2opt(vector<int>& T, int i, int j) {
    // Reverse segment T[i+1 .. j]
    reverse(T.begin() + i + 1, T.begin() + j + 1);
}

// --------------------------------------------------------------------------
// CPU 2-opt with KNN candidate list (clean, no wrap-around)
// Only improves non-wrap segments (i < j, both in [0, n-1])
// This is fast and correct.
// --------------------------------------------------------------------------
double cpu_2opt(vector<int>& T, const vector<City>& C,
                const vector<vector<int>>& knn) {
    int n = T.size();
    vector<int> pos(n);
    for (int i = 0; i < n; i++) pos[T[i]] = i;

    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 0; i < n - 2; i++) {
            int A = T[i], B = T[i+1];
            double dAB = heuc(C, A, B);
            for (int nb : knn[A]) {
                if (heuc(C, A, nb) >= dAB) break; // KNN sorted by distance
                int j = pos[nb];
                if (j <= i + 1 || j >= n) continue; // only non-wrap
                int D = T[j+1 < n ? j+1 : 0];
                if (j + 1 >= n) continue; // skip wrap-around entirely
                double gain = dAB + heuc(C, nb, D)
                            - heuc(C, A, nb) - heuc(C, B, D);
                if (gain > 1e-10) {
                    apply_2opt(T, i, j);
                    for (int x = i+1; x <= j; x++) pos[T[x]] = x;
                    B   = T[i+1];
                    dAB = heuc(C, A, B);
                    improved = true;
                    break;
                }
            }
        }
    }
    double cost = 0;
    for (int i = 0; i < n; i++) cost += heuc(C, T[i], T[(i+1)%n]);
    return cost;
}

// --------------------------------------------------------------------------
// TSPLIB parser
// --------------------------------------------------------------------------
vector<City> readTSP(const string& f) {
    ifstream in(f);
    if (!in) { cerr << "Cannot open: " << f << "\n"; exit(1); }
    string line; bool go = false; vector<City> v;
    while (getline(in, line)) {
        if (line.find("NODE_COORD_SECTION") != string::npos) { go=true; continue; }
        if (!go) continue;
        if (line.find("EOF") != string::npos) break;
        istringstream ss(line); int id; double x, y;
        if (ss >> id >> x >> y) v.push_back({x, y});
    }
    return v;
}

// --------------------------------------------------------------------------
// Build KNN on CPU
// --------------------------------------------------------------------------
vector<vector<int>> build_knn(const vector<City>& C, int k) {
    int n = C.size();
    vector<vector<int>> knn(n, vector<int>(k));
    vector<pair<double,int>> tmp(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) tmp[j] = {heuc(C,i,j), j};
        tmp[i].first = 1e18;
        nth_element(tmp.begin(), tmp.begin()+k, tmp.end());
        sort(tmp.begin(), tmp.begin()+k);
        for (int t = 0; t < k; t++) knn[i][t] = tmp[t].second;
    }
    return knn;
}

// --------------------------------------------------------------------------
// MAIN
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
    string fname = argc > 1 ? argv[1] : "instance.tsp";
    auto C = readTSP(fname);
    int n = C.size();
    cerr << "Instance: " << fname << "  Cities: " << n << "\n";

    // ---- Parameters (auto-tuned by n) ----
    int W, k, max_2opt_rounds, top_K;
    if      (n <= 200)  { W = 512;  k = 15; max_2opt_rounds = 500; top_K = 8; }
    else if (n <= 500)  { W = 256;  k = 20; max_2opt_rounds = 300; top_K = 6; }
    else if (n <= 1000) { W = 128;  k = 20; max_2opt_rounds = 200; top_K = 4; }
    else if (n <= 2000) { W = 64;   k = 25; max_2opt_rounds = 150; top_K = 4; }
    else                { W = 32;   k = 25; max_2opt_rounds = 100; top_K = 4; }

    int pairs = (long long)n*(n-1)/2; // number of 2-opt pairs
    int TPB   = 256;
    int walkerBlocks  = (W + TPB - 1) / TPB;
    int pairBlocks    = (pairs + TPB - 1) / TPB;

    cerr << "W=" << W << "  k=" << k
         << "  2opt_rounds=" << max_2opt_rounds
         << "  pairs=" << pairs << "\n";

    // ---- Build KNN on CPU ----
    cerr << "Building KNN (k=" << k << ")...\n";
    auto knn_host = build_knn(C, k);
    vector<int> knn_flat(n*k);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            knn_flat[i*k+j] = knn_host[i][j];

    // ---- GPU memory ----
    City*        dC;   gpuCheck(cudaMalloc(&dC,   n*sizeof(City)));
    int*         dKNN; gpuCheck(cudaMalloc(&dKNN, n*k*sizeof(int)));
    int*         dT;   gpuCheck(cudaMalloc(&dT,   W*n*sizeof(int)));
    curandState* dRNG; gpuCheck(cudaMalloc(&dRNG, W*sizeof(curandState)));
    // Per-walker 2-opt scratch (best i, j, gain)
    int*         d_bi; gpuCheck(cudaMalloc(&d_bi,  sizeof(int)));
    int*         d_bj; gpuCheck(cudaMalloc(&d_bj,  sizeof(int)));
    double*      d_bg; gpuCheck(cudaMalloc(&d_bg,  sizeof(double)));

    gpuCheck(cudaMemcpy(dC,   C.data(),        n*sizeof(City),  cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(dKNN, knn_flat.data(), n*k*sizeof(int), cudaMemcpyHostToDevice));

    initRNG<<<walkerBlocks, TPB>>>(dRNG, W, 42UL);
    gpuCheck(cudaDeviceSynchronize());

    // ---- Step 1: NN Construction for all walkers ----
    cerr << "GPU: NN construction for " << W << " walkers...\n";
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);

    kernel_nn_construct<<<walkerBlocks, TPB>>>(dC, n, dKNN, k, dT, dRNG, W);
    gpuCheck(cudaDeviceSynchronize());

    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms_nn; cudaEventElapsedTime(&ms_nn, e0, e1);
    cerr << "NN construction: " << ms_nn << " ms\n";

    // ---- Step 2: 2-opt hill climbing for each walker ----
    // For each walker: run up to max_2opt_rounds of TPN 2-opt
    cerr << "GPU: 2-opt hill climbing...\n";

    vector<int>    h_tour(n);
    vector<double> walker_costs(W, 1e18);
    vector<vector<int>> walker_tours(W, vector<int>(n));

    cudaEventRecord(e0);

    for (int w = 0; w < W; w++) {
        int* tourPtr = dT + w*n;
        double cost = 0; // we'll compute on CPU after
        int rounds = 0;

        for (int round = 0; round < max_2opt_rounds; round++) {
            // Reset best gain to 0
            double zero = 0.0;
            int    neg1 = -1;
            gpuCheck(cudaMemcpy(d_bg, &zero, sizeof(double), cudaMemcpyHostToDevice));
            gpuCheck(cudaMemcpy(d_bi, &neg1, sizeof(int),    cudaMemcpyHostToDevice));
            gpuCheck(cudaMemcpy(d_bj, &neg1, sizeof(int),    cudaMemcpyHostToDevice));

            // Launch TPN kernel: all pairs for this walker
            kernel_2opt_round<<<pairBlocks, TPB>>>(
                dC, n, tourPtr, 0.0, d_bi, d_bj, d_bg, pairs
            );
            gpuCheck(cudaDeviceSynchronize());

            // Read back best improvement
            int   bi, bj;
            double bg;
            gpuCheck(cudaMemcpy(&bi, d_bi, sizeof(int),    cudaMemcpyDeviceToHost));
            gpuCheck(cudaMemcpy(&bj, d_bj, sizeof(int),    cudaMemcpyDeviceToHost));
            gpuCheck(cudaMemcpy(&bg, d_bg, sizeof(double), cudaMemcpyDeviceToHost));

            if (bi < 0 || bj < 0 || bg <= 1e-10) break; // converged

            // Apply the 2-opt reversal on GPU: copy tour, reverse, copy back
            gpuCheck(cudaMemcpy(h_tour.data(), tourPtr, n*sizeof(int), cudaMemcpyDeviceToHost));
            apply_2opt(h_tour, bi, bj);
            gpuCheck(cudaMemcpy(tourPtr, h_tour.data(), n*sizeof(int), cudaMemcpyHostToDevice));
            rounds++;
        }

        // Get final tour
        gpuCheck(cudaMemcpy(walker_tours[w].data(), tourPtr, n*sizeof(int), cudaMemcpyDeviceToHost));
        double c = 0;
        for (int i = 0; i < n; i++)
            c += heuc(C, walker_tours[w][i], walker_tours[w][(i+1)%n]);
        walker_costs[w] = c;
        if (w % 16 == 0)
            cerr << "  Walker " << w << "/" << W << " cost=" << c << " rounds=" << rounds << "\n";
    }

    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms_opt; cudaEventElapsedTime(&ms_opt, e0, e1);
    cerr << "2-opt total: " << ms_opt << " ms\n";

    // ---- Pick top-K ----
    vector<int> idx(W); iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a, int b){ return walker_costs[a] < walker_costs[b]; });

    cerr << "\nTop " << top_K << " walkers:\n";
    for (int i = 0; i < top_K; i++)
        cerr << "  [" << i << "] " << walker_costs[idx[i]] << "\n";

    // ---- CPU 2-opt polish on top-K ----
    cerr << "\nCPU polish (KNN 2-opt, no wrap-around)...\n";
    double best_cost = 1e18;
    int    best_i    = 0;
    vector<vector<int>> polished(top_K);

    for (int i = 0; i < top_K; i++) {
        polished[i] = walker_tours[idx[i]];
        double c = cpu_2opt(polished[i], C, knn_host);
        cerr << "  Tour " << i << ": " << walker_costs[idx[i]] << " -> " << c << "\n";
        if (c < best_cost) { best_cost = c; best_i = i; }
    }

    // ---- Output ----
    cout << "\n========== RESULT ==========\n";
    cout << "Instance : " << fname  << "\n";
    cout << "Cities   : " << n      << "\n";
    cout << "NN time  : " << ms_nn  << " ms\n";
    cout << "Opt time : " << ms_opt << " ms\n";
    cout << "Best cost: " << best_cost << "\n";
    cout << "First 20 : ";
    for (int i = 0; i < min(20, n); i++) cout << polished[best_i][i] << " ";
    cout << "...\n";

    cout << "\nFull tour:\n";
    for (int i = 0; i < n; i++)
        cout << polished[best_i][i] << (i<n-1?" ":"\n");

    // Cleanup
    cudaFree(dC); cudaFree(dKNN); cudaFree(dT); cudaFree(dRNG);
    cudaFree(d_bi); cudaFree(d_bj); cudaFree(d_bg);
    return 0;
}
