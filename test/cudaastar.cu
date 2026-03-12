// #include <iostream>
// #include <vector>
// #include <cstdio>
// #include <cstdlib>
// #include <climits>
// #include <cuda_runtime.h>

// #define MAX_N 12
// #define MAX_OPEN 2000000
// #define MAX_THREADS 1024

// // ---------------------------
// // Data Structures
// // ---------------------------
// struct Node {
//     unsigned int visited_mask;
//     int last;
//     float g;
//     float f;
// };

// // ---------------------------
// // Device Globals
// // ---------------------------
// __device__ Node d_open[MAX_OPEN];
// __device__ int d_open_head = 0;
// __device__ int d_open_tail = 0;
// __device__ float d_best_cost = 1e30f;

// // ---------------------------
// // Simple heuristic (placeholder)
// // ---------------------------
// __device__ __forceinline__ float heuristic_lower_bound(int n, unsigned int visited_mask, const float *dist) {
//     return 0.0f;
// }

// // ---------------------------
// // GPU Kernel
// // ---------------------------
// __global__ void naive_a_star_kernel(int n, const float *dist_matrix) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     while (true) {
//         int idx = atomicAdd(&d_open_head, 1);
//         if (idx > d_open_tail) {
//             return;
//         }

//         Node cur = d_open[idx];
//         float global_best = d_best_cost;
//         if (cur.g >= global_best) continue;

//         if ((unsigned int)cur.visited_mask == ((1u << n) - 1)) {
//             float tour_cost = cur.g + dist_matrix[cur.last * n + 0];

//             // safer float atomicMin emulation
//             atomicMin((int*)&d_best_cost, __float_as_int(tour_cost));
//             continue;
//         }

//         for (int nb = 0; nb < n; ++nb) {
//             if (cur.visited_mask & (1u << nb)) continue;

//             Node nxt;
//             nxt.visited_mask = cur.visited_mask | (1u << nb);
//             nxt.last = nb;
//             nxt.g = cur.g + dist_matrix[cur.last * n + nb];
//             nxt.f = nxt.g + heuristic_lower_bound(n, nxt.visited_mask, dist_matrix);

//             int push_idx = atomicAdd(&d_open_tail, 1);
//             if (push_idx < MAX_OPEN) {
//                 d_open[push_idx] = nxt;
//             }
//         }
//     }
// }

// // ---------------------------
// // GPU A* Runner
// // ---------------------------
// void run_naive(int n, float *h_dist) {
//     float *d_dist;
//     cudaMalloc(&d_dist, n * n * sizeof(float));
//     cudaMemcpy(d_dist, h_dist, n * n * sizeof(float), cudaMemcpyHostToDevice);

//     int zero = 0;
//     int one = 1;
//     float inf = 1e30f;
//     cudaMemcpyToSymbol(d_open_head, &zero, sizeof(int));
//     cudaMemcpyToSymbol(d_open_tail, &one, sizeof(int));
//     cudaMemcpyToSymbol(d_best_cost, &inf, sizeof(float));

//     Node start;
//     start.visited_mask = 1u << 0;
//     start.last = 0;
//     start.g = 0.0f;
//     start.f = 0.0f;
//     cudaMemcpyToSymbol(d_open, &start, sizeof(Node));

//     // Check GPU info
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0);
//     printf("Running on GPU: %s\n", prop.name);

//     // ---------------------------
//     // Timing
//     // ---------------------------
//     cudaEvent_t start_event, stop_event;
//     cudaEventCreate(&start_event);
//     cudaEventCreate(&stop_event);

//     dim3 blocks(8);
//     dim3 threads(128);

//     cudaEventRecord(start_event);
//     naive_a_star_kernel<<<blocks, threads>>>(n, d_dist);
//     cudaEventRecord(stop_event);

//     cudaDeviceSynchronize();
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//         printf("CUDA Error: %s\n", cudaGetErrorString(err));

//     cudaEventSynchronize(stop_event);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start_event, stop_event);

//     float best;
//     cudaMemcpyFromSymbol(&best, d_best_cost, sizeof(float));

//     printf("-------------------------------------\n");
//     printf("Best tour cost found (maybe): %f\n", best);
//     printf("Kernel execution time: %.4f ms\n", milliseconds);
//     printf("-------------------------------------\n");

//     cudaFree(d_dist);
//     cudaEventDestroy(start_event);
//     cudaEventDestroy(stop_event);
// }

// // ---------------------------
// // Main
// // ---------------------------
// int main() {
//     int n = 8;
//     float h_dist[MAX_N * MAX_N];
//     srand(123);

//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             if (i == j) h_dist[i * n + j] = 0.0f;
//             else h_dist[i * n + j] = 1.0f + (rand() % 100) / 10.0f;
//         }
//     }

//     printf("Running CUDA A* with %d cities...\n", n);
//     run_naive(n, h_dist);
//     return 0;
// }
// tsp_fully_optimized.cu
// Fully optimized candidate-list-driven multi-start 2-opt GPU solver with TSPLIB parsing
// Features:
//  - TSPLIB .tsp parser (EUC_2D coordinates), and optional known OPT line (COMMENT or TOUR_SECTION ignored)
//  - Grid-based approximate nearest-neighbor candidate builder (fast for large n)
//  - Candidate-list-driven 2-opt evaluation on GPU
//  - Multi-start (island) mode: many independent tours (islands) executed in parallel on GPU (each island maintained on device)
//  - Per-block shared-memory reduction to pick best local move; host reduces per-island and applies best move(s)
//  - Optional CPU-side selective 3-opt polishing on the best result
//  - Benchmark logging for comparing with paper results
// Compile: nvcc -O3 tsp_fully_optimized.cu -o tsp_opt -std=c++11
// Usage: ./tsp_opt <tsp_or_folder_path> <m> <islands> <max_iters> <init_mode> <do_3opt(0/1)>
//   tsp_or_folder_path: single .tsp file or a folder containing .tsp files. If folder, the program runs each .tsp file.
//   m: candidate list length (e.g., 20)
//   islands: number of independent starts to run in parallel (e.g., 64)
//   max_iters: maximum outer iterations per island (e.g., 100000)
//   init_mode: 0=random start, 1=nearest-neighbor start
//   do_3opt: 0 or 1 (if 1, best tour will be polished with simple CPU 3-opt at the end)
// tsp_fully_optimized_chunked.cu
// Fully optimized candidate-list-driven multi-start 2-opt GPU solver with TSPLIB parsing
// Updated: chunked kernel launches to support very large problem sizes, Windows folder listing,
// safe shared-memory layout and portable includes (no bits/stdc++.h), and modern shuffle usage.
// Compile: nvcc -O3 tsp_fully_optimized_chunked.cu -o tsp_opt
// Usage: ./tsp_opt <tsp_or_folder> <m> <islands> <max_iters> <init_mode> <do_3opt>
//   tsp_or_folder: single .tsp file or a folder containing .tsp files. If folder, the program runs each .tsp file.
//   m: candidate list length (e.g., 20)
//   islands: number of independent starts to run in parallel (e.g., 64)
//   max_iters: maximum outer iterations per island (e.g., 100000)
//   init_mode: 0=random start, 1=nearest-neighbor start
//   do_3opt: 0 or 1 (if 1, best tour will be polished with simple CPU 3-opt at the end)

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>
#include <numeric>
#include <sys/stat.h>
#include <climits>
#include <cfloat>
#include <random>
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

#include <cuda_runtime.h>
using namespace std;

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA Error %s:%d: %s
", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

// Kernel: evaluate candidate-driven 2-opt deltas for many islands (island x n x m pairs)
// New: accepts a 64-bit 'offset' so host can launch the kernel on chunks of the pair-space.
extern "C" __global__ void eval_2opt_islands(
    const float *x, const float *y,
    const int *candidates, // n * m
    const int n, const int m,
    const int islands,
    const int *tours,      // islands * n
    const int *pos,        // islands * n
    float *block_best_delta, // per-block
    unsigned long long *block_best_pack, // per-block
    unsigned long long offset_pairs // offset in pairs (0..total_pairs-1)
) {
    // Use a byte-shared buffer and compute pointers to avoid alignment problems
    extern __shared__ unsigned char sbytes[]; // size = blockDim.x * (sizeof(float) + sizeof(unsigned long long))
    float *s_delta = (float*)sbytes; // first blockDim.x floats
    unsigned long long *s_pack = (unsigned long long*)(sbytes + (size_t)blockDim.x * sizeof(float));

    unsigned long long gid = offset_pairs + (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total_pairs = (unsigned long long)islands * (unsigned long long)n * (unsigned long long)m;
    int tid = threadIdx.x;

    float best = 1e38f;
    unsigned long long bestpack = ULLONG_MAX;

    if (gid < total_pairs) {
        unsigned long long rem = gid % (unsigned long long)(n * m);
        int isl = (int)(gid / (unsigned long long)(n * m));
        int i = (int)(rem / (unsigned long long)m);
        int k = (int)(rem % (unsigned long long)m);

        int city_a = tours[ isl * n + i ];
        int city_a1 = tours[ isl * n + ((i+1) % n) ];
        int city_b = candidates[ i * m + k ]; // city id
        int j = pos[ isl * n + city_b ]; // position of city_b in island's tour
        if (j > i+1 && j < n) {
            int city_b1 = tours[ isl * n + ((j+1) % n) ];
            float dx1 = x[city_a] - x[city_b];
            float dy1 = y[city_a] - y[city_b];
            float d_ab = sqrtf(dx1*dx1 + dy1*dy1);
            float dx2 = x[city_a1] - x[city_b1];
            float dy2 = y[city_a1] - y[city_b1];
            float d_a1b1 = sqrtf(dx2*dx2 + dy2*dy2);
            float dx3 = x[city_a] - x[city_a1];
            float dy3 = y[city_a] - y[city_a1];
            float d_a_a1 = sqrtf(dx3*dx3 + dy3*dy3);
            float dx4 = x[city_b] - x[city_b1];
            float dy4 = y[city_b] - y[city_b1];
            float d_b_b1 = sqrtf(dx4*dx4 + dy4*dy4);
            float delta = (d_ab + d_a1b1) - (d_a_a1 + d_b_b1);
            if (delta < best) {
                best = delta;
                unsigned long long pack = (((unsigned long long)isl & 0xFFFFULL) << 48) |
                                          (((unsigned long long)i & 0xFFFFFFULL) << 24) |
                                          ((unsigned long long)j & 0xFFFFFFULL);
                bestpack = pack;
            }
        }
    }

    s_delta[tid] = best;
    s_pack[tid] = bestpack;
    __syncthreads();

    // block reduce (min) simple tree
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = s_delta[tid + s];
            unsigned long long opack = s_pack[tid + s];
            if (other < s_delta[tid]) {
                s_delta[tid] = other;
                s_pack[tid] = opack;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_best_delta[blockIdx.x] = s_delta[0];
        block_best_pack[blockIdx.x] = s_pack[0];
    }
}

// ---------------- Host utilities ----------------
struct TSPLibInstance {
    string name;
    int n;
    vector<float> x, y;
    float opt = -1.0f; // known optimal if provided
};

// Simple TSPLIB parser for EUC_2D type.
bool parse_tsp_file(const string &path, TSPLibInstance &inst) {
    ifstream in(path);
    if (!in) return false;
    string line;
    inst.name = path;
    inst.n = 0;
    // read header until NODE_COORD_SECTION
    int coord_start = -1;
    vector<string> lines;
    while (getline(in, line)) {
        if (line.size() == 0) continue;
        string up = line;
        for (char &c: up) c = toupper(c);
        if (up.find("NODE_COORD_SECTION") != string::npos) {
            coord_start = (int)lines.size();
            lines.push_back(line);
            break;
        }
        lines.push_back(line);
    }
    if (coord_start == -1) {
        // maybe coordinates start immediately after header, fallback: find first numeric line
        in.clear(); in.seekg(0);
        lines.clear();
        while (getline(in, line)) lines.push_back(line);
        int idx = 0; while (idx < (int)lines.size()) {
            string s = lines[idx];
            stringstream ss(s);
            double a,b; if (ss >> a >> b) break; idx++; }
        if (idx >= (int)lines.size()) return false;
        inst.x.clear(); inst.y.clear();
        for (int i=idx;i<(int)lines.size();++i) {
            stringstream ss(lines[i]); double a,b; if (ss >> a >> b) { inst.x.push_back((float)a); inst.y.push_back((float)b); }
        }
        inst.n = (int)inst.x.size();
        return inst.n>0;
    }

    // Now read coordinate lines until EOF or 'EOF' or 'TOUR_SECTION'
    vector<pair<int,pair<float,float>>> coords;
    while (getline(in, line)) {
        if (line.size()==0) continue;
        string up=line; for(char &c:up) c=toupper(c);
        if (up.find("EOF")!=string::npos) break;
        if (up.find("TOUR_SECTION")!=string::npos) break;
        stringstream ss(line);
        int idx; double a,b; if (ss >> idx >> a >> b) {
            coords.push_back({idx,{(float)a,(float)b}});
        }
    }
    sort(coords.begin(), coords.end(), [](auto &A, auto &B){ return A.first < B.first; });
    inst.n = (int)coords.size(); inst.x.resize(inst.n); inst.y.resize(inst.n);
    for (int i=0;i<inst.n;++i) { inst.x[i] = coords[i].second.first; inst.y[i] = coords[i].second.second; }
    return inst.n>0;
}

// compute Euclidean tour cost
float tour_cost(const vector<int>& tour, const vector<float>& x, const vector<float>& y) {
    int n = (int)tour.size(); double c=0.0;
    for (int i=0;i<n;++i) {
        int a=tour[i], b=tour[(i+1)%n]; double dx=x[a]-x[b], dy=y[a]-y[b]; c += sqrt(dx*dx+dy*dy);
    }
    return (float)c;
}

// Nearest neighbor initial tour (single tour)
vector<int> nearest_neighbor_init_single(int n, const vector<float>& x, const vector<float>& y) {
    vector<int> tour; tour.reserve(n); vector<char> used(n,0);
    int cur = rand()%n; tour.push_back(cur); used[cur]=1;
    for (int step=1; step<n; ++step) {
        int best=-1; float bd=1e38f;
        for (int j=0;j<n;++j) if(!used[j]){ double dx=x[cur]-x[j], dy=y[cur]-y[j]; double d=sqrt(dx*dx+dy*dy); if (d<bd) {bd=(float)d; best=j;} }
        tour.push_back(best); used[best]=1; cur=best;
    }
    return tour;
}

// Simple grid-based approximate nearest neighbors (fast O(n + bucketsize))
void build_candidates_grid(const vector<float>& x, const vector<float>& y, int m, vector<int>& candidates) {
    int n = (int)x.size(); candidates.assign(n*m, 0);
    float xmin=1e38f, xmax=-1e38f, ymin=1e38f, ymax=-1e38f;
    for (int i=0;i<n;++i){ xmin=min(xmin,x[i]); xmax=max(xmax,x[i]); ymin=min(ymin,y[i]); ymax=max(ymax,y[i]); }
    float dx = xmax - xmin + 1e-6f; float dy = ymax - ymin + 1e-6f;
    int gridk = max(1, (int) sqrt((float)n));
    int gx = gridk, gy = gridk;
    float cellx = dx / gx, celly = dy / gy;
    vector<vector<int>> buckets(gx*gy);
    for (int i=0;i<n;++i){ int ix = min(gx-1, max(0,(int)((x[i]-xmin)/cellx))); int iy = min(gy-1, max(0,(int)((y[i]-ymin)/celly))); buckets[iy*gx + ix].push_back(i); }

    for (int i=0;i<n;++i) {
        int ix = min(gx-1, max(0,(int)((x[i]-xmin)/cellx))); int iy = min(gy-1, max(0,(int)((y[i]-ymin)/celly)));
        vector<pair<float,int>> found; found.reserve(m*4);
        int radius = 0;
        while ((int)found.size() < m) {
            for (int dyc = -radius; dyc <= radius; ++dyc) {
                for (int dxc = -radius; dxc <= radius; ++dxc) {
                    int nx = ix + dxc; int ny = iy + dyc; if (nx<0||nx>=gx||ny<0||ny>=gy) continue;
                    auto &b = buckets[ny*gx + nx];
                    for (int id : b) if (id != i) {
                        double ddx = x[i]-x[id], ddy = y[i]-y[id]; double d = sqrt(ddx*ddx + ddy*ddy);
                        found.emplace_back((float)d, id);
                    }
                }
            }
            radius++;
            if (radius > max(gx,gy)) break;
        }
        if ((int)found.size() > m) nth_element(found.begin(), found.begin()+m, found.end());
        sort(found.begin(), found.begin()+min((int)found.size(),m));
        int take = min((int)found.size(), m);
        for (int k=0;k<m;++k) {
            if (k < take) candidates[i*m + k] = found[k].second;
            else candidates[i*m + k] = candidates[i*m + (k%take)];
        }
    }
}

// apply 2-opt on host tour (reverse positions i+1 .. j) and update pos map
void apply_2opt_host(vector<int>& tour, vector<int>& pos, int i, int j) {
    int a=i+1, b=j;
    while (a < b) {
        swap(tour[a], tour[b]); pos[tour[a]] = a; pos[tour[b]] = b; a++; b--; }
}

// simple 3-opt CPU polish (limited) for final improvement (checks a few reconnections per triple)
void three_opt_polish(vector<int>& tour, const vector<float>& x, const vector<float>& y, int max_checks=10000) {
    int n = (int)tour.size(); int checks=0; bool improved=true;
    while (improved && checks < max_checks) {
        improved=false;
        for (int i=0;i<n && checks<max_checks;++i) for (int j=i+2;j<n && checks<max_checks;++j) for (int k=j+2;k<n && checks<max_checks;++k) {
            checks++;
            vector<int> r = tour;
            reverse(r.begin()+i+1, r.begin()+j+1);
            reverse(r.begin()+j+1, r.begin()+k+1);
            double oldc = tour_cost(tour, x, y); double newc = tour_cost(r, x, y);
            if (newc + 1e-9 < oldc) { tour = r; improved=true; goto next_round; }
        }
        next_round: ;
    }
}

// process single instance
void process_instance(const TSPLibInstance &inst, int m, int islands, int max_iters, int init_mode, int do_3opt) {
    int n = inst.n; printf("
=== Instance %s (n=%d) ===
", inst.name.c_str(), n);
    if (n < 4) { printf("Trivial instance
"); return; }

    // build candidates (grid approximate)
    vector<int> candidates; build_candidates_grid(inst.x, inst.y, m, candidates);

    // prepare islands' initial tours (host side). We'll create 'islands' different tours.
    vector<int> host_tours(islands * n);
    vector<int> host_pos(islands * n);
    std::mt19937 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());
    for (int isl=0; isl<islands; ++isl) {
        vector<int> t;
        if (init_mode==1) t = nearest_neighbor_init_single(n, inst.x, inst.y);
        else { t.resize(n); iota(t.begin(), t.end(), 0); shuffle(t.begin(), t.end(), rng); }
        for (int i=0;i<n;++i) { host_tours[isl*n + i] = t[i]; host_pos[isl*n + t[i]] = i; }
    }

    // Device allocations
    float *dx_d=0, *dy_d=0; int *cands_d=0; int *tours_d=0; int *pos_d=0;
    CUDA_CHECK(cudaMalloc((void**)&dx_d, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dy_d, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&cands_d, n * m * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&tours_d, islands * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&pos_d, islands * n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dx_d, inst.x.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy_d, inst.y.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cands_d, candidates.data(), n*m*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tours_d, host_tours.data(), islands*n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pos_d, host_pos.data(), islands*n*sizeof(int), cudaMemcpyHostToDevice));

    // Chunking parameters
    const int threads = 256;
    const int maxBlocksPerLaunch = 1 << 16; // 65536 blocks max per launch
    const unsigned long long total_pairs = (unsigned long long)islands * (unsigned long long)n * (unsigned long long)m;
    const unsigned long long chunk_pairs = (unsigned long long)threads * (unsigned long long)maxBlocksPerLaunch; // pairs per kernel launch

    // allocate device buffers for per-block results for up to maxBlocksPerLaunch
    float *block_best_delta_d = nullptr; unsigned long long *block_best_pack_d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&block_best_delta_d, maxBlocksPerLaunch * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&block_best_pack_d, maxBlocksPerLaunch * sizeof(unsigned long long)));

    vector<float> block_best_delta_h; block_best_delta_h.reserve(maxBlocksPerLaunch);
    vector<unsigned long long> block_best_pack_h; block_best_pack_h.reserve(maxBlocksPerLaunch);

    int iter=0; auto t0 = chrono::high_resolution_clock::now();
    cudaEvent_t kstart, kstop; CUDA_CHECK(cudaEventCreate(&kstart)); CUDA_CHECK(cudaEventCreate(&kstop));
    CUDA_CHECK(cudaEventRecord(kstart));

    float global_best_delta=0.0f;
    while (iter < max_iters) {
        ++iter;
        unsigned long long offset = 0;

        // per-island bests for this iteration
        vector<float> isl_best_delta(islands, 1e38f);
        vector<unsigned long long> isl_best_pack(islands, ULLONG_MAX);

        while (offset < total_pairs) {
            unsigned long long remaining = total_pairs - offset;
            unsigned long long this_pairs = remaining < chunk_pairs ? remaining : chunk_pairs;
            int blocks = (int)((this_pairs + threads - 1) / threads);

            size_t shmem = threads * (sizeof(float) + sizeof(unsigned long long));
            // launch kernel for this chunk
            eval_2opt_islands<<<blocks, threads, shmem>>>(dx_d, dy_d, cands_d, n, m, islands, tours_d, pos_d, block_best_delta_d, block_best_pack_d, offset);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // copy back per-block results for this chunk
            block_best_delta_h.resize(blocks);
            block_best_pack_h.resize(blocks);
            CUDA_CHECK(cudaMemcpy(block_best_delta_h.data(), block_best_delta_d, blocks*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(block_best_pack_h.data(), block_best_pack_d, blocks*sizeof(unsigned long long), cudaMemcpyDeviceToHost));

            // reduce into per-island bests
            for (int b=0;b<blocks;++b) {
                float d = block_best_delta_h[b]; unsigned long long p = block_best_pack_h[b];
                if (p == ULLONG_MAX) continue;
                int isl = (int)((p >> 48) & 0xFFFFULL);
                if (d < isl_best_delta[isl]) { isl_best_delta[isl] = d; isl_best_pack[isl] = p; }
            }

            offset += this_pairs;
        }

        // apply best moves per-island (on host)
        bool any_improve=false;
        for (int isl=0; isl<islands; ++isl) {
            float bd = isl_best_delta[isl]; if (bd >= -1e-9f) continue; // no improving
            any_improve = true;
            unsigned long long p = isl_best_pack[isl];
            int i = (int)((p >> 24) & 0xFFFFFFULL);
            int j = (int)(p & 0xFFFFFFULL);
            // copy island tour & pos to host (small transfer)
            vector<int> tour(n), pos(n);
            CUDA_CHECK(cudaMemcpy(tour.data(), tours_d + isl*n, n*sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pos.data(), pos_d + isl*n, n*sizeof(int), cudaMemcpyDeviceToHost));
            apply_2opt_host(tour, pos, i, j);
            // write back
            CUDA_CHECK(cudaMemcpy(tours_d + isl*n, tour.data(), n*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(pos_d + isl*n, pos.data(), n*sizeof(int), cudaMemcpyHostToDevice));
        }

        if (!any_improve) { global_best_delta = 0.0f; break; }

        if ((iter & 63) == 0) {
            float bestcost = 1e38f; for (int isl=0; isl<islands; ++isl) {
                vector<int> tour(n); CUDA_CHECK(cudaMemcpy(tour.data(), tours_d + isl*n, n*sizeof(int), cudaMemcpyDeviceToHost));
                float c = tour_cost(tour, inst.x, inst.y); if (c < bestcost) bestcost = c;
            }
            printf("iter %d bestcost so far=%.6f
", iter, bestcost);
        }
    }

    CUDA_CHECK(cudaEventRecord(kstop)); CUDA_CHECK(cudaEventSynchronize(kstop));
    float kernel_ms=0; CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kstart, kstop));
    auto t1 = chrono::high_resolution_clock::now(); double wall = chrono::duration<double>(t1-t0).count();

    // extract best tour across islands
    float bestcost = 1e38f; vector<int> besttour; besttour.resize(n);
    for (int isl=0; isl<islands; ++isl) {
        vector<int> tour(n); CUDA_CHECK(cudaMemcpy(tour.data(), tours_d + isl*n, n*sizeof(int), cudaMemcpyDeviceToHost));
        float c = tour_cost(tour, inst.x, inst.y);
        if (c < bestcost) { bestcost = c; besttour = tour; }
    }

    // optional 3-opt polish on CPU
    if (do_3opt) {
        printf("Running CPU 3-opt polish (limited)...
");
        three_opt_polish(besttour, inst.x, inst.y, 20000);
        bestcost = tour_cost(besttour, inst.x, inst.y);
    }

    printf("Done iterations=%d kernel=%.3f ms wall=%.3f s bestcost=%.6f
", iter, kernel_ms, wall, bestcost);
    if (inst.opt > 0) { double err = (bestcost - inst.opt) / inst.opt * 100.0; printf("Known OPT=%.6f error=%.6f%%
", inst.opt, err); }

    // validate
    vector<char> seen(n,0); bool ok=true; for (int i=0;i<n;++i) { int c=besttour[i]; if (c<0||c>=n||seen[c]) { ok=false; break; } seen[c]=1; }
    printf("Tour valid: %s
", ok?"YES":"NO");

    // write benchmark line
    FILE *f = fopen("benchmark.csv","a"); if (f) { fprintf(f, "%s,%d,%.6f,%.6f,%d,%.6f
", inst.name.c_str(), n, kernel_ms/1000.0, wall, iter, bestcost); fclose(f); }

    // cleanup
    cudaFree(dx_d); cudaFree(dy_d); cudaFree(cands_d); cudaFree(tours_d); cudaFree(pos_d);
    cudaFree(block_best_delta_d); cudaFree(block_best_pack_d);
}

// helper: list files in folder
vector<string> list_tsp_files(const string &folder) {
    vector<string> out;
#ifdef _WIN32
    WIN32_FIND_DATAA findData;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    string searchPath = folder + "\*.tsp";
    hFind = FindFirstFileA(searchPath.c_str(), &findData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            string name = findData.cFileName;
            string low = name;
            for (char &c : low) c = tolower(c);
            if (low.size() > 4 && low.substr(low.size() - 4) == ".tsp")
                out.push_back(folder + "\" + name);
        } while (FindNextFileA(hFind, &findData) != 0);
        FindClose(hFind);
    }
#else
    DIR *d = opendir(folder.c_str());
    if (!d) return out;
    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        string name = entry->d_name;
        if (name.size() > 4) {
            string low = name;
            for (char &c : low) c = tolower(c);
            if (low.substr(low.size() - 4) == ".tsp")
                out.push_back(folder + "/" + name);
        }
    }
    closedir(d);
#endif
    sort(out.begin(), out.end());
    return out;
}

int main(int argc, char** argv) {
    if (argc < 7) { printf("Usage: %s <tsp_or_folder> <m> <islands> <max_iters> <init_mode> <do_3opt>
", argv[0]); return 1; }
    string path = argv[1]; int m = atoi(argv[2]); int islands = atoi(argv[3]); int max_iters = atoi(argv[4]); int init_mode = atoi(argv[5]); int do_3opt = atoi(argv[6]);

    vector<string> files;
    struct stat sb; if (stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) { files = list_tsp_files(path); } else files.push_back(path);

    for (auto &f: files) {
        TSPLibInstance inst; if (!parse_tsp_file(f, inst)) { fprintf(stderr, "Failed parse %s
", f.c_str()); continue; }
        process_instance(inst, m, islands, max_iters, init_mode, do_3opt);
    }
    return 0;
}
