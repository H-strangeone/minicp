#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cuda_runtime.h>
#define MAX_N 12            
#define MAX_OPEN 2000000    
#define MAX_THREADS 1024
struct Node {
    unsigned int visited_mask; 
    int last;                  
    float g;                   
    float f;                  
};

__device__ Node d_open[MAX_OPEN];
__device__ int d_open_head=0; 
__device__ int d_open_tail=0; 

__device__ float d_best_cost=1e30f; 

__device__ __forceinline__ float heuristic_lower_bound(int n, unsigned int visited_mask, const float *dist) {
    return 0.0f;
}
__global__ void naive_a_star_kernel(int n, const float *dist_matrix) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (true) {
        int idx=atomicAdd(&d_open_head,1);
        if (idx> d_open_tail) {
            return;
        }
        Node cur = d_open[idx];
        float global_best=d_best_cost;
        if (cur.g>=global_best) continue;
        if ((unsigned int)cur.visited_mask==((1u<<n)-1)) {
            float tour_cost=cur.g+dist_matrix[cur.last*n+0];
            float old=atomicMin((int*)&d_best_cost, __float_as_int(tour_cost)); 
            continue;
        }
        for (int nb=0;nb<n;++nb) {
            if (cur.visited_mask & (1u<<nb)) continue;
            Node nxt;
            nxt.visited_mask=cur.visited_mask | (1u<<nb);
            nxt.last=nb;d 
            nxt.g=cur.g+dist_matrix[cur.last*n+nb];
            nxt.f=nxt.g+heuristic_lower_bound(n, nxt.visited_mask, dist_matrix);
            int push_idx=atomicAdd(&d_open_tail, 1);
            if (push_idx<MAX_OPEN) {
                d_open[push_idx]=nxt;
            } else {
            }
        }
    }
}
void run_naive(int n, float *h_dist) {
    float *d_dist;
    cudaMalloc(&d_dist,n*n*sizeof(float));
    cudaMemcpy(d_dist, h_dist, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_open_head, (int[]){0}, sizeof(int));
    cudaMemcpyToSymbol(d_open_tail, (int[]){1}, sizeof(int));
    float inf=1e30f;
    cudaMemcpyToSymbol(d_best_cost, &inf, sizeof(float));
    Node start;
    start.visited_mask=1u<<0;
    start.last=0;
    start.g=0.0f;
    start.f=0.0f;
    cudaMemcpyToSymbol(d_open, &start, sizeof(Node)); 
    dim3 blocks(8);
    dim3 threads(128);
    naive_a_star_kernel<<<blocks,threads>>>(n, d_dist);
    cudaDeviceSynchronize();
    float best;
    cudaMemcpyFromSymbol(&best, d_best_cost, sizeof(float));
    printf(" Best tour cost found (maybe): %f\n", best);
    cudaFree(d_dist);
}
int main() {
    int n=8;
    float h_dist[MAX_N*MAX_N];
    srand(123);
    for (int i=0;i<n;++i) {
        for (int j=0;j<n;++j) {
            if (i==j) h_dist[i*n+j] = 0.0f;
            else h_dist[i*n+j] = 1.0f+(rand()%100)/10.0f;
        }
    }
    run_naive(n, h_dist);
    return 0;
}
