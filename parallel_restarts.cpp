#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include<iomanip>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <queue>
#include <stack>
#include <deque>
#include <random>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <functional>
#include <limits>
#include<omp.h>

using namespace std;
typedef long long ll;

const ll INF = (ll)1e18;
const int MAX_ITERATIONS = 50;

struct City { double x, y; };

enum InitHeuristic { NEAREST_NEIGHBOR = 0, RANDOM_TOUR = 1 };

vector<vector<int>> distmat;
int n = 0;

// ---------- Utilities ----------
int calculate_tour_cost(const vector<int> &tour) {
    int cost = 0;
    for (int i = 1; i < (int)tour.size(); ++i)
        cost += distmat[tour[i]][tour[i-1]];
    cost += distmat[tour.back()][tour.front()];
    return cost;
}

vector<int> random_tour_construction_mt(std::mt19937 &rng) {
    vector<int> tour(n);
    iota(tour.begin(), tour.end(), 0);
    shuffle(tour.begin(), tour.end(), rng);
    return tour;
}

// NOTE: accepts explicit start index; thread-safe
vector<int> nearest_neighbor_construction_start(int start) {
    vector<int> tour;
    tour.reserve(n);
    vector<char> visited(n, 0);
    int curr = start;
    tour.push_back(curr);
    visited[curr] = 1;
    for (int step = 1; step < n; ++step) {
        int next = -1;
        int bestd = INT_MAX;
        for (int j = 0; j < n; ++j) {
            if (!visited[j] && distmat[curr][j] < bestd) {
                bestd = distmat[curr][j];
                next = j;
            }
        }
        if (next == -1) break; // safety
        tour.push_back(next);
        visited[next] = 1;
        curr = next;
    }
    return tour;
}

vector<int> generate_two_opt_neighbor(vector<int> tour, int i, int j) {
    reverse(tour.begin() + i + 1, tour.begin() + j + 1);
    return tour;
}

vector<int> best_improvement(const vector<int> &tour) {
    // find best (i,j) 2-opt with negative delta and apply
    int best_i = -1, best_j = -1;
    int best_delta = 0; // we want negative
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 2; j < n; ++j) {
            if (i == 0 && j == n - 1) continue;
            int a = tour[i], b = tour[i+1], c = tour[j], d = tour[(j+1)%n];
            int delta = distmat[a][c] + distmat[b][d] - distmat[a][b] - distmat[c][d];
            if (delta < best_delta) { best_delta = delta; best_i = i; best_j = j; }
        }
    }
    vector<int> newt = tour;
    if (best_i != -1) reverse(newt.begin() + best_i + 1, newt.begin() + best_j + 1);
    return newt;
}

vector<int> Iterative_Hill_Climb(vector<int> tour, int tour_cost) {
    vector<int> curr = tour;
    int curr_cost = tour_cost;
    for (int it = 0; it < MAX_ITERATIONS; ++it) {
        vector<int> cand = best_improvement(curr);
        int cand_cost = calculate_tour_cost(cand);
        if (cand_cost >= curr_cost) break;
        curr = move(cand);
        curr_cost = cand_cost;
    }
    return curr;
}

// ---------- I/O / Distance matrix ----------
vector<City> read_tsp_lib(const string &filename) {
    ifstream file(filename);
    vector<City> coords;
    string line;
    bool start = false;
    while (getline(file, line)) {
        if (line.find("NODE_COORD_SECTION") != string::npos) { start = true; continue; }
        if (line.find("EOF") != string::npos) break;
        if (!start) continue;
        stringstream ss(line);
        int id; double x,y;
        if (!(ss >> id >> x >> y)) continue;
        coords.push_back({x,y});
    }
    return coords;
}

void build_distance_matrix(const vector<City> &coords) {
    n = (int)coords.size();
    distmat.assign(n, vector<int>(n, 0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) {
                double dx = coords[i].x - coords[j].x;
                double dy = coords[i].y - coords[j].y;
                distmat[i][j] = (int)llround(sqrt(dx*dx + dy*dy));
            }
}

// ---------- Generate initial tours (deterministic mapping for NN starts) ----------
vector<vector<int>> generate_restart_tours(InitHeuristic h, int num_restarts, std::mt19937 &master_rng) {
    vector<vector<int>> tours;
    tours.reserve(num_restarts);
    if (h == NEAREST_NEIGHBOR) {
        // use different start cities (wrap-around) deterministically
        for (int r = 0; r < num_restarts; ++r) {
            int start = r % n;
            tours.push_back(nearest_neighbor_construction_start(start));
        }
    } else {
        // generate random tours using master rng (but each will be shuffled again thread-locally too)
        for (int r = 0; r < num_restarts; ++r) {
            vector<int> t(n);
            iota(t.begin(), t.end(), 0);
            shuffle(t.begin(), t.end(), master_rng);
            tours.push_back(t);
        }
    }
    return tours;
}

// ---------- Main (parallel across restarts) ----------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string tsp_file = "./tests/berlin52.tsp";
    auto coords = read_tsp_lib(tsp_file);
    if (coords.empty()) { cerr << "Failed to read TSP file: " << tsp_file << "\n"; return 1; }
    build_distance_matrix(coords);
    cout << "Loaded " << n << " cities from " << tsp_file << "\n";

    // ask user for heuristic & num restarts (or default)
    int choice = 0, num_restarts = min(50, max(1, n)); // default
    cout << "Choose heuristic (0 = NN, 1 = Random) [default 0]: ";
    if (!(cin >> choice)) choice = 0;
    if (choice == 1) {
        cout << "Enter number of random restarts: ";
        if (!(cin >> num_restarts)) num_restarts = min(50, max(1, n));
    } else {
        // for NN, cap restarts to n (one per start)
        num_restarts = min(num_restarts, n);
    }
    InitHeuristic h = static_cast<InitHeuristic>(choice);

    // master RNG for generating initial tours (deterministic seed optional)
    std::random_device rd;
    std::mt19937 master_rng(rd());

    auto initial_tours = generate_restart_tours(h, num_restarts, master_rng);

    vector<int> global_best_tour;
    int global_best_cost = INT_MAX;

    auto wall_start = chrono::high_resolution_clock::now();

    // Parallel loop across restarts (index-based)
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < (int)initial_tours.size(); ++idx) {
        // thread-local RNG (seed depends on thread id + rd)
        int tid = omp_get_thread_num();
        std::mt19937 rng(rd() ^ (tid + 0x9e3779b9));

        // each thread works on its own copy
        vector<int> my_initial = initial_tours[idx];

        // If heuristic was NN but you want randomized ties, you could perturb here using rng.

        int init_cost = calculate_tour_cost(my_initial);
        vector<int> my_best = Iterative_Hill_Climb(my_initial, init_cost);
        int my_best_cost = calculate_tour_cost(my_best);

        // Update global best safely
        #pragma omp critical
        {
            if (my_best_cost < global_best_cost) {
                global_best_cost = my_best_cost;
                global_best_tour = my_best;
                // Optionally print progress
                cout << "[thread " << tid << "] new global best: " << global_best_cost << " (restart " << idx << ")\n";
            }
        }
    } // end parallel for

    auto wall_end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = wall_end - wall_start;
    cout << "Parallel wall time: " << elapsed.count() << " s\n";

    cout << "Final best cost: " << global_best_cost << "\n";
    cout << "Tour: ";
    for (int v : global_best_tour) cout << v << " ";
    cout << "\n";
    return 0;
}
