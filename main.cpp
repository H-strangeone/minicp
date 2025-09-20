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

using namespace std;
typedef long long ll;

const ll INF = 1e18;
const int MOD = 1e9 + 7;
const int MAX_ITERATIONS = 50;
const int MIN_ITERATIONS = 10;
// Sequential Implementation of Parallel Hill Climbing Algorithm aimed to solve TSP problem for small n (n <= 10)

struct City {
    double x, y;
};

enum InitHeuristic {NEAREST_NEIGHBOR,RANDOM_TOUR};
vector<vector<int>> dist;
vector<int> restarts = {10, 20, 30};
int n,counter=0;

int random_city()
{
    static random_device rd;
    static mt19937 g(rd()); 
    uniform_int_distribution<int> disti(0, n - 1);
    return disti(g); 
}

int calculate_tour_cost(vector<int> &tour)
{
    int cost = 0;
    for (auto i = 1; i < tour.size(); i++)
    {
        cost += dist[tour[i]][tour[i - 1]];
    }
    // We have to also add the distance cost of the edge connecting the last and the first cities as well;
    cost += dist[tour.back()][tour.front()];
    return cost;
}

vector<int>random_tour_construction(){
    vector<int>tour(n);
    iota(tour.begin(),tour.end(),0);
  // Mersenne Twister PRNG
    static random_device rd;
    static mt19937 g(rd());
    shuffle(tour.begin(), tour.end(), g);
    return tour;
}

// vector<int>greedy_tour_construction(){
//     // To be implemented using edges representation
//     return {}
// }

vector<int>nearest_neighbor_construction(){
    vector<int> tour;
    int start = counter;counter++;
    set<int> unvisited;
    tour.push_back(start);

// Fill unvisited set
    for (int i = 0; i < n; i++)
    if (i != start) unvisited.insert(i);
    
    int curr_city=start;
    while (!unvisited.empty()) {
        int next_city = -1;
        int min_dist = INT_MAX;
        // scan only unvisited cities
        for (int u : unvisited) {
            if (dist[curr_city][u] < min_dist) {
                min_dist = dist[curr_city][u];
                next_city = u;
            }
        }
        curr_city = next_city;
        tour.push_back(curr_city);
        unvisited.erase(curr_city);
    }
    return tour;
}
vector<int> generate_initial_tour(InitHeuristic h)
{   
    vector<int>initial_tour;
    switch (h)
    {
        case NEAREST_NEIGHBOR:
        return nearest_neighbor_construction();
        case RANDOM_TOUR:
        return random_tour_construction();
        default:
            cerr << "Invalid heuristic selected. Defaulting to RANDOM_TOUR.\n";
            return random_tour_construction();
    }
    return initial_tour;
}

vector<vector<int>> generate_restart_tours()
{
    vector<vector<int>> restart_tours;
    vector<int> restart_tour;
    // int NUM_RESTARTS=restarts[0];
    cout << "Choose heuristic: (0=NN, 1=Random) ";
    int choice=0,num_restarts=n;cin>>choice;
    if(choice==1)
    {
        cout<<"Please enter the number of random restarts :  "<<endl;
        cin>>num_restarts;
    }
    InitHeuristic h=static_cast<InitHeuristic>(choice);
    for (auto i = 0; i < num_restarts; i++)
    {
        // For every random restart generate an initial tour;
        restart_tour = generate_initial_tour(h);
        restart_tours.push_back(restart_tour);
    }
    // For n cities the generate initial tour generates a tour sequence with the ith city first
    // and since the heuristic for constructing a tour is deterministic (nearest neighbor) the tours will be identical for the same starting cities.
    // Note : For now , the upper bound for the number of restarts for n city input is n unless construct random tours for iterations>N;
    return restart_tours;
}
vector<int> generate_two_opt_neighbor(vector<int> tour, int i, int j)
{
    vector<int> new_tour(n, 0);
    for (auto k = 0; k <= i; k++)
        new_tour[k] = tour[k];
    for (auto k = j; k >= i + 1; k--)
        new_tour[i + 1 + j - k] = tour[k];
    for (auto k = j + 1; k < n; k++)
        new_tour[k] = tour[k];
    return new_tour;
}

vector<int> best_improvement(vector<int> tour)
{
    // Explore all O(n^2) 2-opt neighbors and evaluate the best neighbor and return based on the cost;
    // T.C.=>(O(n^2)) S.C.=>(O(n))
    int cost_delta=0,best_cost_delta=INT_MAX,best_i=-1,best_j=-1;
    for (auto i = 0; i < n - 1; i++)
    {
        for (auto j = i + 2; j < n; j++)
        {
            if (i == 0 && j == n - 1)
                continue;
            // Delta Cost 2-opt strategy 
            cost_delta=dist[tour[i]][tour[j]]+dist[tour[i+1]][tour[(j+1)%n]]-dist[tour[i]][tour[i+1]]-dist[tour[j]][tour[(j+1)%n]];
            if(best_cost_delta>cost_delta)
            {
                best_i=i;
                best_j=j;
                best_cost_delta=cost_delta;
            }
        }
    }
    // reverse the part of the tour containing the [best_i,best_j]
    if (best_cost_delta<0)
    reverse(tour.begin()+best_i+1,tour.begin()+best_j+1);
    return tour;
}

vector<int> Iterative_Hill_Climb(vector<int> tour, int tour_cost)
{
    vector<int> best_tour;
    vector<int> curr_tour = tour;
    int curr_cost = tour_cost;
    int best_cost;
    const int NUM_ITERATIONS = 50; 
    for (auto i = 0; i < NUM_ITERATIONS; i++)
    {
        // Find the best improvement cost neighbor and compute its cost
        best_tour = best_improvement(curr_tour);
        best_cost = calculate_tour_cost(best_tour);
        if (best_cost == curr_cost)
            break;
        curr_cost = best_cost;
        curr_tour = best_tour;
    }
    // For solutions which might get stuck in local minima and never reach global maxima we need to make some random restarts
    return best_tour;
}

// Read coordinates from TSP_LIB file
vector<City> read_tsp_lib(const string &filename) {
    ifstream file(filename);
    vector<City> coords;
    string line;
    bool start = false;

    while (getline(file, line)) {
        if (line.find("NODE_COORD_SECTION") != string::npos) {
            start = true;
            continue;
        }
        if (line.find("EOF") != string::npos) break;
        if (start) {
            stringstream ss(line);
            int id;
            double x, y;
            ss >> id >> x >> y;
            coords.push_back({x, y});
        }
    }
    return coords;
}

// Build Euclidean distance matrix
void build_distance_matrix(const vector<City> &coords) {
    n = coords.size();
    dist.assign(n,vector<int>(n,0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            double dx = coords[i].x - coords[j].x;
            double dy = coords[i].y - coords[j].y;
            dist[i][j] = (int) round(sqrt(dx*dx + dy*dy));
        }
    }
    return;
}

vector<vector<string>> readInput(string fileName)
{
    vector<vector<string>> data;
    ifstream file(fileName);
    if (!file.is_open())
    {
        cerr << "Failed to open file: " << fileName << std::endl;
        return data;
    }

    string line;
    if (getline(file, line))
    {
        n = stoi(line);
    }
    while (getline(file, line))
    {
        vector<string> row;
        stringstream ss(line);
        string cell;

        while (getline(ss, cell, ','))
        {
            row.push_back(cell);
        }

        data.push_back(row);
    }

    file.close();
    return data;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    // Read input by parsing tsp file and also extract the distance matrix and find the min cost tour
    // vector<vector<string>> data = readInput("./matrix_8.csv");
    // int n1 = data.size(), n2 = data[0].size();
    // cout << n1 << " " << n2 << " " << data[0][0] << "\n";
    // dist.assign(n, vector<int>(n, 0));
    string tsp_file = "./tests/berlin52.tsp";

    vector<City> coords = read_tsp_lib(tsp_file);
    build_distance_matrix(coords);

    cout << "Converted " << tsp_file << " -> " << " distance-matrix"
         << " with " << coords.size() << " cities." << endl;
    for (auto i = 0; i < n; i++)
    {
        for (auto j = 0; j < n; j++)
        {
            cout << dist[i][j] << " ";
        }
        cout << "\n";
    }
    vector<vector<int>> initial_tours = generate_restart_tours();
    vector<int> global_best_tour;
    int global_best_cost = INT_MAX;
    // Wall time : (end time-start time)
    cout<<"The different cities"<<"\n";
    for(auto initial_tour : initial_tours)
    {
        cout<<initial_tour[0]<<" ";
    }
    cout<<"\n";
    auto start = chrono::high_resolution_clock::now();
    for (auto initial_tour : initial_tours)
    {
        cout << "Selected Closed Tour is as follows : " << endl;
        for (auto x : initial_tour)
        {
            cout << x << " ";
        }
        cout << initial_tour[0] << " \n";
        int cost_of_initial_tour = calculate_tour_cost(initial_tour);
        cout << "Cost of selected initial tour is : " << cost_of_initial_tour<<"\n";
        vector<int> foundBestTour = Iterative_Hill_Climb(initial_tour, cost_of_initial_tour);
        int best_cost = calculate_tour_cost(foundBestTour);
        if (global_best_tour.empty()||global_best_cost>best_cost)
        {
            global_best_tour = foundBestTour;
            global_best_cost= best_cost;
        }
        cout <<"Global Best tour found so far: ";
        for (auto c : global_best_tour)
        {
            cout << c << " ";
        }cout<<"\n";
        cout<<"Global best cost found so far: "<<global_best_cost<<"\n";
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Sequential Wall Time: " << duration.count() << " seconds\n";
    cout << "Final optimal low cost tour found after running iterative hill climbing algorithm: ";
    for (auto c : global_best_tour)
    {
        cout << c << " ";
    }
    cout << "\n";
    cout << "Best solution found so far: " << global_best_cost;
    cout << "\n";
    return 0;
}