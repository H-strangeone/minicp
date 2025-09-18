#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <queue>
#include <stack>
#include <deque>
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

vector<vector<int>> dist;
vector<int> restarts = {10, 20, 30};
int n;

int random_city(int n)
{
    return rand() % n;
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

vector<int> generate_initial_tour()
{
    vector<int> tour;
    // Using random starting city :
    int start = random_city(n);
    set<int> unvisited;
    tour.push_back(start);
    for (auto i = 0; i < n; i++)
    {
        if (i == start)
            continue;
        unvisited.insert(i);
    }
    while (!unvisited.empty())
    {
        // use the nearest neighbor selection criteria for choosing next city in the tour
        set<pair<int, int>> closer_cities;
        for (auto i = 0; i < n; i++)
        {
            if (i == start)
                continue;
            closer_cities.insert({dist[start][i], i});
        }
        for (auto c : closer_cities)
        {
            if (unvisited.find(c.second) != unvisited.end())
            {
                // Closest City not yet visited and is added to the tour
                start = c.second;
                unvisited.erase(c.second);
                tour.push_back(start);
                break;
            }
        }
    }
    // push the last city of the tour as well
    return tour;
}

vector<vector<int>> generate_restart_tours()
{
    vector<vector<int>> restart_tours;
    vector<int> restart_tour;
    // int NUM_RESTARTS=restarts[0];
    for (auto i = 0; i < n; i++)
    {
        // For every random restart generate an initial tour;
        restart_tour = generate_initial_tour();
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
    // T.C.=>(O(n^3)) S.C.=>(O(n))
    vector<int> best_tour = tour;
    int best_cost = calculate_tour_cost(tour);
    vector<int> neighbor;
    int cost = 0;
    for (auto i = 0; i < n - 1; i++)
    {
        for (auto j = i + 2; j < n; j++)
        {
            if (i == 0 && j == n - 1)
                continue;
            neighbor = generate_two_opt_neighbor(tour, i, j);
            cost = calculate_tour_cost(neighbor);
            if (cost < best_cost)
            {
                best_tour = neighbor;
                best_cost = cost;
            }
        }
    }
    return best_tour;
}

vector<int> Iterative_Hill_Climb(vector<int> tour, int tour_cost)
{
    vector<int> best_tour;
    vector<int> curr_tour = tour;
    int curr_cost = tour_cost;
    int best_cost;
    const int NUM_ITERATIONS = 50; // Lets say 25 for now since 20<25<50 and it lies between the above specified limits of iterations
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
    // seeding RNG
    srand(time(0));
    // Read input by parsing tsp file and also extract the distance matrix and find the min cost tour
    vector<vector<string>> data = readInput("./matrix_8.csv");
    int n1 = data.size(), n2 = data[0].size();
    cout << n1 << " " << n2 << " " << data[0][0] << "\n";
    dist.assign(n, vector<int>(n, 0));
    for (auto i = 0; i < n; i++)
    {
        for (auto j = 0; j < n; j++)
        {
            dist[i][j] = stoi(data[i][j]);
            cout << dist[i][j] << " ";
        }
        cout << "\n";
    }
    vector<vector<int>> initial_tours = generate_restart_tours();
    // vector<int>initial_tour=generate_initial_tour();
    vector<int> global_best_tour;
    int global_best_cost = INT_MAX;
    // Wall time : (end time-start time)
    auto start = std::chrono::high_resolution_clock::now();
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
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Sequential Wall Time: " << duration.count() << " seconds\n";
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