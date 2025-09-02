#include<iostream>
#include<cstdlib>
#include<cmath>
#include<ctime>
#include<omp.h>
using namespace std;

double cost(int x){
    return x*x;
}
int neighbour(int x){
    int step=(rand()%21)-10;
    return x+step;
}
int main(){

    srand(time(0));
    int num_threads = 4;
    int best_global=0;
    double bestcost_global = numeric_limits<double>::max();
    #pragma omp parallel for
    for(int i=0;i<num_threads;i++){
        unsigned seed = time(0) + omp_get_thread_num()*1337;
        srand(seed);
        int current=(rand()%201)-100;
        double currentcost=cost(current);
        #pragma omp critical
        {   cout<< "Thread " << omp_get_thread_num() << " initial solution: x = " << current << ", cost = " <<      currentcost << endl;
        }
        double t=1000;
        double cooling=0.99;
        int best=current;
        double bestcost=currentcost;
        while(t>1e-3){
            int next=neighbour(current);
            double newcost=cost(next);
            if(newcost<bestcost|| exp((currentcost-newcost)/t)>((double)rand()/RAND_MAX)){
                current=next;
                currentcost=newcost;
                if(newcost<bestcost){
                    best=next;
                    bestcost=newcost;
                }
            }
            t=t*cooling;
        }
        #pragma omp critical
        {
            cout << "Thread " << omp_get_thread_num()
                 << " best local solution: x = " << best
                 << ", cost = " << bestcost << endl;
            if(bestcost < bestcost_global){
                best_global = best;
                bestcost_global = bestcost;
            }
        }
    }
    cout << "\nGlobal Best solution: x = " << best_global
         << ", cost = " << bestcost_global << endl;
}