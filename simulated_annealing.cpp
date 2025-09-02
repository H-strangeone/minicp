#include<iostream>
#include<cstdlib>
#include<cmath>
#include<ctime>

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
    int current=(rand()%201)-100;
    double currentcost=cost(current);
    cout << "Initial solution: x = " << current << ", cost = " << currentcost << std::endl;
    double t=1000;
    double cooling=0.99;
    int best=current;
    double bestcost=currentcost;
    while(t>1000*cooling*(pow(cooling,10))){
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
        cout << "Best solution: x = " << best << ", cost = " << bestcost << std::endl;
    }
}