#include<iostream>
#include<stdlib.h>
#include <limits.h>
using namespace std;

int main(){
	double x;
	double y;
	double distance_squared;
	double pi_estimate;
	long long int number_of_tosses= 3000000000;
	long long int number_in_circle = 0;
	for(long long int toss=0;toss < number_of_tosses;toss++){
		x =  2.0 * rand()/RAND_MAX - 1.0 ;
 		y =  2.0 * rand()/RAND_MAX - 1.0 ;
		distance_squared= x*x + y*y;
//	cout<<toss<<" "<<x<<" "<<y<<endl;
		if(distance_squared <= 1){
			number_in_circle++;
		}
	}
	
	pi_estimate = 4 *number_in_circle/((double)number_of_tosses);
	cout<<pi_estimate<<endl;	
	return 0; 

}
