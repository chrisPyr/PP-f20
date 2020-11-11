#include <iostream>
#include <pthread.h>
#include <random>
#include <time.h>
#include <stdlib.h>
using namespace std;
long numthread = 1;
long long int times_per_thread = 0;
long long int total_number_in_circle = 0;
pthread_mutex_t mutexsum; 
int seed=1;


unsigned long long int rand(unsigned long long int* hh) // RAND_MAX assumed to be 32767
{
    *hh = (*hh * 1103515245 + 12345);
    return (unsigned long long int)((*hh) %  2147483647) ;
}

void srand(unsigned long long int* gg,unsigned long long int seed)
{
    *gg = seed;
}
void* thread_count_pi(void*){
	//unsigned long long int seed = time(NULL);
	double x;
	double y;
    unsigned long long int next;
	double distance_squared;
	long long int local_number_in_circle;
	srand(&next,seed);
    	seed++;	
	for(long long int toss = 0; toss<times_per_thread;++toss){
    
		//x = (double)rand_r(&seed)/RAND_MAX;
        //y = (double)rand_r(&seed)/RAND_MAX;
        x = (double)rand(&next)/ 2147483646 ;
        y = (double)rand(&next)/ 2147483646 ;
		distance_squared= x*x + y*y;
		if(distance_squared <= 1){
			local_number_in_circle++;
		}
	}

	pthread_mutex_lock(&mutexsum);
	total_number_in_circle += local_number_in_circle;
	pthread_mutex_unlock(&mutexsum);
	return NULL;
}

int main(int argc, char* argv[]){
	double pi_estimate;
	long thread;
	numthread = atol(argv[1]);
	long long int number_of_tosses = atoll(argv[2]);
	times_per_thread = (number_of_tosses/numthread) + (long long int)(number_of_tosses%numthread);
	pthread_t thd[numthread];
	pthread_mutex_init(&mutexsum, NULL);

	for(thread =0;thread < numthread ; thread++){
		pthread_create(&thd[thread],NULL,thread_count_pi , NULL);
	}
	for(int thread = 0; thread<numthread;thread++){
		pthread_join(thd[thread],NULL);
	}
	pi_estimate = 4.0 * total_number_in_circle/((double)number_of_tosses);
	cout<<pi_estimate<<endl;	
	pthread_exit(NULL);
	return 0; 

}
