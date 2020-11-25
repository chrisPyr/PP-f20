#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>
#include <math.h>
int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    double x,y,distance_squared;
    long long int number_in_circle =0;
    long long int global_count=0;
    long long int local_count=0;
    long long int tmp;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    long long int num_pr = tosses / world_size;
    int k =0;
    int size = world_size;
    while(size!=1){
        size/=2;
        k++;
    }
    
    // TODO: binary tree redunction
    unsigned int seed=world_rank*time(NULL);
    for(int i=0;i<num_pr;++i){
            x =  2.0 * rand_r(&seed)/RAND_MAX - 1.0 ;
 		    y =  2.0 * rand_r(&seed)/RAND_MAX - 1.0 ;
		    distance_squared= x*x + y*y;
		    if(distance_squared <= 1){
			    number_in_circle++;
		    }
            local_count=number_in_circle;
    }

    for(int i=0;i<k;++i){
        if(world_rank%((int)pow(2,(i+1)))==0){
            MPI_Recv(&tmp, 1, MPI_LONG, world_rank+pow(2,i), 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            local_count+=tmp;
        }else if(world_rank%(int)pow(2,(i+1))==(int)pow(2,i)){
            MPI_Send(&local_count, 1, MPI_LONG,world_rank-pow(2,i), 0, MPI_COMM_WORLD);

        }
    }
    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 *local_count/((double)tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
