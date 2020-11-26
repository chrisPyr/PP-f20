#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>
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
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    long long int num_pr = tosses / world_size;

    if (world_rank > 0)
    {   
        unsigned int seed = world_rank*time(NULL);
        // TODO: MPI workers
         for(int i=0;i<num_pr;++i){
            x =  2.0 * rand_r(&seed)/RAND_MAX - 1.0 ;
 		    y =  2.0 * rand_r(&seed)/RAND_MAX - 1.0 ;
		    distance_squared= x*x + y*y;
		    if(distance_squared <= 1){
			    number_in_circle++;
		    }
        }
       
        MPI_Send(&number_in_circle, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size-1];
        MPI_Status status[world_size-1];
        unsigned int seed = world_rank*time(NULL);
        long long int local_count[world_size-1];
        // TODO: master
        for(int i=0;i<num_pr;++i){
            x =  2.0 * rand_r(&seed)/RAND_MAX - 1.0 ;
 		    y =  2.0 * rand_r(&seed)/RAND_MAX - 1.0 ;
		    distance_squared= x*x + y*y;
		    if(distance_squared <= 1){
			    number_in_circle++;
		    }
        }
        for(int i=1;i<world_size;++i){
            MPI_Irecv(&local_count[i-1],1,MPI_LONG,i,0,MPI_COMM_WORLD,&requests[i-1]);          
        }
        MPI_Waitall(world_size-1,requests,status);
        for(int i=1;i<world_size;++i){
            global_count+=local_count[i-1];
        }
        global_count+=number_in_circle;
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 *global_count/((double)tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
