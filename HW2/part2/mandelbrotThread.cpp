#include <stdio.h>
#include <thread>

#include "CycleTimer.h"
static inline int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
} WorkerArgs;

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs *const args)
{

    // TODO FOR PP STUDENTS: Implement the body of the worker
    // thread here. Each thread should make a call to mandelbrotSerial()
    // to compute a part of the output image.  For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.
    int j,k;
    int rows_per_time = (args->height)/1200;
    int pass = rows_per_time * (args->threadId);
    int incre = rows_per_time * (args->numThreads);
    int total_rows_num = (signed int)(args->height)-1;
    for(int i = pass; i <= total_rows_num ; i+=incre){
        k= i;
        if(k+=incre <=total_rows_num){ //if I use(k+=incre), ans will fail, it's weird
		mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width, args->height,i ,rows_per_time, args->maxIterations, args->output);
	}else{
            j=((args->height) % rows_per_time);
            mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width, args->height,i ,j, args->maxIterations, args->output);
        }
    }
    

/*
    int pass=(args->height)/(args->numThreads)*(args->threadId);
    int total_rows= (args->height)/(args->numThreads);
    if(args->threadId==args->numThreads-1) total_rows+=(args->height)%(args->numThreads);
    double startTime = CycleTimer::currentSeconds();
    mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width, args->height,pass ,total_rows, args->maxIterations, args->output);
    double endTime = CycleTimer::currentSeconds();
    printf("Thread %d took %.3f ms to finish\n",args->threadId, (endTime-startTime)*1000);
    */
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i = 0; i < numThreads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;

        args[i].threadId = i;
    }
    
    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 0; i < numThreads; i++)
    {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }

   // workerThreadStart(&args[0]);
    // join worker threads
    for (int i = 0; i < numThreads; i++)
    {
        workers[i].join();
    }
}
