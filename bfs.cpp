#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include<iostream>
#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1


void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int *depth)
{
    int global_count=0;
    int count_lo[10]={0};
    
    #pragma omp parallel 
    {
    #pragma omp for
    for (int i = 0; i < frontier->count; i++)
    {
        
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];
            int index;
            if (distances[outgoing] == NOT_VISITED_MARKER)
            {   
                
                distances[outgoing]= *depth + 1;
                /*do{
                    index = new_frontier ->count;
                }while(!__sync_bool_compare_and_swap(&(new_frontier->count),index,index+1));
                
                new_frontier->vertices[index]= outgoing;*/
            }
        }

    }
    
    
    int local_count =0;
    int id=omp_get_thread_num();
    int *tmp_in = new int[g->num_nodes];
    #pragma omp for
    for(int i = 0; i< g->num_nodes;++i){
       
        if(distances[i]== (*depth)+1){     
            tmp_in[local_count] = i;    
            local_count++;
        }       
    }
        count_lo[id] = local_count;
        #pragma omp barrier
        
        #pragma omp atomic
        global_count+=local_count;
       

    int start_index=0;
    
    for(int i =0; i < id ; ++i){
        start_index += count_lo[i];
    }


    
    int end_index = start_index + count_lo[id];
    int y =0;
   
    for(int i=start_index;i<end_index;++i){
        new_frontier->vertices[i] = tmp_in[y];
        y++;
    }
    delete [] tmp_in;

    }

     new_frontier->count = global_count;
    
   
    
}
 


// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;
    int depth = 0;
    // initialize all nodes to NOT_VISITED
    
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;


    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances,&depth);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        depth++;

    }
}

void bfs_bottom_up_step(
    Graph g,
    vertex_set *frontier,
    int *distances,
    vertex_set *new_frontier,
    int *depth)
    {    
    //int *tmp = new int[g->num_nodes];
    int count_lo[4]={0};
    int global_count=0;
    #pragma omp parallel
    {
        #pragma omp for
    for(int i =0 ;i < g -> num_nodes;++i){
        
        if(distances[i] == NOT_VISITED_MARKER){
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[i + 1];
            for(int j= start_edge ; j<end_edge ; ++j){
                int k = g->incoming_edges[j];
                int index;
                if(distances[k] == *depth){   
                    
                    /*do{
                        index = new_frontier->count;
                    }while(!__sync_bool_compare_and_swap(&(new_frontier->count),index,index+1));   
                    */
                    //new_frontier->vertices[index] = i;
                    //tmp[index] = distances[k]+1;
                   // new_frontier->count++;
                    distances[i] = (*depth)+1;
                    break;
                }
            }
      }  
       
    }
    
    
    /*for(int i =0 ; i < new_frontier->count;++i){
        distances[new_frontier->vertices[i]]=tmp[i];
    }*/
    
    int local_count =0;
    int id=omp_get_thread_num();
    int *tmp_in = new int[g->num_nodes];
    #pragma omp for
    for(int i = 0; i< g->num_nodes;++i){
       
        if(distances[i]== (*depth)+1){     
            tmp_in[local_count] = i;    
            local_count++;
        }       
    }
        count_lo[id] = local_count;
        #pragma omp barrier
        
        #pragma omp atomic
        global_count+=local_count;
       

    int start_index=0;
    
    for(int i =0; i < id ; ++i){
        start_index += count_lo[i];
    }


    
    int end_index = start_index + count_lo[id];
    int y =0;
   
    for(int i=start_index;i<end_index;++i){
        new_frontier->vertices[i] = tmp_in[y];
        y++;
    }
    delete [] tmp_in;

    }

     new_frontier->count = global_count;
    
   
    
    }
    
    
void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    /*int *adj_front = new int[graph->num_nodes];

        for(int i=0;i<graph->num_nodes;++i){
            adj_front[i]=-1;
        }
    */
    
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);
    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;
    int depth=0;
    #pragma omp parellel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
   
   // adj_front[ROOT_NODE_ID] = 0;

    while(frontier->count!=0){
        vertex_set_clear(new_frontier);
        bfs_bottom_up_step(graph,frontier,sol->distances,new_frontier,&depth) ;
         vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        depth++;

    }
    //delete [] adj_front;
    //delete old_count;
    
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;
    int depth =0;
    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    
    while(frontier->count!= 0){

        int mf=0 ;
        int nf=frontier->count;
        int mu=0;
        
        for(int i =0 ; i< frontier->count;++i){
            
            mf += outgoing_size(graph,frontier->vertices[i]);
        }
        for(int i=0;i<graph->num_nodes;++i){
            if(sol->distances[i] == NOT_VISITED_MARKER) mu++;
        }

        vertex_set_clear(new_frontier);
        if(mf > mu /14) {
            bfs_bottom_up_step(graph,frontier,sol->distances,new_frontier,&depth);
        }else if(nf < graph->num_nodes / 24 ){
            top_down_step(graph,frontier,new_frontier,sol->distances,&depth);
        }

        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        depth++;
        
    }

}
