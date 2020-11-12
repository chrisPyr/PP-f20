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
    int *distances)
{
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

            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                distances[outgoing] = distances[node] + 1;
                int index = new_frontier->count++;
                new_frontier->vertices[index] = outgoing;
            }
        }
    }
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

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

int bfs_bottom_up_step(
    Graph g,
    vertex_set *frontier,
    int *distances,
    vertex_set *new_frontier)
    {
        
        /*int front_num = frontier -> count;

        for(int i = *old; i< front_num;++i){

            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];
            
            for(int j=start_edge; j <end_edge;++j){
                adj_front[ g->outgoing_edges[j]]= frontier -> vertices[i];
            }
        }
    int cnt=0;
        for (int i=0; i < g-> num_nodes ;++i){
            
            
            if( distances[i] == NOT_VISITED_MARKER && adj_front[i] != -1 ){
                
                distances[i]=distances[adj_front[i]]+1;
                frontier->vertices[frontier->count] = i;
                frontier ->count++;
                cnt++;
            }

        }
        if(cnt == 0) return 0;
        *old = (frontier -> count) - cnt;    
        return 1;
    */

    int test =0;
    int *tmp = new int[g->num_nodes];
    for(int i =0 ;i < g -> num_nodes;++i){
        
        if(distances[i] == NOT_VISITED_MARKER){
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[i + 1];

            for(int j= start_edge ; j<end_edge ; ++j){
                int k = g->incoming_edges[j];
                if(distances[k] != NOT_VISITED_MARKER){
                    int index = new_frontier->count++;
                    new_frontier->vertices[index] = i;
                    tmp[index] = distances[k]+1;
                    test++;
                    break;
                }
            }
      }   
    }

    for(int i =0 ; i < new_frontier->count;++i){
        distances[new_frontier->vertices[i]]=tmp[i];
    }
    delete [] tmp;
    if (test==0) return 0;
    return 1;
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

    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
   // adj_front[ROOT_NODE_ID] = 0;

    while(frontier->count!=0){
        vertex_set_clear(new_frontier);
        bfs_bottom_up_step(graph,frontier,sol->distances,new_frontier)== 0 ;
         vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

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

    // initialize all nodes to NOT_VISITED
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
            bfs_bottom_up_step(graph,frontier,sol->distances,new_frontier);
        }else if(nf < graph->num_nodes / 24 ){
            top_down_step(graph,frontier,new_frontier,sol->distances);
        }

        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        
    }

}
