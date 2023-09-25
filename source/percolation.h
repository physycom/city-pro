#pragma once
#ifndef PERCOLATION_H
#define PERCOLATION_H
#include <vector>
#include <map>
#include <vector>
#include <boost/graph/adjacency_list.hpp>
#include<carto.h>

/*
GOAL:
    Percolation:
    1) p = P(df_ij/dn<0) [CANNOT DO -> Gonzalez]
    2) p = P(v_ij/v_exp<q) [Try: Havlin]
STEPS:
    1) Download velocity data from OSM
    2) Initialize the graph.
    3) For_each t
        3a) For_each q
            3aa) Eliminate links that satsfy Percolation(2)
            3aa) Save graph_tq
    Initialize:
    P(n_c|q,t) -> couple
    4)For_each t
        4a) For_each q
            4aa) Compute the number of connected components -> push_back(P(n_c|q,t))
 
*/

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                            boost::no_property, boost::property<boost::edge_weight_t, float>> Graph; // graph where to initialize the percolated graph

struct Percolation{
    //GRAPHS
    std::vector<Graph> g_tq;

    std::vector<int>::iterator qi; // Iterator pointing at the ith q for which I decided to compute the percolation èrpcess
    // CONNECTED COMPONENTS
    std::vector<std::vector<int>> time_q_connectedcomps; // For [time][qi] = number of connected components
    std::vector<std::vector<int>> time_q_NEdges1Component; // For [time][qi] = number of edges first connected components
    std::vector<std::vector<int>> time_q_NEdges2Component; // For [time][qi] = number of edges first connected components

    void apply_percolation(std::vector<poly_base>poly,float q);
    void _initialize_time_q_connectedcomps()
    void _initialize_edge_state();
    void _get_data(std::vector<int> polies);
    int _compute_number_connected_components();
    std::map<int,int> _compute_dimension_connected_components();

}

struct city_graph{
    // BOOST GRAPH
    Graph g;
    int t; // time in which I am considering the net.
    void initialize_graph(std::vector<int> polies,int t);
    int q; // traffic parameter to induce probability of percolation
    int number_connected_components; // numbe of connected components in the percolated graph.
    std::map<int,int> component2dimension; // {component0: n_nodes0,...,componentN: n_nodesN} nodes <-> edges
    void PercolationByNode();
    Graph PercolationByEdges(float percentage_velocity);
    // MATRIX REPRESENTATION
    std::vector<std::vector<int>> begin_matrix_B;
    std::vector<std::vector<int>> end_matrix_E;
    std::vector<std::vector<int>> incidence_matrix_I;
    std::vector<std::vector<int>> adjacency_matrix_A;
    std::vector<std::vector<int>> laplacian_matrix_L;
    std::vector<std::vector<int>> distance_matrix_D;
    void _initialize_matrix(std::vector<poly_base>poly);
    // MATRIX REPRESENTATION DUAL
    std::vector<std::vector<int>> dual_begin_matrix_B;
    std::vector<std::vector<int>> dual_end_matrix_E;
    std::vector<std::vector<int>> dual_incidence_matrix_I;
    std::vector<std::vector<int>> dual_adjacency_matrix_A;
    std::vector<std::vector<int>> dual_laplacian_matrix_L;
    std::vector<std::vector<int>> dual_distance_matrix_D;



}


