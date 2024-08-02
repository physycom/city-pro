/* Copyright (C) Nico Curti, Alessandro Fabbri - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Nico Curti <nico.curti2@unibo.it>
* Alessandro Fabbri <alessandro.fabbri27@unibo.it>
* Semptember 2017
*/


#ifndef __UTILS_FEATURESEL_HPP__
#define __UTILS_FEATURESEL_HPP__

#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <boost/function.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/iteration_macros.hpp>

#define ENABLE_PERF false

#if ENABLE_PERF
constexpr int delta_node = 15;
#endif

typedef typename boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::property<boost::vertex_color_t, int>, boost::property<boost::edge_index_t, int>> Graph;//la parola graph è il tipo che la precede
using V = Graph::vertex_descriptor;
// View of Graph that keep_all vertices that satisfy the function = True
using Filtered = boost::filtered_graph<Graph, boost::keep_all, boost::function<bool(V)> >;

/** 
  @brief Feature Selection: It is a template function that determines the subnetwork of homogeneous classes according to the
    4 features from the Fcm Fuzzy clustering algorithm.
    @param ENABLE_PERF: a boolean that represents the choice to divide the subnetworks by:
      In main-city-pro_subnet.cpp:
      NOTE: Given POLY_STAT_TYPE = 0
      (True) -> sub_type {tot} -> sub_fraction {0.1,0.15,0.2} 
      (False) -> sub_type {tot}
      NOTE: POLY_STAT_TYPE = 1
      (True) -> sub_type { "tot", "tot_it", "tot_st" } -> sub_fraction {0.1,0.15,0.2} 
      NOTE: Refer to main_city-pro_subnet.cpp for more information
      NOTE: set to FALSE: default 
  @param Ig: a pointer to a pointer of type T, where T is a generic type. NOTE: Used as [{poly_lid}][{int(NF),int(NT)}]
  @param n_link: an integer that represents the number of links in the network (poly.size())
  @param min_size: an integer that represents the minimum size of the nodes (f*nodes.size()) | f = 0.1,0.15,0.2
  @output: a map {sub_type: vec<[nodeFi,nodTi]>}of string and vector of pair of int, where the string is the label of the subnetwork and the pair of int is the node of the subnetwork
**/
template<typename T>
std::map<std::string, std::vector<std::pair<int, int>>> FeatureSelection(T **Ig, const int &n_link, const int &min_size, bool pruning = true, bool merged = true)//int **i (numero poly,indice polyfront[0],indice tail[1],numero poly,numero_vertici)
{//std::vector(edge)<std::pair<int(nodo1),int(nodo2)>>
  Graph G;
  std::set<V> removed_set;
  // Declaration: Keep the graph that does not contain removed vertices
  Filtered Signature(G, boost::keep_all{}, [&](V v) { return removed_set.end() == removed_set.find(v); }); 
  int L = 0, Ncomp, leave, i = 0;
  int max_key;
  // Declaration: core -> polies that is going to form the subnetwork of homogeneous class.
  // Declaration: components -> Contains the index of the component of each vertex. 
  std::vector<int> components, core;
  std::map<std::string, std::vector<std::pair<int, int>>> sub;

#if ENABLE_PERF
  int min_num = 1;
#else
  int min_num = min_size;
#endif

  while (i < n_link)
  {
    // Add the link to the graph G. NOTE: The indices are ordered by flux. 
    // Ig[i][0] -> polyfront, Ig[i][1] -> polytail
    boost::add_edge(Ig[i][0], Ig[i][1], G);
    // For each added edge:
    while (pruning)
    {
      leave = 0;
      /** 
       Control all vertices if are in the removed_set from the filtered view.
       Description: 
       NOT SURE
      The algorithm starts with G having the edge with the highest flux. i = 0
      G: edges = [(nod0,nod1)], nodes = [nod0,nod1]
      Signature: [node0,node1]
      i = 1 
      G: edges = [(nod0,nod1),(nod2,nod3)], nodes = [nod0,nod1,nod2,node3]
      If node2 or node3 = node1 or node0, then do not remove them as they complete a connected component.
      Otherwise, remove them.
      ....
      NOTE: In this way in the end we will have in the filtered graph just those nodes that are shared by more than one edge.
      NOTE: The number of connected components can be bigger than 1.
      NOTE: removed_set is the set of dangling vertices. (leaf vertices)
      */
      BGL_FORALL_VERTICES(v, Signature, Filtered)
        if (boost::in_degree(v, Signature) < 2)
        {
          removed_set.insert(v);
          ++leave;
        }
      if (leave == 0)
        break;
    }
    // If the graph 
    if (num_vertices(G) - removed_set.size()) 
    {
      // components.size() = number leaf vertices (removed_set) + number of vertices in the signated graph
      components.resize(num_vertices(G));
      // Number of components of the Signated graph: if i = 0, then Ncomp = 0
      Ncomp = boost::connected_components(Signature, &components[0]); 
      if (merged)
      {
        BGL_FORALL_VERTICES(v, Signature, Filtered)
          if (boost::in_degree(v, Signature))
            core.push_back((unsigned int)v);
      }
      else
      {
        std::map<unsigned int, unsigned int> size;
        for (auto &j : components)
          ++size[j];
        max_key = std::max_element(std::begin(size), std::end(size), [](const decltype(size)::value_type & p1, const decltype(size)::value_type & p2) {return p1.second < p2.second; })->first;
        BGL_FORALL_VERTICES(v, Signature, Filtered)//each time Signature and Filtered are evaluated?
          if (components[v] == max_key)
            core.push_back((unsigned int)v);
      }
      L = (int)core.size();
    }

    if (L >= min_num)
    {
      std::stringstream lbl;
      lbl << std::setw(5) << std::setfill('0') << min_num << "_" << std::setw(5) << std::setfill('0') << L;
      BGL_FORALL_EDGES(l, Signature, Filtered)
      {
        if( components[l.m_source] == max_key && components[l.m_target] == max_key )
          sub[lbl.str()].push_back(std::make_pair((int)l.m_source, (int)l.m_target));
      }
#if ENABLE_PERF
      min_num += delta_node;
      if ( min_num > min_size ) break;
#else
      break;
#endif
    }
    components.resize(0);
    ++i;
    removed_set.clear();
    core.resize(0);
  }

  return sub;
}

#endif // __UTILS_FEATURESEL_HPP__
