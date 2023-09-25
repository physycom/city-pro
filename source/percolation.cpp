#include <percolation.h>
#include <carto.h>
extern std::vector<poly_base> poly;

/////////////////////////////////////////// CITY GRAPH ////////////////////////////////////////////////
city_graph::city_graph(std::vector<poly_base>poly,int time){
    t = time;
    for (auto p : poly) {
        int source = p->node_F;
        int target = p->node_T;
        

        boost::add_edge(source, target, velocity, this->g);
    }

}
city_graph::PercolationByEdges(float percentage_velocity){

    
}

//////////////////////////////////////////// PERCOLATION ///////////////////////////////////////////////
Percolation::Percolation(){

}

Percolation::Percolation(std::vector<int> polies,int q){
    for(auto &p = polies.begin();p!=polies.end()){
        node_state.push_back(std::pair(p,false))
    }
}

Percolation::_initialize_node_state(){

}

Percolation::_get_data(std::vector<int> polies){

}
Percolation::_compute_dimension_connected_components(){
    
}

Percolation::apply_percolation(std::vector<poly_base>poly,int t){
    for(int t=0;t<time;t++){
        city_graph city(std::vector<poly_base>poly,int t);
        for(auto q:vecq){
            g_tq.push_back(PerocolationByEdges(q))
        }
    }
    


}

Percolation::_compute_dimension_connected_components(){

}

// MATRIX REPRESENTATION
city_grap::_initialize_matrix(std::vector<poly_base> poly){
    for(auto p:poly){
        int source = p->node_F;
        int target = p->node_T;
        begin_matrix_B[source][target] = 1;
        end_matrix_E[target][source] = 1;
        incidence_matrix_I[source][target] = 1;
        incidence_matrix_I[target][source] = 1;
        adjacency_matrix_A[source][target] = 1;
        adjacency_matrix_A[target][source] = 1;
        laplacian_matrix_L[source][target] = 1;
        laplacian_matrix_L[target][source] = 1;
        distance_matrix_D[source][target] = 1;
        distance_matrix_D[target][source] = 1;
    }
}