# include <iostream>
#include "stdafx.h"
#include "data_analysis.h"
#include "config.h"
#include "carto.h"
#include "data_reading.h"
#include "form.h"
#include "frame.h"
#include <FL/Fl.H>
#include <physycom/time.hpp>
#include <jsoncons/json.hpp>

struct analyzer{
    config config_;
    // DATA READING
    std::vector<poly_base> poly; // polies analysis

    std::vector<activity_base> activity; // activity
    std::vector<polygon_base> polygon;
    std::map<unsigned long long int, int> poly_cid2lid;
    // CARTO
    double lat0, lon0, dlat, dlon, zoom_start;
    std::vector <node_base> node;
    std::vector <arc_base> arc;
    std::map<unsigned long long int, int> node_cid2id;
    std::vector<std::vector<mapping_base>> A; int jmax, imax; // A[jmax][imax]
    double ds_lat, ds_lon, c_ris1, c_ris2;
    // DATA ANALYSIS
    std::vector<traj_base> traj;
    std::vector<presence_base> presence;
    std::vector<centers_fcm_base> centers_fcm;
    std::vector<traj_base> pawns_traj;
    std::vector<traj_base> wheels_traj;
    double sigma = 0.0;
    std::vector<cluster_base> data_notoncarto;
    data_loss dataloss;
    std::map<string, vector<int>> subnets;
    std::vector<std::map<int,int>> classes_flux;

};