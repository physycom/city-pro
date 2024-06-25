#pragma once

#include "stdafx.h"
#include "carto.h"
# include "config.h"
//extern config config_;

// RECORD //
struct record_base {
  size_t itime;
  int indx, state;
  double lat, lon, t;
  string type;
  double speed;
  double accel;
};

// ACTIVITY //
struct activity_base {
//in the code I have vector<activity_base> activity;
//for &r:
//activity.push_back()

  bool ok = true;
  long long int id_act;
//dt is of interest as it is one of the parameters that influences the choiche of different travellers
  int indx, dt;  
  double length;
  double average_speed;  
  vector <record_base> record;
};

// PRESENCE //
struct presence_base {  
  double lat, lon;
  long long int id_act;
  size_t itime_start, itime_end;
  int row_n_rec;
  presence_base() {};
  presence_base(double lat_, double lon_, long long int id_act_, size_t t_start, size_t t_stop, int row_n_rec_);
};

// HEADING //
struct heading_base {
  double x = 1.0, y = 0.0;
};

// CLUSTER //
struct cluster_base {
  vector<record_base> points;
  record_base centroid;
  size_t duration;
  heading_base heading;
  double inst_speed; //inst speed from previous cluster, the first is always 0.0;
  double inst_accel; 
  bool on_carto = true;
  bool visited = false;
  polyaffpro_base pap;
  void add_point(record_base rec);
  cluster_base() {};
  void replace(const cluster_base &other) {
    centroid = other.centroid;
    duration = other.duration;
    heading = other.heading;
    inst_speed = other.inst_speed;
    inst_accel = other.inst_accel;
    on_carto = other.on_carto;
    visited = other.visited;
    pap = other.pap;
  }
};

heading_base measure_heading(cluster_base c1, cluster_base c0);


// TRAJ //
struct traj_base {
  bool ok = true;
  long long int id_act;
  int n_poly_unique;
  double length;
  int time;
  int row_n_rec;
  vector <record_base> record;
  vector <cluster_base> stop_point;
  string type;
  list <pair<int, double>> path;      // <id_poly, time entering(decimal hours)>
  void add_cluster(cluster_base &C , int n);
  double average_speed;
  double average_inst_speed; // average speed of the inst_speeds of the clusters
  double average_accel; // average acceleration of the inst_accel of the clusters
  double v_max;
  double v_min;
  double a_max;
  double a_min;
  double sigma_speed;
  double sigma_accel;
  double sinuosity;
  vector<double> p_cluster;
  int means_class;
  int means_new_class;
  double means_p;
  std::map<int,int> class_countpoly;
  
};
//----------------------------------------------------------------------------------------
double distance_record(record_base r1, record_base r2);
//----------------------------------------------------------------------------------------
// DATALOSS //
struct data_loss {
  int n_data_tot = 0;
  int n_data_meter = 0;
  int n_data_outcarto = 0;
  int n_data_oncarto = 0;
  int n_data_single_record = 0;
  int n_data_threshold = 0;
  int n_data_no_single_record = 0;

  int n_traj_tot = 0;
  int n_traj_poly_thresh = 0;
  void dump();
};

//----------------------------------------------------------------------------------------

