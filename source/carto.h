#pragma once
#include "stdafx.h"
#include "global_params.h"
#include <jsoncons/json.hpp>
#include "config.h"
//#include "dato.h"

using namespace std;
struct poly_base;
struct arc_nase;
// POINT //
struct point_base {
  double lon, lat;
  point_base();
  point_base(const double &lat, const double &lon);
  point_base(const int &lat, const int &lon);
};


// ARCAFFPRO //
struct arcaffpro_base {

  // a = affinity; s = distance from start of poly; d = distance from point considered(lat,lon)
  arcaffpro_base();
  double s, d, a;
  void set(double s, double d, double a) { this->s = s; this->d = d; this->a = a; }
  void add(arcaffpro_base aap);
};

// ARC //
struct arc_base {
  int id_poly;
  double dx, dy, length, s0;
  point_base a, b;
  void set(int id_poly, double s0, point_base a, point_base b,std::vector<poly_base> &poly);
  double measure_dist(double x, double y);
  bool measure_affinity(double lon, double lat, arcaffpro_base &aap);
};

// FL2D
struct Fl2D {
  double FT, TF;
  Fl2D();
};

// POLY //
struct poly_base {
  unsigned long long int cid_poly;
  unsigned long long int cid_Fjnct, cid_Tjnct;
  int id, id_local;
  int type, oneway; // oneway=0,1,2,3 --- doppio senso,FT,TF,ZTL
  int node_F, node_T;
  double length;
  double weightFT, weightTF;
  string name;
  int cntFT, cntTF;
  bool ok = true;
  bool             visible = false;
  bool             path_visible = false;
  int              n_traj_TF, n_traj_FT;
  std::map<int, int> classes_flux;
  std::vector<std::pair<int, int>> timed_fluxes;
  std::vector<std::pair<std::vector<double>, std::vector<double>>> all_velocity;
  std::vector<std::pair<double, double>> density,velocity,time_percorrence;
  std::vector<std::pair<std::vector<double>, std::vector<double>>> all_time_percorrence;
  std::vector<std::pair<int,int>> subnets_couple;
  //ALBI
  std::vector<double> velocities;
  double av_velocity = -1;
  std::map<int,std::vector<double>> time2velocities;
  std::map<int,double> time2av_vel;
  std::map<int,double> time2timepercorrence;


  // ALBI
  void set(int id_, unsigned long long int cid_, vector <point_base> punto_);
  void set(unsigned long long int cid_Fjnct_, unsigned long long int cid_Tjnct_, float meters_, int oneway_, string name_);
  void set(unsigned long long int cid_Fjnct_, unsigned long long int cid_Tjnct_);

  map <string, Fl2D> fluxes;
  double           diffMO_EV = 0;
  vector <point_base> points, delta_points;
  poly_base(void);
  void reverse();
  void clear(void);
  void measure_length(void);
};

// POLYAFFPRO //
struct polyaffpro_base {
  int id_poly;
  double a, d, s;
  list <pair<int, double>> path; // pair< id_poly, time_in >
  double path_weight;
  void clear();
};// describe poly crossed and crossed time

//NODE
struct node_base {
  node_base(void);
  ~node_base(void);
  unsigned long long int cid_node;
  int id_node;
  double lon, lat;
  vector <int> id_nnode, id_nlink;
  void set(unsigned long long int cid_node, int id_node);
  void add_link(int id_nv, int id_l);
  void remove_link(void);
  int get_n_link();
  double measure_dist(double lon, double lat);
};

//---------------------------------------------------------------------
struct node_near_base {
  int id_node;
  double distance;
};
//---------------------------------------------------------------------
bool comp_near_node (const node_near_base& a, const node_near_base& b);

// MAPPING
struct mapping_base {
  vector <int> node_id, arc_id, traj_end, traj_start;
  //ALBI
  vector <int> poly_id;
  //ALBI
};

int A_put_arc(int n);
int A_put_node(int n,std::vector<node_base> &node,config &config);
void make_mapping(std::vector<poly_base> &poly,std::vector<node_base> &node);
void make_arc(std::vector<poly_base> &poly);
int  find_near_poly(double x, double y, double &dist, int &id_poly);
bool find_near_node(double x, double y, double &dist, int &id_nodo,std::vector<node_base> &node);
bool find_polyaff(double lon, double lat, polyaffpro_base &pap);
void map_poly2grid(vector<int> poly_sub,std::vector<poly_base> &poly);

// POLYGON //
//----------------------------------------------------------------------------
struct polygon_base {
  int id;
  int tag_type; //1 for center area, 2 for coasts, 3 for other cities, 4 station
  std::vector<point_base> points;
  double area, perimeter;
  std::map<std::string, std::string> pro;
  std::vector<int> polylid_in;
  polygon_base();
  polygon_base(const jsoncons::json &feature);
  int is_in_wn(double lat, double lon); // wn!=0 inside polygon, wn==0 outside polygon
};
//---------------------------------------------------------------------------- 


// METHODS
void set_geometry(config &config_);
void make_node(std::vector<poly_base> &poly,std::map<unsigned long long,int> &node_cid2id,std::vector<node_base> &node);
void dump_poly_geojson(const std::string &basename,std::vector<poly_base> &poly);
//----------------------------------------------------------------------------
FILE *my_fopen(char *fname, char *mode);