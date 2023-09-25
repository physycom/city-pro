#pragma once
#include "record.h"
// ALBI
#include "carto.h"
#include "config.h"
// ALBI
#include <physycom/string.hpp>

//---------------------------------------------------------------------

// SEED //
struct seed_pro_base
{
  double distance;
  int id_node, node_bv, link_bv;
  void set(int id_node, int node_bv, int link_bv, double distance);
};
//---------------------------------------------------------------------
struct seed_base
{
  double dd; // dd = dist + d_eu (d_eu= a_eu*dist, 0 < a_eu < 1)
  int cnt;
  bool operator<(const seed_base &s) const { return dd > s.dd; }
};

// CENTERS FCM //
struct centers_fcm_base
{
  vector<double> feat_vector;
  double sigma = 0.0;
  int cnt = 0;
  int idx;
  std::map<std::string, int> cnt_polygons;
};
//---------------------------------------------------------------------
// POLYSTAT //
struct polystat_base
{
  enum
  {
    OFFSET_ID = 0,
    OFFSET_IDLOC = 1,
    OFFSET_NF = 2,
    OFFSET_NT = 3,
    OFFSET_LENGTH = 4,
    OFFSET_FLUXTOT = 5
    // these are the columns that appear in file somewhere for the creation of the network
  };

  int id, id_local;
  long long int nF, nT;
  double length;
  map<string, int> flux;

  polystat_base(){};
  polystat_base(const string &line)
  {
    vector<string> tokens;
    physycom::split(tokens, line, string(";"), physycom::token_compress_off);
    id = stoi(tokens[OFFSET_ID]);
    id_local = stoi(tokens[OFFSET_IDLOC]);
    nF = stoll(tokens[OFFSET_NF]);
    nT = stoll(tokens[OFFSET_NT]);
    length = stod(tokens[OFFSET_LENGTH]);
    flux["tot"] = stoi(tokens[OFFSET_FLUXTOT]);
    if (tokens.size() != 6)
      for (int idxc = 6; idxc < tokens.size(); idxc++)
      {
        string name = "class_" + std::to_string(idxc - 6);
        flux[name] = stoi(tokens[idxc]);
      }
  }
};

//--------------------------------------------------------------------------------------------------------

void sort_activity(std::vector<activity_base> &activity);
void bin_activity(std::vector<activity_base> &activity);
void make_traj(std::vector<activity_base> &activity,data_loss &dataloss,std::vector<traj_base> &traj,std::vector<cluster_base> &data_notoncarto,std::vector<presence_base> &presence);
bool best_poly(cluster_base &d1, cluster_base &d2,std::vector<poly_base> &poly,std::vector<node_base> &node);
void make_bp_traj(std::vector<traj_base> &traj,config &config_,double &sigma,data_loss &dataloss,std::vector<poly_base> &poly,std::vector<centers_fcm_base> &centers_fcm,std::vector<node_base> &node,std::vector<std::map<int,int>> &classes_flux);
void make_fluxes(std::vector<traj_base> &traj,double &sigma,std::vector<poly_base> &poly,std::vector<centers_fcm_base> &centers_fcm,std::vector<std::map<int,int>> &classes_flux);
void make_polygons_analysis(config &config_,std::vector<centers_fcm_base> &centers_fcm,std::vector<traj_base> &traj,std::vector<polygon_base> &polygon);
void make_multimodality(std::vector<traj_base> &traj,config &config_,std::vector<centers_fcm_base> &centers_fcm);
void dump_fluxes(std::vector<traj_base> &traj,config &config_,std::vector<centers_fcm_base> &centers_fcm,std::vector<poly_base> &poly,std::vector<std::map<int,int>> &classes_flux);
map<string, vector<int>> make_subnet(config &config_);
void make_MFD(jsoncons::json jconf,std::vector<traj_base> &traj,std::vector<centers_fcm_base> &centers_fcm);
double measure_representativity(const string &label,std::vector<poly_base> &poly,map<string, vector<int>> subnets);
// ALBI
void make_FD();
void dump_FD(std::vector<poly_base> &poly);
void velocity_subnet(std::vector<int> poly_subnet, std::vector<poly_base> &poly, int time_step, int num_bin, std::string save_label);
void make_multimodality_subnet();
void subnet_intersection(std::vector<int> v1, std::vector<int> v2, std::vector<int> &v3);
void subnet_complementary(std::vector<int> v1, std::vector<int> intersection, std::vector<int> &v3);
void subnet_union(std::vector<int> v1, std::vector<int> v2, std::vector<int> &v3);
void deeper_intersections(std::string dir_subnet1, std::string dir_subnet2,std::vector<traj_base> &traj);
void fcm_subnets(config &config_, std::vector<traj_base> traj_subnet, std::vector<int> subnet, std::string label_save,std::vector<traj_base> &traj);
void compute_complete_intersection(std::pair<int, std::vector<int>> sub_neti, std::vector<int> &complete_intersection, std::vector<std::pair<int, std::vector<int>>> vector_pair_intersection);
void compute_complete_complement(std::vector<int> &complete_complement, std::vector<std::pair<int, std::vector<int>>> vector_pair_intersection, std::map<std::string, std::vector<int>> subnets, config config_);
std::vector<traj_base> selecttraj_from_vectorpolysubnet_velsubnet(std::vector<int> poly_subnet, std::vector<traj_base> &traj, std::vector<poly_base> &poly, std::string label_save);
void analysis_subnets(std::vector<traj_base> &traj,std::vector<poly_base> &poly,map<string, vector<int>> &subnets);
void dump_longest_traj(std::vector<traj_base> &traj);
void hierarchical_deletion_of_intersection(std::map<std::string, std::vector<int>> subnets80);
void assign_new_class(std::vector<traj_base> &traj,std::vector<poly_base> &poly,std::map<std::string, std::vector<int>> subnets80);
// ALBI
