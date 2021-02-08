#pragma once
#include "record.h"
#include <physycom/string.hpp>

void sort_activity();
void bin_activity();
void make_traj();
bool best_poly(cluster_base &d1, cluster_base &d2);
void make_bp_traj();
void make_fluxes();
void make_polygons_analysis();
void make_multimodality();
void dump_fluxes();
void make_subnet();
//---------------------------------------------------------------------

// SEED //
struct seed_pro_base {
  double distance;
  int id_node, node_bv, link_bv;
  void set(int id_node, int node_bv, int link_bv, double distance);
};
//---------------------------------------------------------------------
struct seed_base {
  double dd;  // dd = dist + d_eu (d_eu= a_eu*dist, 0 < a_eu < 1)
  int cnt;
  bool operator<(const seed_base &s) const { return dd > s.dd; }
};

// CENTERS FCM //
struct centers_fcm_base {
  vector<double> feat_vector;
  double sigma = 0.0;
  int cnt = 0;
  int idx;
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

  };

  int id, id_local;
  long long int nF, nT;
  double length;
  map<string, int> flux;

  polystat_base() {};
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
    if (tokens.size()!=6)
      for (int idxc = 6; idxc<tokens.size(); idxc++){
        string name = "class_" + std::to_string(idxc - 6);
        flux[name] = stoi(tokens[idxc]);
      }
  }
};
