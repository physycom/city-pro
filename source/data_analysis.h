#pragma once
#include "record.h"

void sort_activity();
void bin_activity();
void make_traj();
bool best_poly(cluster_base &d1, cluster_base &d2);
void make_bp_traj();
void make_fluxes();
void make_polygons_analysis();
void make_multimodality();
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
