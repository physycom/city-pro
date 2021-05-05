#pragma once
#include <jsoncons/json.hpp>

struct config {

  std::string city_tag;
  std::string file_pro;
  std::string file_pnt;
  std::vector<std::string> file_data;
  std::string file_polygons;
  std::string cartout_basename;
  std::string name_pro;
  std::vector<std::string> polygons_code; // 2 args [location, code number (0 start or stop, 1 just start, 2 just stop)] // 3 args [location_start, location_sto, code_number (0 both way, 1 oneway)]


  std::string start_datetime;
  std::string end_datetime;
  double bin_time;
  size_t start_time;
  size_t end_time;
  std::vector<double> slice_time;

  double lat_max;
  double lat_min;
  double lon_max;
  double lon_min;
  double dslat;
  double dslon;

  double map_resolution;
  double grid_resolution;
  double l_gauss;
  double min_data_distance;
  double max_inst_speed;
  double min_node_distance;
  double min_poly_distance;

  double threshold_v;
  int    threshold_t;
  int    threshold_n;
  int    threshold_polyunique;

  bool enable_multimodality;
  bool enable_slow_classification;
  int num_tm;            // number of cluster for multimodality classification
  double threshold_p;    // threshold of p for multimodality classification (0.0-1.0)

  bool enable_threshold;
  bool enable_bin_act;
  bool enable_fluxes_print;
  bool enable_subnet;
  std::string file_subnet;
  bool enable_print;

  config();
  void set_config(jsoncons::json _jconf);

  void info();
};