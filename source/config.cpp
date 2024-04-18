
#include <physycom/time.hpp>
#include "config.h"
#include "global_params.h"

config::config() {}

void config::set_config(jsoncons::json jconf)
{
  this->city_tag = jconf.contains("city_tag") ? jconf["city_tag"].as<std::string>() : "city_name";
  this->file_pro = jconf.contains("file_pro") ? jconf["file_pro"].as<std::string>() : "roads.pro";
  this->file_pnt = jconf.contains("file_pnt") ? jconf["file_pnt"].as<std::string>() : "roads.pnt";
  if (jconf.contains("file_data"))
    this->file_data = jconf["file_data"].as<std::vector<std::string>>();
  else
    this->file_data = { "data.csv" };
  this->cartout_basename = jconf.contains("cartout_basename") ? jconf["cartout_basename"].as<std::string>() :  "output/";

  this->file_polygons = jconf.contains("file_polygons") ? jconf["file_polygons"].as<std::string>() : "file_polygons";
  if (jconf.contains("polygons_code"))
    this->polygons_code = jconf["polygons_code"].as<std::vector<std::string>>();
  else
    this->polygons_code = {};

  this->start_datetime = jconf.contains("start_time") ? jconf["start_time"].as<std::string>() : "2017-07-13 10:00:00";
  this->end_datetime = jconf.contains("end_time") ? jconf["end_time"].as<std::string>() : "2017-07-13 10:00:00";
  this->start_time = physycom::date_to_unix(this->start_datetime);
  this->end_time = physycom::date_to_unix(this->end_datetime);
  this->bin_time = jconf.contains("bin_time") ? jconf["bin_time"].as<double>() : 15.0;
  if (jconf.contains("slice_time"))
    this->slice_time = jconf["slice_time"].as<std::vector<double>>();
  else
    this->slice_time = {};


  this->lat_max = jconf.contains("lat_max") ? jconf["lat_max"].as<double>() : 45.451410;
  this->lat_min = jconf.contains("lat_min") ? jconf["lat_min"].as<double>() : 45.351410;
  this->lon_max = jconf.contains("lon_max") ? jconf["lon_max"].as<double>() : 12.372170;
  this->lon_min = jconf.contains("lon_min") ? jconf["lon_min"].as<double>() : 12.272170;
  this->dslat = 111053.8;
  this->dslon = this->dslat*cos((this->lat_max + this->lat_min) / 2 * PI_180);

  this->map_resolution = jconf.contains("map_resolution") ? jconf["map_resolution"].as<double>() : 60.0;
  this->grid_resolution = jconf.contains("grid_resolution") ? jconf["grid_resolution"].as<double>() : 150.0;
  this->l_gauss = jconf.contains("l_gauss") ? jconf["l_gauss"].as<double>() : 10.0;
  this->min_data_distance = jconf.contains("min_data_distance") ? jconf["min_data_distance"].as<double>() : 50.0;
  this->max_inst_speed = jconf.contains("max_inst_speed") ? jconf["max_inst_speed"].as<double>() : 37.0;
  this->min_node_distance = jconf.contains("min_node_distance") ? jconf["min_node_distance"].as<double>() : 20.0;
  this->min_poly_distance = jconf.contains("min_poly_distance") ? jconf["min_poly_distance"].as<double>() : 50.0;

  this->enable_threshold = jconf.contains("enable_threshold") ? jconf["enable_threshold"].as<bool>() : true;
  this->threshold_v = jconf.contains("threshold_v") ? jconf["threshold_v"].as<double>() : 50.0;
  this->threshold_t = jconf.contains("threshold_t") ? jconf["threshold_t"].as<int>() : 3600;
  this->threshold_n = jconf.contains("threshold_n") ? jconf["threshold_n"].as<int>() : 3;
  this->threshold_polyunique = jconf.contains("threshold_polyunique") ? jconf["threshold_polyunique"].as<int>() : 3;

  this->enable_multimodality = jconf.contains("enable_multimodality") ? jconf["enable_multimodality"].as<bool>() : false;
  this->enable_slow_classification = jconf.contains("enable_slow_classification") ? jconf["enable_slow_classification"].as<bool>() : false;
  this->num_tm = jconf.contains("num_tm") ? jconf["num_tm"].as<int>() : 2;
  this->threshold_p = jconf.contains("threshold_p") ? jconf["threshold_p"].as<double>() : 0.85;

  this->dump_dt = jconf.contains("dump_dt") ? jconf["dump_dt"].as<int>() : 1440;

  this->enable_bin_act = jconf.contains("enable_bin_act") ? jconf["enable_bin_act"].as<bool>() : false;
  this->enable_fluxes_print = jconf.contains("enable_fluxes_print") ? jconf["enable_fluxes_print"].as<bool>() : false;
  this->enable_subnet = jconf.contains("enable_subnet") ? jconf["enable_subnet"].as<bool>() : false;
  this->file_subnet = jconf.contains("file_subnet") ? jconf["file_subnet"].as<std::string>() : "fluxes.sub";
  this->enable_print = jconf.contains("enable_print") ? jconf["enable_print"].as<bool>() : false;
  this->enable_MFD= jconf.contains("enable_MFD") ? jconf["enable_MFD"].as<bool>() : false;
  this->enable_FD= jconf.contains("enable_FD") ? jconf["enable_FD"].as<bool>() : false;
  this->multimodality_subnet= jconf.contains("multimodality_subnet") ? jconf["multimodality_subnet"].as<bool>() : false;
  this->num_tm_subnet  = jconf.contains("num_tm_subnet") ? jconf["num_tm_subnet"].as<int>() : 3;
  this->all_subnets_speed  = jconf.contains("all_subnets_speed") ? jconf["all_subnets_speed"].as<bool>() : true;
  this->complete_intersection_speed  = jconf.contains("complete_intersection_speed") ? jconf["complete_intersection_speed"].as<bool>() : true;
  this->complete_complement_speed  = jconf.contains("complete_complement_speed") ? jconf["complete_complement_speed"].as<bool>() : true;
  this->jump2subnet_analysis  = jconf.contains("jump2subnet_analysis") ? jconf["jump2subnet_analysis"].as<bool>() : false;


  vector<string> tokens;
  vector<string> tokens1;
  physycom::split(tokens, this->file_data[0], string("/"), physycom::token_compress_off);
  physycom::split(tokens1, tokens[tokens.size() - 1], string("."), physycom::token_compress_off);
  this->name_pro = tokens1[0];

  info();
}
void config::info()
{
  std::cout << "******** DATA INFO ***************" << std::endl;
  for (const auto &i: file_data) std::cout << "* Data file  : " << i << std::endl;
  std::cout << "* Time range : " << start_datetime << " " << end_datetime << std::endl;
  std::cout << "* LAT range  : " << lat_min << " " << lat_max << std::endl;
  std::cout << "* LON range  : " << lon_min << " " << lon_max << std::endl;
  std::cout << "**********************************" << std::endl;
}
