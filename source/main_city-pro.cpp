#include "stdafx.h"
#include "data_analysis.h"
//#include "analisi_modello.h"
//#include "analisi_presenze.h"
#include "config.h"
#include "carto.h"
#include "data_reading.h"
#include "form.h"
//#include "Globalmaptiles.h"
#include "frame.h"
//#include "mappa.h"
//#include "ale/subnet_gra.h"

#include <FL/Fl.H>
#include <physycom/time.hpp>
#include <jsoncons/json.hpp>

using namespace jsoncons;

constexpr int MAJOR = 1;
constexpr int MINOR = 2;

config config_;

extern Frame *scene;
extern bool   re_draw;
//-------------------------------------------------------------------------------------------------
void idle_cb(void*)
{
  if (re_draw) scene->redraw();
}
//-------------------------------------------------------------------------------------------------
void usage(const char *progname)
{
  string pn(progname);
  cerr << "Usage: " << pn.substr(pn.find_last_of("/\\") + 1) << " path/to/json/config" << endl;
  cerr << R"(JSON CONFIG SAMPLE
{
  
  "city_tag"        : "NameCity",
  ///// cartography setup
  "file_pro"        : "roads.pro",  // poly properties input file
  "file_pnt"        : "roads.pnt",  // poly geometry input file
  "file_data"       : ["data.csv"], // name of data file
  "file_polygons"   : "polygons.geojson",  // polygons geometry

  ///// time restrictions
  "start_time"      : "2020-08-07 00:00:00",        // date-time analysis start
  "end_time"        : "2020-08-16 00:00:00",        // date-time analysis end
  "bin_time"        : "15.0",                       // time interval for binning (minutes)

  ///// space restricions
  "lat_min" : 43.96519,
  "lat_max" : 44.16400,
  "lon_min" : 12.43720,
  "lon_max" : 12.73840,

  ///// distances and resolutions
  "map_resolution"    : 60.0,                       // meters for map resolution
  "grid_resolution"   : 150.0,                      // meters for grid resolution
  "l_gauss"           : 10.0,                       // meters for GPS data precision
  "min_data_distance" : 49.0,                       // minimum distance between two consecutive data
  "min_node_distance" : 20.0,                       // minimum distance datum-node
  "min_poly_distance" : 50.0,                       // minimum distance datum-poly


///// functionalities
  "enable_print"       : true,                       //print output file
}
)";
}
//-------------------------------------------------------------------------------------------------

int main(int argc, char **argv) {

  cout << "city-pro v" << MAJOR << "." << MINOR << endl;

  string conf;
  if (argc == 2)
  {
    conf = argv[1];
  }
  else
  {
    usage(argv[0]);
    exit(1);
  }

  try
  {
    json jconf;
    jconf = json::parse_file(conf);
    config_.set_config(jconf);
    clock_t begin = clock();

    set_geometry();
    load_poly();
    make_node();
    make_mapping();
    load_polygon();
    load_data();
    sort_activity(); 
    if(config_.enable_bin_act) 
      bin_activity();
    make_traj();
    make_polygons_analysis();
    if (config_.enable_multimodality)
      make_multimodality();
    make_bp_traj();
    
    if (config_.enable_fluxes_print)
      dump_fluxes();

    if (config_.enable_subnet)
      make_subnet();

    load_subnet();

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Main: elapsed_secs = " << elapsed_secs << endl;

    make_window();
    Fl::add_idle(idle_cb, 0);
    return Fl::run();
    return 0;
  }
  catch (exception &e)
  {
    cerr << "EXC: " << e.what() << endl;
    exit(1);
  }
}
