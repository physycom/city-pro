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
#include <analyzer_object.h>
#include <FL/Fl.H>
#include <physycom/time.hpp>
#include <jsoncons/json.hpp>
#include<map>
using namespace jsoncons;


constexpr int MAJOR = 1;
constexpr int MINOR = 2;
config config_;

extern Frame *scene;
extern bool   re_draw;

// DATA READING
std::vector<poly_base> poly;

std::vector<activity_base> activity;
std::vector<polygon_base> polygon;
std::map<unsigned long long int, int> poly_cid2lid;
// CARTO
extern int  screen_w, screen_h;
extern int  screen_w, screen_h;
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
std::map<int,std::map<int,double>> classes_flux;
// FCM

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
  "enable_print"       : true                       //print output file
}
)";
}
//-------------------------------------------------------------------------------------------------

int main(int argc, char **argv) {

  //CONFIG
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
    static analyzer analysis_container;
    analysis_container.poly.reserve(112966);
//    analysis_container.activity.reserve(400000);
    analysis_container.node.reserve(86798);
    analysis_container.arc.reserve(631382);
//    analysis_container.traj.reserve(130000);
//    analysis_container.presence.reserve(200000);
    json jconf;
    jconf = json::parse_file(conf);
    config_.set_config(jconf);
    clock_t begin = clock();
    set_geometry(config_);
    analysis_container.config_ = config_;
//ho ottenuto dlat,dlon e ho settato la proporzionalità tra screen_w e dlon e screen_h e lat.
//vado su data_reading.cpp
    load_poly(analysis_container.config_,analysis_container.poly,analysis_container.poly_cid2lid);
//ho l'insieme di tutte le poly con i rispettivi poly[n].cid_poly,id_poly,points,cid_Fjnct, cid_Tjnct, meters, oneway, name poi qua trovo definite:
//vector <poly_base> poly;
//vector <activity_base> activity;
//vector <polygon_base> polygon;
//map <unsigned long long int, int> poly_cid2lid;(questo è per avere un grafo più portatile probabilmente)

// vado su carto.cpp
    make_node(analysis_container.poly,analysis_container.node_cid2id,analysis_container.node);

//
    make_mapping(analysis_container.poly,analysis_container.node);
    load_polygon(analysis_container.polygon,analysis_container.config_);
    load_data(analysis_container.activity,analysis_container.config_);
    sort_activity(analysis_container.activity); //for auto a:activity PROPERTIES: 1) a.itime<a++.itime, 2) measure a.length,.speed  3) measure a.record.speed,.acc
//posso salvarmi in sort_activity le informazioni che mi servono per descrivere le traiettorie, le velocità. (l'inizializzazione di indx per activity and record è un'opzione per la funzione sotto.)
    if(config_.enable_bin_act)
      bin_activity(analysis_container.activity);

    make_traj(analysis_container.activity,analysis_container.dataloss,analysis_container.traj,analysis_container.data_notoncarto,analysis_container.presence);//for t in traj_temp 1) activity->traj_temp 2) t.record -> t.stop_point 3)t.stop_point = sp_on_carto (vector clster_base)
    if (config_.enable_multimodality)
      {
//        cout << "make multimodality" << endl;
        make_multimodality(analysis_container.traj,analysis_container.config_,analysis_container.centers_fcm);}
    make_polygons_analysis(analysis_container.config_,analysis_container.centers_fcm,analysis_container.traj,analysis_container.polygon);
//    cout << "make best path trajectories" << endl;
    make_bp_traj(analysis_container.traj,analysis_container.config_,analysis_container.sigma,analysis_container.dataloss,analysis_container.poly,analysis_container.centers_fcm,analysis_container.node,analysis_container.classes_flux);
    //dump_longest_traj(analysis_container.traj);
//    cout <<"make fluxes" <<endl;
    make_fluxes(analysis_container.traj,analysis_container.sigma,analysis_container.poly,analysis_container.centers_fcm,analysis_container.classes_flux);
// OBSOLETO ora utilizzo dump_fluxes_file
    if (config_.jump2subnet_analysis == false){
        if (0==1)//(config_.enable_fluxes_print)
          {
            cout << "dump fluxes" << endl;
            dump_fluxes(analysis_container.traj,analysis_container.config_,analysis_container.centers_fcm,analysis_container.poly,analysis_container.classes_flux);};
//OBSOLETO
        if (config_.enable_FD) dump_FD(poly);
        if (config_.enable_MFD)
        {
          cout << "make MFD" << endl;
          make_MFD(jconf,analysis_container.traj,analysis_container.centers_fcm);};
        if (config_.enable_subnet)
        {
          analysis_container.subnets = make_subnet(analysis_container.config_);
        };
        cout << "dump poly geojson" << endl;
        dump_poly_geojson("bologna-provincia",analysis_container.poly); //"city-pro-carto"
      }
    cout << "load subnet" << endl;
    load_subnet(analysis_container.config_,analysis_container.subnets);
    if (config_.multimodality_subnet && config_.enable_subnet){cout <<"start calculation multimodality in subnet"<<endl;
      analysis_subnets(analysis_container.traj,analysis_container.poly,analysis_container.subnets);
      //make_multimodality_subnet();
//      cout<<"velocity subnet" << endl;
//      velocity_subnet();
    }

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Main: elapsed_secs = " << elapsed_secs << endl;

  //  make_window();
  //   Fl::add_idle(idle_cb, 0);
  //   return Fl::run();
  //   return 0;
  }
  catch (exception &e)
  {
    cerr << "EXC: " << e.what() << endl;
    exit(1);
  }
}
