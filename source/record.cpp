#include "stdafx.h"
#include "record.h"
#include "config.h"

extern config config_;


//----------------------------------------------------------------------------------------
double distance_record(record_base r1, record_base r2) {
  double dx, dy, ds2;
  dx = config_.dslon*(r2.lon - r1.lon); dy = config_.dslat*(r2.lat - r1.lat);
  ds2 = dx * dx + dy * dy;
  if (ds2 > 0) return sqrt(ds2);
  else return 0.0;
}

//----------------------------------------------------------------------------------------
heading_base measure_heading(cluster_base c1, cluster_base c0) {
  heading_base hw;
  double dx, dy, ds2, ds;
  dx = config_.dslon*(c1.points.front().lon - c0.points.front().lon);
  dy = config_.dslat*(c1.points.front().lat - c1.points.front().lat);
  ds2 = dx * dx + dy * dy;
  if (ds2 < 1.0e-16) {
    std::cout << "Heading: error ds2 = " << ds2 << " too small! ds = " << distance_record(c1.points.front(), c0.points.front()) << endl;
    hw.x = 1.0;
    hw.y = 0.0;
  }
  else { ds = sqrt(ds2); hw.x = dx / ds; hw.y = dy / ds; }
  return hw;
}
// DATALOSS //
void data_loss::dump() {
  ofstream out_data("../output/dataloss/" + config_.name_pro + ".csv");
  ofstream out_traj("../output/trajloss/" + config_.name_pro + ".csv");
  out_data << "n_data_tot;n_data_meter;n_data_outcarto;n_data_oncarto;n_data_single_record;n_data_threshold" << endl;
  out_data << n_data_tot << ";" << n_data_meter << ";" << n_data_outcarto << ";" << n_data_oncarto << ";" << n_data_single_record << ";" << n_data_threshold << endl;
  out_traj << "n_traj_tot;n_traj_poly_thresh" << endl;
  out_traj << n_traj_tot << ";" << n_traj_poly_thresh << endl;
  out_data.close();
  out_traj.close();
}
// PRESENCE //
presence_base::presence_base(double lat_, double lon_, long long int id_act_, size_t t_start, size_t t_stop, int row_n_rec_) {
  this->lat = lat_;
  this->lon = lon_;
  this->id_act = id_act_;
  this->itime_start = t_start;
  this->itime_end = t_stop;
  this->row_n_rec = row_n_rec_;
}

// CLUSTER //
void cluster_base::add_point(record_base rec) {

  points.push_back(rec);

  //update centroid
  centroid.lat = points.front().lat;
  centroid.lon = points.front().lon;

  //update duration
  if (points.size() > 1)
    duration = points.back().itime - points.front().itime;
  else
    duration = 0;

}
//----------------------------------------------------------------------------------------

// TRAJ //
void traj_base::add_cluster(cluster_base &C, int n) {//n indica l'iterazione del ciclo in cui cambio i record di una certa traiettoria

  if (stop_point.size() != 0) {
    bool find_cor = false;
    for (auto &sp : stop_point) { // fai questo loop al contrario per ottimizzare (più probabile alla fine)
      if (distance_record(sp.centroid, record[n]) <= config_.min_data_distance) {
        double inst_speed_rec = distance_record(record[n], record[n - 1]) / (record[n].itime - record[n - 1].itime); // il caso in cui ho due dati a distanza temporale elevata e magari uno si è fermato per un tot nell'altro punto?
        if (inst_speed_rec < config_.max_inst_speed) {
          sp.add_point(record[n]);
          sp.visited = true;
          find_cor = true;
//devo aver già calcolato la velocità del record nel momento che vado a riempire activity
          C.inst_speed = C.points.front().speed;
          if (C.inst_speed < config_.max_inst_speed && !C.visited)
            stop_point.push_back(C);
          C.replace(sp);
          break;
        }
        else {
          find_cor = true;
          break;
        }
      }
    }
    if (!find_cor) {
      C.inst_speed = C.points.front().speed;
      if (C.inst_speed < config_.max_inst_speed && !C.visited)
        stop_point.push_back(C);
      C.points.clear();
      C.add_point(record[n]);
    }
  }
  else {
    C.inst_speed = 0.0;
    stop_point.push_back(C);
    C.points.clear();//I clear the cluster point that I have, I now have it stored in the memory of traj
    C.add_point(record[n]);//to continue the activity I add the point that is distant enough to the previous centroid, now I have the new centroid.
  }//still to know how to handle the name of the user.

}



//----------------------------------------------------------------------------------------
//albi
/*
vector<record_base> calc_center_of_mass(vector<record_base> rb){
  record_base total,cm;
  vector<record_base> center_of_mass;
  int iteration;
  for (auto &r: rb){
    iteration++;
    total.lat+=r.lon;
    total.lon+=r.lon;
    cm.lat=total.lat/iteration;
    cm.lon=total.lon/iteration;
    center_of_mass.push_back(cm);
};
return center_of_mass;
};

vector<record_base> difference(vector<record_base> &r1,vector<cluster_base> &r2){
vector<record_base> centroid_sp;
for (cluster_base &c: r2){
  centroid_sp.push_back(c.centroid);
};

vector<record_base> difference;
int number=r1.size();
difference.reserve(number);
if r1.size()==centroid_sp.size(){
for (int i=0;i<r1.size();i++){
  difference[i].lat=r1[i].lat-centroid_sp[i].lat;
  difference[i].lon=r1[i].lon-centroid_sp[i].lon;
  };
}
else
std::cerr<<"the dimension of the record_base dont match "<<std::endl;

return difference;

};

vector<record_base> traj_base:: difference(){
vector<record_base> difference;
difference.reserve(center_mass.size());
if (center_mass.size()==stop_point.size())
for (int i=0;i<center_mass.size();i++){
  difference[i].lat=r1[i].lat-stop_point[i].centroid.lat;
  difference[i].lon=r1[i].lon-stop_point[i].centroid.lon;
}
else std::cerr<<'the number of records in the center of mass dont match with the number of stop points'<<std::endl;
return difference;

};



vector <record_base> traj_base::calc_center_mass(){
  record_base total,cm;
  vector<record_base> center_mass;
  int iteration;
  for (auto &sp:stop_point){
    iteration++;
    total.lat+=sp.centroid.lon;
    total.lon+=sp.centroid.lon;
    cm.lat=total.lat/iteration;
    cm.lon=total.lon/iteration;
    center_mass.push_back(cm);
    std::cout<<'center mass size'<<center_mass.size()<<'\n id traj:\t '<<id_act<<endl;

};
  return center_mass;
};


vector <double> traj_base::calc_gyration_radius(){
  calc_center_mass();
  vector<record_base> differences;
  differences=difference(this->center_mass,this->stop_point);
  int iterator;
  vector<record_base> zeros;
  zeros.reserve(differences.size());
  for (int i=0;differences.size();i++){zeros[i].lat=0;zeros[i].lon=0;}
for (auto &df:differences){
gyration_radius.push_back(distance_record(df[iterator],zeros[iterator])**2/(iterator+1));
std::cout<<'sto calcolando il raggio di giration iterazione'<<iterator<<'raggio'<<radius_gyration[iterator]<<std::endl;
iterator++;
}
  zeros.clear();
  differences.clear();
    };




double traj_base::calc_inertia(){return 0;

};
  double traj_base::calc_entropy(){

  return 0;};
  double traj_base::calc_rdm_entropy(){

  return 0;};
  double traj_base::calc_temp_unc_entropy(){

  return 0;};

*/