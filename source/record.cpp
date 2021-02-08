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
void traj_base::add_cluster(cluster_base &C, int n) {

  if (stop_point.size() != 0) {
    bool find_cor = false;
    for (auto &sp : stop_point) { // fai questo loop al contrario per ottimizzare (più probabile alla fine)
      if (distance_record(sp.centroid, record[n]) <= config_.min_data_distance) {
        double inst_speed_rec = distance_record(record[n], record[n - 1]) / (record[n].itime - record[n - 1].itime);
        if (inst_speed_rec < config_.max_inst_speed) {
          sp.add_point(record[n]);
          sp.visited = true;
          find_cor = true;
          C.inst_speed = C.points.front().speed;
          if (C.inst_speed < config_.max_inst_speed && !C.visited)
            stop_point.push_back(C);
          C = sp;
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
    C.points.clear();
    C.add_point(record[n]);
  }

}
//----------------------------------------------------------------------------------------

