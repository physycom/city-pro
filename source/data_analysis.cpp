#include "stdafx.h"
#include <iostream>
#include <algorithm>
#include "carto.h"
#include "record.h"
#include "data_analysis.h"
#include "config.h"
#include "utils/physycom/histo.hpp"
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include "fcm.cpp"
#include <random>
#include <iterator>
#include <functional>

using namespace Eigen;
using namespace std;

extern vector <poly_base> poly;
extern vector <polygon_base> polygon;
extern vector <activity_base> activity;
extern vector<node_base> node;
extern vector<centers_fcm_base> centers_fcm;
//extern map<string, vector<int>> subnets;
extern config config_;
vector <traj_base> traj;
vector <presence_base> presence;
vector <centers_fcm_base> centers_fcm;

//temp
vector<traj_base> pawns_traj;
vector<traj_base> wheels_traj;

double sigma = 0.0;
vector<cluster_base> data_notoncarto;
data_loss dataloss;

//----------------------------------------------------------------------------------------------------
bool comp_rec_itime(const record_base& a, const record_base& b) { return (a.itime < b.itime); }
//----------------------------------------------------------------------------------------------------
bool comp_traj_lenght(const traj_base& a, const traj_base& b) { return (a.length > b.length); }
//----------------------------------------------------------------------------------------------------
void sort_activity() {

  // sort data in each activity
  for (auto &a : activity) sort(a.record.begin(), a.record.end(), comp_rec_itime);

  // add index for activity and records
  int j = 0, k;
  for (auto &a : activity) {
    a.indx = j++;
    k = 0;
    for (auto &r : a.record)
      r.indx = k++;
  }

  // add state for each record
  for (auto &a : activity) {
    if (a.record.size() == 1) a.record[0].state = 3;
    else {
      for (auto &r : a.record)  r.state = 1;
      a.record.front().state = 0;
      a.record.back().state = 2;
    }
  }

  // measure lenght and average speed
  for (auto &a : activity) {
    a.length = 0.0;
    for (auto r = 0; r != a.record.size() - 1; r++) {
      a.length += distance_record(a.record[r], a.record[r + 1]);
    }
    a.average_speed = a.length / double(a.dt);
  }
}
//----------------------------------------------------------------------------------------------------
void bin_activity() {
  double delta_s = config_.bin_time * 60;
  map <int, int> bintime_data;
  for (auto &a : activity) {
    int start_index = int((a.record.front().itime - config_.start_time) / delta_s);
    int end_index = int((a.record.back().itime - config_.start_time) / delta_s);
    for (int i = start_index; i <= end_index; ++i)
      bintime_data[i]++;
  }
  ofstream out_bin(config_.cartout_basename + config_.name_pro + ".csv");
  out_bin << "Time;Occurence" << endl;
  for (auto &m : bintime_data)
    out_bin << m.first*config_.bin_time << ";" << m.second << endl;
  out_bin.close();
  cout << "Time binning of data: done." << endl;
}
//----------------------------------------------------------------------------------------------------
void make_traj() {

  vector <traj_base> traj_temp;
  traj_base tw;
  for (auto &a : activity) {
    tw.record.clear();
    tw.id_act = a.id_act;
    tw.record = a.record;
    tw.row_n_rec = int(tw.record.size());
    traj_temp.push_back(tw);
  }

  activity.clear(); activity.shrink_to_fit();// clean memory, the info are passed to traj and presence

  //for (auto &t : traj_temp)
  //  if (t.id_act == 27858){
  //    for (int r=1; r<t.record.size(); ++r){
  //      std::cout  <<"  "<<distance_record(t.record[r], t.record[r-1])/(t.record[r].itime- t.record[r-1].itime) << std::endl;
  //    }
  //  }
  //for (auto &t : traj_temp)
  //  if (t.id_act == 27858)
  //    for (auto &r:t.record)
  //      std::cout << r.lat<<"  " << r.lon << std::endl;
  //std::cin.get();


  // filter data on distance and inst_speed
  int cnt_tot_data = 0;
  int cnt_tot_sp = 0;
  for (auto &t : traj_temp) {
    cnt_tot_data += int(t.record.size());
    cluster_base C;
    C.add_point(t.record.front());
    for (int n = 1; n < t.record.size(); ++n) {
      if (distance_record(C.centroid, t.record[n]) < config_.min_data_distance) {
        C.add_point(t.record[n]);
      }
      else if (distance_record(C.centroid, t.record[n]) >= config_.min_data_distance) {
        t.add_cluster(C, n);
      }
    }
    double last_speed = C.points.front().speed;
    if (last_speed < config_.max_inst_speed && !C.visited)
      t.stop_point.push_back(C);
    cnt_tot_sp += int(t.stop_point.size());
  }
  std::cout << "Distance filter:       " << double(cnt_tot_sp) / cnt_tot_data * 100. << "%" << std::endl;
  dataloss.n_data_tot = cnt_tot_data;
  dataloss.n_data_meter = cnt_tot_sp;

  //ofstream out_spot(config_.cartout_basename + "_presence.csv");
  //out_spot << "time;lat;lon" << std::endl;
  //for (auto &t : traj_temp)
  //  for (auto &sp : t.stop_point)
  //    out_spot << sp.points.front().itime << ";" << sp.points.front().lat << ";" << sp.points.front().lon << std::endl;

  //filter stops on carto geolocalization
  for (auto &t : traj_temp) {
    vector<cluster_base> sp_oncarto;
    for (auto &sp : t.stop_point) {
      sp.on_carto = find_polyaff(sp.centroid.lon, sp.centroid.lat, sp.pap);
      if (sp.on_carto && sp.pap.d > config_.min_poly_distance) sp.on_carto = false;
      if (!sp.on_carto) data_notoncarto.push_back(sp);
      if (sp.on_carto) sp_oncarto.push_back(sp);
    }
    t.stop_point = sp_oncarto;
  }
  dataloss.n_data_outcarto = int(data_notoncarto.size());
  std::cout << "Georeferencing filter out carto: " << double(dataloss.n_data_outcarto) / cnt_tot_sp * 100. << "%" << std::endl;

  // split data in 2 classes: traj (more than 1 point) and presence (1 point)
  for (auto &t : traj_temp) {
    t.record.clear(); t.record.shrink_to_fit();// clean memory!
    if (t.stop_point.size() == 1) {
      //transform in presence and then push
      dataloss.n_data_oncarto++;
      presence_base pr(t.stop_point.front().centroid.lat, t.stop_point.front().centroid.lon, t.id_act, t.stop_point.front().points.front().itime, t.stop_point.front().points.back().itime, t.row_n_rec);
      presence.push_back(pr);
      dataloss.n_data_single_record++;
    }
    else if (t.stop_point.size() > 1) {
      dataloss.n_traj_tot++;
      dataloss.n_data_oncarto += int(t.stop_point.size());
      t.time = int(t.stop_point.back().points.back().itime - t.stop_point.front().points.front().itime);
      t.length = distance_record(t.stop_point.back().points.back(), t.stop_point.front().points.front());
      //for (int sp = 0; sp < t.stop_point.size() - 1; ++sp){
      //  t.stop_point[sp].heading = measure_heading(t.stop_point[sp + 1], t.stop_point[sp]);
      //}
      //t.stop_point.back().heading = t.stop_point[t.stop_point.size() - 2].heading;
      if (config_.enable_threshold) {
        if (t.stop_point.size() > config_.threshold_n && t.time <= config_.threshold_t && (t.length / t.time) < config_.threshold_v) {
          traj.push_back(t);
          dataloss.n_data_threshold += int(t.stop_point.size());
        }
      }
      else
        traj.push_back(t);
    }
  }
  traj_temp.clear(); traj_temp.shrink_to_fit();

  std::cout << "Georeferencing filter on carto : " << double(dataloss.n_data_oncarto) / cnt_tot_sp * 100. << "%" << std::endl;
  std::cout << "Activiry with single record    : " << double(dataloss.n_data_single_record) / cnt_tot_sp * 100. << "%" << std::endl;
  std::cout << "Threshold filter:      " << double(dataloss.n_data_threshold) / cnt_tot_sp * 100. << "%" << std::endl;


  std::cout << "Num Presence:               " << presence.size() << std::endl;
  std::cout << "Num Traj:                   " << traj.size() << std::endl;

  // check
  for (auto &t : traj) {
    for (int n = 0; n < t.stop_point.size() - 1; ++n) {
      double dist = distance_record(t.stop_point[n].centroid, t.stop_point[n + 1].centroid);
      if (dist < 30.0) {
        std::cout << "Distance Error, improve the algos!" << std::endl;
        std::cout << "id: " << t.id_act << " dist " << dist << " nrec " << t.record.size() << " nstops:" << t.stop_point.size() << std::endl;
        std::cin.get();
      }
    }
  }
  for (auto &t : traj) {
    t.time = int(t.stop_point.back().points.back().itime - t.stop_point.front().points.front().itime);
    t.length = distance_record(t.stop_point.back().points.back(), t.stop_point.front().points.front());
  }


  if (config_.enable_print) {
    ofstream out_stats(config_.cartout_basename + config_.name_pro + "_stats.csv");
    out_stats << "id_act;length;time;av_speed;ndat" << std::endl;
    for (auto &t : traj) {
      t.time = int(t.stop_point.back().points.back().itime - t.stop_point.front().points.front().itime);
      t.length = distance_record(t.stop_point.back().points.back(), t.stop_point.front().points.front());
      t.average_speed = t.length / t.time;
      int time2 = int(t.stop_point.back().points.back().itime - t.stop_point[1].points.front().itime);
      t.average_accel = (t.stop_point.back().points.back().speed - t.stop_point[1].points.front().speed) / time2;
      out_stats << t.id_act << ";" << t.length << ";" << t.time << ";" << t.average_speed << ";" << t.stop_point.size() << std::endl;
    }
    out_stats.close();
  }

}
//-------------------------------------------------------------------------------------------------
// SEED //
void seed_pro_base::set(int id_node, int node_bv, int link_bv, double distance) {
  this->distance = distance;
  this->id_node = id_node;
  this->node_bv = node_bv;
  this->link_bv = link_bv;
}
//-------------------------------------------------------------------------------------------------
bool best_poly(cluster_base &d1, cluster_base &d2) {

  polyaffpro_base t1 = d1.pap;
  polyaffpro_base t2 = d2.pap;

  int ipoly1, ipoly2;
  double s1, s2, p1l, p2l;
  ipoly1 = t1.id_poly;
  ipoly2 = t2.id_poly;

  p1l = poly[ipoly1].weightTF;
  p2l = poly[ipoly2].weightTF;
  s1 = t1.s;
  s2 = t2.s;

  list <pair<int, double>> poly_crossed;

  if (ipoly1 == ipoly2) {
    if (poly_crossed.size() > 0 && d1.pap.path.size() > 0) {
      if (poly_crossed.front() == d1.pap.path.back())
        poly_crossed.pop_front();
    }

    d2.pap.path = d1.pap.path;
    d2.pap.path.splice(d2.pap.path.end(), poly_crossed);
    d2.pap.path_weight = d1.pap.path_weight + abs(s2 - s1);
    d1.pap.clear();
    return true; // i'm moving on the same poly. I must not return it.
  }

  static int *index;
  seed_base node_v;                  // node_v: visited node
  seed_pro_base node_v_pro;          // nodo_v_pro: properties of visited node
  vector <seed_pro_base> list_node_pro;
  vector <int> visited;
  int nw, inw, nw_near, i_poly, iv, iter = 0;
  double x1, y1, x2, y2, dx, dy, dist, d_eu, a_eu = 0.0;
  double distance;
  static bool first_time = true;
  // index 0 initialization
  if (first_time) {
    first_time = false;
    index = new int[int(node.size())]();
  }

  bool goal_F = false;
  bool goal_T = false;

  priority_queue<seed_base> heap;

  int n1F = poly[ipoly1].node_F;
  int n1T = poly[ipoly1].node_T;
  int n2F = poly[ipoly2].node_F;
  int n2T = poly[ipoly2].node_T;

  x2 = d2.centroid.lon;
  y2 = d2.centroid.lat;

  // push fake node start
  iv = 0;
  node_v_pro.set(0, 0, 0, 0.0);
  list_node_pro.push_back(node_v_pro);

  // push node n1 front
  x1 = node[n1F].lon;
  y1 = node[n1F].lat;
  dx = config_.dslon*(x1 - x2);
  dy = config_.dslat*(y1 - y2);
  d_eu = a_eu * sqrt(dx*dx + dy * dy);
  iv++;
  node_v_pro.set(n1F, 0, -ipoly1, s1);
  list_node_pro.push_back(node_v_pro);
  node_v.cnt = iv;
  node_v.dd = s1 + d_eu;
  visited.push_back(n1F);
  index[n1F] = node_v.cnt;
  heap.push(node_v);

  // push node n1 tail
  x1 = node[n1T].lon;
  y1 = node[n1T].lat;
  dx = config_.dslon*(x1 - x2);
  dy = config_.dslat*(y1 - y2);
  d_eu = a_eu * sqrt(dx*dx + dy * dy);
  iv++;
  node_v_pro.set(n1T, 0, ipoly1, p1l - s1);
  list_node_pro.push_back(node_v_pro);
  node_v.cnt = iv;
  node_v.dd = p1l - s1 + d_eu;
  visited.push_back(n1T);
  index[n1T] = node_v.cnt;
  heap.push(node_v);

  while (!heap.empty()) {
    node_v = heap.top();
    heap.pop();
    iter++;
    nw = list_node_pro[node_v.cnt].id_node;
    distance = list_node_pro[node_v.cnt].distance; // distance from n1 along the path

    inw = index[nw];
    if (inw > 0 && list_node_pro[inw].distance < distance) continue;

    index[nw] = node_v.cnt;

    if (nw == n2F) goal_F = true;
    if (nw == n2T) goal_T = true;
    if (goal_F && goal_T) break;                   //goal reached

    for (int n = 0; n < node[nw].id_nnode.size(); n++) {
      nw_near = node[nw].id_nnode[n];
      i_poly = node[nw].id_nlink[n];
      if (i_poly > 0) dist = distance + poly[abs(i_poly)].weightFT;
      else            dist = distance + poly[abs(i_poly)].weightTF;
      inw = index[nw_near];
      if (inw > 0 && (list_node_pro[inw].distance < dist)) continue;

      // push nw_near
      x1 = node[nw_near].lon;
      y1 = node[nw_near].lat;
      dx = config_.dslon*(x1 - x2);
      dy = config_.dslat*(y1 - y2);
      d_eu = a_eu * sqrt(dx*dx + dy * dy);
      iv++;
      node_v_pro.set(nw_near, nw, i_poly, dist);
      list_node_pro.push_back(node_v_pro);
      node_v.cnt = iv;
      node_v.dd = dist + d_eu;
      visited.push_back(nw_near);
      index[nw_near] = node_v.cnt;
      heap.push(node_v);
    }
  }

  // reconstruction of path
  int n;
  double delta_t = d2.points.front().t - d1.points.front().t;
  double dist_2F = list_node_pro[index[n2F]].distance;
  double dist_2T = list_node_pro[index[n2T]].distance;
  int i_poly_2F = list_node_pro[index[n2F]].link_bv;
  int i_poly_2T = list_node_pro[index[n2T]].link_bv;

  if (i_poly_2F == ipoly2)  std::cout << " error:  i_poly_2F ==  ipoly2 " << endl;
  if (i_poly_2T == -ipoly2) std::cout << " error:  i_poly_2T == -ipoly2 " << endl;

  dist_2F += s2;
  dist_2T += p2l - s2;
  pair<int, double> pw;
  if (dist_2F < dist_2T) {
    n = n2F;
    distance = dist_2F;
    pw.first = ipoly2;
    pw.second = d1.points.front().t + delta_t * (list_node_pro[index[n2F]].distance) / distance;
  }
  else {
    n = n2T;
    distance = dist_2T;
    pw.first = -ipoly2;
    pw.second = d1.points.front().t + delta_t * (list_node_pro[index[n2T]].distance) / distance;
  }
  poly_crossed.push_front(pw);

  n = list_node_pro[index[n]].node_bv;
  i_poly = list_node_pro[index[n]].link_bv;
  double ss0 = 0;
  while (i_poly != 0) {
    pw.first = i_poly;
    if (distance >= 1.e-4) pw.second = d1.points.front().t + delta_t * (list_node_pro[index[n]].distance / distance);
    else 	        	       pw.second = d1.points.front().t;
    poly_crossed.push_front(pw);
    ss0 = list_node_pro[index[n]].distance;
    n = list_node_pro[index[n]].node_bv;
    i_poly = list_node_pro[index[n]].link_bv;
  }
  poly_crossed.front().second = d1.points.front().t;

  double ss1 = 0;

  for (int i = 0; i < visited.size(); i++) index[visited[i]] = 0;
  visited.clear();
  list_node_pro.clear();

  if (poly_crossed.front().first == d1.pap.path.back().first) 	poly_crossed.pop_front();
  d2.pap.path = d1.pap.path;
  d2.pap.path.splice(d2.pap.path.end(), poly_crossed);
  d2.pap.path_weight = d1.pap.path_weight + distance;
  d1.pap.clear();

  return true;
}
//----------------------------------------------------------------------------------------------------
void make_fluxes() {
  for (auto t : traj) {
    for (auto j : t.path) {
      if (j.first > 0) poly[j.first].n_traj_FT++;
      else             poly[-j.first].n_traj_TF++;
    }
  }
  //measure sigma for standardize color in draw fluxes 
  for (auto &p : poly)
    sigma += (p.n_traj_FT + p.n_traj_TF)*(p.n_traj_FT + p.n_traj_TF);

  sigma = sqrt(sigma / int(poly.size()));
  std::cout << "Make fluxes:    sigma = " << sigma << std::endl;

  if (config_.enable_multimodality) {
    // empty initialization
    for (auto &p : poly)
      for (int n = 0; n < config_.num_tm; ++n)
        p.classes_flux[n] = 0.0;

    //fill with value
    for (auto &t : traj)
      for (auto &j : t.path) {
        if (j.first > 0) poly[j.first].classes_flux[t.means_class]++;
        else  poly[-j.first].classes_flux[t.means_class]++;
      }

    //measure sigma for standardize color in draw fluxes 
    for (auto &p : poly)
      for (int n = 0; n < config_.num_tm; ++n)
        centers_fcm[n].sigma += (p.classes_flux[n] * p.classes_flux[n]);

    for (auto &c : centers_fcm) c.sigma = sqrt(c.sigma / poly.size());

  }
}
//----------------------------------------------------------------------------------------------------
void make_bp_traj() {
  list <pair<int, double>> path, join_path;
  int cnt = 0, cnt_bestpoly = 0;

  std::cout << "**********************************" << std::endl;
  for (auto &n : traj) {
    int ipoly_old = -8000000;
    path.clear();
    join_path.clear();

    n.stop_point[0].pap.clear();
    for (int k = 0; k < n.stop_point.size() - 1; ++k) {
      best_poly(n.stop_point[k], n.stop_point[k + 1]);
      cnt_bestpoly++;
    }
    n.path = n.stop_point.back().pap.path;
    n.stop_point.back().pap.path.clear();
  }

  // count number of poly crossed and delete activity on the same polys
  if (config_.enable_threshold) {
    for (auto &t : traj) {
      list<int> path_temp;
      for (auto l : t.path) path_temp.push_back(abs(l.first));
      path_temp.sort();
      path_temp.unique();
      t.n_poly_unique = int(path_temp.size());
    }

    for (vector <traj_base>::iterator it = traj.begin(); it != traj.end(); ) {
      if (it->n_poly_unique < config_.threshold_polyunique)
        it = traj.erase(it);
      else
        it++;
    }
    dataloss.n_traj_poly_thresh = int(traj.size());
    std::cout << "Num Traj after poly unique: " << traj.size() << std::endl;
  }

  make_fluxes();

}
//----------------------------------------------------------------------------------------------------
void make_polygons_analysis() {
  if (polygon.size() == 0) return;

  vector<traj_base> traj_tmp;

  // case: 2 args [location, code number (0 start or stop, 1 just start, 2 just stop)] 
  if (config_.polygons_code.size() == 2) {
    int cnt_pg_center = 0;
    int cnt_pg_cities = 0;
    int cnt_pg_coast = 0;
    int cnt_pg_other = 0;
    int cnt_pg_station = 0;
    int cnt_from = 0;

    int location_type = 3;
    if (config_.polygons_code[0].find("center") == 0) location_type = 1;
    else if (config_.polygons_code[0].find("coast") == 0) location_type = 2;
    else if (config_.polygons_code[0].find("station") == 0) location_type = 4;

    std::cout << "Polygons code: " << config_.polygons_code[0] << " , " << config_.polygons_code[1] << std::endl;
    for (const auto &t : traj) {
      int wn_start = 0;
      int wn_stop = 0;
      int tag_start = 0;
      int tag_stop = 0;
      std::string polyg_start, polyg_stop;
      for (auto &pl : polygon) {
        wn_start = pl.is_in_wn(t.stop_point.front().centroid.lat, t.stop_point.front().centroid.lon);
        if (wn_start != 0) {
          tag_start = pl.tag_type;
          polyg_start = pl.pro["name"];
          break;
        }
      }
      for (auto &pl : polygon) {
        wn_stop = pl.is_in_wn(t.stop_point.back().centroid.lat, t.stop_point.back().centroid.lon);
        if (wn_stop != 0) {
          polyg_stop = pl.pro["name"];
          tag_stop = pl.tag_type;
          break;
        }
      }
      // 0 : start or stop point
      if (config_.polygons_code[1] == "0") {
        if (tag_start == location_type || tag_stop == location_type) {
          cnt_from++;
          if (tag_stop == 0 || tag_start == 0) {
            cnt_pg_other++;
            traj_tmp.push_back(t);
          }
          else if (tag_stop == 1 || tag_start == 1) {
            cnt_pg_center++;
            traj_tmp.push_back(t);
          }
          else if (tag_stop == 2 || tag_start == 2) {
            cnt_pg_coast++;
            traj_tmp.push_back(t);
          }
          else if (tag_stop == 3 || tag_start == 3) {
            cnt_pg_cities++;
            traj_tmp.push_back(t);
          }
          else if (tag_stop == 4 || tag_start == 4) {
            cnt_pg_station++;
          }
          else {
            std::cout << "tag_stop: " << tag_stop << std::endl;
            std::cin.get();
          }
        }
      }
      // 1: start point
      else if (config_.polygons_code[1] == "1") {
        if (tag_start == location_type) {
          cnt_from++;
          if (tag_stop == 0) {
            cnt_pg_other++;
            traj_tmp.push_back(t);
          }
          else if (tag_stop == 1) {
            cnt_pg_center++;
            traj_tmp.push_back(t);
          }
          else if (tag_stop == 2) {
            cnt_pg_coast++;
            traj_tmp.push_back(t);
          }
          else if (tag_stop == 3) {
            cnt_pg_cities++;
            traj_tmp.push_back(t);
          }
          else if (tag_stop == 4) {
            cnt_pg_station++;
          }
          else {
            std::cout << "tag_stop: " << tag_stop << std::endl;
            std::cin.get();
          }
        }
      }
      // 2: stop point
      else if (config_.polygons_code[1] == "2") {
        if (tag_stop == location_type) {
          cnt_from++;
          if (tag_start == 0) {
            cnt_pg_other++;
            traj_tmp.push_back(t);
          }
          else if (tag_start == 1) {
            cnt_pg_center++;
            traj_tmp.push_back(t);
          }
          else if (tag_start == 2) {
            cnt_pg_coast++;
            traj_tmp.push_back(t);
          }
          else if (tag_start == 3) {
            cnt_pg_cities++;
            traj_tmp.push_back(t);
          }
          else if (tag_start == 4) {
            cnt_pg_station++;
          }
          else {
            std::cout << "tag start: " << tag_start << std::endl;
            std::cin.get();
          }
        }
      }
      else {
        std::cout << "Second item in polygons code not valid!" << std::endl;
        std::cin.get();
      }
    }
    std::cout << "Polygons:    Center Traj   " << (cnt_pg_center / double(cnt_from))*100.0 << "%" << std::endl;
    std::cout << "Polygons:    Cities Traj   " << (cnt_pg_cities / double(cnt_from)) * 100.0 << "%" << std::endl;
    std::cout << "Polygons:    Coast Traj    " << (cnt_pg_coast / double(cnt_from)) * 100.0 << "%" << std::endl;
    std::cout << "Polygons:    Station Traj  " << (cnt_pg_station / double(cnt_from))*100.0 << "%" << std::endl;
    std::cout << "Polygons:    Other Traj    " << (cnt_pg_other / double(cnt_from))*100.0 << "%" << std::endl;
    std::cout << "Num Traj after polygons    " << traj_tmp.size() << std::endl;
  }
  // 3 args [location_start, location_sto, code_number (0 both way, 1 oneway)]
  else if (config_.polygons_code.size() == 3) {
    int cnt_stop2start = 0;
    int cnt_start2stop = 0;
    int cnt_from = 0;
    int cnt_pg_other = 0;

    //initialize start type
    int start_type = 3;
    if (config_.polygons_code[0].find("center") == 0) start_type = 1;
    else if (config_.polygons_code[0].find("coast") == 0) start_type = 2;
    else if (config_.polygons_code[0].find("station") == 0) start_type = 4;

    //initialize stop type
    int stop_type = 3;
    if (config_.polygons_code[1].find("center") == 0) stop_type = 1;
    else if (config_.polygons_code[1].find("coast") == 0) stop_type = 2;
    else if (config_.polygons_code[1].find("station") == 0) stop_type = 4;

    std::cout << "Polygons code: " << config_.polygons_code[0] << " , " << config_.polygons_code[1] << " , " << config_.polygons_code[2] << std::endl;

    for (const auto &t : traj) {
      int wn_start = 0;
      int wn_stop = 0;
      int tag_start = 0;
      int tag_stop = 0;
      std::string polyg_start, polyg_stop;
      for (auto &pl : polygon) {
        wn_start = pl.is_in_wn(t.stop_point.front().centroid.lat, t.stop_point.front().centroid.lon);
        if (wn_start != 0) {
          tag_start = pl.tag_type;
          polyg_start = pl.pro["name"];
          break;
        }
      }
      for (auto &pl : polygon) {
        wn_stop = pl.is_in_wn(t.stop_point.back().centroid.lat, t.stop_point.back().centroid.lon);
        if (wn_stop != 0) {
          polyg_stop = pl.pro["name"];
          tag_stop = pl.tag_type;
          break;
        }
      }
      // 0 : both directions
      if (config_.polygons_code[2] == "0") {
        if ((tag_start == start_type || tag_stop == stop_type) && tag_start != tag_stop) {
          cnt_from++;
          if (tag_stop == 0 || tag_start == 0) {
            cnt_pg_other++;
          }
          else if (tag_start == start_type && tag_stop == stop_type) {
            cnt_start2stop++;
            traj_tmp.push_back(t);
          }
          else if (tag_stop == start_type && tag_start == stop_type) {
            cnt_stop2start++;
            traj_tmp.push_back(t);
          }
        }
      }
      // 1: from start to stop
      else if (config_.polygons_code[2] == "1") {
        if ((tag_start == start_type || tag_stop == stop_type) && tag_start != tag_stop) {
          cnt_from++;
          if (tag_stop == 0 || tag_start == 0) {
            cnt_pg_other++;
          }
          else if (tag_start == start_type && tag_stop == stop_type) {
            cnt_start2stop++;
            traj_tmp.push_back(t);
          }
          else if (tag_stop == start_type && tag_start == stop_type) {
            cnt_stop2start++;
          }
        }
      }
      else {
        std::cout << "Second item in polygons code not valid!" << std::endl;
        std::cin.get();
      }
    }
    std::cout << "Polygons:    Start-stop traj " << (cnt_start2stop / double(cnt_from))*100.0 << "%" << std::endl;
    std::cout << "Polygons:    Stop-start traj " << (cnt_stop2start / double(cnt_from)) * 100.0 << "%" << std::endl;
    std::cout << "Polygons:    Other Traj      " << (cnt_pg_other / double(cnt_from))*100.0 << "%" << std::endl;
    std::cout << "Num Traj after polygons      " << traj_tmp.size() << std::endl;
  }
  else {
    std::cout << " Code for polygon analysis missed or wrong!" << std::endl;
  }

  traj = traj_tmp;
  traj_tmp.clear();
  traj_tmp.shrink_to_fit();
  return;
}
//----------------------------------------------------------------------------------------------------
void make_multimodality() {
  // prepare features for Transport Means Recognition
  vector<float> features_data;
  for (auto &t : traj) {
    t.v_max = 0.0;
    t.v_min = 1000.0;
    t.a_max = 0.0;
    t.a_min = 1000.0;
    for (int sp = 1; sp < t.stop_point.size(); ++sp) {
      double speed = t.stop_point[sp].points.front().speed;
      if (speed >= t.v_max)
        t.v_max = speed;
      if (speed < t.v_min)
        t.v_min = speed;

      if (sp >= 2) {
        double accel = t.stop_point[sp].points.front().accel;
        if (accel >= t.a_max)
          t.a_max = accel;
        if (accel < t.a_min)
          t.a_min = accel;
      }
    }
  }

  for (auto &t : traj) features_data.push_back(float(t.average_speed));
  for (auto &t : traj) features_data.push_back(float(t.v_max));
  for (auto &t : traj) features_data.push_back(float(t.v_min));
  //for (auto &t : traj) features_data.push_back(float(t.a_max));

  std::cout << "**********************************" << std::endl;
  std::cout << "Multimodality num classes:      " << config_.num_tm << std::endl;

  int num_tm = config_.num_tm;
  int num_N = int(traj.size());
  int num_feat = 3; //v_average, v_max, v_min
  double epsilon_fcm = 0.005;
  FCM *fcm;
  fcm = new FCM(2, epsilon_fcm);
  Map<MatrixXf> data_tr(features_data.data(), num_N, num_feat);
  MatrixXf data = data_tr;
  fcm->set_data(&data);
  fcm->set_num_clusters(num_tm);

  //initialize U[0] randomly
  random_device rnd_device;
  mt19937 mersenne_engine{ rnd_device() };  // Generates random integers
  uniform_real_distribution<float> dist{ 0.0, 1.0 };
  auto gen = [&dist, &mersenne_engine]() {
    return dist(mersenne_engine);
  };
  vector<float> vec_rnd(num_N*num_tm);
  generate(begin(vec_rnd), end(vec_rnd), gen);
  Map<MatrixXf> membership_temp(vec_rnd.data(), num_N, num_tm);
  MatrixXf membership = membership_temp;
  fcm->set_membership(&membership);

  double diff = 1.0;
  fcm->compute_centers();
  fcm->update_membership();
  while (diff > epsilon_fcm) {
    fcm->compute_centers();
    diff = fcm->update_membership();
  }

  for (int n = 0; n < num_tm; ++n) {
    centers_fcm_base cw;
    cw.idx = n;
    for (int m = 0; m < num_feat; ++m)
      cw.feat_vector.push_back((*(fcm->get_cluster_center()))(n, m));
    centers_fcm.push_back(cw);
  }

  std::cout << "Multimodality p threshold:      " << config_.threshold_p << std::endl;
  // save results
  int cnt_recong = 0;
  for (int n = 0; n < traj.size(); ++n) {
    int max_idx;
    double max_p = 0.0;
    for (int m = 0; m < num_tm; ++m) {
      traj[n].p_cluster.push_back((*(fcm->get_membership()))(n, m));
      if ((*(fcm->get_membership()))(n, m) > max_p) {
        max_idx = m;
        max_p = (*(fcm->get_membership()))(n, m);
      }
    }
    if (max_p < config_.threshold_p)
      traj[n].means_class = 10; //fake class index for hybrid traj
    else{
      traj[n].means_class = max_idx;
      cnt_recong++;
    }
    traj[n].means_p = max_p;
  }
  for (auto &t : traj) centers_fcm[t.means_class].cnt++;
  std::cout << "Multimodality traj recognized: " << cnt_recong << std::endl;

  double dunn_i = measure_dunn_index();
  std::cout << "Multimodality dunn index:       " << dunn_i << std::endl;

  //temp
  //ofstream out_scan(config_.cartout_basename + "scan_param.csv", ios::out | ios::app);
  //out_scan << config_.num_tm << ";" << config_.threshold_p << ";" << dunn_i << std::endl;

  if (config_.enable_print) {
    ofstream out_fcm_center(config_.cartout_basename + config_.name_pro + "_fcm_centers.csv");
    out_fcm_center << "idx;av_speed;v_max;v_min;cnt" << std::endl;
    for (int c = 0; c < centers_fcm.size(); ++c){
      out_fcm_center << c;
      for (auto &cc : centers_fcm[c].feat_vector)
        out_fcm_center << ";" << cc;
      out_fcm_center << ";" << centers_fcm[c].cnt << std::endl;
    }
    out_fcm_center.close();

    ofstream out_fcm(config_.cartout_basename + config_.name_pro + "_fcm.csv");
    out_fcm << "lenght;time;av_speed;v_max;v_min;class;p" << std::endl;
    for (auto &t : traj)
      out_fcm << t.length << ";" << t.time << ";" << t.average_speed << ";" << t.v_max << ";" << t.v_min << ";" << t.means_class << ";" << t.means_p << std::endl;
    out_fcm.close();
  }
}
//----------------------------------------------------------------------------------------------------
