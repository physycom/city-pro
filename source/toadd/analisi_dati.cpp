#include "stdafx.h"
#include <iostream>
#include <algorithm>
#include "polyline.h"
#include "nodo.h"
#include "dato.h"
#include "mappa.h"
#include "analisi_dati.h"
#include "bestPath.h"
#include "metodi.h"
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

extern vector <PolyLine> polyline;  extern int n_poly;
extern vector <Poligono> polygon;
extern vector <People> people;
extern vector<Nodo> nodo;
vector <Trajectory> traj;   int n_traj;
vector <Dato> mob_presence;
vector<Trajectory> pawns_traj;
vector<Trajectory> wheels_traj;
extern map<string, vector<int>> subnets;
double sigma_MTSR, sigma_STIT;
int cnt_errore = 0;
vector <pair<int, double>> nl_poly;
vector<Dato> data_notonmap;
extern config config_;
map<size_t, int> geid_idx;
DataLoss dataloss;

extern vector<AttrPoint> attrpoints;
extern vector<AttrPoint> testpoints;
extern vector<Client> clients;
extern map<int, int> super_proxy;
vector<Presence> presence;




//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
void StampaFlussiAll(const string &fname) {
  ofstream out(fname);
  //out << "p.id;p.id_local;p.n2c;p.n_traj_FT;p.n_traj_TF; FT+TF; length" << endl;
  out << "#lid cid n_FT  n_TF" << endl;
  for (auto p : polyline) {
    //out << p.id << ";" << p.id_local << ";" << p.n2c << ";";
    //out << p.n_traj_FT << ";" << p.n_traj_TF << ";" << p.n_traj_FT + p.n_traj_TF << ";";
    //out << p.length << endl;
    out << p.id_local - 1 << "\t" << p.cid_poly << "\t" << p.n_traj_FT << "\t" << p.n_traj_TF << endl;
  }
  out.close();
}
//----------------------------------------------------------------------------------------------------
void StampaFlussiSubnet(const string &fname) {
  ofstream out(fname);
  out << "p.id;p.id_local;nodeF;nodeT;p.n_traj_FT;p.n_traj_TF; FT+TF; length" << endl;
  for (auto p : polyline) {
    out << p.id << ";" << p.id_local << ";" << p.cid_Fjnct << ";" << p.cid_Tjnct << ";";
    if (p.n2c == 6) out << 0 << ";" << 0 << ";" << 0 << ";";
    else            out << p.n_traj_FT << ";" << p.n_traj_TF << ";" << p.n_traj_FT + p.n_traj_TF << ";";
    out << p.length << endl;
  }
  out.close();
}
//----------------------------------------------------------------------------------------------------
void StampaFlussiSubnet_IT_ST(const string &fname) {
  ofstream out(fname);
  out << "p.id;p.id_local;nodeF;nodeT;p.n_traj_FT_IT;p.n_traj_TF_IT; FT+TF_IT;p.n_traj_FT_ST;p.n_traj_TF_ST; FT+TF_ST; length" << endl;
  for (auto p : polyline) {
    out << p.id << ";" << p.id_local << ";" << p.cid_Fjnct << ";" << p.cid_Tjnct << ";";
    if (p.n2c == 6) out << 0 << ";" << 0 << ";" << 0 << ";" << 0 << ";" << 0 << ";" << 0 << ";";
    else {
      out << p.flussi_distinti["IT"].FT << ";" << p.flussi_distinti["IT"].TF << ";" << p.flussi_distinti["IT"].FT + p.flussi_distinti["IT"].TF
        << ";" << p.flussi_distinti["ST"].FT << ";" << p.flussi_distinti["ST"].TF << ";" << p.flussi_distinti["ST"].FT + p.flussi_distinti["ST"].TF << ";";
    }
    out << p.length << endl;
  }
  out.close();
}
//----------------------------------------------------------------------------------------------------
void StampaFlussiSubnet_IT_ST_MT_SR(const string &fname) {
  ofstream out(fname);
  out << "p.id;p.id_local;nodeF;nodeT;p.n_traj_FT_IT;p.n_traj_TF_IT; FT+TF_IT;p.n_traj_FT_ST;p.n_traj_TF_ST; FT+TF_ST;p.n_traj_FT_MT;p.n_traj_TF_MT; FT+TF_MT;p.n_traj_FT_SR;p.n_traj_TF_SR; FT+TF_SR; length" << endl;
  for (auto p : polyline) {
    out << p.id << ";" << p.id_local << ";" << p.cid_Fjnct << ";" << p.cid_Tjnct << ";";
    if (p.n2c == 6) out << 0 << ";" << 0 << ";" << 0 << ";" << 0 << ";" << 0 << ";" << 0 << ";" << 0 << ";" << 0 << ";" << 0 << ";" << 0 << ";" << 0 << ";" << 0 << ";";
    else {
      out << p.flussi_distinti["IT"].FT << ";" << p.flussi_distinti["IT"].TF << ";" << p.flussi_distinti["IT"].FT + p.flussi_distinti["IT"].TF
        << ";" << p.flussi_distinti["ST"].FT << ";" << p.flussi_distinti["ST"].TF << ";" << p.flussi_distinti["ST"].FT + p.flussi_distinti["ST"].TF << ";"
        << p.flussi_distinti["MT"].FT << ";" << p.flussi_distinti["MT"].TF << ";" << p.flussi_distinti["MT"].TF + p.flussi_distinti["MT"].FT << ";"
        << p.flussi_distinti["SR"].FT << ";" << p.flussi_distinti["SR"].TF << ";" << p.flussi_distinti["SR"].TF + p.flussi_distinti["SR"].FT << ";";
    }
    out << p.length << endl;
  }
  out.close();
}
//----------------------------------------------------------------------------------------------------
double CalcolaRappresentanza_media(const string &label)
{
  double all = 0, sub = 0;
  for (const auto &p : polyline) {
    if (p.n2c == 6) continue;
    all += (p.n_traj_FT + p.n_traj_TF)*p.length;
    if (find(subnets[label].begin(), subnets[label].end(), p.id_local) != subnets[label].end())
      sub += (p.n_traj_FT + p.n_traj_TF)*p.length;
  }
  return sub / all;
}
//----------------------------------------------------------------------------------------------------
double CalcolaSubnet(const string &label)
{
  double all = 0, sub = 0;
  for (const auto &p : polyline) {
    if (p.n2c == 6) continue;
    all += p.length;
    if (find(subnets[label].begin(), subnets[label].end(), p.id_local) != subnets[label].end())
      sub += p.length;
  }
  return sub / all;
}
//----------------------------------------------------------------------------------------------------
double CalcolaRappresentanza_rapporto(const string &label)
{
  double length_path = 0, length_sub = 0;
  for (const auto &t : traj) {
    for (const auto &p : t.allPath) {
      if (polyline[abs(p.first)].n2c == 6) continue;
      length_path += polyline[abs(p.first)].length;
      if (find(subnets[label].begin(), subnets[label].end(), abs(p.first)) != subnets[label].end())
        length_sub += polyline[abs(p.first)].length;
    }
  }
  return length_sub / length_path;
}
//----------------------------------------------------------------------------------------------------
void CalcolaFlussi() {
  double ST_tot = 0., IT_tot = 0.;
  double MT_tot = 0., SR_tot = 0.;
  double soglia_day = 14.0;
  for (auto n : traj) {
    for (auto j : n.allPath) {
      if (j.first > 0) polyline[j.first].n_traj_FT++;
      else             polyline[-j.first].n_traj_TF++;
      double temp = int(j.second / 24);
      if (j.second - (24 * temp) < soglia_day) {
        MT_tot++;
        if (j.first > 0) polyline[j.first].flussi_distinti["MT"].FT++;
        else             polyline[-j.first].flussi_distinti["MT"].TF++;
      }
      else {
        SR_tot++;
        if (j.first > 0) polyline[j.first].flussi_distinti["SR"].FT++;
        else             polyline[-j.first].flussi_distinti["SR"].TF++;
      }
    }
    if (n.tipo == "IT") {
      for (auto j : n.allPath) {
        if (j.first > 0) polyline[j.first].flussi_distinti[n.tipo].FT++;
        else             polyline[-j.first].flussi_distinti[n.tipo].TF++;
        IT_tot++;
      }
    }
    if (n.tipo == "ST") {
      for (auto j : n.allPath) {
        if (j.first > 0) polyline[j.first].flussi_distinti[n.tipo].FT++;
        else             polyline[-j.first].flussi_distinti[n.tipo].TF++;
        ST_tot++;
      }
    }
  }
  std::cout << "CalcolaFlussi: percentuale di flussi stranieri: " << 100 * ST_tot / (IT_tot + ST_tot) << " %" << endl;
  std::cout << "CalcolaFlussi: percentuale di mattina pomeriggio: " << MT_tot << "  " << SR_tot << endl;
  double media_STIT = 0.0; int cnt_STIT = 0;
  double media_MTSR = 0.0; int cnt_MTSR = 0;
  sigma_MTSR = 0.0;
  sigma_STIT = 0.0;

  sigma_MTSR = 0.0;
  sigma_STIT = 0.0;

  for (int p = 1; p < n_poly; p++) {
    if (polyline[p].n2c == 6) continue;
    pair<int, double> temp;
    temp.first = polyline[p].id_local;
    temp.second = polyline[p].length*(polyline[p].n_traj_FT + polyline[p].n_traj_TF);
    nl_poly.push_back(temp);
    polyline[p].diffST_IT = (polyline[p].flussi_distinti["IT"].FT + polyline[p].flussi_distinti["IT"].TF) - (polyline[p].flussi_distinti["ST"].FT + polyline[p].flussi_distinti["ST"].TF)*IT_tot / ST_tot;
    polyline[p].diffMT_SR = (polyline[p].flussi_distinti["SR"].FT + polyline[p].flussi_distinti["SR"].TF) - (polyline[p].flussi_distinti["MT"].FT + polyline[p].flussi_distinti["MT"].TF)* SR_tot / MT_tot;
    cnt_MTSR++;
    cnt_STIT++;
    media_MTSR += polyline[p].diffMT_SR;
    media_STIT += polyline[p].diffST_IT;
    sigma_MTSR += polyline[p].diffMT_SR*polyline[p].diffMT_SR;
    sigma_STIT += polyline[p].diffST_IT*polyline[p].diffST_IT;
  }

  sort(nl_poly.begin(), nl_poly.end(), [](const pair<int, double> &a, const pair<int, double> &b) -> bool
  {
    return a.second > b.second;
  });

  sigma_MTSR = sqrt(sigma_MTSR / cnt_MTSR);
  sigma_STIT = sqrt(sigma_STIT / cnt_STIT);
  media_MTSR /= cnt_MTSR;
  media_STIT /= cnt_STIT;
  std::cout << "CalcolaFlussi: media_MTSR = " << media_MTSR << endl;
  std::cout << "CalcolaFlussi: media_STIT = " << media_STIT << endl;
  std::cout << "CalcolaFlussi: sigma_MTSR = " << sigma_MTSR << endl;
  std::cout << "CalcolaFlussi: sigma_STIT = " << sigma_STIT << endl;

  if (config_.enable_multimod) {
    for (auto n : pawns_traj) {
      for (auto j : n.allPath) {
        if (j.first > 0) polyline[j.first].flussi_distinti["pawns"].FT++;
        else             polyline[-j.first].flussi_distinti["pawns"].TF++;
      }
    }

    for (auto n : wheels_traj) {
      for (auto j : n.allPath) {
        if (j.first > 0) polyline[j.first].flussi_distinti["wheels"].FT++;
        else             polyline[-j.first].flussi_distinti["wheels"].TF++;
      }
    }
  }
}


///////////////////////////////////////////////////////////////////////////
typedef std::pair<int, double> pair_;
struct comp {
  bool operator()(const pair_& x, const pair_& y) const {
    if (x.second != y.second)
      return x.second < y.second;
    return x.first < y.first;
  }
};

