#include "stdafx.h"
#include "record.h"
#include "carto.h"
#include "data_reading.h"
//#include "analisi_dati.h"
#include "config.h"
#include <boost/algorithm/string.hpp>
#include <physycom/geometry.hpp>

vector <poly_base> poly;
vector <activity_base> activity;
vector <polygon_base> polygon;
map <unsigned long long int, int> poly_cid2lid;

extern config config_;
//------------------------------------------------------------------------------------------------------
void load_poly()
{
  string poly_pnt_file = config_.file_pnt;
  string poly_pro_file = config_.file_pro;

  FILE *fp0; fp0 = fopen(poly_pnt_file.c_str(), "r");
  if (fp0 == NULL) { cout << "fopen error in " << poly_pnt_file << endl; exit(5); }

  int id, nPT, ilat, ilon;
  long long int cid;
  vector <point_base> points; poly_base poly_w;
  poly_w.clear();
  // add empty poly at the position 0
  points.clear();
  poly_w.set(0, 0, points);
  poly.push_back(poly_w);
  poly_w.clear();

  // read pnt file
  while (fscanf(fp0, " %lld %d %d ", &cid, &id, &nPT) != EOF) {
    for (int i = 0; i < nPT; i++) {
      if (fscanf(fp0, " %d %d", &ilat, &ilon) == EOF) {
        cout << "reading error in " << poly_pnt_file << endl; fflush(stdout); exit(5);
      }
      point_base point_w(ilat, ilon);
      points.push_back(point_w);
    }
    poly_w.set(id, cid, points);
    poly.push_back(poly_w);
    points.clear(); poly_w.clear();
  }
  fclose(fp0);
  cout << "Poly:     " << poly.size() - 1 << endl;

  // read pro file
  fp0 = fopen(poly_pro_file.c_str(), "r");
  if (fp0 == NULL) { cout << "fopen error in " << poly_pro_file << endl; exit(5); }
  if (fscanf(fp0, "%*[^\n]\n")) {}; // skip header

  int frc, n2c, fow, oneway, kmh, lanes; char *name = new char[70];
  unsigned long long int id_poly, cid_Fjnct, cid_Tjnct;
  float meters;

  int n = 0;
  // front tail VS start end
  while (fscanf(fp0, " %lld %lld %lld %f %f %d %d %d %d %d %s",
    &id_poly, &cid_Fjnct, &cid_Tjnct, &meters, &frc, &n2c, &fow, &oneway, &kmh, &lanes, name) != EOF) {
    n++;
    if (poly[n].cid_poly == id_poly) {
      poly[n].set(cid_Fjnct, cid_Tjnct, meters, oneway, name);
      poly[n].measure_length();
      poly_cid2lid[id_poly] = n;
    }
    else { cout << " index error in poly [" << n << "] " << poly[n].cid_poly << " vs " << id_poly << endl; exit(5); }
  }
  fclose(fp0);
}
//------------------------------------------------------------------------------------------------------
void load_data()
{
  int cnt_in = 0;
  int cnt_out = 0;

  for (auto &i : config_.file_data) {
    string data_file = i;
    ifstream fp_in(data_file.c_str());
    string line;
    vector<string> strs;
    record_base rw;
    map<size_t, vector<record_base>> activity_collect;

    size_t id_act = 0;
    int cnt = 0;
    if (!fp_in) {
      cout << "load_data: error in reading " << data_file << endl;
      exit(1);
    }
    int cnt_row = 0;
    getline(fp_in, line); // salto l'header
    while (fp_in) {
      getline(fp_in, line);
      boost::split(strs, line, boost::is_any_of(";"), boost::token_compress_off);
      cnt_row++;
      if (strs.size() > 1) {
        id_act = stoll(strs[1]);
        rw.itime = stol(strs[3]);
        rw.lat = stod(strs[4]);
        rw.lon = stod(strs[5]);
        if ((rw.lat > config_.lat_min && rw.lat < config_.lat_max) && (rw.lon > config_.lon_min && rw.lon < config_.lon_max))
        {
          if ((rw.itime > config_.start_time) && (rw.itime < config_.end_time))
          {
            rw.t = (double)((rw.itime - config_.start_time) / 3600.);
            int n_day = int(rw.t / 24.0);
            if (config_.slice_time.size() != 0) {
              if (rw.t <= config_.slice_time[0] + n_day * 24.00 && rw.t > config_.slice_time[1] + n_day * 24.0) {
                cnt_in++;
                activity_collect[id_act].push_back(rw);
              }
              cnt_out++;
            }
            else {
              activity_collect[id_act].push_back(rw);
            }
          }
        }
        if (cnt_row % 1000000 == 0) std::cout << "cnt_row:    " << cnt_row << std::endl;
      }
    }
    fp_in.close();

    std::cout << "**********************************" << std::endl;
    if (config_.slice_time.size() != 0)
      std::cout << "slice time:  " << cnt_in << "/" << cnt_out << std::endl;

    activity_base aw;
    for (const auto &n : activity_collect) {
      aw.id_act = n.first;
      aw.record = n.second;
      activity.push_back(aw);
    }
  }
  cout << "Activity: " << int(activity.size()) << endl;
}

//------------------------------------------------------------------------------------------------------
void load_polygon() {
  if (config_.file_polygons == "file_polygons") return;
  int n_polygon = 0;
  std::ifstream polyifs(config_.file_polygons);
  auto jpoly = jsoncons::json::parse(polyifs);
  for (const auto &feature : jpoly["features"].array_range()) {
    if (!feature["geometry"]["coordinates"].size()) continue;
    auto type = feature["geometry"]["type"].as<std::string>();
    if (type == "LineString")
    {
      ++n_polygon;
      auto pol = feature["geometry"]["coordinates"];
      polygon_base pw = polygon_base(pol);
      for (const auto &pro : feature["properties"].object_range())
        pw.pro[std::string(pro.key())] = pro.value().as<std::string>();
      pw.id = n_polygon;
      if (pw.pro["name"] == "center") pw.tag_type = 1;
      else if (pw.pro["name"].find("coast") == 0) pw.tag_type = 2;
      else if (pw.pro["name"].find("station") == 0) pw.tag_type = 4;
      else pw.tag_type = 3;
      polygon.push_back(pw);
    }
    else
    {
      std::cerr << "GEOJSON feature type " << type << " unsupported" << std::endl;
      return;
    }
  }
  std::cout << "Polygon:  " << polygon.size() << std::endl;
}
