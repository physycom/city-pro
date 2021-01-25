#include "stdafx.h"

#include <FL/Fl.H>
#include <FL/gl.h>
#include <FL/glu.h>
#include <FL/Fl_Output.H>
#include "carto.h"
#include "record.h"
#include "draw.h"
#include "gltrans.h"
#include "data_analysis.h"
#include "config.h"
//#include "bestPath.h"
//#include "ale/subnet_gra.h"

//#include <physycom/geometry.hpp>

#define SQUARE           1
#define SMALL_SQUARE     2
#define SS_SQUARE        3
#define DISK             4

extern vector <poly_base> poly;
extern vector <polygon_base> polygon;
extern vector <node_base> node;
extern vector <traj_base> traj;
extern vector <cluster_base> data_notoncarto;
extern vector <activity_base> activity;
extern list <int> path;
extern config config_;
extern map<size_t, int> activity_idx;
extern vector<centers_fcm_base> centers_fcm;

extern bool show_data, show_outofcarto, show_path, show_poly, show_node, show_traj;
extern bool show_fluxes;
extern bool show_startstop;
extern bool show_subnet;
extern bool show_polygons;
extern bool re_draw;

extern int    graphicInView, timeInView;
extern double alfa_zoom, delta_lon, delta_lat, lat0, lon0, dlon, dlat, time_value;

extern Fl_Output *line1, *line2;
extern int poly_work, node_work, traj_work, means_work;

extern int screen_width, screen_height;
string name;
extern int node_1, node_2;
extern double sigma;

// subnet
//extern map<string, vector<int>> subnets;
//extern string subnet_select;

// ********************************************************************************************************
void draw_init(void)
{
  gl_font(FL_HELVETICA_BOLD, 26);
  double LX, LY;

  LX = 0.006*dlon / alfa_zoom; LY = 0.7*LX;
  glNewList(SQUARE, GL_COMPILE);
  glBegin(GL_QUADS);
  glVertex3d(-LX, -LY, 0.1); glVertex3d(LX, -LY, 0.1);
  glVertex3d(LX, LY, 0.1); glVertex3d(-LX, LY, 0.1);
  glEnd();
  glEndList();

  LX = 0.5*0.003*dlon / alfa_zoom; LY = LX;
  glNewList(SMALL_SQUARE, GL_COMPILE);
  glBegin(GL_QUADS);
  glVertex3d(-LX, -LY, 0.1); glVertex3d(LX, -LY, 0.1);
  glVertex3d(LX, LY, 0.1); glVertex3d(-LX, LY, 0.1);
  glEnd();
  glEndList();

  LX = 0.0006*dlon / alfa_zoom; LY = 0.7*LX;
  glNewList(SS_SQUARE, GL_COMPILE);
  glBegin(GL_QUADS);
  glVertex3d(-LX, -LY, 0.1); glVertex3d(LX, -LY, 0.1);
  glVertex3d(LX, LY, 0.1); glVertex3d(-LX, LY, 0.1);
  glEnd();
  glEndList();

  glNewList(DISK, GL_COMPILE);
  GLUquadricObj *disk; disk = gluNewQuadric();
  glScaled(1.0, 0.7, 1.0);
  gluDisk(disk, 0, 2 * LX, 10, 1); // gluDisk(disk, inDiameter, outDiameter, vertSlices, horizSlices);
  glEndList();
}
// ********************************************************************************************************
void draw_poly()
{
  glPushMatrix();
  for (int i = 1; i<int(poly.size()); i++) {
    glColor3d(0.4, 0.4, 0.4);
    glBegin(GL_LINE_STRIP);
    for (int j = 0; j < poly[i].points.size(); j++)
      glVertex3d(poly[i].points[j].lon, poly[i].points[j].lat, 0.05);
    glEnd();
  }
  glPopMatrix();
  glColor3d(1.0, 1.0, 1.0);
}
// ********************************************************************************************************
void draw_fluxes()
{
  if (config_.enable_multimodality) {
    glPushMatrix();
    for (int i = 1; i<int(poly.size()); i++) {
      double fluxes = poly[i].classes_flux[means_work];
      if (fluxes > 4.5 * centers_fcm[means_work].sigma) { glColor3d(1.0, 1.0, 0.0); glLineWidth(5); }
      else if (fluxes > 2.0 * centers_fcm[means_work].sigma) { glColor3d(1.0, 0.0, 0.0); glLineWidth(4); }
      else if (fluxes > 1.0 * centers_fcm[means_work].sigma) { glColor3d(0.0, 1.0, 0.0); glLineWidth(3); }
      else if (fluxes > 0.5 * centers_fcm[means_work].sigma) { glColor3d(0.0, 0.0, 1.0); glLineWidth(2); }
      else { glColor3d(0.4, 0.4, 0.4); glLineWidth(1); }
      glBegin(GL_LINE_STRIP);
      for (int j = 0; j < poly[i].points.size(); j++)
        glVertex3d(poly[i].points[j].lon, poly[i].points[j].lat, 0.1);
      glEnd();
    }
    glPopMatrix();

    glDisable(GL_DEPTH_TEST);
    string s2 = " class idx = " + to_string(means_work) + " center_features: ";
    for (auto &m : centers_fcm[means_work].feat_vector) s2 += to_string(m) + "  ";
    s2 += "cnt: " + to_string(centers_fcm[means_work].cnt);
    line2->value(s2.c_str());
    glEnable(GL_DEPTH_TEST);

  }
  else {
    glPushMatrix();
    for (int i = 1; i<int(poly.size()); i++) {
      double fluxes = poly[i].n_traj_FT + poly[i].n_traj_TF;
      if (fluxes > 4.5 * sigma) { glColor3d(1.0, 1.0, 0.0); glLineWidth(5); }
      else if (fluxes > 2.5 * sigma) { glColor3d(1.0, 0.0, 0.0); glLineWidth(4); }
      else if (fluxes > 1.5 * sigma) { glColor3d(0.0, 1.0, 0.0); glLineWidth(3); }
      else if (fluxes > 0.5 * sigma) { glColor3d(0.0, 0.0, 1.0); glLineWidth(2); }
      else { glColor3d(0.4, 0.4, 0.4); glLineWidth(1); }
      glBegin(GL_LINE_STRIP);
      for (int j = 0; j < poly[i].points.size(); j++)
        glVertex3d(poly[i].points[j].lon, poly[i].points[j].lat, 0.1);
      glEnd();
    }
    glPopMatrix();
  }
  
  glColor3d(1.0, 1.0, 1.0); 
  glLineWidth(1);
}
// ********************************************************************************************************
void draw_startstop() {
  if (config_.enable_multimodality) {
    for (auto &t : traj) {
      if (t.means_class == means_work) {
        glPushMatrix();
        glTranslated(t.stop_point.front().centroid.lon, t.stop_point.front().centroid.lat, 0.1);
        glColor3d(1.0, 0.0, 1.0);
        glCallList(SS_SQUARE);
        glPopMatrix();
        glPushMatrix();
        glTranslated(t.stop_point.back().centroid.lon, t.stop_point.back().centroid.lat, 0.1);
        glColor3d(0.0, 1.0, 0.0);
        glCallList(SS_SQUARE);
        glPopMatrix();
      }
    }

    glDisable(GL_DEPTH_TEST);
    string s2 = "purple = start traj;  green = stop traj, class: "+to_string(means_work);
    line2->value(s2.c_str());
    glEnable(GL_DEPTH_TEST);
  }
  else {
    for (auto &t : traj) {
      glPushMatrix();
      glTranslated(t.stop_point.front().centroid.lon, t.stop_point.front().centroid.lat, 0.1);
      glColor3d(1.0, 0.0, 1.0);
      glCallList(SS_SQUARE);
      glPopMatrix();
      glPushMatrix();
      glTranslated(t.stop_point.back().centroid.lon, t.stop_point.back().centroid.lat, 0.1);
      glColor3d(0.0, 1.0, 0.0);
      glCallList(SS_SQUARE);
      glPopMatrix();
    }

    glDisable(GL_DEPTH_TEST);
    string s2 = "purple = start traj;  green = stop traj";
    line2->value(s2.c_str());
    glEnable(GL_DEPTH_TEST);
  }
  glColor3d(1.0, 1.0, 1.0);
}
// ********************************************************************************************************
void draw_path() {
  int np = int(path.size()); if (np <= 0) return;
  glColor3d(1.0, 1.0, 0.0);
  glPushMatrix();
  for (auto n : path) {
    glBegin(GL_LINE_STRIP);
    for (int j = 0; j < poly[abs(n)].points.size(); j++)
      glVertex3d(poly[abs(n)].points[j].lon, poly[abs(n)].points[j].lat, 1.0);
    glEnd();
  }
  glPopMatrix();
  glColor3d(1.0, 1.0, 1.0);

  glDisable(GL_DEPTH_TEST);
  string s2 = " node1 cid= " + to_string(node[node_1].cid_node) + " nodo2 cid= " + to_string(node[node_2].cid_node);
  line2->value(s2.c_str());
  glEnable(GL_DEPTH_TEST);
}
// ********************************************************************************************************
void draw_node_work() {

  int i = node_work; if (i < 1 || i > node.size() - 1) return;
  glColor3d(1.0, 0.0, 0.0);
  glPushMatrix();
  glTranslated(node[i].lon, node[i].lat, 0.2);
  glCallList(SQUARE);
  glPopMatrix();
  glColor3d(1.0, 1.0, 1.0);

  glDisable(GL_DEPTH_TEST);
  string s2 = " node cid= " + to_string(node[i].cid_node) + " node id= " + to_string(node[i].id_node);
  line2->value(s2.c_str());
  glEnable(GL_DEPTH_TEST);

}
// ********************************************************************************************************
void draw_poly_work() {
  int i = poly_work;
  if (i <1 || i > poly.size() - 1) return;

  glColor3d(1.0, 1.0, 1.0);
  glLineWidth(2);
  glPushMatrix();
  glBegin(GL_LINE_STRIP);
  for (int j = 0; j < poly[i].points.size(); j++)
    glVertex3d(poly[i].points[j].lon, poly[i].points[j].lat, 1.0);
  glEnd();
  glPopMatrix();
  glLineWidth(1);

  glDisable(GL_DEPTH_TEST);
  string s2 = to_string(i) + "   " + poly[i].name + " poly cid= " + to_string(poly[i].cid_poly) + " length: " + to_string(poly[i].length);
  line2->value(s2.c_str());
  glEnable(GL_DEPTH_TEST);

}
// ********************************************************************************************************
void draw_data() {
  for (auto &a : activity) {
    glPushMatrix();
    glTranslated(a.record.front().lon, a.record.front().lat, 0.01);
    glColor3d(1.0, 0.0, 1.0);
    glCallList(SS_SQUARE);
    glPopMatrix();
    glPushMatrix();
    glTranslated(a.record.back().lon, a.record.back().lat, 0.01);
    glColor3d(0.0, 1.0, 0.0);
    glCallList(SS_SQUARE);
    glPopMatrix();
    for (int d = 1; d < a.record.size() - 1; ++d) {
      glPushMatrix();
      glTranslated(a.record[d].lon, a.record[d].lat, 0.01);
      glColor3d(1.0, 0.0, 0.0);
      glCallList(SS_SQUARE);
      glPopMatrix();
    }
  }

  glDisable(GL_DEPTH_TEST);
  string s2 = "yellow = start activity;  green = stop activity";
  line2->value(s2.c_str());
  glEnable(GL_DEPTH_TEST);

  glColor3d(1.0, 1.0, 1.0);

}
// ********************************************************************************************************
void draw_outofcarto() {
  glColor3d(1.0, 0.0, 0.0);

  for (auto &s : data_notoncarto) {
    glPushMatrix();
    glTranslated(s.centroid.lon, s.centroid.lat, 0.5);
    glCallList(SMALL_SQUARE);
    glPopMatrix();
  }
  glDisable(GL_DEPTH_TEST);
  string s2 = to_string(data_notoncarto.size()) + " points are farther than " + to_string(config_.min_poly_distance) + " metri.";
  line2->value(s2.c_str());
  glEnable(GL_DEPTH_TEST);

  glColor3d(1.0, 1.0, 1.0);
}
// ********************************************************************************************************
void draw_traj_work() {

  while (traj[traj_work].row_n_rec < 50) traj_work++;

  int i = traj_work; if (i < 0 || i > traj.size() - 1) return;
  for (auto sp : traj[i].stop_point) {
    for (auto &spp : sp.points) {
      glPushMatrix();
      glColor3d(0.0, 1.0, 0.0);
      glTranslated(spp.lon, spp.lat, 0.05);
      glCallList(SMALL_SQUARE);
      glPopMatrix();
    }
  }
  for (auto sp : traj[i].stop_point) {
    glPushMatrix();
    glColor3d(1.0, 0.0, 0.0);
    glTranslated(sp.centroid.lon, sp.centroid.lat, 0.08);
    glCallList(SMALL_SQUARE);
    glPopMatrix();
  }

  glColor3d(1.0, 1.0, 1.0);

  glDisable(GL_DEPTH_TEST);

  string s2 = "n = " + to_string(traj_work);
  s2 += "  activity_id = " + to_string(traj[i].id_act);
  s2 += "  nDat= " + to_string(traj[i].row_n_rec);
  s2 += "  length= " + to_string(traj[i].length);
  s2 += "  time= " + to_string(traj[i].time);
  s2 += "  n points = " + to_string(traj[i].stop_point.size());
  line2->value(s2.c_str());
  glEnable(GL_DEPTH_TEST);
}
// ********************************************************************************************************
void draw_record_intime()
{
  double delta_t = (1 * 3600) / 3600;

  glColor3d(1.0, 0.0, 0.0);
  vector <int> po(int(poly.size()), 0);
  list <pair<int, double>>::iterator pn, pn1, pEnd;
  vector <record_base> intime;

  for (auto &k : traj)
    for (auto &l : k.record)
      if (l.t > time_value && l.t < time_value + delta_t) {
        intime.push_back(l);
      }


  for (auto &s : intime) {
    glPushMatrix();
    glTranslated(s.lon, s.lat, 0.5);
    glCallList(SMALL_SQUARE);
    glPopMatrix();
  }
  glColor3d(1.0, 1.0, 1.0);

  glDisable(GL_DEPTH_TEST);
  string s2 = "Daily Hours: " + to_string(int(time_value));
  line2->value(s2.c_str());
  glEnable(GL_DEPTH_TEST);

  intime.clear();
}
// ********************************************************************************************************
void draw_polygons() {

  glPushMatrix();
  for (const auto &p : polygon)
  {
    glColor3d(0.0, 0.0, 1.0);
    if (p.tag_type == 1) glColor3d(1.0, 1.0, 0.0);
    glBegin(GL_LINE_STRIP);
    for (const auto &pp : p.points) {
      glVertex3d(pp.lon, pp.lat, 0.1);
    }
    glEnd();
  }
  glPopMatrix();
}
// ********************************************************************************************************
void draw_scene() {
  re_draw = false;
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glPushMatrix();
  glTranslated(lon0, lat0, 0.0);
  glScaled(alfa_zoom, alfa_zoom, 1.0);
  glTranslated(-lon0, -lat0, 0.0);
  glTranslated(delta_lon, delta_lat, 0.0);
  save_model();
  glPushMatrix();
  draw_poly();
  if (show_data)       draw_data();
  if (show_outofcarto) draw_outofcarto();
  if (show_node)       draw_node_work();
  if (show_poly)       draw_poly_work();
  if (show_traj)       draw_traj_work();
  if (show_path)       draw_path();
  //if (show_record_intime) draw_record_intime();
  if (show_fluxes)    draw_fluxes();
  if (show_startstop)  draw_startstop();
  //if (show_subnet) draw_subnet();
  if (show_polygons) draw_polygons();
  glPopMatrix();
  glPopMatrix();
  glDisable(GL_BLEND);
}

// General purpose ****************************************************************************************
void color_palette(const int &i)
{
  switch (i % 5)
  {
  case 0:  glColor3d(198.f / 255.f, 44.f / 255.f, 1.f / 255.f); break; // red
  case 1:  glColor3d(1.f / 255.f, 1.f / 255.f, 198.f / 255.f); break; // blue
  case 2:  glColor3d(172.f / 255.f, 44.f / 255.f, 247.f / 255.f); break; // purple
  case 3:  glColor3d(255.f / 255.f, 249.f / 255.f, 81.f / 255.f); break; // yellow
  case 4:  glColor3d(1.f / 255.f, 198.f / 255.f, 1.f / 255.f); break;    // green
  default: glColor3d(255.f / 255.f, 255.f / 255.f, 255.f / 255.f); break; // white
  }
}