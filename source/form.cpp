#include "stdafx.h"
#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Slider.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Output.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Round_Button.H>
#include <FL/Fl_Menu_Item.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_Sys_Menu_Bar.H>
#include <FL/Fl_Choice.H>
#include "global_params.h"
#include "data_reading.h"
#include "frame.h"
#include "form.h"
#include "record.h"
#include "draw.h"
#include "config.h"

Fl_Window       *form;
Fl_Box          *mia;
Frame           *scene = NULL;
Fl_Sys_Menu_Bar *menu;
Fl_Value_Slider *zoom, *time_slider;
Fl_Output       *line1, *line2;
Fl_Button       *button_exit, *button_plus, *button_minus;
Fl_Check_Button *button_data, *button_outofcarto, *button_path, *button_poly, *button_node;
Fl_Check_Button *button_traj, *button_fluxes, *button_startstop;
Fl_Check_Button *button_polygons;
Fl_Check_Button *button_subnet;

extern list <int> path;
extern int poly_work;
extern config config_;
extern vector<traj_base> traj;
extern vector<activity_base> activity;
extern map<string, vector<int>> subnets;

int screen_w = SCREEN_WIDTH, screen_h = SCREEN_HEIGHT;
int w_est, h_est, space, b_w, b_h;
int offset_h, r_offset_h;

bool re_draw = true;
bool show_data = false, show_outofcarto = false, show_path = false, show_poly = false, show_node = false;
bool show_traj = false;
bool show_fluxes = true;
bool show_startstop = false;
bool show_subnet = false;
bool show_polygons = false;

double alfa_zoom = ZOOM_START, time_value = 0.0;
int traj_work = 0;
int means_work = 0;
int subnet_work = 0;
string subnet_select;

//-------------------------------------------------------------------------------------------------
void set_off()
{
  button_data->clear(); 
  button_outofcarto->clear(); 
  button_path->clear(); 
  button_poly->clear(); 
  button_node->clear();
  button_traj->clear(); 
  button_fluxes->clear();
  button_startstop->clear();
  button_polygons->clear();
  button_subnet->clear();
  show_data = show_outofcarto = show_path = show_poly = show_node = show_traj = false;
  show_fluxes = show_polygons = show_startstop = show_subnet = false;
  re_draw = true;
  line2->value("");
}
//-------------------------------------------------------------------------------------------------
void plus_cb(Fl_Widget*)
{
  if (show_traj) { traj_work++; if (traj_work >= traj.size())  traj_work = int(traj.size()) - 1; }
  if (show_fluxes || show_startstop) { means_work++; if (means_work >= config_.num_tm+1)  means_work = config_.num_tm; }
  if (show_subnet) { subnet_work++; if (subnet_work >= subnets.size()) means_work = int(subnets.size()-1); }
  re_draw = true;
}
void minus_cb(Fl_Widget*)
{
  if (show_traj) { traj_work--; if (traj_work < 0)  traj_work = 0; }
  if (show_fluxes || show_startstop) { means_work--; if (means_work < 0)  means_work = 0; }
  if (show_subnet) { subnet_work--; if (subnet_work < 0)  subnet_work = 0; }
  re_draw = true;
}
void exit_cb(Fl_Widget*) { form->hide(); }
void zoom_cb(Fl_Widget*) { alfa_zoom = zoom->value()*zoom->value(); re_draw = true; }
void time_cb(Fl_Widget*) { time_value = time_slider->value(); re_draw = true; }
void show_data_cb(Fl_Widget*) { set_off(); button_data->set(); show_data = !show_data; }
void show_outofcarto_cb(Fl_Widget*) { set_off(); button_outofcarto->set(); show_outofcarto = !show_outofcarto; }
void show_path_cb(Fl_Widget*) { set_off(); button_path->set(); show_path = !show_path; path.clear(); }
void show_poly_cb(Fl_Widget*) { set_off(); button_poly->set(); show_poly = !show_poly; }
void show_node_cb(Fl_Widget*) { set_off(); button_node->set(); show_node = !show_node; }
void show_traj_cb(Fl_Widget*) { set_off(); button_traj->set(); show_traj = !show_traj; }
void show_fluxes_cb(Fl_Widget*) { set_off(); button_fluxes->set(); show_fluxes = !show_fluxes; }
void show_startstop_cb(Fl_Widget*) { set_off(); button_startstop->set(); show_startstop = !show_startstop; }
void show_polygons_cb(Fl_Widget*) { set_off(); button_polygons->set(); show_polygons = !show_polygons; }
void show_subnet_cb(Fl_Widget*) { set_off(); button_subnet->set(); show_subnet= !show_subnet; }
//-------------------------------------------------------------------------------------------------
void make_window(void)
{
  // Get screen resolution and adjust window dimension to 90% of screen width and height
  int form_x, form_y, form_w, form_h;
  Fl::screen_work_area(form_x, form_y, form_w, form_h, 0);
  w_est = int(form_w * 0.95);
  h_est = int(form_h * 0.9);
  screen_w = int(w_est * 0.8);
  screen_h = int(h_est * 0.8);
  space = int(0.01 * w_est);
  b_w = int(0.06 * w_est);
  b_h = int(0.04 * h_est);

  form = new Fl_Window(10, 10, w_est, h_est, "");
  mia = new Fl_Box(FL_NO_BOX, 0, 0, w_est, h_est, "");
  new Fl_Box(FL_DOWN_FRAME, space + b_w + space - 3, space - 3, screen_w + 6, screen_h + 6, "SR");
  scene = new Frame(space + b_w + space, space, screen_w, screen_h, 0);
  form->resizable(mia);

  // ------------  Menu  --------------------------------------------------------------
  menu = new Fl_Sys_Menu_Bar(0, 0, b_w, b_h, "");
  offset_h = 2 * b_h;
  // ------------  Text  --------------------------------------------------------------
  line1 = new Fl_Output(space, 2 * space + screen_h, screen_w / 3, b_h, "");
  line2 = new Fl_Output(2 * space + screen_w / 3, 2 * space + screen_h, 2 * screen_w / 3 - space, b_h, "");
  // ------------  Time- --------------------------------------------------------------
  button_minus = new Fl_Button(2 * space + screen_w, 2 * space + screen_h, b_h, b_h, "-");
  button_minus->callback(minus_cb);
  // ------------  Time+ --------------------------------------------------------------
  button_plus = new Fl_Button(3 * space + screen_w + b_h, 2 * space + screen_h, b_h, b_h, "+");
  button_plus->callback(plus_cb);
  // ------------  Zoom ---------------------------------------------------------------
  zoom = new Fl_Value_Slider(space, h_est - 2 * space - b_h, 4 * b_w, b_h, "Zoom");
  zoom->type(FL_HOR_SLIDER);  zoom->bounds(0.5, 10.0); zoom->value(ZOOM_START);      zoom->callback(zoom_cb);
  alfa_zoom = zoom->value()*zoom->value();
  // ------------  Time ---------------------------------------------------------------
  time_slider = new Fl_Value_Slider(2 * space + 4 * b_w, h_est - 2 * space - b_h, 4 * b_w, b_h, "Time");
  time_slider->type(FL_HOR_SLIDER);  
  time_slider->bounds(0, 24); 
  time_slider->value(0.0);              
  time_slider->callback(time_cb);
  time_value = time_slider->value();
  // ------------  Exit ---------------------------------------------------------------
  button_exit = new Fl_Button(w_est - space - b_w, h_est - space - b_w, b_w, b_w, "Exit");
  button_exit->callback(exit_cb);
  // ------------  Subnet  ------------------------------------------------------------
  //CreaSubnet();
  // ------------  Data    -----------------------------------------------------------
  r_offset_h = space;
  button_data = new Fl_Check_Button(4 * space + screen_w + b_w, r_offset_h, b_w, b_h, "Data");
  r_offset_h += b_h;
  button_data->callback(show_data_cb);
  if (show_data) button_data->set();
  // ------------  Out of Map --------------------------------------------------------------
  button_outofcarto = new Fl_Check_Button(4 * space + screen_w + b_w, r_offset_h, b_w, b_h, "Out of Carto"); 
  r_offset_h += b_h;
  button_outofcarto->callback(show_outofcarto_cb);
  if (show_outofcarto) button_outofcarto->set();
  // ------------  Fluxes --------------------------------------------------------------
  button_fluxes = new Fl_Check_Button(4 * space + screen_w + b_w, r_offset_h, b_w, b_h, "Fluxes"); 
  r_offset_h += b_h;
  button_fluxes->callback(show_fluxes_cb);
  if (show_fluxes) button_fluxes->set();
  // ------------  Start Stop ----------------------------------------------------------
  button_startstop = new Fl_Check_Button(4 * space + screen_w + b_w, r_offset_h, b_w, b_h, "StartStop");
  r_offset_h += b_h;
  button_startstop->callback(show_startstop_cb);
  if (show_startstop) button_startstop->set();
  // ------------  Traj -----------------------------------------------------------------
  button_traj = new Fl_Check_Button(4 * space + screen_w + b_w, r_offset_h, b_w, b_h, "Traj");
  r_offset_h += b_h;
  button_traj->callback(show_traj_cb);
  if (show_traj) button_traj->set();
  // ------------  Poly -----------------------------------------------------------------
  button_poly = new Fl_Check_Button(4 * space + screen_w + b_w, r_offset_h, b_w, b_h, "Poly");
  r_offset_h += b_h;
  button_poly->callback(show_poly_cb);
  if (show_poly) button_poly->set();
  // ------------  Node -----------------------------------------------------------------
  button_node = new Fl_Check_Button(4 * space + screen_w + b_w, r_offset_h, b_w, b_h, "Node"); 
  r_offset_h += b_h;
  button_node->callback(show_node_cb);
  if (show_node) button_node->set();
  // ------------  Path -----------------------------------------------------------------
  button_path = new Fl_Check_Button(4 * space + screen_w + b_w, r_offset_h, b_w, b_h, "Path"); 
  r_offset_h += b_h;
  button_path->callback(show_path_cb);
  if (show_path) button_path->set();
  // ------------  Poly -----------------------------------------------------------------
  button_polygons = new Fl_Check_Button(4 * space + screen_w + b_w, r_offset_h, b_w, b_h, "Polygons");
  r_offset_h += b_h;
  button_polygons->callback(show_polygons_cb);
  if (show_polygons) button_polygons->set();
  // ------------  Subnet  -----------------------------------------------------------------
  button_subnet = new Fl_Check_Button(4 * space + screen_w + b_w, r_offset_h, b_w, b_h, "Subnet");
  r_offset_h += b_h;
  button_subnet->callback(show_subnet_cb);
  if (show_subnet) button_subnet->set();
  //-----------------------------------------------------------------------------------
  form->end();
  form->show();
  scene->show();
}
//-------------------------------------------------------------------------------------------------
