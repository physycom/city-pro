#include "stdafx.h"
#include <FL/Fl.H>
#include <FL/gl.h>
#include <FL/glu.h>
#include <FL/Fl_Output.H>
#include <FL/Fl_Value_Slider.H>
#include "frame.h"
#include "gltrans.h"
#include "carto.h"

extern bool show_poly, show_node, show_path, re_draw;
extern Fl_Output *line1;
extern int screen_w, screen_h;
extern vector <poly_base> poly;
extern double alfa_zoom_circle;
extern void draw_init();
extern void draw_scene();
extern double lat0, lon0, dlat, dlon;
extern double alfa_zoom;
extern Fl_Value_Slider *zoom;

list <int> path;
double delta_lon = 0.0, delta_lat = 0.0;
int poly_work = 0, node_work = 0;
GLdouble x_mouse, y_mouse, z_mouse;
int node_1, node_2;

//-------------------------------------------------------------------------------------------------
void Frame::draw() {

  if (!valid()) {
    glClearColor(1.0, 1.0, 0.95, 1);                        // Turn the background color black
    //glClearColor(0.0, 0.0, 0.0, 1);                        // Turn the background color black
    glViewport(0, 0, w(), h());                               // Make our viewport the whole window
    glMatrixMode(GL_PROJECTION);                           // Select The Projection Matrix
    glLoadIdentity();                                      // Reset The Projection Matrix
    gluOrtho2D(lon0 - dlon, lon0 + dlon, lat0 - dlat, lat0 + dlat);
    glMatrixMode(GL_MODELVIEW);                            // Select The Modelview Matrix
    glLoadIdentity();                                      // Reset The Modelview Matrix
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    // Clear The Screen And The Depth Buffer
    glLoadIdentity();                                      // Reset The View
    glEnable(GL_DEPTH_TEST);

    draw_init();
    valid(1);
  }
  draw_scene();
}
//-------------------------------------------------------------------------------------------------
int Frame::handle(int evento)
{
  int i_mouse, j_mouse, ierr;
  float iz;

  static char message[300];

  switch (evento)
  {
  case FL_MOVE:
    re_draw = true;
    i_mouse = Fl::event_x();  j_mouse = screen_h - Fl::event_y();
    ierr = unproject(GLdouble(i_mouse), GLdouble(j_mouse), GLdouble(iz), &x_mouse, &y_mouse, &z_mouse);
    sprintf(message, " %10.6lf %10.6lf ", y_mouse, x_mouse);
    glDisable(GL_DEPTH_TEST); line1->value(message); glEnable(GL_DEPTH_TEST);
    break;
  case FL_PUSH:
    re_draw = true;
    i_mouse = Fl::event_x(); 
    j_mouse = screen_h - Fl::event_y();
    ierr = unproject(GLdouble(i_mouse), GLdouble(j_mouse), GLdouble(iz), &x_mouse, &y_mouse, &z_mouse);

    if (Fl::event_button() == 3) {     //  1 left  2 both 3 right click
      delta_lon = -(x_mouse - lon0);
      delta_lat = -(y_mouse - lat0);
      break;
    }
    else if (Fl::event_button() == 1) {
      double poly_dist, node_dist;
      if (show_poly || show_path) find_near_poly(x_mouse, y_mouse, poly_dist, poly_work);
      if (show_node || show_path) find_near_node(x_mouse, y_mouse, node_dist, node_work);
      std::cout << " distance = " << poly_dist << " poly near = " << poly_work << std::endl;
      std::cout << " distance = " << node_dist << " node near = " << node_work << std::endl;
      if (show_path) {
        static bool first = true;
        if (first) {
          first = !first; 
          node_1 = node_work;
          path.clear();
        }
        else {
          first = !first;
          node_2 = node_work;
          //path = bestPoly(nodo_1, nodo_2);
        }
      }
    }
    break;
  case FL_MOUSEWHEEL:
    alfa_zoom -= 0.125 * Fl::event_dy();
    if (alfa_zoom < zoom->minimum()) alfa_zoom = zoom->minimum();
    if (alfa_zoom > zoom->maximum()) alfa_zoom = zoom->maximum();
    zoom->value(sqrt(alfa_zoom));
    re_draw = true;
  default:
    break;
  }
  return 1;
}

