#include "stdafx.h"
#include "global_params.h"
#include "carto.h"
#include "config.h"
//#include "dato.h"

using namespace std;

extern int  screen_w, screen_h;

double lat0, lon0, dlat, dlon, zoom_start;

extern config config_;
extern vector <poly_base> poly;
vector <node_base> node;
vector <arc_base> arc;
//extern vector <People> people;

map<unsigned long long int, int> node_cid2id;

mapping_base **A; int jmax, imax; // A[jmax][imax]
double ds_lat, ds_lon, c_ris1, c_ris2;

// POINT //
point_base::point_base() : lat(0.0), lon(0.0) {}
//- ---------------------------------------------------------------------------------------
point_base::point_base(const double &lat, const double &lon) : lat(lat), lon(lon) {}
//- ---------------------------------------------------------------------------------------
point_base::point_base(const int &ilat, const int &ilon) : lat(1.0E-6*double(ilat)), lon(1.0E-6*double(ilon)) {}

// ARC  //
void arc_base::set(int id_poly, double s0, point_base a, point_base b) {
  this->id_poly = id_poly;
  this->s0 = s0;
  this->a = a;
  this->b = b;
  dx = config_.dslon*(b.lon - a.lon);  
  dy = config_.dslat*(b.lat - a.lat);
  length = sqrt(dx*dx + dy * dy);
  if (length > 0.01) { dx /= length; dy /= length; }
  else {
    cout << " id_poly= " << id_poly << " cid=" << poly[id_poly].cid_poly << " arc length= " << length << " meters " << endl;
    cout << fixed << setprecision(6);
    cout << a.lat << "   " << a.lon << endl;
    cout << b.lat << "   " << b.lon << endl;
    exit(8);
  }
}
//----------------------------------------------------------------------------------------
double arc_base::measure_dist(double lon, double lat) {  //meters

  double dxa = config_.dslon*(lon - a.lon); 
  double dya = config_.dslat*(lat - a.lat);
  double ds = dxa * dx + dya * dy; 

  double dist;
  if (ds < 0) dist = sqrt(dxa*dxa + dya * dya);
  else if (ds > length) {
    double dxb = config_.dslon*(lon - b.lon); 
    double dyb = config_.dslat*(lat - b.lat);
    dist = sqrt(dxb*dxb + dyb * dyb);
  }
  else dist = fabs(dxa*dy - dya * dx);

  return dist;
}
//----------------------------------------------------------------------------------------
bool arc_base::measure_affinity(double lon, double lat, arcaffpro_base &aap) {
  // s = distance between start of poly and P, where P is the intersection between the lat lon and the poly.
  // d = distance between the lat lon and arc.
  // a = affinity
  aap.set(s0, config_.map_resolution, 0.0);  // (s,d,a)
  double lon_c = (a.lon + b.lon) / 2; 
  double lat_c = (a.lat + b.lat) / 2;
  double xc = config_.dslon*(lon - lon_c);  
  double yc = config_.dslat*(lat - lat_c);

  double x, y;  // coordinates
  y = abs(xc*dy - yc * dx);
  if (y > config_.map_resolution) return false;
  x = (xc*dx + yc * dy);  // sign can be neglected
  double l = 0.5*this->length; // half arc

  if (x < -l) { 
    aap.s = s0 + EPSILON;       
    aap.d = sqrt(y*y + (x + l)*(x + l)); 
  }
  else if (x > l) {
    aap.s = s0 + l + l - EPSILON; 
    aap.d = sqrt(y*y + (x - l)*(x - l));
  }
  else { 
    aap.s = s0 + l + x;    
    aap.d = y;
  }
  if (aap.d > config_.map_resolution) return false;

  x /= config_.l_gauss;
  y /= config_.l_gauss; 
  l /= config_.l_gauss;
  aap.a = exp(-y * y)*(erf(l - x) + erf(l + x)) / 2; // affinity of point on infinite lenght arc is 1
  return true;
}

// ARCAFFPRO //

arcaffpro_base::arcaffpro_base() {
  this->set(0.0, config_.map_resolution, 0.0);
}
//----------------------------------------------------------------------------------------
void arcaffpro_base::add(arcaffpro_base aap) {
  this->a += aap.a;
  if (this->d > aap.d) {
    this->d = aap.d; 
    this->s = aap.s;
  }
}
//----------------------------------------------------------------------------------------

// FL2D //
Fl2D::Fl2D() {
  this->FT = 0;
  this->TF = 0;
}
//----------------------------------------------------------------------------------------

// POLY //
poly_base::poly_base(void) {

  n_traj_TF = n_traj_FT = 0;
  clear();
}
//----------------------------------------------------------------------------------------
void poly_base::clear(void) {
  points.clear(); delta_points.clear();
  visible = false;
  name = "__";
}
//----------------------------------------------------------------------------------------
void poly_base::set(int id, unsigned long long int cid, vector <point_base> points) {
  this->id = id;
  this->cid_poly = cid;
  this->points = points;
}
//----------------------------------------------------------------------------------------
void poly_base::set(unsigned long long int cid_Fjnct, unsigned long long int cid_Tjnct, float meters, int oneway_,string name_) 
{
  this->cid_Fjnct = cid_Fjnct;
  this->cid_Tjnct = cid_Tjnct;
  this->length = meters;
  this->oneway = oneway_;
  this->name = name_;
}
//----------------------------------------------------------------------------------------
void poly_base::set(unsigned long long int cid_Fjnct, unsigned long long int cid_Tjnct) {
  this->cid_Fjnct = cid_Fjnct;
  this->cid_Tjnct = cid_Tjnct;
}
//----------------------------------------------------------------------------------------
void poly_base::measure_length(void) {
  double sum = 0.0;
  for (int k = 0; k < int(points.size() - 1); k++) {
    double dx = config_.dslon*(points[k + 1].lon - points[k].lon);
    double dy = config_.dslat*(points[k + 1].lat - points[k].lat);
    double ds = sqrt(dx*dx + dy * dy);
    sum += ds;
  }
  length = sum;
  weightFT = weightTF = length;
}
//----------------------------------------------------------------------------
void poly_base::reverse() {
  vector <point_base> pw;
  for (vector<point_base>::reverse_iterator rit = points.rbegin(); rit != points.rend(); ++rit) pw.push_back(*rit);
  points = pw;
}
//----------------------------------------------------------------------------
// POLYAFFPRO //
void polyaffpro_base::clear() {
  this->path_weight = 0;
  this->path.clear();
}
//----------------------------------------------------------------------------------------
bool comppap(const polyaffpro_base& a, const polyaffpro_base& b) { return (a.a > b.a); }


// NODE //
node_base::node_base(void) { id_node = 0; }
//--------------------------------------------------------------------
node_base::~node_base(void) { id_nnode.clear(); id_nlink.clear();}
//--------------------------------------------------------------------
void node_base::set(unsigned long long int cid_node, int id_node) {
  this->cid_node = cid_node;
  this->id_node = id_node;
}
//--------------------------------------------------------------------
void node_base::add_link(int id_nv, int id_lv) {
  id_nnode.push_back(id_nv);
  id_nlink.push_back(id_lv);
}
//--------------------------------------------------------------------
void node_base::remove_link(void) {
  id_nnode.pop_back();
  id_nlink.pop_back();
}
//--------------------------------------------------------------------
int node_base::get_n_link(void) { return int(id_nlink.size()); }
//----------------------------------------------------------------------------  
double node_base::measure_dist(double lon, double lat) {
  double dx = config_.dslon*(this->lon - lon);
  double dy = config_.dslat*(this->lat - lat);
  double ds2 = dx * dx + dy * dy;
  if (ds2 > 0) return sqrt(ds2);
  else          return 0.0;
}
//----------------------------------------------------------------------------  
bool comp_near_node(const node_near_base& a, const node_near_base& b) { return (a.distance < b.distance); }

// MAPPING //
void make_arc() {
  arc_base aw; 
  arc.clear(); 
  double s0;
  for (int i = 1; i<int(poly.size()); i++) {
    s0 = 0.0;
    for (int k = 0; k<int(poly[i].points.size()) - 1; k++) {
      aw.set(i, s0, poly[i].points[k], poly[i].points[k + 1]);
      arc.push_back(aw);
      s0 += aw.length;
    }
  }
  cout << "Arc:      " << arc.size() <<endl;
}
//----------------------------------------------------------------------------------------------------
int A_put_arc(int n) {

  int tw, n_arc_put = 0;
  int ia = int((arc[n].a.lon - config_.lon_min) / ds_lon);
  int ja = int((arc[n].a.lat - config_.lat_min) / ds_lat);
  int ib = int((arc[n].b.lon - config_.lon_min) / ds_lon);
  int jb = int((arc[n].b.lat - config_.lat_min) / ds_lat);
  if (ib < ia) { tw = ia; ia = ib; ib = tw; }        
  if (jb < ja) { tw = ja; ja = jb; jb = tw; }
  int i0 = (ia - 1 > 0 ? ia - 1 : 0);  int i1 = (ib + 2 < imax ? ib + 2 : imax);
  int j0 = (ja - 1 > 0 ? ja - 1 : 0);  int j1 = (jb + 2 < jmax ? jb + 2 : jmax);

  for (int j = j0; j < j1; j++) {
    double lat_c = config_.lat_min + (j + 0.5)*ds_lat;
    for (int i = i0; i < i1; i++) {
      double lon_c = config_.lon_min + (i + 0.5)*ds_lon;
      if (arc[n].measure_dist(lon_c, lat_c) < c_ris1) {
        A[j][i].arc_id.push_back(n); n_arc_put++;
      }
    }
  }
  return n_arc_put;
}
//------------------------------------------------------------------------------------------------------
int A_put_node(int n)
{
  double x, y; 
  int n_node_put = 0;
  x = node[n].lon;
  y = node[n].lat;

  int ia = int((x - config_.lon_min) / ds_lon);
  int ja = int((y - config_.lat_min) / ds_lat);
  int i0 = (ia - 1 > 0 ? ia - 1 : 0);  
  int i1 = (ia + 2 < imax ? ia + 2 : imax);
  int j0 = (ja - 1 > 0 ? ja - 1 : 0);  
  int j1 = (ja + 2 < jmax ? ja + 2 : jmax);

  for (int j = j0; j < j1; j++) {
    double lat_c = config_.lat_min + (j + 0.5)*ds_lat;
    double dya = config_.dslat*(y - lat_c);
    for (int i = i0; i < i1; i++) {
      double lon_c = config_.lon_min + (i + 0.5)*ds_lon;
      double dxa = config_.dslon*(x - lon_c);
      if (dxa*dxa + dya * dya < c_ris2) {
        A[j][i].node_id.push_back(n); n_node_put++;
      }
    }
  }
  return n_node_put;
}
//------------------------------------------------------------------------------------------------------

void make_mapping(void)
{
  static bool first = true;
  if (first) first = false;
  else { for (int j = 0; j < jmax; j++) delete[]A[j]; delete[]A; }
  make_arc();

  int n_arc_put = 0; 
  int n_node_put = 0;
  c_ris1 = 1.72*config_.map_resolution; 
  c_ris2 = c_ris1 * c_ris1;
  ds_lat = config_.map_resolution / config_.dslat;
  ds_lon = config_.map_resolution / config_.dslon;
  jmax = int(1 + (config_.lat_max - config_.lat_min) / ds_lat);
  imax = int(1 + (config_.lon_max - config_.lon_min) / ds_lon);
  cout << "Map:      [" << imax << "," << jmax << "]";
  cout << "  LX[km]= " << (config_.lat_max - config_.lat_min)*config_.dslat / 1000;
  cout << "  LY[km]= " << (config_.lon_max - config_.lon_min)*config_.dslon / 1000 << endl;

  A = new mapping_base*[jmax]; 
  for (int j = 0; j < jmax; j++) A[j] = new mapping_base[imax];  // A[jmax][imax]
  for (int n = 0; n< int(arc.size()); n++) n_arc_put += A_put_arc(n);
  for (int n = 1; n < node.size(); n++) n_node_put += A_put_node(n);
}
//------------------------------------------------------------------------------------------------------
int find_near_poly(double x, double y, double &dist, int &id_poly) {
  dist = 1.0e10; id_poly = 0;
  int i = int((x - config_.lon_min) / ds_lon);
  int j = int((y - config_.lat_min) / ds_lat);
  int n_near = int(A[j][i].arc_id.size());
  if (n_near  == 0) return n_near ;
  dist = arc[A[j][i].arc_id[0]].measure_dist(x, y); id_poly = arc[A[j][i].arc_id[0]].id_poly;
  for (int k = 1; k < n_near; k++) {
    double dd = arc[A[j][i].arc_id[k]].measure_dist(x, y);
    if (dd < dist) { dist = dd; id_poly = arc[A[j][i].arc_id[k]].id_poly; }
  }
  return n_near;
}
//------------------------------------------------------------------------------------------------------
bool find_near_node(double lon, double lat, double &dist, int &id_node) {
  dist = 1.0e8; id_node = 0;
  list <node_near_base> node_near;
  int i = int((lon - config_.lon_min) / ds_lon); int j = int((lat - config_.lat_min) / ds_lat);
  int n_near = int(A[j][i].node_id.size()); if (n_near == 0) return false;
  for (auto n : A[j][i].node_id) {
    double node_dist = node[n].measure_dist(lon, lat);
    if (dist > node_dist) { id_node = n; dist = node_dist; }
  }
  if (id_node == 0) 
    return false; 
  else 
    return true;
}
//----------------------------------------------------------------------------------------------------
bool find_polyaff(double lon, double lat, polyaffpro_base &pap) {
  // caso 0 = free; caso 1 = on_vapo; caso 2 out_vapo;
  pap.a = 0; 
  pap.d = 1.0e8; 
  pap.id_poly = 80000000; 
  pap.s = 0;
  list <polyaffpro_base> pap_list;
  int i = int((lon - config_.lon_min) / ds_lon); int j = int((lat - config_.lat_min) / ds_lat);
  int n_near = int(A[j][i].arc_id.size());
  if (n_near < 1) return false;

  map<int, arcaffpro_base> poly_aap; arcaffpro_base aapw;
  for (auto &n : A[j][i].arc_id) {
    if (arc[n].measure_affinity(lon, lat, aapw)) poly_aap[arc[n].id_poly].add(aapw);
  }
  if (poly_aap.size() < 1) return false;

  polyaffpro_base paw;
  for (auto &n : poly_aap) {
    paw.id_poly = n.first; 
    paw.a = n.second.a; 
    paw.d = n.second.d; 
    paw.s = n.second.s;
    pap_list.push_back(paw);
  }
  poly_aap.clear();
  pap_list.sort(comppap);

  pap = pap_list.front(); // default:  return first in the list sorted on affinity.

  return true;
}
//----------------------------------------------------------------------------------------------------

// POLYGON //
polygon_base::polygon_base() {}
//----------------------------------------------------------------------------
polygon_base::polygon_base(const jsoncons::json &polygonlist)
{
  for (const auto &pt : polygonlist.array_range())
  {
    points.emplace_back(pt[1].as<double>(), pt[0].as<double>());
  }
}
//----------------------------------------------------------------------------
double IsLeft(const double &lat_0, const double &lon_0, const double &lat_1, const double &lon_1, const double &lat_p, const double &lon_p)
{
  return ((lat_1 - lat_0) * (lon_p - lon_0) - (lat_p - lat_0) * (lon_1 - lon_0));
}
//----------------------------------------------------------------------------
int polygon_base::is_in_wn(double lat_p, double lon_p) {
  int wn = 0;    // the  winding number counter
  for (int i = 0; i < points.size() - 1; ++i) // edge from polygon[i] to  polygon[i+1]
  {
    if (points[i].lon <= lon_p)
    {
      if (points[i + 1].lon > lon_p)     // an upward crossing
        if (IsLeft(points[i].lat, points[i].lon, points[i + 1].lat, points[i + 1].lon, lat_p, lon_p) > 0)  // P left of  edge
          ++wn;  // have  a valid up intersect
    }
    else {                        // start y > P.y (no test needed)
      if (points[i + 1].lon <= lon_p)     // a downward crossing
        if (IsLeft(points[i].lat, points[i].lon, points[i + 1].lat, points[i + 1].lon, lat_p, lon_p) < 0)  // P right of  edge
          --wn; // have  a valid down intersect
    }
    //cout << "wn:  " << wn << endl;
  }

  return wn;
}

// METHODS //
void set_geometry()
{
  dlat = 0.5*(config_.lat_max - config_.lat_min) / ZOOM_START; 	lat0 = 0.5*(config_.lat_min + config_.lat_max);
  dlon = 0.5*(config_.lon_max - config_.lon_min) / ZOOM_START;   	lon0 = 0.5*(config_.lon_min + config_.lon_max);

  if (config_.dslat*dlat > config_.dslon*dlon) { dlon = (config_.dslat / config_.dslon)*dlat*double(screen_w) / screen_h; }
  else { dlat = (config_.dslon / config_.dslat)*dlon*double(screen_h) / screen_w; }

}
//----------------------------------------------------------------------------------------------------
void make_node()
{
  for (int i = 1; i < int(poly.size()); i++) {
    poly[i].id_local = i;
    node_cid2id[poly[i].cid_Fjnct] = 0;
    node_cid2id[poly[i].cid_Tjnct] = 0;
  }
  int cnt = 1; 
  for (auto &i : node_cid2id) 
    i.second = cnt++;
  
  node.resize(int(node_cid2id.size()) + 1);
  for (int i = 1; i < int(poly.size()); i++) {
    int id_nodeF = node_cid2id[poly[i].cid_Fjnct];
    int id_nodeT = node_cid2id[poly[i].cid_Tjnct];

    node[id_nodeF].add_link(id_nodeT, i);
    node[id_nodeT].add_link(id_nodeF, -i);
    node[id_nodeF].id_node = id_nodeF; node[id_nodeF].cid_node = poly[i].cid_Fjnct;
    node[id_nodeT].id_node = id_nodeT; node[id_nodeT].cid_node = poly[i].cid_Tjnct;
    node[id_nodeF].lat = poly[i].points.front().lat;
    node[id_nodeF].lon = poly[i].points.front().lon;
    node[id_nodeT].lat = poly[i].points.back().lat;
    node[id_nodeT].lon = poly[i].points.back().lon;
  }
  cout << "Node:     " << node.size() << endl;

  map <unsigned long long int, int> cid2id;
  for (int i = 1; i < node.size(); ++i) 	cid2id[node[i].cid_node] = i;
  for (int i = 1; i < poly.size(); i++) {
    poly[i].node_F = cid2id[poly[i].cid_Fjnct];
    poly[i].node_T = cid2id[poly[i].cid_Tjnct];
  }
  cid2id.clear();
}
//----------------------------------------------------------------------------------------------------
FILE *my_fopen(char *fname, char *mode)
{
  FILE *fp;

  if ((fp = fopen(fname, mode)) == NULL)
  {
    fprintf(stderr, "error - impossible open file %s in %s\n",
      fname, (mode[0] == 'r') ? "reading" : "writing");
    exit(1);
  }
  return fp;
}
//----------------------------------------------------------------------------------------------------------------

