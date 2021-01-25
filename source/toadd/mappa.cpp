#include "stdafx.h"
#include "polyline.h"
#include "nodo.h"
#include "dato.h"
#include "mappa.h"
#include "config.h"

extern vector <PolyLine> polyline;  extern int n_poly;
extern vector <Nodo> nodo;  extern int n_nodi;
extern vector <Trajectory> traj; extern int n_traj;
extern config config_;

vector <Segmento> segmento;


//------------------------------------------------------------------------------------------------------
int A_put_traj_start(Trajectory t)
{
	double x, y; int n_traj_put = 0;
	x = t.dato.front().lon; y = t.dato.front().lat;

	int ia = int((x - config_.lon_min) / ds_lon);
	int ja = int((y - config_.lat_min) / ds_lat);
	int i0 = (ia - 1 > 0 ? ia - 1 : 0);  int i1 = (ia + 2 < imax ? ia + 2 : imax);
	int j0 = (ja - 1 > 0 ? ja - 1 : 0);  int j1 = (ja + 2 < jmax ? ja + 2 : jmax);

	for (int j = j0; j<j1; j++) {
		double lat_c = config_.lat_min + (j + 0.5)*ds_lat;
		double dya = config_.dslat*(y - lat_c);
		for (int i = i0; i<i1; i++) {
			double lon_c = config_.lon_min + (i + 0.5)*ds_lon;
			double dxa = config_.dslon*(x - lon_c);
			if (dxa*dxa + dya*dya < c_ris2) {
				A[j][i].traj_start.push_back(t.id); n_traj_put++;
			}
		}
	}
	return n_traj_put;
}
//------------------------------------------------------------------------------------------------------
int A_put_traj_end(Trajectory t)
{
	double x, y; int n_traj_put = 0;
	x = t.dato.back().lon; y = t.dato.back().lat;

	int ia = int((x - config_.lon_min) / ds_lon);
	int ja = int((y - config_.lat_min) / ds_lat);
	int i0 = (ia - 1 > 0 ? ia - 1 : 0);  int i1 = (ia + 2 < imax ? ia + 2 : imax);
	int j0 = (ja - 1 > 0 ? ja - 1 : 0);  int j1 = (ja + 2 < jmax ? ja + 2 : jmax);

	for (int j = j0; j<j1; j++) {
		double lat_c = config_.lat_min + (j + 0.5)*ds_lat;
		double dya = config_.dslat*(y - lat_c);
		for (int i = i0; i<i1; i++) {
			double lon_c = config_.lon_min + (i + 0.5)*ds_lon;
			double dxa = config_.dslon*(x - lon_c);
			if (dxa*dxa + dya*dya < c_ris2) {
				A[j][i].traj_end.push_back(t.id); n_traj_put++;
			}
		}
	}
	return n_traj_put;
}

//----------------------------------------------------------------------------------------------------
void CreaMappaTraj(void)
{

	int n_traj_start = 0; int n_traj_end = 0;
	c_ris1 = 1.72*config_.map_resolution; c_ris2 = c_ris1*c_ris1;
	ds_lat = config_.map_resolution / config_.dslat;
	ds_lon = config_.map_resolution / config_.dslon;
	jmax = int(1 + (config_.lat_max - config_.lat_min) / ds_lat);
	imax = int(1 + (config_.lon_max - config_.lon_min) / ds_lon);
	cout << "CreaMappaTraj: " << endl;

	for (auto t : traj) { n_traj_start += A_put_traj_start(t); n_traj_end += A_put_traj_end(t); }

	cout << "CreaMappa: ho inserito " << n_traj_start << " traj_start e " << n_traj_end << " traj_end " << endl;

}
//----------------------------------------------------------------------------------------------------
int CercaTrajStartVicine(Dato dato, list <int> &vicini) {
  int tempo_raccolta = 3600;
	vicini.clear();
	int i = int((dato.lon - config_.lon_min) / ds_lon); int j = int((dato.lat - config_.lat_min) / ds_lat);
	int n_vicini = int(A[j][i].traj_start.size()); if (n_vicini == 0) return n_vicini;
	for (int k = 0; k < n_vicini; k++) {
		if (traj[A[j][i].traj_start[k]].dato.front().itime > dato.itime1 - 1 && traj[A[j][i].traj_start[k]].dato.front().itime < dato.itime1 + tempo_raccolta)
			vicini.push_back(A[j][i].traj_start[k]);
	}
	vicini.sort(); vicini.unique();
	return int(vicini.size());
}
//----------------------------------------------------------------------------------------------------
int CercaTrajEndVicine(double lon, double lat, list <int> &vicini) {
	vicini.clear();
	int i = int((lon - config_.lon_min) / ds_lon); int j = int((lat - config_.lat_min) / ds_lat);
	int n_vicini = int(A[j][i].traj_start.size()); if (n_vicini == 0) return n_vicini;
	for (int k = 0; k< n_vicini; k++) vicini.push_back(A[j][i].traj_end[k]);
	vicini.sort(); vicini.unique();
	return int(vicini.size());
}

