#include "stdafx.h"
#include <iostream>
#include <algorithm>
#include "carto.h"
#include "record.h"
#include "data_analysis.h"
#include "config.h"
#include <physycom/histo.hpp>
#include <boost/algorithm/string.hpp>
#include <physycom/time.hpp>
#include <Eigen/Dense>
#include "fcm.cpp"
#include <random>
#include <iterator>
#include <functional>
#include <set>
#include <featsel.hpp>
#include "global_params.h"
#include <unistd.h>
#include <type_traits>
#include <chrono>
#include <thread>
#include<map>
using namespace Eigen;
using namespace std;
extern config config_;
extern double ds_lat, ds_lon;
// ALBI
/*std::map<int,std::vector<int>> count_flux_per_poly(std::vector<int> poly_ids,std::vector<int> classes,std::vector<int> fcm_centers_id){
    std::vector<int> considered_polies;
    std::map<int,std::vector<int>> poly2class_count;
    for (auto fc:fcm_centers_id){
        std::cout <<"fcm center id: "<<fc<<std::endl;
    }

    for(int i=0;i<poly_ids.size();i++){
        poly2class_count[poly_ids[i]]=std::vector<int>(4,0);
    }
    for(int i=0;i<poly_ids.size();i++){

        for (int fc = 0; fc<fcm_centers_id.size();fc++){
            if(classes[i]==fcm_centers_id[fc]){
                if (poly_ids[i]<0){
                    std::cout<<"poly negative id: "<<poly_ids[i]<<std::endl;
                    poly2class_count[-poly_ids[i]][fc]++;}
                else{poly2class_count[poly_ids[i]][fc]++;}
                }
            }
        }
    return poly2class_count;
    }
*/

void dump_coords_traj(config &config_, vector<traj_base> &traj)
{
    std::cout << " dump coords traj" << std::endl;
    long long int ids[] = {692410, 846737, 504205, 173098401, 261751914, 265019034, 165465954};
    ofstream out_coords_traj(config_.cartout_basename + config_.name_pro + "_coordinates_possible_errors.csv");
    for (auto i : ids)
    {
        std::cout << "identification " << to_string(i) << std::endl;
        out_coords_traj << to_string(i) << ";";
        for (auto &t : traj)
        {
            if (t.id_act == i)
            {
                for (auto &sp : t.stop_point)
                {
                    out_coords_traj << to_string(sp.centroid.lat) << ";" << to_string(sp.centroid.lon) << ";" << to_string(sp.points.front().itime) << ";";
                }
                out_coords_traj << std::endl;
            }
        }
    }
    out_coords_traj.close();
}

// ALBI

//----------------------------------------------------------------------------------------------------
bool comp_rec_itime(const record_base &a, const record_base &b) { return (a.itime < b.itime); }
//----------------------------------------------------------------------------------------------------
bool comp_traj_lenght(const traj_base &a, const traj_base &b) { return (a.length > b.length); }
//----------------------------------------------------------------------------------------------------
void sort_activity(std::vector<activity_base> &activity)
{

    // sort data in each activity
    for (auto &a : activity)
        sort(a.record.begin(), a.record.end(), comp_rec_itime);

    // add index for activity and records
    int j = 0, k;
    for (auto &a : activity)
    {
        a.indx = j++; // aggiungo l'ordine tra le diverse attività
        k = 0;
        for (auto &r : a.record)
            r.indx = k++; // aggiungo l'ordine nei record per ogni attività per poter poi accedere direttamente a record consecutivi per indice facendo loop e incrementando una variabile
    }

    // add state for each record
    for (auto &a : activity)
    {
        if (a.record.size() == 1)
            a.record[0].state = 3; // se ho che l'attività è composta da un record allora la classifico come 3
        else
        { // in mezzo lo state è 1
            for (auto &r : a.record)
                r.state = 1;
            a.record.front().state = 0; // il front ha state 0
            a.record.back().state = 2;  // il tail ha state 3
        }
    }
    // measure lenght and average speed
    for (auto &a : activity)
    {
        a.length = 0.0;
        for (auto r = 0; r != a.record.size() - 1; r++)
        {
            a.length += distance_record(a.record[r], a.record[r + 1]);
        }
        a.dt = int(a.record.back().itime - a.record.front().itime);
        a.average_speed = a.length / double(a.dt);
    }

    // measure speed and accel
    for (auto &a : activity)
    {
        for (int i = 1; i < a.record.size(); ++i)
        {
            a.record[i].speed = distance_record(a.record[i], a.record[i - 1]) / (a.record[i].itime - a.record[i - 1].itime);
            if (i >= 2)
                a.record[i].accel = (a.record[i].speed - a.record[i - 1].speed) / (a.record[i].itime - a.record[i - 1].itime);
        }
    }

    // temporarily added for analysis
    // if (1==1) {
    //   ofstream out_dur(config_.cartout_basename + config_.city_tag + "_duration_pre.csv");
    //   ofstream out_deltat(config_.cartout_basename + config_.city_tag + "_deltat_s_pre.csv");
    //   out_dur << "id_act;duration;n_rec" << std::endl;
    //   out_deltat << "id_act;deltat;deltas" << std::endl;
    //   for (auto &a : activity) {
    //     out_dur << a.id_act << ";" << a.dt << ";" << a.record.size() << std::endl;
    //     for (int s = 1; s < a.record.size(); ++s)
    //       out_deltat << a.id_act << ";" << a.record[s].itime - a.record[s - 1].itime <<";"<< distance_record(a.record[s-1], a.record[s]) << std::endl;
    //   }
    // }

    if (0 == 1)
    {
        ofstream out_activity(config_.cartout_basename + config_.name_pro + "_activity_startstop.csv");
        out_activity << "id;timestart;timestop" << std::endl;
        for (auto &a : activity)
            out_activity << a.id_act << ";" << a.record.front().itime << ";" << a.record.back().itime << std::endl;
        out_activity.close();
    }
}
//----------------------------------------------------------------------------------------------------
void bin_activity(std::vector<activity_base> &activity)
{
    double delta_s = config_.bin_time * 60;
    map<int, int> bintime_data;
    for (auto &a : activity)
    {
        int start_index = int((a.record.front().itime - config_.start_time) / delta_s);
        int end_index = int((a.record.back().itime - config_.start_time) / delta_s);
        for (int i = start_index; i <= end_index; ++i)
            bintime_data[i]++;
    }
    ofstream out_bin(config_.cartout_basename + config_.name_pro + ".csv");
    out_bin << "Time;Occurence" << endl;
    for (auto &m : bintime_data)
        out_bin << m.first * config_.bin_time << ";" << m.second << endl;
    out_bin.close();
    cout << "Time binning of data: done." << endl;
}
//----------------------------------------------------------------------------------------------------
void make_traj(std::vector<activity_base> &activity,data_loss &dataloss,std::vector<traj_base> &traj,std::vector<cluster_base> &data_notoncarto,std::vector<presence_base> &presence)
{/*activity: emptied
traj: filled
dataloss: filled
data_notcarto:filled
presence: filled*/
    vector<traj_base> traj_temp;
    traj_base tw;
    std::cout << "position tw: " << &tw << std::endl;   
    std::cout << "position traj_temp: " << &traj_temp << std::endl;
    std::cout << "position activity: " << &activity << std::endl;
    std::cout << "position config_: " << &config_ << std::endl;
    for (auto &a : activity)
    {
        if(a.record.size()<MAX_RECORD)
        {        
            tw.record.clear();
            tw.id_act = a.id_act;
            tw.record.reserve(MAX_RECORD);
            tw.record = a.record;
            tw.row_n_rec = int(tw.record.size()); // in the trajectory I have the count of record to show how many of them I 'loose' when I put them together
            traj_temp.push_back(tw);
        }
        else{std::cout << "I do not consider the activity "<< a.id_act<<std::endl;}
    }

    activity.clear();
    activity.shrink_to_fit(); // clean memory, the info are passed to traj and presence
    // filter data on distance and inst_speed
    int cnt_tot_data = 0;
    int cnt_tot_sp = 0;
    for (auto &t : traj_temp)
    {                                         // per ogni traiettoria mi  dichiaro un cluster_base
        cnt_tot_data += int(t.record.size()); // conto il numero totale di dati in entrata dalle activity
        cluster_base C;
        C.add_point(t.record.front()); // aggiungo il primo elemento dei record,inizializzo il centroide(lat,lon, no time,no indx)
        for (int n = 1; n < t.record.size(); ++n)
        { // per gli altri record calcolo la distanza dal  centroide che è di default quel record oppure quello successivo distante almeno min_data_distance+0.01
            if (distance_record(C.centroid, t.record[n]) < config_.min_data_distance)
            {
                C.add_point(t.record[n]); // aggiungo all'oggeto cluster il punto che è a distanza minore del threshold
            }
            else if (distance_record(C.centroid, t.record[n]) >= config_.min_data_distance)
            {
                t.add_cluster(C, n); // Qua sto creando il cluster con il punto e aggiungendolo allla traiettoria.Inizializzando il cluster con centroide,lat e lon
            }                        // sto agendo sullo stesso spazio di memoria delle traiettorie esistenti. Se nella traiettoria che sto utilizzando ho che stop_point is not yet initialized then I add the stop_point variable, that is a cluster_base type. It means that
                                     //
        }
        double last_speed = C.points.front().speed;
        if (last_speed < config_.max_inst_speed && !C.visited) // il visited del cluster è cambiato quando faccio add_cluster? Quindi in questo ciclo non considero i punti che hanno distanza maggiore dal punto iniziale? Vuol dire che i trattori sono lenti.
            // il punto di stop c'è quando lo stato di cluster visited = False e la velocità del cluster appena creato con i record dell'attività (distanti almeno config_.min_data_distance=49 m)
            t.stop_point.push_back(C);
        // ci metto i punti che hanno percorso meno spazio della distanza massima nell'intervallo di tempo tra un record e l'altro: da chiarire come si calcola la velocità istantanea
        cnt_tot_sp += int(t.stop_point.size());
    }
    std::cout << "Record before filters :  " << cnt_tot_data << std::endl;
    std::cout << "Record after filters :  " << cnt_tot_sp << std::endl;
    std::cout << "Distance filter       :  " << double(cnt_tot_sp) / cnt_tot_data * 100. << "%" << std::endl;
    dataloss.n_data_tot = cnt_tot_data;
    dataloss.n_data_meter = cnt_tot_sp;

    // ofstream out_spot(config_.cartout_basename + config_.name_pro + "_filtered_data.csv");
    // out_spot << "id_act;time;lat;lon" << std::endl;
    // for (auto &t : traj_temp)
    //   for (auto &sp : t.stop_point)
    //     out_spot << t.id_act << ";" << sp.points.front().itime << ";" << sp.points.front().lat << ";" << sp.points.front().lon << std::endl;

    // ofstream out_nogeoref(config_.cartout_basename + config_.name_pro + "_nogeoref.csv");
    // out_nogeoref << "id_act;lat;lon;time" << std::endl;

    // filter stops on carto geolocalization
    for (auto &t : traj_temp)
    {
        vector<cluster_base> sp_oncarto;
        for (auto &sp : t.stop_point)
        {
            sp.on_carto = find_polyaff(sp.centroid.lon, sp.centroid.lat, sp.pap);
            if (sp.on_carto && sp.pap.d > config_.min_poly_distance)
                sp.on_carto = false;
            if (!sp.on_carto)
                data_notoncarto.push_back(sp);
            if (sp.on_carto)
                sp_oncarto.push_back(sp);
            // temporarly added for duration analysis
            // if (sp.on_carto == false) out_nogeoref << t.id_act << ";" << sp.points.front().lat<<";"<<sp.points.front().lon<<";"<<sp.points.front().itime << std::endl;
        }
        t.stop_point = sp_oncarto; // filter stop points and consider just tthose that are pn cartp
    }
    dataloss.n_data_outcarto = int(data_notoncarto.size());
    std::cout << "Georeferencing filter out carto: " << double(dataloss.n_data_outcarto) / cnt_tot_sp * 100. << "%" << std::endl;

    // temporarly added
    // ofstream out_nothresh(config_.cartout_basename + config_.name_pro + "_nothresh.csv");
    // out_nothresh << "lenght;time;av_speed" << std::endl;
    //

    // split data in 2 classes: traj (more than 1 point) and presence (1 point)
    for (auto &t : traj_temp)
    {
        t.record.clear();
        t.record.shrink_to_fit(); // clean memory!
        if (t.stop_point.size() == 1)
        {
            // transform in presence and then push
            dataloss.n_data_oncarto++;
            presence_base pr(t.stop_point.front().centroid.lat, t.stop_point.front().centroid.lon, t.id_act, t.stop_point.front().points.front().itime, t.stop_point.front().points.back().itime, t.row_n_rec);
            presence.push_back(pr);
            dataloss.n_data_single_record++;
        }
        else if (t.stop_point.size() > 1)
        {
            dataloss.n_traj_tot++;
            dataloss.n_data_no_single_record += int(t.stop_point.size());
            dataloss.n_data_oncarto += int(t.stop_point.size());
            t.time = int(t.stop_point.back().points.back().itime - t.stop_point.front().points.front().itime); // tempo totale di un user a cui è associato un certo id_act
            // t.length = distance_record(t.stop_point.back().points.back(), t.stop_point.front().points.front());
            t.length = 0;
            for (int sp = 0; sp < t.stop_point.size() - 1; ++sp)
            {
                // t.stop_point[sp].heading = measure_heading(t.stop_point[sp + 1], t.stop_point[sp]);
                t.length += distance_record(t.stop_point[sp].points.front(), t.stop_point[sp + 1].points.front());
            }
            // t.stop_point.back().heading = t.stop_point[t.stop_point.size() - 2].heading;
            if (config_.enable_threshold)
            {
                if (t.stop_point.size() > config_.threshold_n && t.time <= config_.threshold_t && (t.length / t.time) < config_.threshold_v)
                {
                    traj.push_back(t);
                    dataloss.n_data_threshold += int(t.stop_point.size());
                }
                // else
                // out_nothresh << t.length << ";" << t.time << ";" << t.length / t.time << std::endl;
            }
            else
                traj.push_back(t);
        }
    }
    traj_temp.clear();
    traj_temp.shrink_to_fit();

    std::cout << "Georeferencing filter on carto : " << double(dataloss.n_data_oncarto) / cnt_tot_sp * 100. << "%" << std::endl;
    std::cout << "Activity with single record    : " << double(dataloss.n_data_single_record) / cnt_tot_sp * 100. << "%" << std::endl;
    std::cout << "Data of Activity with more than 1 record: " << double(dataloss.n_data_no_single_record) / cnt_tot_sp * 100. << "%" << std::endl;
    std::cout << "Threshold filter:      " << double(dataloss.n_data_threshold) / cnt_tot_sp * 100. << "%" << std::endl;

    std::cout << "Num Presence:               " << presence.size() << std::endl;
    std::cout << "Num Traj:                   " << traj.size() << std::endl;

    // print presence
    ofstream out_pres(config_.cartout_basename + config_.name_pro + "_presence.csv");
    out_pres << "id_act;timestart;timeend;lat;lon" << std::endl;
    for (auto pp : presence)
        out_pres << pp.id_act << ";" << pp.itime_start << ";" << pp.itime_end << ";" << pp.lat << ";" << pp.lon << std::endl;

    // check
    for (auto &t : traj)
    {
        for (int n = 0; n < t.stop_point.size() - 1; ++n)
        {
            double dist = distance_record(t.stop_point[n].centroid, t.stop_point[n + 1].centroid);
            if (dist < config_.min_data_distance)
            {
                std::cout << "Distance Error, improve the algos!" << std::endl;
                std::cout << "id: " << t.id_act << " dist " << dist << " nrec " << t.record.size() << " nstops:" << t.stop_point.size() << std::endl;
                // std::cin.get();
            }
        }
    }
    for (auto &t : traj)
    {
        t.time = int(t.stop_point.back().points.front().itime - t.stop_point.front().points.front().itime);
        t.length = 0.0;
        int time_ = 0;
        for (int sp = 1; sp < t.stop_point.size(); ++sp)
        {
            double dist_pp = distance_record(t.stop_point[sp - 1].points.front(), t.stop_point[sp].points.front());
            t.length += dist_pp;
            time_ +=t.stop_point[sp].points.front().itime - t.stop_point[sp - 1].points.front().itime;
            t.stop_point[sp].inst_speed = distance_record(t.stop_point[sp - 1].points.front(), t.stop_point[sp].points.front()) / (t.stop_point[sp].points.front().itime - t.stop_point[sp - 1].points.front().itime);
            if (sp > 1)
                t.stop_point[sp].inst_accel = (t.stop_point[sp].points.front().speed - t.stop_point[sp - 1].points.front().speed) / (t.stop_point[sp].points.front().itime - t.stop_point[sp - 1].points.front().itime);
        }
        t.average_speed = t.length / t.time;
        int time2 = int(t.stop_point.back().points.back().itime - t.stop_point[1].points.front().itime);
        t.average_accel = (t.stop_point.back().inst_speed - t.stop_point[1].inst_speed) / time2;
    }
    for (auto &t : traj)
    {
        t.v_max = 0.0;
        t.v_min = 1000.0;
        t.a_max = 0.0;
        t.a_min = 1000.0;
        t.sigma_speed = 0.0;
        t.average_inst_speed = 0.0;
        int window_size = 2;
        for (int sp = window_size; sp < t.stop_point.size(); ++sp)
        {
            double speed = 0.0;
            for (int n = 0; n < window_size; ++n){
                speed += t.stop_point[sp - n].inst_speed;
                }
            speed /= window_size;
            if (speed >= t.v_max)
                t.v_max = speed;
            if (speed < t.v_min)
                t.v_min = speed;
            t.sigma_speed += pow((speed - t.average_speed), 2);
            t.average_inst_speed += speed;

            if (sp >= window_size + 1)
            {
                double accel = 0.0;
                for (int n = 0; n < window_size; ++n)
                    accel += t.stop_point[sp - n].inst_accel;
                accel /= window_size;

                if (accel >= t.a_max)
                    t.a_max = accel;
                if (accel < t.a_min)
                    t.a_min = accel;
                t.sigma_accel += pow((accel - t.average_accel), 2);
            }
        }

        t.sigma_speed /= (t.stop_point.size() - window_size);
        t.average_inst_speed /= (t.stop_point.size() - window_size);
        t.sigma_speed = sqrt(t.sigma_speed);

        t.sigma_accel /= (t.stop_point.size() - window_size);
        t.sigma_accel = sqrt(t.sigma_accel);
    }
    if (config_.enable_print)
    {
    ofstream out_stats(config_.cartout_basename + config_.name_pro + "_stats.csv");
    out_stats << "id_act;lenght;time;av_speed;ndat;front_lat;front_lon;tail_lat;tail_lon;start_time;end_time" << std::endl;
    for (auto &t : traj)
        {if(t.average_inst_speed<t.v_max && t.average_inst_speed>t.v_min)            
            out_stats << t.id_act << ";" << t.length << ";" << t.time << ";" << t.average_inst_speed << ";" << t.stop_point.size() << ";" << t.stop_point.front().centroid.lat << ";" << t.stop_point.front().centroid.lon << ";" << t.stop_point.back().centroid.lat << ";" << t.stop_point.back().centroid.lon << ";" << t.stop_point.front().points.front().itime << ";" << t.stop_point.back().points.back().itime << std::endl;
        else std::cout<<"average speed out of bound: "<< t.id_act << " vmin " << t.v_min << " vmax " <<t.v_max << " av_inst_speed "<< t.average_inst_speed<<std::endl;
        }
        out_stats.close();
        }
    dump_coords_traj(config_, traj);
}
//-------------------------------------------------------------------------------------------------
// SEED //
void seed_pro_base::set(int id_node, int node_bv, int link_bv, double distance)
{
    this->distance = distance;
    this->id_node = id_node;
    this->node_bv = node_bv;
    this->link_bv = link_bv;
}
//-------------------------------------------------------------------------------------------------
bool best_poly(cluster_base &d1, cluster_base &d2,std::vector<poly_base> &poly,std::vector<node_base> &node)
{
    polyaffpro_base t1 = d1.pap; //  int id_poly;double a, d, s;list <pair<int, double>> path; // pair< id_poly, time_in >
                                 //  double path_weight;  void clear(); are the attributes and functions of this type.
    polyaffpro_base t2 = d2.pap;

    int ipoly1, ipoly2;
    double s1, s2, p1l, p2l;
    ipoly1 = t1.id_poly;
    ipoly2 = t2.id_poly;

    p1l = poly[ipoly1].weightTF; // mi prendo la poli tail front
    p2l = poly[ipoly2].weightTF;
    s1 = t1.s; // distance between start of the poly and intersection.
    s2 = t2.s;

    list<pair<int, double>> poly_crossed;

    if (ipoly1 == ipoly2)
    {
        if (d1.pap.path.size() > 0)
        {
            if (poly_crossed.size() > 0)
            {
                if (poly_crossed.front() == d1.pap.path.back())
                    poly_crossed.pop_front();
            }
            // else {
            //   bool oneway;
            //   if (s2 >= s1)
            //     oneway = true;
            //   else
            //     oneway = false;
            //   if ((d1.pap.path.back().first > 0 && !oneway) | (d1.pap.path.back().first < 0 && oneway)) {
            //     int poly_sign = -d1.pap.path.back().first;
            //     double time_val = d1.pap.path.back().second + d2.points.front().t;
            //     d1.pap.path.push_back(make_pair(poly_sign, time_val));
            //   }
            // }
        }
        else if (d1.pap.path.size() == 0)
        {
            int poly_way;
            double time = d1.points.front().t;
            if (s2 >= s1)
                poly_way = ipoly1;
            else
                poly_way = -ipoly1;
            d1.pap.path.push_back(std::make_pair(poly_way, time));
        }

        d2.pap.path = d1.pap.path;
        d2.pap.path.splice(d2.pap.path.end(), poly_crossed);
        d2.pap.path_weight = d1.pap.path_weight + abs(s2 - s1);
        d1.pap.clear();
        return true; // i'm moving on the same poly. I must not return it.
    }

    static int *index;
    seed_base node_v;         // node_v: visited node
    seed_pro_base node_v_pro; // nodo_v_pro: properties of visited node
    vector<seed_pro_base> list_node_pro;
    vector<int> visited;
    int nw, inw, nw_near, i_poly, iv, iter = 0;
    double x1, y1, x2, y2, dx, dy, dist, d_eu, a_eu = 0.0;
    double distance;
    static bool first_time = true;
    // index 0 initialization
    if (first_time)
    {
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
    dx = config_.dslon * (x1 - x2);
    dy = config_.dslat * (y1 - y2);
    d_eu = a_eu * sqrt(dx * dx + dy * dy);
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
    dx = config_.dslon * (x1 - x2);
    dy = config_.dslat * (y1 - y2);
    d_eu = a_eu * sqrt(dx * dx + dy * dy);
    iv++;
    node_v_pro.set(n1T, 0, ipoly1, p1l - s1);
    list_node_pro.push_back(node_v_pro);
    node_v.cnt = iv;
    node_v.dd = p1l - s1 + d_eu;
    visited.push_back(n1T);
    index[n1T] = node_v.cnt;
    heap.push(node_v);

    while (!heap.empty())
    {
        node_v = heap.top();
        heap.pop();
        iter++;
        nw = list_node_pro[node_v.cnt].id_node;
        distance = list_node_pro[node_v.cnt].distance; // distance from n1 along the path

        inw = index[nw];
        if (inw > 0 && list_node_pro[inw].distance < distance)
            continue;

        index[nw] = node_v.cnt;

        if (nw == n2F)
            goal_F = true;
        if (nw == n2T)
            goal_T = true;
        if (goal_F && goal_T)
            break; // goal reached

        for (int n = 0; n < node[nw].id_nnode.size(); n++)
        {
            nw_near = node[nw].id_nnode[n];
            i_poly = node[nw].id_nlink[n];
            if (i_poly > 0)
                dist = distance + poly[abs(i_poly)].weightFT;
            else
                dist = distance + poly[abs(i_poly)].weightTF;
            inw = index[nw_near];
            if (inw > 0 && (list_node_pro[inw].distance < dist))
                continue;

            // push nw_near
            x1 = node[nw_near].lon;
            y1 = node[nw_near].lat;
            dx = config_.dslon * (x1 - x2);
            dy = config_.dslat * (y1 - y2);
            d_eu = a_eu * sqrt(dx * dx + dy * dy);
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
    double delta_t = d2.points.front().t - d1.points.front().t; // tempo che ci vuole ad andare dal punto di inizio al punto di fine
    double dist_2F = list_node_pro[index[n2F]].distance;
    double dist_2T = list_node_pro[index[n2T]].distance;
    int i_poly_2F = list_node_pro[index[n2F]].link_bv;
    int i_poly_2T = list_node_pro[index[n2T]].link_bv;

    if (i_poly_2F == ipoly2)
        std::cout << " error:  i_poly_2F ==  ipoly2 " << endl;
    if (i_poly_2T == -ipoly2)
        std::cout << " error:  i_poly_2T == -ipoly2 " << endl;

    dist_2F += s2;
    dist_2T += p2l - s2;

    pair<int, double> pw;
    if (dist_2F < dist_2T)
    {
        n = n2F;
        distance = dist_2F;
        pw.first = ipoly2;
        pw.second = d1.points.front().t + delta_t * (list_node_pro[index[n2F]].distance) / distance;
    }
    else
    {
        n = n2T;
        distance = dist_2T;
        pw.first = -ipoly2;
        pw.second = d1.points.front().t + delta_t * (list_node_pro[index[n2T]].distance) / distance;
    }
    poly_crossed.push_front(pw);
    int i_poly_first = list_node_pro[index[n]].link_bv;
    n = list_node_pro[index[n]].node_bv;
    i_poly = list_node_pro[index[n]].link_bv;
    double ss0 = 0;
    if (i_poly == 0 && i_poly_first != 0)
    {
        pw.first = i_poly_first;
        if (i_poly_first > 0)
            pw.second = d1.points.front().t; // manca quel piccolo passo sulla poly nel t
        else
            pw.second = d1.points.front().t; // manca quel piccolo passo sulla poly nel t
        poly_crossed.push_front(pw);
    }
    while (i_poly != 0)
    {
        pw.first = i_poly;
        if (distance >= 1.e-4)
            pw.second = d1.points.front().t + delta_t * (list_node_pro[index[n]].distance / distance); // il senso di questo distance? dovrebbe essere la delta L
        else
            pw.second = d1.points.front().t;
        poly_crossed.push_front(pw);
        ss0 = list_node_pro[index[n]].distance;
        n = list_node_pro[index[n]].node_bv;
        i_poly = list_node_pro[index[n]].link_bv;
    }

    poly_crossed.front().second = d1.points.front().t;

    double ss1 = 0;

    for (int i = 0; i < visited.size(); i++)
        index[visited[i]] = 0;
    visited.clear();
    list_node_pro.clear();

    if (poly_crossed.front().first == d1.pap.path.back().first)
        poly_crossed.pop_front();
    d2.pap.path = d1.pap.path;
    d2.pap.path.splice(d2.pap.path.end(), poly_crossed);
    d2.pap.path_weight = d1.pap.path_weight + distance;
    d1.pap.clear();

    return true;
}

void sleep()
{
    std::cout << "sleep" << std::endl;
    std::cout << std::flush << std::endl;
    auto now = std::chrono::system_clock::now();

    // Add a duration of 5 seconds to the current time point
    auto wakeupTime = now + std::chrono::seconds(10);

    std::this_thread::sleep_until(wakeupTime);
    std::cout << "awaken" << std::endl;
}

std::vector<int> unique_vector(std::vector<int> classes){
    std::vector<int> unique;
    for(auto &c:classes){
        if (std::find(unique.begin(),unique.end(),c)!=unique.end()) continue;
        else unique.push_back(c);
    }
    std::sort(unique.begin(),unique.end());
    for(auto &u:unique){
        std::cout<<"unique: "<<u<<std::endl;
    }
    return unique;
}

void write_mil_file_poly2class(std::vector<poly_base> &poly,std::vector<traj_base> &traj,config &config_){
//   Description: write the file poly_with_fcm.csv
//   Input: poly and traj
//   Output: poly_with_fcm.csv
//   Note: the file is used to make the .FLUXES file

    ofstream out_poly2fcm(config_.cartout_basename+config_.name_pro + "_poly_with_fcm.csv");

    std::cout << "Debug" << std::endl;
    std::cout << config_.cartout_basename+config_.name_pro + "_poly_with_fcm.csv" << std::endl;

    out_poly2fcm << "av_speed;class;poly_id;nodeF;nodeT" << std::endl;
    std::cout <<"writing POLY_WITH_FCM.csv"<<std::endl;
    if(out_poly2fcm.is_open()){ 
        for (auto &t : traj){   
            int stop_point_it = 0;
            for (auto &j:t.path){
                if (j.first<0){
                    int index_ = -j.first;
                    out_poly2fcm << t.average_inst_speed << ";" << t.means_class << ";" << index_ << ";" <<poly[index_].cid_Fjnct << ";" <<poly[index_].cid_Tjnct << std::endl;//<<t.stop_point[stop_point_it].points.front().itime<<std::endl;
                    stop_point_it++;
                    }
                else
                    out_poly2fcm << t.average_inst_speed << ";" << t.means_class <<";"<< j.first<< ";" <<poly[j.first].cid_Fjnct <<";"<< poly[j.first].cid_Tjnct<<std::endl;// <<t.stop_point[stop_point_it].points.front().itime<<std::endl;
                    stop_point_it++;
            }
        }
        out_poly2fcm.flush();
        out_poly2fcm.close();
    }
    else throw std::runtime_error("Could not open poly_with_fcm file");
}

void dump_fluxes_file(std::vector<poly_base> &poly,config &config_,std::vector<std::vector<int>> &polyclass){
//  Description: create the file .fluxes -> 1) read mil_file: with poly,class,nodeF-T,velocity for each point of all trajectories
//  Input: poly and centers_fcm
    std::cout << "dumping .fluxes file" << std::endl;
    // DECLARE USED VARIABLES
    std::vector<int> classes;
    std::vector<int> poly_ids;
    std::vector<unsigned long long int> nodeFs;
    std::vector<unsigned long long int> nodeTs;
    std::vector<double> velocities;
    std::vector<int> fcm_centers_id;
//    std::vector<int> unique_poly_ids;
//    std::vector<unsigned long long int> unique_nodeFs;
//    std::vector<unsigned long long int> unique_nodeTs;
    // READ FROM FILE
    std::fstream read_fcm(config_.cartout_basename +config_.name_pro + "_poly_with_fcm.csv");
    if(!read_fcm.is_open()) throw std::runtime_error("Could not open fcm file");
    if (read_fcm.good()){
        std::string line;
        std::getline(read_fcm,line); // get columns name
        std::string token;
        while (std::getline(read_fcm,line))
        {   std::vector<std::string> tokens;
            physycom::split(tokens, line, string(";"), physycom::token_compress_on);                      
            velocities.push_back(std::stod(tokens[0]));
            classes.push_back(std::stoi(tokens[1]));
            poly_ids.push_back(std::stoi(tokens[2]));
            nodeFs.push_back(std::stoull(tokens[3]));
            nodeTs.push_back(std::stoull(tokens[4]));
        }
    }
//    std::cout << "velocities: " << velocities.size() << " type: " << typeid(velocities[velocities.size()-1]).name() << " classes: "<< classes.size() << " type: " << typeid(classes[0]).name() << " poly_ids: " << poly_ids.size()-1 << " type: " << typeid(poly_ids[poly_ids.size()-1]).name() << "nodeFs: " << nodeFs.size() << " type: " << typeid(nodeFs[nodeFs.size()-1]).name() << "nodeTs: " << nodeTs.size() << " type: " << typeid(nodeTs[nodeTs.size()-1]).name() << std::endl;
    read_fcm.close();
    fcm_centers_id = unique_vector(classes);
    std::sort(fcm_centers_id.begin(),fcm_centers_id.end());
    // CREATE UNIQUE NODES and POLIES
//    for (int i= 0;i< poly_ids.size();++i){
//        if (std::find(unique_poly_ids.begin(), unique_poly_ids.end(), poly_ids[i]) != unique_poly_ids.end()) {
//            unique_poly_ids.push_back(poly_ids[i]);
//            unique_nodeFs.push_back(nodeFs[i]);
//            unique_nodeTs.push_back(nodeTs[i]);
//        }
//        else continue;
//    }
//    std::cout<< "size unique: " << unique_poly_ids.size() <<" "<< unique_nodeFs.size()<<" "<<fcm_centers_id.size() <<std::endl;
    // COUNT FLUX PER POLY
    for(int i=0;i<poly_ids.size()-1;++i){
        for (int fc = 0; fc<fcm_centers_id.size();fc++){
            if(classes[i]==fcm_centers_id[fc] && (fcm_centers_id[fc]!=10 && fcm_centers_id[fc]!=11)){
                if (poly_ids[i]<0){
                    std::cout<<"poly negative id: "<<poly_ids[i]<<std::endl;
                    polyclass[-poly_ids[i]][fcm_centers_id[fc]]++;}
                else{polyclass[poly_ids[i]][fcm_centers_id[fc]]++;
                    if(poly_ids[i]>poly.size()) std::cout << "identification poly_ids bigger then poly size" <<std::endl; }
                }
            }
        }
    // count flux per poly
    //count_flux_per_poly(poly_ids,classes,fcm_centers_id,poly2class_count);
    std::cout << config_.cartout_basename + "/weights/" + config_.name_pro + ".fluxes" << std::endl;
    std::ofstream out(config_.cartout_basename + "/weights/" + config_.name_pro + ".fluxes");
    if (!out.is_open())
        throw std::runtime_error("Could not open fluxes file");
    else{    
        std::cout <<polyclass[polyclass.size()-1][0] << " "<<poly_ids[poly_ids.size()]<< std::endl;
        out << "id;id_local;nodeF;nodeT;lenght;total_fluxes";
        for (auto &c : fcm_centers_id){
            std::cout <<"index fcm: "<<c << std::endl;
            if (c!=10 && c!=11) 
            {out << ";"<< "class_" + to_string(c);}
        }        
        out << std::endl;        
        for (int i=0;i<poly.size();++i)
            {   
            int total_fluxes=0;
            for(int fc = 0; fc<polyclass[i].size();fc++){
                total_fluxes+=polyclass[i][fc];
    //            for(int c=0;c<poly2class_count[p].size();c++){
    //                total_fluxes+=poly2class_count[p][c];
            }
            out << i << ";" << poly[i].id_local << ";" << poly[i].cid_Fjnct << ";" << poly[i].cid_Tjnct << ";" << poly[i].length << ";" << total_fluxes;
    //            ofstream class10file(config_.cartout_basename + "/"+config_.name_pro + "_class10.fluxes");
    //            class10file << total_fluxes -poly[p].n_traj_FT - poly[p].n_traj_TF << ";"<< total_fluxes<<std::endl;
            for (int fc = 0; fc<fcm_centers_id.size();fc++)
                if (fcm_centers_id[fc] != 10 && fcm_centers_id[fc] != 11){
                    out << ";" << polyclass[i][fcm_centers_id[fc]];
                    
                }
            out << std::endl;
//            std::cout << "is open: " << i << " poly "<<poly_ids[i]<<" " <<polyclass[i][0]<<std::endl;
        }
    }
    out.flush();
    out.close();
    classes.clear();
    velocities.clear();
    nodeFs.clear();
    nodeTs.clear();
    poly_ids.clear();
    fcm_centers_id.clear();

//    class10file.close();
}

void init_default_timed_fluxes(std::vector<poly_base> &poly,config &config){
    std::vector<std::pair<int, int>> poly_timetable; // un vettore il cui numero di elementi è la suddivisione in intervalli di un'ora dell'intervallo totale di tempo considerato.    
    int num_bin = int((config_.end_time - config_.start_time) / (config_.dump_dt * 60));
    for (int i = 0; i < num_bin; ++i)
    {
        poly_timetable.push_back(std::make_pair(0, 0));
    }
    for (auto &p : poly)
        p.timed_fluxes = poly_timetable;

}

void initialize_timed_fluxes(std::vector<poly_base> &poly,std::vector<traj_base> &traj,double &sigma,config &config_){
//   Description: initialize the timed fluxes for each poly
//   Input: poly and traj
//   Output: timed fluxes for each poly {t0:[FT,TF],t1:[FT,TF],...} t0: int, FT: int, TF: int
    std::cout << "initializing FT-TF fluxes" <<std::endl;
    int num_bin = int((config_.end_time - config_.start_time) / (config_.dump_dt * 60));
    for (auto &t : traj)
    {
        for (auto &j : t.path)
        {
            int bin_w = int(int(j.second * 3600) / (config_.dump_dt * 60));
            if (bin_w < num_bin){
                if (j.first >= 0)
                {
                    poly[j.first].n_traj_FT++;
                    poly[j.first].timed_fluxes[bin_w].first++;
                }
                else
                {
                    poly[-j.first].n_traj_TF++;
                    poly[-j.first].timed_fluxes[bin_w].second++;
                }
            }
        }
    }
    // measure sigma for standardize color in draw fluxes
    for (auto &p : poly)
        sigma += (p.n_traj_FT + p.n_traj_TF) * (p.n_traj_FT + p.n_traj_TF);
    sigma = sqrt(sigma / int(poly.size()));
    std::cout << "Make fluxes:    sigma = " << sigma << std::endl;


}

void count_flux_per_poly(std::vector<int> poly_ids,std::vector<int> classes,std::vector<int> fcm_centers_id,std::map<int,std::vector<int>> &poly2class_count){   
    for(int i=0;i<poly_ids.size()-1;i++){
        poly2class_count[poly_ids[i]]=std::vector<int>(fcm_centers_id.size(),0);
    }
//    std::cout << "size initialization vector: " << poly2class_count[poly_ids.back()].size() << " "<<poly2class_count[poly_ids.front()].size() << std::endl;
    for(int i=0;i<poly_ids.size()-1;i++){
        for (int fc = 0; fc<fcm_centers_id.size();fc++){
//            std::cout << "poly id: " << poly_ids[i] << " class: " << classes[i] << " fcm id: " << fcm_centers_id[fc] << std::endl;
            if(classes[i]==fcm_centers_id[fc] && (fcm_centers_id[fc]!=10 && fcm_centers_id[fc]!=11)){
                if (poly_ids[i]<0){
                    std::cout<<"poly negative id: "<<poly_ids[i]<<std::endl;
                    poly2class_count[-poly_ids[i]][fc]++;}
                else{poly2class_count[poly_ids[i]][fc]++;}
                }
            }
        }
    }


void dump_timed_fluxes(std::vector<poly_base> &poly,config &config_){
    ofstream out_timed(config_.cartout_basename + "/" + config_.name_pro + "_timed_fluxes.csv");
    std::cout << "timed fluxes" << std::endl;
    //   out << "id;id_local;nodeF;nodeT;length;total_fluxes;n_traj_FT;n_traj_TF;cid" << endl;
    if (out_timed.is_open()){
        out_timed << "time;id;id_local;nodeF;nodeT;length;total_fluxes;n_traj_FT;n_traj_TF;cid" << endl;
        for (auto &p : poly)
        {
            for (int n = 0; n < p.timed_fluxes.size(); ++n)
            {
                string datetime = physycom::unix_to_date(size_t(n * config_.dump_dt * 60 + config_.start_time));
                out_timed << datetime << ";" << p.id << ";" << p.id_local << ";" << p.cid_Fjnct << ";" << p.cid_Tjnct << ";" << p.length << ";" << p.timed_fluxes[n].first + p.timed_fluxes[n].second << ";" << p.timed_fluxes[n].first << ";" << p.timed_fluxes[n].second << ";" << p.cid_poly << std::endl;
            }
        }
        out_timed.flush();
        out_timed.close();
    }
    else throw std::runtime_error("Could not open timed fluxes file");

} 

//----------------------------------------------------------------------------------------------------
void make_fluxes(std::vector<traj_base> &traj,double &sigma,std::vector<poly_base> &poly,std::vector<centers_fcm_base> &centers_fcm,std::vector<std::map<int,int>> &classes_flux)
{
    std::cout << "centers fcm size: " << centers_fcm.size() << std::endl;
    int size_v = centers_fcm.size();
    std::vector<std::vector<int>> polyclass(poly.size(),std::vector<int> (size_v,0));
    write_mil_file_poly2class(poly,traj,config_);
    init_default_timed_fluxes(poly,config_);
    initialize_timed_fluxes(poly,traj,sigma,config_); // use to write timed_fluxes.csv
    dump_fluxes_file(poly,config_,polyclass); // use to write .fluxes
    std::cout << " dumped fluxes"<< std::endl;
    polyclass.clear();
    std::cout << " polyclass freed"<< std::endl;

    dump_timed_fluxes(poly,config_);
    std::cout << "dumped timed fluxes" << std::endl;
     // use to write timed_fluxes.csv
    // temporary added for direction analysis of rimini fluxes
    // std::map<std::string, int> fluxes_direttrici;
    // std::vector<std::string> rimini_direttrici;
    // rimini_direttrici.push_back("Via_Emilia");
    // rimini_direttrici.push_back("Via_Marecchiese");
    // rimini_direttrici.push_back("Via_Consolare_Rimini-San_Marino");
    // rimini_direttrici.push_back("Via_Flaminia");
    // rimini_direttrici.push_back("Via_Popilia");
    // for (auto &rd : rimini_direttrici)
    //  fluxes_direttrici[rd] = 0;
    //
    // double direttrici_total = 0.;
    // for (auto &p : poly) {
    //  if (std::find(rimini_direttrici.begin(), rimini_direttrici.end(), p.name) != rimini_direttrici.end()){
    //
    //    fluxes_direttrici[p.name] += (p.n_traj_FT + p.n_traj_TF);
    //    direttrici_total += (p.n_traj_FT + p.n_traj_TF);
    //  }
    //}
    // std::cout << "total: "<<direttrici_total << std::endl;
    // for (auto &fd : fluxes_direttrici)
    //  std::cout << fd.first << "   " << (fd.second/direttrici_total)*100<<" %" << std::endl;
}
//----------------------------------------------------------------------------------------------------

void dump_longest_traj(std::vector<traj_base> &traj)
{
    ofstream outtraj(config_.cartout_basename + config_.name_pro + "_traj2plot.csv");
    ofstream outtraj1(config_.cartout_basename + config_.name_pro + "_traj2plot1.csv");
    outtraj << "CALL_ID;timestamp;LAT;LON" << std::endl;
    outtraj1 << "CALL_ID;timestamp;LAT;LON" << std::endl;

    int l_max = 0;
    int idx = 0;
    int c = 0;
    for (auto &t : traj)
    {
        if (t.stop_point.size() > l_max)
        {
            l_max = t.stop_point.size();
            idx = c;
        }
        else
            continue;
        c++;
    }
    for (auto &sp : traj[idx].stop_point)
    {
        outtraj << traj[idx].id_act << ";" << sp.points.front().itime << ";" << sp.centroid.lat << ";" << sp.centroid.lon << std::endl;
    }
    for (auto &t : traj)
    {
        //    if (t.id_act==4294926734)
        for (auto &sp : t.stop_point)
            outtraj1 << t.id_act << ";" << sp.points.front().itime << ";" << sp.centroid.lat << ";" << sp.centroid.lon << std::endl;
        //    else continue;
    }
    outtraj.close();
    outtraj1.close();
}

void make_bp_traj(std::vector<traj_base> &traj,config &config_,double &sigma,data_loss &dataloss,std::vector<poly_base> &poly,std::vector<centers_fcm_base> &centers_fcm,std::vector<node_base> &node,std::vector<std::map<int,int>> &classes_flux)
{
    list<pair<int, double>> path, join_path;
    int cnt = 0, cnt_bestpoly = 0;

    std::cout << "**********************************" << std::endl;
    for (auto &n : traj)
    {
        int ipoly_old = -8000000;
        path.clear();
        join_path.clear();
        // ogni traiettoria è composta da diversi stoppoint di tipo cluster caratterizzati :
        // vettori di punti e centroide:(lat,lon,t,indx,state,itime,speed,accel,type),duration,pap
        n.stop_point[0].pap.clear();
        for (int k = 0; k < n.stop_point.size() - 1; ++k)
        {
            // if (n.id_act == 13181)
            //     std::cout << n.stop_point[k].pap.id_poly<<"  "<< n.stop_point[k+1].pap.id_poly <<"   "<< n.stop_point[k].pap.s << "   " << n.stop_point[k+1].pap.s << std::endl;
            best_poly(n.stop_point[k], n.stop_point[k + 1],poly,node);
            cnt_bestpoly++;
        }
//      DEBUG
//        std::cout<<"survived all points traj: "<<n.id_act<<std::endl;
//        if (n.id_act == 3260072392){
//            std::cout << "Class: " << n.means_class << std::endl;
//            for(auto &pol:n.stop_point.back().pap.path){std::cout << "id poly: " << pol.first << " time: " << pol.second << std::endl;}
//        }
//      END DEBUG
        n.path = n.stop_point.back().pap.path;
        n.stop_point.back().pap.path.clear();

        // if (n.id_act == 13181){
        //   for (auto &m : n.path)
        //     std::cout << m.first << std::endl;
        //   std::cin.get();
        // }
    }
    std::cout << "end traj" << std::endl;
    // count number of poly crossed and delete activity on the same polys
    std::cout << "enable threshold" << std::endl;
    if (config_.enable_threshold)
    {
        for (auto &t : traj)
        {
            list<int> path_temp;
            for (auto l : t.path)
                path_temp.push_back(abs(l.first));
            path_temp.sort();
            path_temp.unique();
            t.n_poly_unique = int(path_temp.size());
        }

        for (vector<traj_base>::iterator it = traj.begin(); it != traj.end();)
        {
            if (it->n_poly_unique < config_.threshold_polyunique)
                it = traj.erase(it);
            else
                it++;
        }
        dataloss.n_traj_poly_thresh = int(traj.size());
        std::cout << "Num Traj after poly unique: " << traj.size() << std::endl;
    }
    std::cout << "start make_fluxes" << std::endl;
//    dump_longest_traj(traj);
//    make_fluxes(traj,sigma,poly,centers_fcm,classes_flux);
    // make_FD();
}
//----------------------------------------------------------------------------------------------------
void make_polygons_analysis(config &config_,std::vector<centers_fcm_base> &centers_fcm,std::vector<traj_base> &traj,std::vector<polygon_base> &polygon)
{
    if (polygon.size() == 0)
        return;

    vector<traj_base> traj_tmp;

    // case: 2 args [location, code number (0 start or stop, 1 just start, 2 just stop)]
    if (config_.polygons_code.size() == 2)
    {
        int cnt_pg_center = 0;
        int cnt_pg_cities = 0;
        int cnt_pg_coast = 0;
        int cnt_pg_other = 0;
        int cnt_pg_station = 0;
        int cnt_from = 0;
        for (auto c : centers_fcm)
        {
            std::cout << c.idx << std::endl;
            c.cnt_polygons["center"] = 0;
            c.cnt_polygons["cities"] = 0;
            c.cnt_polygons["coast"] = 0;
            c.cnt_polygons["station"] = 0;
            c.cnt_polygons["other"] = 0;
            c.cnt_polygons["from"] = 0;
        }

        int location_type = 3;
        if (config_.polygons_code[0].find("center") == 0)
            location_type = 1;
        else if (config_.polygons_code[0].find("coast") == 0)
            location_type = 2;
        else if (config_.polygons_code[0].find("station") == 0)
            location_type = 4;

        std::cout << "Polygons code: " << config_.polygons_code[0] << " , " << config_.polygons_code[1] << std::endl;
        for (const auto &t : traj)
        {
            int wn_start = 0;
            int wn_stop = 0;
            int tag_start = 0;
            int tag_stop = 0;
            std::string polyg_start, polyg_stop;
            for (auto &pl : polygon)
            {
                wn_start = pl.is_in_wn(t.stop_point.front().centroid.lat, t.stop_point.front().centroid.lon);
                if (wn_start != 0)
                {
                    tag_start = pl.tag_type;
                    polyg_start = pl.pro["name"];
                    break;
                }
            }
            for (auto &pl : polygon)
            {
                wn_stop = pl.is_in_wn(t.stop_point.back().centroid.lat, t.stop_point.back().centroid.lon);
                if (wn_stop != 0)
                {
                    polyg_stop = pl.pro["name"];
                    tag_stop = pl.tag_type;
                    break;
                }
            }
            // 0 : start or stop point
            if (config_.polygons_code[1] == "0")
            {
                if (tag_start == location_type || tag_stop == location_type)
                {
                    cnt_from++;
                    if (centers_fcm.size() != 0)
                        if (t.means_class != 10 && t.means_class != 11)
                            centers_fcm[t.means_class].cnt_polygons["from"]++;

                    if ((tag_start == location_type && tag_stop == 0) || (tag_stop == location_type && tag_start == 0))
                    {
                        cnt_pg_other++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["other"]++;
                        traj_tmp.push_back(t);
                    }
                    else if ((tag_start == location_type && tag_stop == 1) || (tag_stop == location_type && tag_start == 1))
                    {
                        cnt_pg_center++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10&& t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["center"]++;
                        traj_tmp.push_back(t);
                    }
                    else if ((tag_start == location_type && tag_stop == 2) || (tag_stop == location_type && tag_start == 2))
                    {
                        cnt_pg_coast++;

                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10&& t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["coast"]++;
                        traj_tmp.push_back(t);
                    }
                    else if ((tag_start == location_type && tag_stop == 3) || (tag_stop == location_type && tag_start == 3))
                    {
                        cnt_pg_cities++;

                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["cities"]++;
                        traj_tmp.push_back(t);
                    }
                    else if ((tag_start == location_type && tag_stop == 4) || (tag_stop == location_type && tag_start == 4))
                    {
                        cnt_pg_station++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["station"]++;
                    }
                    else
                    {
                        std::cout << "tag_stop: " << tag_stop << std::endl;
                        // std::cin.get();
                    }
                }
            }
            // 1: start point
            else if (config_.polygons_code[1] == "1")
            {
                if (tag_start == location_type)
                {
                    cnt_from++;
                    if (centers_fcm.size() != 0)
                        if (t.means_class != 10 && t.means_class != 11)
                            centers_fcm[t.means_class].cnt_polygons["from"]++;

                    if (tag_stop == 0)
                    {
                        cnt_pg_other++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["other"]++;
                        traj_tmp.push_back(t);
                    }
                    else if (tag_stop == 1)
                    {
                        cnt_pg_center++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["center"]++;
                        traj_tmp.push_back(t);
                    }
                    else if (tag_stop == 2)
                    {
                        cnt_pg_coast++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["coast"]++;
                        traj_tmp.push_back(t);
                    }
                    else if (tag_stop == 3)
                    {
                        cnt_pg_cities++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["cities"]++;
                        traj_tmp.push_back(t);
                    }
                    else if (tag_stop == 4)
                    {
                        cnt_pg_station++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["station"]++;
                    }
                    else
                    {
                        std::cout << "tag_stop: " << tag_stop << std::endl;
                        std::cin.get();
                    }
                }
            }
            // 2: stop point
            else if (config_.polygons_code[1] == "2")
            {
                if (tag_stop == location_type)
                {
                    cnt_from++;
                    if (centers_fcm.size() != 0)
                        if (t.means_class != 10 && t.means_class != 11)
                            centers_fcm[t.means_class].cnt_polygons["from"]++;

                    if (tag_start == 0)
                    {
                        cnt_pg_other++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["other"]++;
                        traj_tmp.push_back(t);
                    }
                    else if (tag_start == 1)
                    {
                        cnt_pg_center++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["center"]++;
                        traj_tmp.push_back(t);
                    }
                    else if (tag_start == 2)
                    {
                        cnt_pg_coast++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["coast"]++;
                        traj_tmp.push_back(t);
                    }
                    else if (tag_start == 3)
                    {
                        cnt_pg_cities++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["cities"]++;
                        traj_tmp.push_back(t);
                    }
                    else if (tag_start == 4)
                    {
                        cnt_pg_station++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["station"]++;
                    }
                    else
                    {
                        std::cout << "tag start: " << tag_start << std::endl;
                        std::cin.get();
                    }
                }
            }
            else
            {
                std::cout << "Second item in polygons code not valid!" << std::endl;
                std::cin.get();
            }
        }
        std::cout << "Polygons:    Center Traj            " << (cnt_pg_center / double(cnt_from)) * 100.0 << "%" << std::endl;
        std::cout << "Polygons:    Cities Traj            " << (cnt_pg_cities / double(cnt_from)) * 100.0 << "%" << std::endl;
        std::cout << "Polygons:    Coast Traj             " << (cnt_pg_coast / double(cnt_from)) * 100.0 << "%" << std::endl;
        std::cout << "Polygons:    Station Traj  " << (cnt_pg_station / double(cnt_from)) * 100.0 << "%" << std::endl;
        std::cout << "Polygons:    Other Traj             " << (cnt_pg_other / double(cnt_from)) * 100.0 << "%" << std::endl;
        std::cout << "Num Traj after polygons             " << traj_tmp.size() << std::endl;
        for (auto c : centers_fcm)
        {
            std::cout << "----------------------------------" << std::endl;
            std::cout << "Class " << c.idx << ": Polygons:    Center Traj   " << (c.cnt_polygons["center"] / double(c.cnt_polygons["from"])) * 100.0 << "%" << std::endl;
            std::cout << "Class " << c.idx << ": Polygons:    Cities Traj   " << (c.cnt_polygons["cities"] / double(c.cnt_polygons["from"])) * 100.0 << "%" << std::endl;
            std::cout << "Class " << c.idx << ": Polygons:    Coast Traj    " << (c.cnt_polygons["coast"] / double(c.cnt_polygons["from"])) * 100.0 << "%" << std::endl;
            std::cout << "Class " << c.idx << ": Polygons:    Station Traj      " << (c.cnt_polygons["station"] / double(c.cnt_polygons["from"])) * 100.0 << "%" << std::endl;
            std::cout << "Class " << c.idx << ": Polygons:    Other Traj    " << (c.cnt_polygons["other"] / double(c.cnt_polygons["from"])) * 100.0 << "%" << std::endl;
        }
    }
    // 3 args [location_start, location_sto, code_number (0 both way, 1 oneway)]
    else if (config_.polygons_code.size() == 3)
    {
        int cnt_stop2start = 0;
        int cnt_start2stop = 0;
        int cnt_from = 0;
        int cnt_pg_other = 0;
        for (auto c : centers_fcm)
        {
            c.cnt_polygons["stop2start"] = 0;
            c.cnt_polygons["start2stop"] = 0;
            c.cnt_polygons["other"] = 0;
            c.cnt_polygons["from"] = 0;
        }

        // initialize start type
        int start_type = 3;
        if (config_.polygons_code[0].find("center") == 0)
            start_type = 1;
        else if (config_.polygons_code[0].find("coast") == 0)
            start_type = 2;
        else if (config_.polygons_code[0].find("station") == 0)
            start_type = 4;

        // initialize stop type
        int stop_type = 3;
        if (config_.polygons_code[1].find("center") == 0)
            stop_type = 1;
        else if (config_.polygons_code[1].find("coast") == 0)
            stop_type = 2;
        else if (config_.polygons_code[1].find("station") == 0)
            stop_type = 4;

        std::cout << "Polygons code: " << config_.polygons_code[0] << " , " << config_.polygons_code[1] << " , " << config_.polygons_code[2] << std::endl;

        for (const auto &t : traj)
        {
            int wn_start = 0;
            int wn_stop = 0;
            int tag_start = 0;
            int tag_stop = 0;
            std::string polyg_start, polyg_stop;
            for (auto &pl : polygon)
            {
                wn_start = pl.is_in_wn(t.stop_point.front().centroid.lat, t.stop_point.front().centroid.lon);
                if (wn_start != 0)
                {
                    tag_start = pl.tag_type;
                    polyg_start = pl.pro["name"];
                    break;
                }
            }
            for (auto &pl : polygon)
            {
                wn_stop = pl.is_in_wn(t.stop_point.back().centroid.lat, t.stop_point.back().centroid.lon);
                if (wn_stop != 0)
                {
                    polyg_stop = pl.pro["name"];
                    tag_stop = pl.tag_type;
                    break;
                }
            }
            // 0 : both directions
            if (config_.polygons_code[2] == "0")
            {
                if ((tag_start == start_type || tag_start == stop_type) && tag_start != tag_stop)
                {
                    cnt_from++;
                    if (centers_fcm.size() != 0)
                        if (t.means_class != 10 && t.means_class != 11)
                            centers_fcm[t.means_class].cnt_polygons["from"]++;

                    if (tag_stop == 0 || tag_start == 0)
                    {
                        cnt_pg_other++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["other"]++;
                    }
                    else if (tag_start == start_type && tag_stop == stop_type)
                    {
                        cnt_start2stop++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["start2stop"]++;
                        traj_tmp.push_back(t);
                    }
                    else if (tag_stop == start_type && tag_start == stop_type)
                    {
                        cnt_stop2start++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["stop2start"]++;
                        traj_tmp.push_back(t);
                    }
                }
            }
            // 1: from start to stop
            else if (config_.polygons_code[2] == "1")
            {
                if ((tag_start == start_type || tag_stop == stop_type) && tag_start != tag_stop)
                {
                    cnt_from++;
                    if (centers_fcm.size() != 0)
                        if (t.means_class != 10 && t.means_class != 11)
                            centers_fcm[t.means_class].cnt_polygons["from"]++;

                    if (tag_stop == 0 || tag_start == 0)
                    {
                        cnt_pg_other++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["other"]++;
                    }
                    else if (tag_start == start_type && tag_stop == stop_type)
                    {
                        cnt_start2stop++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["start2stop"]++;
                        traj_tmp.push_back(t);
                    }
                    else if (tag_stop == start_type && tag_start == stop_type)
                    {
                        cnt_stop2start++;
                        if (centers_fcm.size() != 0)
                            if (t.means_class != 10 && t.means_class != 11)
                                centers_fcm[t.means_class].cnt_polygons["stop2start"]++;
                    }
                }
            }
            else
            {
                std::cout << "Second item in polygons code not valid!" << std::endl;
                std::cin.get();
            }
        }
        std::cout << "Polygons:    Start-stop traj " << (cnt_start2stop / double(cnt_from)) * 100.0 << "%" << std::endl;
        std::cout << "Polygons:    Stop-start traj " << (cnt_stop2start / double(cnt_from)) * 100.0 << "%" << std::endl;
        std::cout << "Polygons:    Other Traj      " << (cnt_pg_other / double(cnt_from)) * 100.0 << "%" << std::endl;
        std::cout << "Num Traj after polygons      " << traj_tmp.size() << std::endl;

        for (auto c : centers_fcm)
        {
            std::cout << "----------------------------------" << std::endl;
            std::cout << "Class " << c.idx << ": Polygons:    Start - stop traj " << (c.cnt_polygons["start2stop"] / double(c.cnt_polygons["from"])) * 100.0 << "%" << std::endl;
            std::cout << "Class " << c.idx << ": Polygons:    Stop-start traj   " << (c.cnt_polygons["stop2start"] / double(c.cnt_polygons["from"])) * 100.0 << "%" << std::endl;
            std::cout << "Class " << c.idx << ": Polygons:    Other Traj        " << (c.cnt_polygons["other"] / double(c.cnt_polygons["from"])) * 100.0 << "%" << std::endl;
        }
    }
    else
    {
        std::cout << " Code for polygon analysis missed or wrong!" << std::endl;
    }

    traj = traj_tmp;
    traj_tmp.clear();
    traj_tmp.shrink_to_fit();
    return;
}
//----------------------------------------------------------------------------------------------------
/*
make_multimodality description:
Input:
    traj: vector of traj_base (contains informations useful in input:
                                    - average_inst_speed,v_min,v_max,sinuosity 
                                informations about output:
                                    - means_class (int of the class the trajectory belongs to), means_p (vector float of probability size of num_classes = config.num_tm) )
    config: Configuration file
    centers_fcm: vector of size num_classes = config.num_tm
Description:

Goal:
    Assign each traj the values of means_class (According to the Fuzzy kmean) and means_p
    
NOTE:
IMPORTANT -> The classes are in order of velcoity in cw (vector of centers_fcm) from the slowest to quickest .
It is important as going on with the algorithm, you clusterize the network by the distinction of the trajectories and order will
remain. 
*/
void make_multimodality(std::vector<traj_base> &traj,config &config_,std::vector<centers_fcm_base> &centers_fcm)
{
    // Prepare features for Transport Means Recognition
// HO AGGIUNTO QUI         if (t.average_inst_speed < config_.max_inst_speed) ogni volta che devo aggiungere alle features un grado di libertà
    vector<float> features_data;
    for (auto &t : traj)
        t.sinuosity = distance_record(t.stop_point.front().points.front(), t.stop_point.back().points.front()) / t.length;
    for (auto &t : traj)//        if (t.average_inst_speed < config_.max_inst_speed){
            features_data.push_back(float(t.average_inst_speed));
    for (auto &t : traj)//        if (t.average_inst_speed < config_.max_inst_speed)
            features_data.push_back(float(t.v_max));
    for (auto &t : traj)//        if (t.average_inst_speed < config_.max_inst_speed)
            features_data.push_back(float(t.v_min));
    for (auto &t : traj)//        if (t.average_inst_speed < config_.max_inst_speed)
            features_data.push_back(float(t.sinuosity));
    ofstream out_features(config_.cartout_basename + config_.name_pro + "out_features.csv");
    out_features <<"id_act;average_speed;v_max;v_min;sinuosity" << std::endl;
    for (auto &t : traj)    
        {if(t.average_inst_speed >t.v_min && t.average_inst_speed < t.v_max)
            out_features << t.id_act << ";"<<t.average_inst_speed<<";"<<t.v_max<<";"<<t.v_min<< ";"<<t.sinuosity <<std::endl;
        else std::cout<<"average speed out of bound: "<< t.id_act << " vmin " << t.v_min << " vmax " <<t.v_max << " av_inst_speed "<< t.average_inst_speed<<std::endl;
        }
    out_features.close();
    // for (auto &t : traj) features_data.push_back(float(t.length));
    // for (auto &t : traj) features_data.push_back(float(t.time));
// END PREPARATION FEATURES

    std::cout << "**********************************" << std::endl;
    std::cout << "Multimodality num classes:      " << config_.num_tm << std::endl;

// START FUZZY ALGORITHM
    int num_tm = config_.num_tm;                                    // number classes
    int num_N = int(traj.size());                                   // number trajectories
    int num_feat = 4;                                               // v_average, v_max, v_min, sinuosity
    double epsilon_fcm = 0.005;                                     // threshold used to create initial points in FCM
    FCM *fcm;                                                       // fcm object   \\\\\NOTE: Problem?//////
    fcm = new FCM(2, epsilon_fcm);                                  // Can it be done? Declare and then initialized with new?
    Map<MatrixXf> data_tr(features_data.data(), num_N, num_feat);   // matrix initialized with (features_data, shape (N traj, N features))
    MatrixXf data = data_tr;                                        // 
    fcm->set_data(&data);                                           // Copy the memory to fcm.m_data (N traj, N features)
    fcm->set_num_clusters(num_tm);                                  // Set cluster vector in fcm. m_cluster_center: (N clusters, N features);
    random_device rnd_device;                                       // Declare random device
    mt19937 mersenne_engine{rnd_device()};                          // Generates random integers
    mersenne_engine.seed(5);                                        // Set seed
    uniform_real_distribution<float> dist{0.0, 1.0};                // Uniform distribution from 0 to 1
    auto gen = [&dist, &mersenne_engine]()                          
    {                                                               //
        return dist(mersenne_engine);                               //
    };                                                              //
    vector<float> vec_rnd(num_N * num_tm);                          // 1D vector, is going to contain probabilities to belong to class i
    generate(begin(vec_rnd), end(vec_rnd), gen);                    // Generates the starting random configuration of probability for each traj
    Map<MatrixXf> membership_temp(vec_rnd.data(), num_N, num_tm);   // 2D Matrix, contains for each traj the prob of belonging to class i
    MatrixXf membership = membership_temp;                          // 2D matrix, copy (Is it going to create a problem?)
    fcm->set_membership(&membership);                               // m_membership (2D matrix (traj,n of classes))
    double diff = 1.0;                                              //
    fcm->compute_centers();                                         // compute centers
    fcm->update_membership();                                       // Update membership
    while (diff > epsilon_fcm)
    {
        fcm->compute_centers();                                     // Recomopute centers 
        diff = fcm->update_membership();                            // And re-update until the distance of all points converge to 0
    }
    fcm->reorder_cluster_centers();// NOTE: This following line is necessary to have slowest class indexed by 0, ..., quickest by config.num_tm
    // END FUZZY ALGORITHM

    // START TRANSCRIPTION FUZZY ALGORITHM TO TRAJECTORIES
    // 1) Transcript information centers in center_fcm_base vector. <(<v>_0,vmin_0,vmax_0,sin_0),...,(<v>_(num_feat),vmin_(num_feat),vmax_(num_feat),sin_(num_feat))>
    for (int n = 0; n < num_tm; ++n)
    {
        centers_fcm_base cw;
        cw.idx = n;
        for (int m = 0; m < num_feat; ++m)
            {cw.feat_vector.push_back((*(fcm->get_cluster_center()))(fcm->get_reordered_map_centers_value(n), m));
            }
        centers_fcm.push_back(cw);
    }
     // Transcript information about class
    std::cout << "Multimodality p threshold:      " << config_.threshold_p << std::endl;
    int cnt_recong = 0;    
    for (int n = 0; n < traj.size(); ++n)
    {
        int max_idx;
        double max_p = 0.0;
        for (int m = 0; m < num_tm; ++m)
        {
            traj[n].p_cluster.push_back((*(fcm->get_membership()))(n, fcm->get_reordered_map_centers_value(m))); //metti in ordine p_cluster  (pcluster[0] = 0: slowest, ...)
            if ((*(fcm->get_membership()))(n, fcm->get_reordered_map_centers_value(m)) > max_p){// NOTA: la massima probabilità viene assegnata con ordine randomico, l'indice che do ora è quello mappato con l'ordine di centers_fcm
                max_idx = m; // instead of class index of the order of fcm -> I take to the new order of the map
                max_p = (*(fcm->get_membership()))(n, fcm->get_reordered_map_centers_value(m)); // I want just the index changed not the probability
            }
        }
        if (max_p < config_.threshold_p)
            traj[n].means_class = 10; // fake class index for hybrid traj
        else
        {
            traj[n].means_class = max_idx;
            cnt_recong++;
        }
        traj[n].means_p = max_p;
    }
    for (auto &t : traj)
        centers_fcm[t.means_class].cnt++;
    std::cout << "Multimodality traj recognized: " << cnt_recong << " (" << (double(cnt_recong) / int(traj.size())) * 100 << "%)" << std::endl;
    // END TRANSCRIPTION FUZZY ALGORITHM TO TRAJECTORIES

    if (config_.enable_slow_classification)
    {
        // Get Velocity Slowest Class
        int slow_id, medium_id;
        double min_v = 10000.0;
        for (auto &c : centers_fcm)
            if (c.feat_vector[0] < min_v)
            {
                slow_id = c.idx;
                min_v = c.feat_vector[0];
            }
        for (auto &c : centers_fcm)
            if (c.feat_vector[0] > min_v && c.feat_vector[0] < 20.0){
                medium_id = c.idx;
                std::cout << "medium_id: " << medium_id << " min_v: " << min_v << " velocity: " << centers_fcm[c.idx].feat_vector[0] << std::endl;
                }
        
        // Prepare features for Transport Means Recognition for slower classification (different days of data may need different treatments)
        vector<float> features_data2;
        for (auto &t : traj)
            if (t.means_class == slow_id)
                features_data2.push_back(float(t.average_inst_speed));
        for (auto &t : traj)
            if (t.means_class == slow_id)
                features_data2.push_back(float(t.v_max));
        for (auto &t : traj)
            if (t.means_class == slow_id)
                features_data2.push_back(float(t.v_min));
        for (auto &t : traj)
            if (t.means_class == slow_id)
                features_data2.push_back(float(t.sinuosity));
        std::cout << "**********************************" << std::endl;

        // START FUZZY ALGORITHM
        int num_tm2 = 2;
        int num_N2 = int(centers_fcm[slow_id].cnt);
        std::cout << "Slower Multimodality num classes:      " << num_tm2 << std::endl;
        std::cout << "Slower Multimodality num samples:      " << num_N2 << std::endl;
        int num_feat2 = 4; // v_average, v_max, v_min, sinuosity
        double epsilon_fcm2 = 0.005;
        FCM *fcm2;
        fcm2 = new FCM(2, epsilon_fcm2);
        Map<MatrixXf> data_tr2(features_data2.data(), num_N2, num_feat2);
        MatrixXf data2 = data_tr2;
        fcm2->set_data(&data2);
        fcm2->set_num_clusters(num_tm2);
        random_device rnd_device2;
        mt19937 mersenne_engine2{rnd_device2()}; // Generates random integers
        uniform_real_distribution<float> dist2{0.0, 1.0};
        auto gen2 = [&dist2, &mersenne_engine2]()
        {
            return dist2(mersenne_engine2);
        };
        vector<float> vec_rnd2(num_N2 * num_tm2);
        generate(begin(vec_rnd2), end(vec_rnd2), gen2);
        Map<MatrixXf> membership_temp2(vec_rnd2.data(), num_N2, num_tm2);
        MatrixXf membership2 = membership_temp2;
        fcm2->set_membership(&membership2);
        double diff2 = 1.0;
        fcm2->compute_centers();
        fcm2->update_membership();
        while (diff2 > epsilon_fcm2)
        {
            fcm2->compute_centers();
            diff2 = fcm2->update_membership();
        }
        fcm2->reorder_cluster_centers();
        // END FUZZY ALGORITHM
        
        vector<centers_fcm_base> centers_fcm_slow;
        for (int n = 0; n < num_tm2; ++n)
        {
            centers_fcm_base cw;
            cw.idx = n;
            for (int m = 0; m < num_feat2; ++m)
                cw.feat_vector.push_back((*(fcm2->get_cluster_center()))(fcm2->get_reordered_map_centers_value(n), m));
            centers_fcm_slow.push_back(cw);
        }
        for (auto &c : centers_fcm_slow){
            std::cout<< "centers fcm slow: " << c.idx << " velocity: " << c.feat_vector[0] << std::endl;
        }
        
//        for (auto &c : centers_fcm_slow)
//            std::cout << c.idx << " velocity: " << c.feat_vector[0] << std::endl;
        // update results IN THE CASE I WANT: slow_id2 is always the smallest velocity class
/*        if (centers_fcm_slow[0].feat_vector[0] < centers_fcm_slow[1].feat_vector[0])
        {
            slow_id2 = 0; // I expect to be dropped here
            medium_id2 = 1;
        }
        else
        {
            slow_id2 = 1;
            medium_id2 = 0;
            std::cout << "ERROR in the order of fcm of slower classes"  << std::endl;
        }
*/
        int slow_id2 = 0; int medium_id2 = 1;
         

        centers_fcm[slow_id].feat_vector = centers_fcm_slow[slow_id2].feat_vector;
        
        for (auto &c: centers_fcm_slow){
            std::cout << "index " << c.idx << "velocity " << c.feat_vector[0] << std::endl;
        }
        for (auto &c : centers_fcm){
            if (c.idx >= 1 && c.idx != 10){
                c.idx += 1;
            }
            std::cout<< "centers fcm: " << c.idx << " velocity: " << c.feat_vector[0] << std::endl;

        }

        centers_fcm.insert(centers_fcm.begin()+1,centers_fcm_slow[1]);
        for (auto &c : centers_fcm){
            std::cout<< "centers fcm: " << c.idx << " velocity: " << c.feat_vector[0] << std::endl;
        }

//        centers_fcm.push_back(centers_fcm_slow[medium_id2]);
        // update centers_fcm: idx
        centers_fcm[1].idx = 1;
//        centers_fcm[num_tm].idx = num_tm;
        // update centers_fcm: cnt

        // EXTEND VECTOR P_CLUSTER for each trajectory
        int idx_2fcm = 0;
        for (int n = 0; n < traj.size(); ++n)
        {
//          DEBUG
//            int i = 0;
//            for (auto &p:traj[n].p_cluster){
//                std::cout<<"Prob index: "<< p << " counter " << i << " " << std::endl;
//                i++;
//            }
//          END DEBUG
            std::vector<double>::iterator it_0 = traj[n].p_cluster.begin();
            std::vector<double>::iterator it = traj[n].p_cluster.begin() + 1;
            if (traj[n].means_class == slow_id){
                traj[n].p_cluster.insert(it,(*(fcm2->get_membership()))(idx_2fcm, medium_id2));
                traj[n].p_cluster[slow_id2] = (*fcm2->get_membership())(idx_2fcm, slow_id2);
                traj[n].p_cluster[medium_id2] = (*fcm2->get_membership())(idx_2fcm, medium_id2);
                // DEBUG on the PROBABILITY OF BELONGING TO A CLUSTER: CHECK -> OK
//                std::cout << "Trajectory number: " << n << std::endl;
//                std::cout << "Velocity: " << traj[n].average_speed << std::endl;
//                std::cout << "Info class slow: " << slow_id << std::endl;
//                std::cout << "Velocity: " << centers_fcm_slow[slow_id2].feat_vector[0] << std::endl;
//                std::cout << "Probability: " << traj[n].p_cluster[slow_id2] << std::endl;
//                std::cout << "Info class medium: " << medium_id2 << std::endl;
//                std::cout << "Velocity: " << centers_fcm_slow[medium_id2].feat_vector[0] << std::endl;
//                std::cout << "Probability: " << traj[n].p_cluster[medium_id2] << std::endl;
//                std::cout << "Probability 2 classes: " << traj[n].p_cluster[medium_id2] + traj[n].p_cluster[slow_id] << std::endl;
//              END DEBUG
                // NORMALIZE THE PROBABILiTY OF p_cluster
                float total_prob_with_all_classes = 0.;
                for(auto &p:traj[n].p_cluster){
                    total_prob_with_all_classes += p;
                }
                for(auto &p:traj[n].p_cluster){
                    p = p/total_prob_with_all_classes;
                }
                total_prob_with_all_classes = 0.;
                for(auto &p:traj[n].p_cluster){
                    total_prob_with_all_classes += p;
                }
//              DEBUG probability normalized: CHECK
//                std::cout << "Probability all classes: " << total_prob_with_all_classes << std::endl;

                idx_2fcm++; 
            }
            else{
                traj[n].p_cluster.insert(it,0.0);
            }

//        for (auto &p : traj[n].p_cluster){
//            std::cout <<"Prob Associated: "<< p << std::endl;
//        }

        }

// DEBUG
/*        ofstream control_size_cluster(config_.cartout_basename + config_.name_pro + "_control.txt");
        for (int n = 0; n < traj.size(); ++n)
            if(traj[n].p_cluster.size() == 5){
                control_size_cluster << "traj: " << n << " size: " <<  traj[n].p_cluster.size() << " velocty: "<< centers_fcm[traj[n].means_class].feat_vector[0];
                for(auto &p:traj[n].p_cluster){
                    control_size_cluster << p << ";" <<std::endl;
                    }
                }
        control_size_cluster.close();
*/
        std::cout << "fcm2 txt" << std::endl;
        ofstream out_fcm2(config_.cartout_basename + config_.name_pro + "_fcm2.txt");
        idx_2fcm = 0;         
        for (int n = 0; n < traj.size(); ++n)
        {
            float max_probability = 0.;
            int index_cluster = 0;
            int index_chosen = 0;
            // DA CONTROLLARE !!!!!!!! CHECK
//            std::cout << "Trajectory number: " << n << std::endl;
            if(traj[n].means_class!=10 || traj[n].means_class!=11){        
                for(auto p: traj[n].p_cluster){
                    if(p>max_probability){
                        max_probability = p;
                        index_chosen = index_cluster;
                    }
                    index_cluster +=1;
    //                std::cout << "Class: " << index_cluster << " probability: " << p << std::endl;
                }
                traj[n].means_class = index_chosen;
                traj[n].means_p = max_probability;
            }
//              DEBUG
//            std::cout << "Velocity: " << traj[n].average_speed << std::endl;
//            std::cout << "Class: " << traj[n].means_class << std::endl;
//            std::cout << "control output before pressing a key to go on: " << std::endl;
//            int number;
//            std::cin >> number;

/*
            if (traj[n].means_class == slow_id2)
            {  
                traj[n].p_cluster[slow_id2] = (*(fcm2->get_membership()))(idx_2fcm, slow_id2);
                //traj[n].p_cluster.push_back((*(fcm2->get_membership()))(idx_2fcm, medium_id2));
                if ((*(fcm2->get_membership()))(idx_2fcm, slow_id2) > (*(fcm2->get_membership()))(idx_2fcm,medium_id2))
                {
                    traj[n].means_class = fcm2->get_reordered_map_centers_value(slow_id2);
                    traj[n].means_p = traj[n].p_cluster[fcm2->get_reordered_map_centers_value(slow_id2)];
                   out_fcm2 << " chosen class: " << traj[n].means_class << " probability: " << traj[n].means_p << " velocity: " << centers_fcm[traj[n].means_class].feat_vector[0] <<std::endl;
                }
                else
                {
//                    traj[n].means_class = num_tm;
//                    traj[n].means_p = traj[n].p_cluster[num_tm];
                    traj[n].means_class = fcm2->get_reordered_map_centers_value(medium_id2);
                    traj[n].means_p = traj[n].p_cluster[fcm2->get_reordered_map_centers_value(medium_id2)];
                   out_fcm2 << " chosen class: " << traj[n].means_class << " probability: " << traj[n].means_p << " velocity: " << centers_fcm[traj[n].means_class].feat_vector[0] <<std::endl;

                }
                idx_2fcm++;
            }
//ADDED ALBI ORDERING CENTERS_FCM
            else{   
                traj[n].means_class += 1;
                traj[n].means_p = traj[n].p_cluster[traj[n].means_class];
               out_fcm2 << " chosen class: " << traj[n].means_class << " probability: " << traj[n].means_p << " velocity: " << centers_fcm[traj[n].means_class].feat_vector[0] <<std::endl;
            }
//ADDED ALBI ORDERING CENTERS_FCM
*/
            out_fcm2<< " trajectory number: "<< n <<std::endl;
            for(auto &p:traj[n].p_cluster){
                out_fcm2 << p << ";";
            }
            if(traj[n].means_class!=10 && traj[n].means_class!=11)
            out_fcm2 << " velocity: " << centers_fcm[traj[n].means_class].feat_vector[0] << " class: "<< traj[n].means_class << " average speed: " << traj[n].average_inst_speed<<std::endl;        
        }
        out_fcm2.close();
        for (auto &c : centers_fcm)
        c.cnt = 0;
            // update centers_fcm: feat vector
        for (auto &t : traj)
            centers_fcm[t.means_class].cnt++;
    }
    for (auto &c : centers_fcm){
        std::cout << "slow mobilty upgraded class: " << c.idx << " velocity: " << c.feat_vector[0] << " number of trajectories: "<< c.cnt << std::endl;
    }
 
    // measure validation parameter ( intercluster dist/intracluster dist)
    // double dunn_index = measure_dunn_index();
    // std::cout << "Dunn index                      :      " << dunn_index << std::endl;

    if (config_.enable_print)
    {
        ofstream out_fcm_center(config_.cartout_basename + config_.name_pro + "_fcm_centers.csv");
        for (int c = 0; c < centers_fcm.size(); ++c)
        {
            out_fcm_center << c;
            for (auto &cc : centers_fcm[c].feat_vector)
                out_fcm_center << ";" << cc;
            out_fcm_center << ";" << centers_fcm[c].cnt << std::endl;
        }
        out_fcm_center.close();
        std::cout << config_.cartout_basename + config_.name_pro << std::endl;
        ofstream out_fcm(config_.cartout_basename + config_.name_pro + "_fcm.csv");
        out_fcm << "id_act;lenght;time;av_speed;v_max;v_min;cnt;av_accel;a_max;class;p" << std::endl;
        for (auto &t : traj)
        {
            out_fcm << t.id_act << ";" << t.length << ";" << t.time << ";" << t.average_inst_speed << ";" << t.v_max << ";" << t.v_min << ";" << t.stop_point.size() << ";" << t.average_accel << ";" << t.a_max << ";" << t.means_class << ";" << t.means_p <<std::endl;
        }
        out_fcm.close();

        if (1 == 0)
        {
            for (auto c = 0; c < centers_fcm.size(); ++c)
            {
                ofstream out_classes(config_.cartout_basename + config_.name_pro + "_class_" + to_string(c) + ".csv");
                ofstream out_classes_points(config_.cartout_basename + config_.name_pro + "_class_" + to_string(c) + "points.csv");
                out_classes << "id_act;lenght;time;speed;v_max;v_min;p_cluster;n_stop_points" << std::endl;
                out_classes_points << "id_act;lat;lon;time" << std::endl;
                for (auto &t : traj)
                    if (t.means_class == c)
                    {
                        out_classes << t.id_act << ";" << t.length << ";" << t.time << ";" << t.average_inst_speed << ";" << t.v_max << ";" << t.v_min << ";" << t.p_cluster[t.means_class] << ";" << t.stop_point.size() << std::endl;
                        for (auto &sp : t.stop_point)
                            out_classes_points << t.id_act << ";" << sp.centroid.lat << ";" << sp.centroid.lon << ";" << sp.points.front().itime << std::endl;
                    }
                out_classes.close();
            }
        }
    }
    if (0 == 1)
    {
        ofstream out_activity(config_.cartout_basename + config_.name_pro + "_activity.csv");
        out_activity << "id_act;lat;lon;timestamp;class" << std::endl;
        for (auto &t : traj)
        {
            for (auto &p : t.stop_point)
            {
                out_activity << t.id_act << ";" << p.centroid.lat << ";" << p.centroid.lon << ";" << p.points.front().itime << ";" << t.means_class << std::endl;
            }
        }
        out_activity.close();
    }
}


// ALBI

/**
 * Calculates the intersection of two vectors v1 and v2 and stores the result in vector v3.
 *
 * @param v1 the first vector
 * @param v2 the second vector
 * @param v3 the vector to store the intersection result
 */
void subnet_intersection(std::vector<int> v1, std::vector<int> v2, std::vector<int> &v3)
{
    std::set_intersection(v1.begin(), v1.end(),
                          v2.begin(), v2.end(),
                          back_inserter(v3));
}

/**
 * Generates the complementary set of elements in v1 relative to the intersection set.
 *
 * @param v1 vector of integers to generate the complementary set from
 * @param intersection vector of integers representing the intersection set
 * @param v3 reference to a vector of integers to store the complementary set
 *
 * @return void
 *
 * @throws None
 */
void subnet_complementary(std::vector<int> v1, std::vector<int> intersection, std::vector<int> &v3)
{
    std::sort(v1.begin(), v1.end());
    std::sort(intersection.begin(), intersection.end());
    for (auto &e1 : v1)
    {
        bool found = false;
        for (auto &e : intersection)
        {
            if (e == e1)
            {
                found = true;
                break;
            }
            else if (e1 < e)
            {
                break;
            } // this works beacouse the elements are ordered -> I cannot find e anymore once I overcome e
            else
                continue;
        }
        if (!found)
        {
            v3.push_back(e1);
        }
    }
}
/* Description selecttraj_from_vectorpolysubnet_velsubnet:
* Generates a vector of trajectories that are contained by the subnet whose label is save_label 
*/

std::vector<traj_base> selecttraj_from_vectorpolysubnet_velsubnet(std::vector<int> poly_subnet, std::vector<traj_base> &traj, std::vector<poly_base> &poly, std::string label_save)
{
    std::cout << "case: " << label_save << " initial number of trajectories analyzed: " << traj.size() << " number of polies subnet: " << poly_subnet.size() << " total number polies: " << poly.size() << std::endl;
    int time_step = config_.bin_time * 60; // bin_time
    int num_bin = int((config_.end_time - config_.start_time) / time_step);
    std::vector<traj_base> traj_subnet;
    // INITIALIZE time2av_vel,time2velocities,time2timepercorrence {bin_0: 0.0, bin_1: 0.0, ...} , {bin_0: vector<double> 0., bin_1: vector<double> 0., ...} ,{bin_0: 0.0, bin_1: 0.0, ...} 
    for (auto pol : poly_subnet)
    {
        for (int i = 0; i < num_bin; i++)
            poly[pol].time2av_vel[i] = 0.;
        std::vector<double> vel;
        for (int i = 0; i < num_bin; i++)
            poly[pol].time2velocities[i] = vel;
        for (int i = 0; i < num_bin; i++)
            poly[pol].time2timepercorrence[i] = 0.;
    }
    // collect data of trajectories of interest
    std::cout << "number of recognized trajectories: " << traj_subnet.size() << std::endl;
    for (auto &t : traj)
    {
        std::vector<int> found_polies;
        for (auto &sp : t.path)
        { // for(auto &sp:t.stop_point){
            for (auto pol : poly_subnet)
            {
                if (sp.first == pol && std::find(found_polies.begin(), found_polies.end(), sp.first) == found_polies.end())
                { 
                    // add the trajectory that live in the subnet and their speed.
                    found_polies.push_back(sp.first);
                    traj_subnet.push_back(t);
                    poly[pol].velocities.push_back(t.average_inst_speed);
                    for (int i = 0; i < num_bin; i++)
                    {
                        if (int(config_.start_time) + int(i * time_step) < int(t.stop_point.front().points.front().itime) && int(config_.start_time + (i + 1) * time_step) > int(t.stop_point.front().points.front().itime))
                        {
                            poly[pol].time2velocities[i].push_back(t.average_inst_speed);
                            break;
                        }
                        else
                            continue;
                    }
                    break;
                }
                else
                    continue;
            }

            // if point path is in the set of intersected polies and it
        }
    }
    // OUTPUT: time2velocities: {bin_0: vector<double> some, bin_1: vector<double> some, ...} -> READY TO COMPUTE AV SPEED POLY
    // FORMAT RICHIESTO DA REGIONE id_act;lenght;time;av_speed;ndat;front_lat;front_lon;tail_lat;tail_lon;start_time;end_time
    if(0==1){
        ofstream dati_regione(config_.cartout_basename + config_.name_pro + "_" + label_save + "dati_regione.csv");
        dati_regione << "id_act;lenght;time;av_speed;ndat;front_lat;front_lon;tail_lat;tail_lon;start_time;end_time" << std::endl;
        for (auto &t : traj_subnet)
        {
            dati_regione << t.id_act << ";" << t.length << ";" << t.time << ";" << t.average_inst_speed << ";" << t.stop_point.size() << ";" << t.stop_point.front().centroid.lat << ";" << t.stop_point.front().centroid.lon << ";" << t.stop_point.back().centroid.lat << ";" << t.stop_point.back().centroid.lon << ";" << t.stop_point.front().points.front().itime << ";" << t.stop_point.back().points.back().itime << std::endl;
        }
        dati_regione.close();
    }
    // time2av_vel: {bin_0: 0.0, bin_1: 0.0, ...} -> READY TO COMPUTE AV SPEED POLY
    for (auto pol : poly_subnet)
    {
        for (int i = 0; i < num_bin; i++)
        {
            if (poly[pol].time2velocities[i].size() != 0)
            {
                poly[pol].time2av_vel[i] = std::accumulate(poly[pol].time2velocities[i].begin(), poly[pol].time2velocities[i].end(), 0) / poly[pol].time2velocities[i].size();
                if (poly[pol].time2av_vel[i] != 0)
                    poly[pol].time2timepercorrence[i] = poly[pol].length / poly[pol].time2av_vel[i];
                else
                    poly[pol].time2timepercorrence[i] = -1;
            }
            else
            {
                poly[pol].time2timepercorrence[i] = -1;
                poly[pol].time2av_vel[i] = -1;
            }
        }
    }
    std::cout << "velocity subnet: " << label_save << std::endl;
    velocity_subnet(poly_subnet, poly, time_step, num_bin, label_save);
    return traj_subnet;
}
/* Description velocity_subnet:

*/

void velocity_subnet(std::vector<int> poly_subnet, std::vector<poly_base> &poly, int time_step, int num_bin, std::string save_label)
{

    //  per ogni sottonet
    ofstream oo(config_.cartout_basename + config_.name_pro + "_" + save_label + "velocity_subnet.csv");
    oo << "start_bin;end_bin;poly_id;number_people_poly;total_number_people;av_speed;time_percorrence" << std::endl;
    std::map<int, int> time2numberpeople; // number of people per timestep
    for (int i = 0; i < num_bin; i++)
    {
        time2numberpeople[i] = 0;
        for (auto &pol : poly_subnet)
        {
            time2numberpeople[i] += poly[pol].time2velocities[i].size();
        }
//        std::cout << "number of people in interval " << i << " : " << time2numberpeople[i] << std::endl;
    }
    for (auto &pol : poly_subnet)
    {
        for (int i = 0; i < num_bin; i++)
        {
            if (poly[pol].time2velocities[i].size() != 0)
            {
                //        std::cout <<  config_.start_time + i*time_step << ";"<< config_.start_time + (i+1)*time_step << ";" << poly[pol].id_local << ";" <<poly[pol].time2velocities[i].size() <<";"<<time2numberpeople[i] << ";"<< poly[pol].time2av_vel[i] << ";" << poly[pol].time2timepercorrence[i]<< std::endl;
                oo << config_.start_time + i * time_step << ";" << config_.start_time + (i + 1) * time_step << ";" << poly[pol].id_local << ";" << poly[pol].time2velocities[i].size() << ";" << time2numberpeople[i] << ";" << poly[pol].time2av_vel[i] << ";" << poly[pol].time2timepercorrence[i] << std::endl;
            }
            else
            {
                oo << config_.start_time + i * time_step << ";" << config_.start_time + (i + 1) * time_step << ";" << poly[pol].id_local << ";"
                   << "0"
                   << ";"
                   << "0"
                   << ";"
                   << "-1"
                   << ";"
                   << "-1" << std::endl;
            }
            poly[pol].time2velocities[i].clear();
            poly[pol].time2av_vel[i] = 0;
            poly[pol].time2timepercorrence[i] = 0;
        }
    }
    oo.close();
}
/* Description simplifies_c_intersect:
Input: 
    - subnets80: map of subnets (those initialized by make_subnets) (load_subnets if already computed)
    - traj: vector of all trajectories (those initialized by make_traj)
    - poly: vector of polies (those initialized by make_poly)
Output:
    - complete_intersect: vector of polies belonging to all classes i (i = [1,...,fcm_centers.size()-1])
    - complete_intersection_velocity_subnet.csv: velocity of polies belonging just to the class i (i = [1,...,fcm_centers.size()-1]) FORMAT {id_local;time;av_speed;time_percorrence} 
*/

std::vector<int> simplifies_c_intersect(std::map<std::string, std::vector<int>> subnets80,std::vector<traj_base> &traj,std::vector<poly_base> &poly)
{
    std::vector<int> subnet_class0 = subnets80.begin()->second; //starting with class 0
    std::vector<int> net_complete_intersection = subnet_class0; 
    std::cout << "initial size of class " << subnets80.begin()->first << " : " << subnets80.begin()->second.size() << std::endl;
    for (std::map<std::string, std::vector<int>>::iterator i = subnets80.begin(); i != subnets80.end(); ++i)
    {
        std::vector<int> tmp_intersection;
        subnet_intersection(net_complete_intersection, i->second, tmp_intersection); // update net_complete_intersection with the intersection of the successive class
        net_complete_intersection = tmp_intersection;
        std::cout << "size of intersection at class " << i->first << " : " << net_complete_intersection.size() << std::endl;
        tmp_intersection.clear();
    }
    // OUTPUT std::vector<int> net_complete_intersection
    ofstream spaced_file_subnet_complete_intersection(config_.cartout_basename + "/" + config_.name_pro + "_complete_intersection.txt");
    for (auto &poly_intersect : net_complete_intersection)
    {
        spaced_file_subnet_complete_intersection << poly_intersect << " ";
    }
    spaced_file_subnet_complete_intersection.close();
    std::vector<traj_base> traj_complete_intersection;
    std::string label_save = "complete_intersection_";
    traj_complete_intersection = selecttraj_from_vectorpolysubnet_velsubnet(net_complete_intersection, traj, poly, label_save);

    return net_complete_intersection;
}
/* Description simplifies_c_complement:
Input: 
    - subnets80: map of subnets (those initialized by make_subnets) (load_subnets if already computed)
    - traj: vector of all trajectories (those initialized by make_traj)
    - poly: vector of polies (those initialized by make_poly)
Output:
    ITERATION FOR ALL CLASSES:
    - complete_complement: vector of polies belonging just to the class i (i = [1,...,fcm_centers.size()-1])
    - complete_complement_i_velocity_subnet.csv: velocity of polies belonging just to the class i (i = [1,...,fcm_centers.size()-1]) FORMAT {id_local;time;av_speed;time_percorrence} 
*/

/*
Description hierarchical_deletion_of_intersection: builds and save file.txt containing:
    quickest subnet
    second quickest subnet without the quickest
    3 quickest without 1,2
    and so on until the slowest subnet
DESCRIPTION:
    step1: for (std::map<std::string, std::vector<int>>::iterator i = subnets80.begin(); i != subnets80.end(); ++i)
        I iterate from ordered subnets (vector of int) -> subnets80[0] = (poly00,...,polyn0) <slowest class> 
    step2: for (std::map<std::string, std::vector<int>>::iterator j = subnets80.begin(); j != subnets80.end(); ++j)
        I extract the intersection recursively obtaining the slowe
        st subnet
*/      


std::map<std::string,std::vector<int>> hierarchical_deletion_of_intersection(std::map<std::string, std::vector<int>> subnets80){
    int idx_subnet2cut = 0;
    std::map<std::string,std::vector<int>> hierarchically_selected_subnets;
    for (std::map<std::string, std::vector<int>>::iterator i = subnets80.begin(); i != subnets80.end(); ++i){
//        std::cout << "hierarchical intersection:\nclass: " << i->first << " size: " << i->second.size() << std::endl;
        std::vector<int> starting_subnet = i->second;
        int idx_subnet2compare = 0;
        for (std::map<std::string, std::vector<int>>::iterator j = subnets80.begin(); j != subnets80.end(); ++j)
            {
            std::vector<int> subnet_intersection_class;

            if (idx_subnet2cut<idx_subnet2compare && idx_subnet2cut<4){
                std::vector<int> temporary_complement;
                subnet_intersection(starting_subnet, j->second, subnet_intersection_class); // update net_complete_intersection with the intersection of the successive class
                subnet_complementary(starting_subnet, subnet_intersection_class, temporary_complement);
                starting_subnet = temporary_complement;
                subnet_intersection_class.clear();
                temporary_complement.clear();
            }
        idx_subnet2compare+=1;
        }
    if (idx_subnet2cut<4){
        std::cout << "SAVING FILE: " << config_.cartout_basename + "/" + config_.name_pro + std::to_string(idx_subnet2cut) + "_class_subnet.txt" << std::endl;
        std::cout <<"idx: " << idx_subnet2cut << " size subnet to be cut: "<< starting_subnet.size()<<std::endl;
        std::cout << "index extracted graph: " << idx_subnet2compare << " size subnet: "<< starting_subnet.size()<<std::endl;
        hierarchically_selected_subnets[std::to_string(idx_subnet2cut)] = starting_subnet;
        ofstream spaced_file_subnet_complete_intersection(config_.cartout_basename + "/" + config_.name_pro + std::to_string(idx_subnet2cut) + "_class_subnet.txt");
        for (auto &poly_intersect : starting_subnet)
        {
            spaced_file_subnet_complete_intersection << poly_intersect << " ";
        }
        spaced_file_subnet_complete_intersection.close();
        idx_subnet2cut+=1;
        }
    }    // OUTPUT std::vector<int> net_complete_intersection
    return hierarchically_selected_subnets;
}

/*
assign_new_class: assign new class to trajectories that live in the subnet.
Associates to the trajcetories that are of the slower classes the class that most represents them. 
Hopefully, we find that the slower classes contain trajectories that actually are in the quickest part of the network
This part of the code needs to be used to analyze the fondamental diagram. That now contains all the different classs.
*/

void assign_new_class(std::vector<traj_base> &traj,std::vector<poly_base> &poly,std::map<std::string, std::vector<int>> subnets80){
    std::cout << "assign new class" << std::endl;
    ofstream newclassification(config_.cartout_basename + "/" + config_.name_pro + "_fcm_new.csv");
    newclassification <<"id_act;class";
    for(auto &s:subnets80){
        newclassification << ";" << s.first;
    }
    newclassification<<std::endl;

            for (std::map<std::string, std::vector<int>>::iterator i = subnets80.begin(); i != subnets80.end(); ++i){
                    std::cout << "Sono in funzione e controllo ordine subnets " << std::stoi(i->first) << "\n Mi spetto siano in ordine" << std::endl;
            }
    

    for (auto &t : traj)
    {            
        std::vector<int> count_point_per_class(subnets80.size(),0);
        if(t.means_class!=subnets80.size()-1){
            for (std::map<std::string, std::vector<int>>::iterator i = subnets80.begin(); i != subnets80.end(); ++i){
                int subnet_index = std::stoi(i->first);        
                std::vector<int> subnet_polies = i->second;
                for (auto &sp : t.path)
                { // for(auto &sp:t.stop_point){
                    for (auto pol : subnet_polies)
                    {
                        if (sp.first == pol)
                        { 
                            // add the trajectory that live in the subnet and their speed.
                            count_point_per_class[subnet_index] +=1;
                            break;
                        }
                        else
                            continue;
                    }
                }
            }
            int index_class;
            int max_count = 0;
            for(int i=0;i<count_point_per_class.size();i++){
                if(count_point_per_class[i]>max_count){
                    max_count = count_point_per_class[i];
                    index_class = i;
                }
            }
            newclassification << t.id_act << ";" << index_class;
            for(auto c:count_point_per_class){
                newclassification << ";" << c;}
            newclassification << std::endl;
            //TODO cicla per scrivere la frazione di strade per ogni classe
        }
        else{
            newclassification << t.id_act << ";" << t.means_class;
            for(int c = 0;c <count_point_per_class.size();c++){
                if(c==count_point_per_class.size()-1)
                    newclassification << ";" << t.path.size();
                else 
                    newclassification << ";" << 0;
                }
            newclassification << std::endl;

        }
    }
    newclassification.close();
}

std::map<std::string, std::vector<int>> simplifies_c_complement(std::map<std::string, std::vector<int>> subnets80,std::vector<traj_base> &traj,std::vector<poly_base> &poly)
{
    std::map<std::string, std::vector<int>> complete_complement;
    for (std::map<std::string, std::vector<int>>::iterator k = subnets80.begin(); k != subnets80.end(); ++k)
    {
        std::string key1 = k->first;
        complete_complement[key1] = k->second;
        // compute the complete complement for the kth class -> subnetwork
        for (std::map<std::string, std::vector<int>>::iterator l = subnets80.begin(); l != subnets80.end(); ++l)
        {
            if (k->first != l->first)
            {
                std::vector<int> temp_vect;
                temp_vect.clear();
                subnet_complementary(complete_complement[key1], l->second, temp_vect);
                complete_complement[key1] = temp_vect;
                std::cout << "complementary of " << k->first << "with " << l->first << std::endl;
                std::cout << "size " << complete_complement[key1].size() << std::endl;
            }
            else
                continue;
        }
//        ofstream spaced_file_subnet_complete_complement(config_.cartout_basename + "/" + config_.name_pro + k->first + "_complete_complement.txt");
//        std::cout << "save file with " << complete_complement[key1].size() << " polies" << std::endl;
//        for (auto &poly_complement : complete_complement[key1])
//        {
//            spaced_file_subnet_complete_complement << poly_complement << " ";
//        }
//        spaced_file_subnet_complete_complement.close();
        // Take the complete complementary and calculate the velocity of polies. (Look at the population and time of these data.)
        std::vector<traj_base> traj_complete_complement;
        std::string label_save = "complete_complement_" + k->first + "_";
        traj_complete_complement = selecttraj_from_vectorpolysubnet_velsubnet(complete_complement[key1], traj, poly, label_save);
        //      complete_complement[key1].clear();
    }
    return complete_complement;
}
/* Description analysis_subnet:
Input: 
    - traj: vector of all trajectories (those initialized by make_traj)
    - poly: vector of polies (those initialized by make_poly)
    - subnets: map of subnets (those initialized by make_subnets) (load_subnets if already computed)

Order calls:
    1- config_["all_subnets_speed"] (a) -> selecttraj_from_vectorpolysubnet_velsubnet (b)-> velocity_subnet
    ITERATED FOR EACH CLASS: fcm_centers.size()-1
    2- simplifies_c_intersect (a)-> subnet_intersecton (b)-> selecttraj_from_vectorpolysubnet_velsubnet (c)-> velocity_subnet
    3- simplifies_c_complement (a)-> subnet_complement (b)-> selecttraj_from_vectorpolysubnet_velsubnet (c)-> velocity_subnet

Output:
    1 (a) -> traj_of_interest (b)-> class_i_velocity_subnet.csv -- with i = [1,...,fcm_centers.size()-1] FORMAT {id_local;time;av_speed;time_percorrence}
    ITERATED FOR EACH CLASS: fcm_centers.size()-1
    2 (a) -> polies_of_interest (b) -> traj_of_interest (c)-> complete_intersection_velocity_subnet.csv 
    3 (a) -> polies_of_interest(b) -> traj_of_interest (c)-> complete_complement_i_velocity_subnet.csv -- with i = [1,...,fcm_centers.size()-1] FORMAT {id_local;time;av_speed;time_percorrence}


*/

void analysis_subnets(std::vector<traj_base> &traj,std::vector<poly_base> &poly,map<string, vector<int>> &subnets)  
{
    // FILL SUBNETS80 {"class":vector<int>polies}
    std::map<std::string, std::vector<int>> subnets80;
    for (std::map<std::string, std::vector<int>>::iterator i = subnets.begin(); i != subnets.end(); ++i)
    {
        std::vector<std::string> tokens;
        std::string lab = i->first;
        std::vector<int> polies = i->second;
        physycom::split(tokens, lab, string("_"), physycom::token_compress_on);
        if (tokens[0] != "tot")
        {
            if (tokens[2] == std::to_string(80))
            {
                subnets80.insert(std::pair<std::string, std::vector<int>>(tokens[1], polies));
            }
            else
                continue;
        }
        else
            continue;
    }
    std::map<std::string,std::vector<int>> hierarchically_selected_subnets; 
    hierarchically_selected_subnets = hierarchical_deletion_of_intersection(subnets80);
    assign_new_class(traj,poly,hierarchically_selected_subnets);
    hierarchically_selected_subnets.clear();
    // VELOCITY TIME SUBNET FOR EACH CLASS
    if(config_.all_subnets_speed == true){
        for (std::map<std::string, std::vector<int>>::iterator i = subnets.begin(); i != subnets.end(); ++i){
            std::vector<traj_base> traj_subnet_tmp;
            std::vector<std::string> tokens;
            std::string lab = i->first; // takes values in "sub_60","sub_70","sub_80","tot_80"
            std::vector<int> polies = i->second; // takes int values of polies
            physycom::split(tokens, lab, string("_"), physycom::token_compress_on);
            if (tokens[0] != "tot")
            {
                if (tokens[2] == std::to_string(80))
                {
                    ofstream spaced_file_subnet(config_.cartout_basename + "/" + config_.name_pro + "class_"+tokens[1]+".txt");
                    for (auto &p : polies)
                    {
                        spaced_file_subnet << p << " ";
                    }
                    spaced_file_subnet.close(); 
                    traj_subnet_tmp = selecttraj_from_vectorpolysubnet_velsubnet(polies, traj, poly, "class_"+tokens[1]);

                }
                else
                    continue;
            }
            else
                continue;

            traj_subnet_tmp.clear();
        }
    }
    // COMPUTE INTERSECTION // PRODUCE VELOCITY AND TIME SUBNETS:
    if(config_.complete_intersection_speed==true){
        std::vector<int> total_intersection = simplifies_c_intersect(subnets80,traj,poly);
        total_intersection.clear();
    }
    // COMPUTE COMPLEMENT: // PRODUCE VELOCITY AND TIME SUBNETS:
    if(config_.complete_complement_speed==true){
        std::map<std::string, std::vector<int>> total_complement = simplifies_c_complement(subnets80,traj,poly);
        total_complement.clear();
        }
    
}
/* Description dump_FD:
*/

void dump_FD(std::vector<poly_base> &poly)
{
    ofstream out_fd(config_.cartout_basename + "/" + config_.name_pro + "_foundamental_diagram.csv");
    out_fd << "time;id;id_local;lenght;velocity_FT;velocity_TF;time_percorrence_FT;time_percorrence_TF" << endl;
    for (auto &p : poly)
    {
        for (int n = 0; n < p.velocity.size(); ++n)
        {
            string datetime = physycom::unix_to_date(size_t(n * config_.dump_dt * 60 + config_.start_time));
            out_fd << datetime << ";" << p.id << ";" << p.id_local << ";" << p.length << ";" << p.velocity[n].first << ";" << p.velocity[n].second << ";" << p.time_percorrence[n].first << ";" << p.time_percorrence[n].second << std::endl;
        }
    }
    out_fd.close();
}
/*
std::map<int, std::vector<long long int>> get_class2idact_from_fcm_csv(confit &config_){
    std::map<int, std::vector<long long int>> p2id_act;
    std::ifstream file(config_.cartout_basename + "/" + config_.name_pro +"_fcm.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file." << std::endl;
        return 1;
    }

    // Read and process each line of the CSV file
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string token;
        std::vector<std::string> tokens;

        // Split the line into tokens using ";" as the delimiter
        while (std::getline(lineStream, token, ';')) {
            tokens.push_back(token);
        }

        // Check if the line has enough columns
        if (tokens.size() >= 11) {
            int p = std::stoi(tokens[10]);  // Assuming "p" is in the 11th column (0-based indexing)
            long long int id_act = std::stoll(tokens[0]);  // Assuming "id_act" is in the 1st column (0-based indexing)

            // Add the id_act to the corresponding "p" key in the map
            p2id_act[p].push_back(id_act);
        }
    }

    // Close the file
    file.close();
    return p2id_act;
}

//
initialize_class2countpoly_traj: 
Per ogni classe vedere a quale sottorete le traiettorie appartengono.
Supponiamo per id = 110000000 
Ho che 20 pt sono in 0 (di questi )
15 in 1 
10 in 2
5 in 3
//


void initialize_class2countpoly_traj(std::vector<traj_base> &traj){
    // NOTA: per semplicità e non tirare fuori di nuovo subnets e iterare li, prendo direttamente il range di classi che so mi servano, nel mio caso 4 
    std::vector<std::string> tokens {"0","1","2","3","4"};
    for (auto &tok:tokens){
        std::ifstream spaced_file_subnet(config_.cartout_basename + "/" + config_.name_pro + "class_"+tok+".txt");
        if spaced_file_subnet.is_open(){
            std::vector<int> polies;
            polies.clear();
            std::string line;
            while (std::getline(spaced_file_subnet, line)){
                std::istringstream lineStream(line);
                std::string token;
                while (std::getline(lineStream, token, ' ')){
                    polies.push_back(std::stoi(token));
                    }
                for (auto &t : traj){
                    for (auto &sp:t.path)
                        if (std::find(polies.begin(), polies.end(), sp.first) != polies.end()){
                            t.class2countpoly_traj[std::stoi(tok)] += 1;
                    }
                }
            }
        }
        else{
            std::cout  << config_.cartout_basename + "/" + config_.name_pro + "class_"+tok+".txt"<< " not found, continue..." <<std::endl;
            continue;}
    }
    spaced_file_subnet.close();
    std::ofstream out_class2countpoly_traj(config_.cartout_basename + "/" + config_.name_pro + "_reinit_class.csv");
    out_class2countpoly_traj << "id_act;" << std::endl;     
    for (auto &tok:tokens){
        std::ifstream spaced_file_subnet(config_.cartout_basename + "/" + config_.name_pro + "class_"+tok+".txt");
        if spaced_file_subnet.is_open(){
            out_class2countpoly_traj <<

        }    
    for (auto &t : traj){
        int biggest_count_class = 0;
        int new_class;
        for (auto &c : t.class2countpoly_traj){
            if (c.second > biggest_count_class){
                biggest_count_class = c.second;
                new_class = c.first;
            }
        }

        out_class2countpoly_traj << t.id_act << ";" << new_class << std::endl;
    }
    out_class2countpoly_traj.close();
    }
}
void reclassify_traj_(std::vector<traj_base> &traj,std::vector<poly_base> &poly,config &config_)
{
    std::map<int, std::vector<long long int>> p2id_act;
    p2id_act = get_class2idact_from_fcm_csv(config_);

}
*/
// ALBI

//----------------------------------------------------------------------------------------------------
void dump_fluxes(std::vector<traj_base> &traj,config &config_,std::vector<centers_fcm_base> &centers_fcm,std::vector<poly_base> &poly,std::vector<std::map<int,int>> &classes_flux)
{
    //  std::cout << "dump fluxes start" << std::endl;
    //  std::cout << config_.cartout_basename + "/weights/" + config_.name_pro + ".fluxes" << std::endl;
    ofstream out(config_.cartout_basename + "/weights/" + config_.name_pro + ".fluxes");
    ofstream out_timed(config_.cartout_basename + "/" + config_.name_pro + "_timed_fluxes.csv");
    if (config_.enable_multimodality)
    {
        out << "id;id_local;nodeF;nodeT;lenght;total_fluxes";
        for (auto &c : centers_fcm)
            out << ";"
                << "class_" + to_string(c.idx);
        out << std::endl;
        for (auto &p : poly)
        {
            out << p.id << ";" << p.id_local << ";" << p.cid_Fjnct << ";" << p.cid_Tjnct << ";" << p.length << ";" << p.n_traj_FT + p.n_traj_TF;
            for (auto &pc : p.classes_flux)
                if (pc.first != 10)
                    out << ";" << pc.second;
            out << std::endl;
        }
    }
    else
    {
        std::cout << "timed fluxes" << std::endl;
        //   out << "id;id_local;nodeF;nodeT;length;total_fluxes;n_traj_FT;n_traj_TF;cid" << endl;
        out << "id;id_local;nodeF;nodeT;length;total_fluxes" << endl;
        out_timed << "time;id;id_local;nodeF;nodeT;length;total_fluxes;n_traj_FT;n_traj_TF;cid" << endl;
        for (auto &p : poly)
        {
            // out << p.id << ";" << p.id_local << ";" << p.cid_Fjnct << ";" << p.cid_Tjnct << ";" << p.length << ";" << p.n_traj_FT + p.n_traj_TF << ";" << p.n_traj_FT << ";" << p.n_traj_TF << ";" << p.cid_poly << std::endl;
            out << p.id << ";" << p.id_local << ";" << p.cid_Fjnct << ";" << p.cid_Tjnct << ";" << p.length << ";" << p.n_traj_FT + p.n_traj_TF << std::endl;
            for (int n = 0; n < p.timed_fluxes.size(); ++n)
            {
                string datetime = physycom::unix_to_date(size_t(n * config_.dump_dt * 60 + config_.start_time));
                out_timed << datetime << ";" << p.id << ";" << p.id_local << ";" << p.cid_Fjnct << ";" << p.cid_Tjnct << ";" << p.length << ";" << p.timed_fluxes[n].first + p.timed_fluxes[n].second << ";" << p.timed_fluxes[n].first << ";" << p.timed_fluxes[n].second << ";" << p.cid_poly << std::endl;
            }
        }
        
    }
    out_timed.close();
    out.close();
    if (0 == 1)
    {
        ofstream out_crossed(config_.cartout_basename + "/" + config_.name_pro + "_crossed_poly.csv");
        out_crossed << "id_car;cid_poly;onaway" << std::endl;
        for (auto &t : traj)
        {
            for (auto &c : t.path)
            {
                if (c.first < 0)
                    out_crossed << t.id_act << ";" << poly[-c.first].cid_poly << ";"
                                << "TF" << std::endl;
                else
                    out_crossed << t.id_act << ";" << poly[c.first].cid_poly << ";"
                                << "FT" << std::endl;
            }
        }
        out_crossed.close();
    }
}
//----------------------------------------------------------------------------------------------------
void make_MFD(jsoncons::json jconf,std::vector<traj_base> &traj,std::vector<centers_fcm_base> &centers_fcm)
{
    std::cout << "Make MFD" << std::endl;
    double lat_max_MFD = jconf.has_member("lat_max_MFD") ? jconf["lat_max_MFD"].as<double>() : 44.08189;
    double lat_min_MFD = jconf.has_member("lat_min_MFD") ? jconf["lat_min_MFD"].as<double>() : 44.04698;
    double lon_max_MFD = jconf.has_member("lon_max_MFD") ? jconf["lon_max_MFD"].as<double>() : 12.55489;
    double lon_min_MFD = jconf.has_member("lon_min_MFD") ? jconf["lon_min_MFD"].as<double>() : 12.61257;

    // initialize MFD_collector
    std::map<int, std::map<int, std::vector<std::pair<double, int>>>> MFD_collection;
    int slice_time = 1800;
    std::cout << config_.start_time << "  " << config_.end_time << std::endl;
    for (auto &c : centers_fcm)
        for (int tt = config_.start_time; tt < config_.end_time; tt += slice_time)
        {
            int time_idx = int((tt - config_.start_time) / slice_time);
            std::vector<std::pair<double, int>> vec_temp;
            MFD_collection[c.idx][time_idx] = vec_temp;
        }

    ofstream out_mfd(config_.cartout_basename + config_.name_pro + "_mfd.csv");
    out_mfd << "class;time_idx;L_total;T_total;density;speed_av" << std::endl;

    for (const auto &t : traj)
    {
        traj_base tw;
        tw.means_class = t.means_class;

        // space filter
        for (const auto &n : t.stop_point)
        {
            if (n.centroid.lat > lat_min_MFD && n.centroid.lat < lat_max_MFD && n.centroid.lon > lon_min_MFD && n.centroid.lon < lon_max_MFD)
            {
                record_base rec;
                rec.lat = n.centroid.lat;
                rec.lon = n.centroid.lon;
                rec.itime = n.points.front().itime;
                tw.record.push_back(rec);
            }
        }
        if (tw.record.size() != 0)
        {
            double T = double(tw.record.back().itime - tw.record.front().itime);
            double L = 0.0;
            for (int r = 0; r < tw.record.size() - 1; ++r)
                L += distance_record(tw.record[r + 1], tw.record[r]);
            int time_idx = int((tw.record.front().itime - config_.start_time) / slice_time);
            MFD_collection[tw.means_class][time_idx].push_back(make_pair(L, T));
        }
    }

    for (auto &idx : MFD_collection)
        for (auto &t_idx : idx.second)
        {
            int density = t_idx.second.size();
            double L_total = 0.0;
            double T_total = 0;
            for (auto &i : t_idx.second)
            {
                L_total += i.first;
                T_total += i.second;
            }
            out_mfd << idx.first << ";" << t_idx.first << ";" << L_total << ";" << T_total << ";" << density << ";" << centers_fcm[idx.first].feat_vector[0] << std::endl;
        }
}
//----------------------------------------------------------------------------------------------------
//------------------------- POLYSTAT -----------------------------------------------------------------
set<long long int> nodes;
map<long long int, map<long long int, int>> node_poly;
map<int, long long int> lid_cid;
map<long long int, int> cid_lid;

void make_node_map(const vector<polystat_base> &poly)
{
    for (const auto &p : poly)
    {
        nodes.insert(p.nF);
        nodes.insert(p.nT);
        node_poly[p.nF][p.nT] = p.id_local;
        node_poly[p.nT][p.nF] = p.id_local;
    }

    int node_lid = 0;
    for (const auto &n : nodes)
    {
        lid_cid[node_lid] = n;
        cid_lid[n] = node_lid++;
    }
}
//----------------------------------------------------------------------------------------------------
vector<string> sub_types({"tot"});
vector<polystat_base> import_poly_stat(const string &filename)
{
    ifstream input(filename);
    if (!input)
    {
        cerr << "make subnet error to open input : " << filename << endl;
        exit(4);
    }

    string line;
    vector<polystat_base> polystat;
    getline(input, line); // skip header
    vector<string> tokens;
    physycom::split(tokens, line, string(";"), physycom::token_compress_off);
    if (tokens.size() > 6)
        for (int idx = 6; idx < tokens.size(); ++idx)
            sub_types.push_back(tokens[idx]);
    getline(input, line); // skip first fake poly
    while (getline(input, line))
    {
        polystat.emplace_back(line);
    }
    make_node_map(polystat);
    return polystat;
}
//----------------------------------------------------------------------------------------------------
map<string, vector<int>> make_subnet(config &config_)
{
    string file_fluxes = config_.cartout_basename + "/weights/" + config_.name_pro + ".fluxes";
    std::map<string, vector<int>> subnets;
    auto poly = import_poly_stat(file_fluxes); // leggo dal file che ho creato sopra.
    std::cout <<"imported poly_stats" << std::endl;
    // vector<double> sub_fractions({ 0.1, 0.15, 0.2 });
    vector<double> sub_fractions({0.8});//{0.6,0.7,0.8} // sto dicendo che percentuale di flussi controllo

    std::cout << "**********************************" << std::endl;
    cout << "Subnet poly parsed : " << poly.size() << endl;
    cout << "Subnet node parsed : " << nodes.size() << endl;


    // allocate indices matrix
    int **ind;
    ind = new int *[poly.size()]; // prendo in memoria esattamente il numero di poly
    for (int i = 0; i < (int)poly.size(); ++i)
        ind[i] = new int[2]; // per ogni poly ho la coppia di nodi

    for (const auto &t : sub_types)
    {
        std::cout << "subtype size" << sub_types.size() << endl;
        int total_crossings = 0;
        for (const auto &n : poly)
         {if (n.flux.at(t)>0)
            total_crossings += n.flux.at(t) * n.length;
            }
        std::cout << "Total crossings per meters: " << total_crossings << std::endl; // totale di metri percorsi dai flussi.
        for (const auto &f : sub_fractions)
        {
            string label = t + "_" + to_string(int(f * 100));   // è per l'output
            cout << "Processing : " << t << " @ " << f << endl; // class_i
            sort(poly.begin(), poly.end(), [t](const polystat_base &p1, const polystat_base &p2)
                 { return p1.flux.at(t) > p2.flux.at(t); }); // metto in ordine la poly dalla più numerosa alla meno numerosa
            std::set<int> num_nodes;
            double cumulative = 0.0;
            for (const auto &p : poly)
            {
                cumulative += p.flux.at(t) * p.length;
                if ((cumulative / total_crossings) < f)
                {
                    num_nodes.insert(p.nF);
                    num_nodes.insert(p.nT); // essentially I am taking from the first f*100 per cent of the lenght of the total fluxes
                }
                else
                    break;
            }
            for (int i = 0; i < (int)poly.size(); ++i)
            {
                ind[i][0] = cid_lid[poly[i].nF];
                ind[i][1] = cid_lid[poly[i].nT];
            }
            std::cout << "num nodes " << num_nodes.size() << std::endl;
            std::cout << "num nodes old " << int(f * nodes.size()) << std::endl;

            std::cout << "poly size: " << (int)poly.size() << " num nodes: " <<  int(num_nodes.size()) << std::endl;

            auto nodesel = FeatureSelection(ind, (int)poly.size(), int(num_nodes.size()), true, false);
            std::cout << "size: " << nodesel.begin()->second.size() << std::endl;
            // auto nodesel = FeatureSelection(ind, (int)poly.size(), int(f * nodes.size()), true, false);
            for (const auto &p : nodesel.begin()->second){
                subnets[label].push_back(node_poly[lid_cid[p.first]][lid_cid[p.second]]);
                std::cout << "pushed " << node_poly[lid_cid[p.first]][lid_cid[p.second]] << std::endl;
            }
            sort(subnets[label].begin(), subnets[label].end());
            cout << "Selected poly : " << subnets[label].size() << endl;
        }
        std::cout << "----------------------------------" << std::endl;
    }
    ofstream out(file_fluxes + ".sub");
    for (const auto &i : subnets)
    {
        out << i.first << "\t";
        for (const auto &p : i.second)
            out << p << "\t";
        out << endl;
    }
    out.close();
    std::cout << "done" << endl;
return subnets;
}
//----------------------------------------------------------------------------------------------------
double measure_representativity(const string &label,std::vector<poly_base> &poly,map<string, vector<int>> subnets)
{
    double all = 0, sub = 0;
    for (const auto &p : poly)
    {
        std::vector<std::string> tokens;
        physycom::split(tokens, label, string("_"), physycom::token_compress_on);
        int class_idx = stoi(tokens[1]);
        all += p.classes_flux.at(class_idx) * p.length;
        if (find(subnets[label].begin(), subnets[label].end(), p.id_local) != subnets[label].end())
            sub += p.classes_flux.at(class_idx) * p.length;
    }
    return sub / all;
}
//----------------------------------------------------------------------------------------------------
