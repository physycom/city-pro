#include "stdafx.h"
#include "record.h"
#include "carto.h"
#include "data_reading.h"
#include "config.h"
#include <boost/algorithm/string.hpp>
#include <physycom/geometry.hpp>
#include <physycom/string.hpp>
#include "global_params.h"
//------------------------------------------------------------------------------------------------------
void load_poly(config config_,std::vector<poly_base> &poly,std::map<unsigned long long int, int> &poly_cid2lid)
{
    // inizializzo con stringhe dei file pnt e pro che ho creato da cartography-data tramite il file di configurazione dato in lettura i.e. trattori.json
    string poly_pnt_file = config_.file_pnt;
    string poly_pro_file = config_.file_pro;
    // apro in posizione di lettura poly_pnt_file e inizializzo il puntatore a file fp0
    FILE *fp0;
    fp0 = fopen(poly_pnt_file.c_str(), "r");
    if (fp0 == NULL)
    {
        cout << "fopen error in " << poly_pnt_file << endl;
        exit(5);
    }

    int id, nPT, ilat, ilon;
    long long int cid;
    // inizializzazioni in carto.h
    vector<point_base> points;
    poly_base poly_w;
    poly_w.clear();
    // add empty poly at the position 0
    points.clear();
    poly_w.set(0, 0, points);
    poly.push_back(poly_w);
    poly_w.clear();

    // read pnt file
    //%lld is the cid (8 bytes), %d id increasing from 1 (4 bytes), %d number of points
    while (fscanf(fp0, " %lld %d %d ", &cid, &id, &nPT) != EOF)
    {
        for (int i = 0; i < nPT; i++)
        {
            if (fscanf(fp0, " %d %d", &ilat, &ilon) == EOF)
            {
                cout << "reading error in " << poly_pnt_file << endl;
                fflush(stdout);
                exit(5);
            }
            // prendo le coordinate per ogni riga e ci butto il punt e poi lo pushback nei points.  Per ogni riga contata da nPT
            point_base point_w(ilat, ilon);
            points.push_back(point_w);
        }
        // finito il ciclo di punti mi prendo poly_w e ci attacco id,cid e points [carto.cpp linea 140]
        poly_w.set(id, cid, points);
        // push backo anche la poly
        poly.push_back(poly_w);
        points.clear();
        poly_w.clear();
    }
    fclose(fp0);
    cout << "Poly:     " << poly.size() - 1 << endl;

    // read pro file
    fp0 = fopen(poly_pro_file.c_str(), "r");
    if (fp0 == NULL)
    {
        cout << "fopen error in " << poly_pro_file << endl;
        exit(5);
    }
    if (fscanf(fp0, "%*[^\n]\n"))
    {
    }; // skip header

    int frc, n2c, fow, oneway, lanes;
    char *name = new char[70];
    double kmh;
    unsigned long long int id_poly, cid_Fjnct, cid_Tjnct;
    float meters;

    int n = 0;
    // front tail VS start end
    while (fscanf(fp0, " %llu %llu %llu %f %d %d %d %d %lf %d %s",
                  &id_poly, &cid_Fjnct, &cid_Tjnct, &meters, &frc, &n2c, &fow, &oneway, &kmh, &lanes, name) != EOF)
    {
        n++;
        std::cout << "n: " << n << std::endl;
        std::cout << "id poly: " << id_poly << " cid_Fjnct: " << cid_Fjnct << " cid_Tjnct: " << cid_Tjnct << std::endl;
//        std::cout << "size_of id poly: " << sizeof(id_poly) << " size_of cid_Fjnct: " << sizeof(cid_Fjnct) << " size_of cid_Tjnct: " << sizeof(cid_Tjnct) << std::endl;
        std::cout << "meters: " << meters << " frc: " << frc << " n2c: " << n2c << " fow: " << fow << " oneway: " << oneway << " kmh: " << kmh << " lanes: " << lanes << " name: " << name<<std::endl;
        std::cout << "size_of meters: " << sizeof(meters) << " size_of frc: " << sizeof(frc) << " size_of n2c: " << sizeof(n2c) << " size_of fow: " << sizeof(fow) << " size_of oneway: " << sizeof(oneway) << " size_of kmh: " << sizeof(kmh) << " size_of lanes: " << sizeof(lanes) << " size_of name: " << sizeof(name) << std::endl;
        if (poly[n].cid_poly == id_poly)
        {
            // inizializzo gli altri parametri di poly_base tramite il riconoscimento del numero della poly id di .pnt.
            // In questo momento questi sono la cid_Front (quello del grafo),la cid_tail(sempre quello del grafox1)
            poly[n].set(cid_Fjnct, cid_Tjnct, meters, oneway, name);
            poly[n].measure_length();
            poly_cid2lid[id_poly] = n;
        }
        else
        {
            cout << " index error in poly [" << n << "] " << poly[n].cid_poly << " vs " << id_poly << endl;
            exit(5);
        }
        std::cout << "end of scope" << std::endl;
    }
    fclose(fp0);
}
//------------------------------------------------------------------------------------------------------
void load_data(std::vector<activity_base> &activity,config &config_)
{
    int cnt_in = 0;
    int cnt_out = 0;
    int max_size_records = 0;

    for (auto &i : config_.file_data)
    {
        // per ogni file csv in cui ho id_act;id_act;something;timestamp;lat;lon;other_variables
        string data_file = i;
        // apro il data_file (la c:str dell'indirizzo in memoria aperto da fp_in)
        ifstream fp_in(data_file.c_str());
        string line;
        vector<string> strs;
        record_base rw;
        // per ogni riga del csv mi prendo l'id e poi i record
        map<long long int, vector<record_base>> activity_collect;

        long long int id_act = 0;
        int cnt = 0;
        if (!fp_in)
        {
            cout << "load_data: error in reading " << data_file << endl;
            exit(1);
        }
        int cnt_row = 0;
        getline(fp_in, line); // skip header
        while (fp_in)
        {
            getline(fp_in, line);
            boost::split(strs, line, boost::is_any_of(";"), boost::token_compress_off);
            cnt_row++;
            if (strs.size() > 1)
            { // se la stringa inizializzata dal file contiene almeno un elemento
                // main version (WORKS FOR VIASAT)
                //        id_act = stoll(strs[1]);
                //        rw.itime = stol(strs[3]);
                //        rw.lat = stod(strs[4]);
                //        rw.lon = stod(strs[5]);

                // new version (IMPORTANT TO USE THE TIMESTAMP FORMAT FOR THE TIME  )
                id_act = stoll(strs[0]);
                rw.itime = stol(strs[1]);
                rw.lat = stod(strs[2]);
                rw.lon = stod(strs[3]);

                // temp version (you have to fix bologna_post_fix)
                // id_act = stoll(strs[1]);
                // rw.itime = stol(strs[2]);
                // rw.lat = stod(strs[3]);
                // rw.lon = stod(strs[4]);
                //        cout << "start time\t" << config_.start_time <<endl;
                //        cout << "end time\t" << config_.end_time <<endl;
                //        cout << "id act\t" << id_act;
                //        cout << "lat \t"<<rw.lat <<"\t lon \t"  <<rw.lon << endl;
                if ((rw.lat > config_.lat_min && rw.lat < config_.lat_max) && (rw.lon > config_.lon_min && rw.lon < config_.lon_max))
                { // ho appena guardato che il mio  record sia all'interno dei limiti della mia cartografia.
                    //          cout << "time record \t" << rw.itime << endl;
                    if ((rw.itime > config_.start_time) && (rw.itime < config_.end_time))
                    {
                        rw.t = (double)((rw.itime - config_.start_time) / 3600.); // il tempo passato dall'inizio in ore
                        int n_day = int(rw.t / 24.0);
                        if (config_.slice_time.size() != 0)
                        {
                            if (cnt_row % 1000000 == 0)
                                cout << "config slice time\t" << config_.slice_time.size() << endl;
                            if (rw.t <= config_.slice_time[1] + n_day * 24.00 && rw.t > config_.slice_time[0] + n_day * 24.0)
                            {
                                cnt_in++; // sto contando il numero di record che sono ammissibili secondo i parametri del configuration file in un intervallo di tempo [14.00,16.00] all'interno dei giorni che voglio
                                activity_collect[id_act].push_back(rw);
                            }
                            cnt_out++; // sto contando il numero di record che sono ammissibili secondo i parametri del configuration file in un intervallo di tempo [14.00,16.00] al di fuori dei giorni che voglio
                        }
                        else
                        {
                            //              if (cnt_row % 1000000 == 0) cout<< "id act \t" << id_act << "\n timestamp \t" << rw.itime << endl;
                            //              cout << "id act \t" << id_act << "\n timestamp \t" << rw.itime << endl;
                            activity_collect[id_act].push_back(rw);

                        }
                    }
                }
                if (cnt_row % 1000000 == 0)
                    std::cout << "cnt_row:    " << cnt_row << std::endl;
                //        std::cout << "act\t " << id_act << "\ntime\t" << rw.itime<< "\nlat,lon\t" << rw.lat<< "," << rw.itime << std::endl;}
            }
        }
        fp_in.close();

        std::cout << "**********************************" << std::endl;
        if (config_.slice_time.size() != 0)
            std::cout << "slice time:  " << cnt_in << "/" << cnt_out << std::endl;

        activity_base aw;
        for (const auto &n : activity_collect)
        {
            aw.id_act = n.first;
            aw.record.reserve(MAX_RECORD);
            if(n.second.size()>MAX_RECORD)
                {cout <<"excluding a trajectory with more than 12000"<<endl;}
            else
            {
            aw.record = n.second;
            activity.push_back(aw);
            if (aw.record.size() > max_size_records)
                max_size_records = aw.record.size();}
        }
    }
    cout << "Activity: " << int(activity.size()) << endl;
    cout << "maximal size records activity: " << max_size_records << endl;
    if (max_size_records>MAX_RECORD) cout << "change record.reserve() in carto.h" <<endl;


} // ho inizializzato le mie attività direttamente dal file di configurazione. La mia id ha tanti record quanti sono quelli che ho trovato senza ancora distinguere  quelli di stop e quelli di attività vera.

//------------------------------------------------------------------------------------------------------
void load_polygon(std::vector<polygon_base> &polygon,config &config_)
{
    if (config_.file_polygons == "file_polygons")
        return;
    int n_polygon = 0;
    std::ifstream polyifs(config_.file_polygons);
    auto jpoly = jsoncons::json::parse(polyifs);
    for (const auto &feature : jpoly["features"].array_range())
    {
        if (!feature["geometry"]["coordinates"].size())
            continue;
        auto type = feature["geometry"]["type"].as<std::string>();
        if (type == "LineString")
        {
            ++n_polygon;
            auto pol = feature["geometry"]["coordinates"];
            polygon_base pw = polygon_base(pol);
            for (const auto &pro : feature["properties"].object_range())
                pw.pro[std::string(pro.key())] = pro.value().as<std::string>();
            pw.id = n_polygon;
            if (pw.pro["name"] == "center")
                pw.tag_type = 1;
            else if (pw.pro["name"].find("coast") == 0)
                pw.tag_type = 2;
            // else if (pw.pro["name"].find("BorgoSanGiuliano") == 0) pw.tag_type = 4;
            else if (pw.pro["name"].find("station") == 0)
                pw.tag_type = 4;
            else
                pw.tag_type = 3;
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
//------------------------------------------------------------------------------------------------------
vector<string> subnet_label;
void load_subnet(config &config_,map<string, vector<int>> &subnets)
{
    ifstream sub(config_.file_subnet);
    if (!sub)
    {
        cout << "load_data: error in reading: " << config_.file_subnet << endl;
        return;
    }

    subnets.clear();

    string line;
    vector<string> tokens;
    while (getline(sub, line))
    {
        physycom::split(tokens, line, string("\t"), physycom::token_compress_on);
        for (int i = 1; i < (int)tokens.size(); ++i)
            subnets[tokens[0]].push_back(stoi(tokens[i]));
    }

    for (auto &p : subnets)
    {
        subnet_label.push_back(p.first);
        sort(p.second.begin(), p.second.end());
        p.second.erase(unique(p.second.begin(), p.second.end()), p.second.end());
    }
}
