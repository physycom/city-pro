# city-pro
Goal:
    This project is a C++ project whose goal is multiple and can be summed up in these follwoing steps.
    1) READING:
        1a) Trajectory information:
            Reads .csv files containing information about mobilephone data containing the following columns:
            [iD,lat,lon,time]
            NOTE: Usually, we need to create this dataset. TIM gives another format of data that we need to preprocess and create these columns.
            Further informations about preprocessing needed...
        1b) Cartography information:
            It creates a graph of the city from cartography.pnt and .pro. Essentially these are used to create objects that are contained
            in carto.h. Whose attributes and functions are initialized in carto.cpp
    2) EXTRACT TRAJECTORIES:
        



Input: (Described in Description)
    REQUIRED:
        I1) cartography.pnt, cartography.pro
        R1) DatiTelecom.csv


Description:
    I1)
        To produce these files:
            WORKSPACE/cartography-data/
        - (cartography.pnt,cartography.pro):
            Contain all informations needed to build the road network in such a way that the program is able
            to read these informations from them.
        - cartography.pnt: 
            Contains informations about where the points of the road are.
        - cartography.pro:
            Contains informations about links.

Sorgente esperimento, do il nome di source perchè così non devo
modificare il make ogni volta.
NOTA:
Gli script sono fatti per analizzare un giorno alla volta. La struttura delle cartelle rispecchia questo. Per ogni giorno analizzato ho una cartella in work_geo/bologna_mdt_detailed e output/bologna_mdt_detailed

Description:
Telecom gives initial dataset day by day with a lot of fields and zipped. I have created mdt_converter.py that essentially takes the dataset, and extract [iD,lat,lon,time] and saves it in a csv that will be given to analysis.cpp

Description:
    R1)

    I1)

ANALISI DATI BOLOGNA MDT:
    cd /codice/city-pro
1----------------------------------------------- PREPROCESSING PYTHON
Input:
    /path/to/gzip/files = ([G:/bologna_mdt/dir1,...,G:/bologna_mdt/dirn]) or (/nas/homes/albertoamaduzzi/dati_mdt_bologna/)
Launch:
    python3 ./python/mdt_converter.py
NOTE: insert manuallly the dates in LIST_START, LIST_END depending on the dates you have.
Output:
    /path/to/raw/files = [/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data] [file_raw_data1,...,file_raw_datan] -> STRUCTURE: [id_user,timestamp,lat,lon]

2----------------------------------------------- PROCESSING CPP (FOR EACH) SINGLE DAY
Input: raw_file in raw_files
    ./bin/city-pro ./work_geo/bologna_mdt_detailed/date/config_bologna.json ----- ./launch_all_analysis.sh
Output:
    subnet file: .fluxes.sub
    (class_i)_complete_complement.csv
    presence.csv
    timed_fluxes.csv
    stats.csv
    fcm.csv

3------------------------------------------------ POSTPROCESSING PYTHON (FOR EACH) SINGLE DAY
3a.
    ./python/config_subnet_create.py (README in the file)
Output:
    all_subnets.sh
    work_geo/bologna_mdt_detailed/date/plot_subnet
--------------------------------------------------
3b. analysis.ipynb (non è il top affatto)
Bisogna inserire manualmente gli indirizzi dove è salvata la roba nella prima cella. Fatto questo si possono runnare le altre celle.
Poi lanciare cella per cella:
Input:
    fcm.csv
    stats.csv
    timed_fluxes.csv
Output:
    distribuzione velocità per ogni classe
    distribuzione lunghezze e tempi per ogni classe
    fondamental diagram per ogni classe

4-------------------------------------------------- POSTPROCESSING AGGREGATION PYTHON
COMMAND:
    python3 fondamental_diagram_aggregated.py -c config_fundamental_diagram_aggregated.json
Input:
    class_i_velocity_subnet.csv
    _fcm.csv


--------------------------------------------------------------------------------------------------
OVERVIEW OUTPUT:
NETWORK INFORMATIONS:
1)
..._class_i_velocity_subnet.csv (as many files as many classes usually 4-5)
    start_bin;end_bin;poly_id;number_people_poly;total_number_people;av_speed;time_percorrence
Description:
    Information about network via poly_id in time for the subnet i. i belongs to {0,..,num_tm} and increases by velocity of percurrence
2)...class_i.txt (as many files as many classes usually 4-5)
Description:
    "Space separated" poly ids of the subnet of class i
3)
...iclass_subnet.txt (as many files as many classes usually 4-5)
Description:
    "Space separated" poly ids of the subnet of class i that is freed from the network of higher velocity.
    In this way we have a "hierarchy" of subnetwork, that is, if I consider a poly that is contained in multiple subnetwork
    it will be assigned to the quickest subnet. -> This hopefully will help us find traffic via fondamental diagram.


TRAJECTORY INFORMATIONS:

1)
..._presence.csv
    id_act;timestart;timeend;lat;lon
Description:
    Autoexplicative
2)
..._fcm_centers.csv
Description:
    Contains informations about the centers in the feature space.
    Are ordered from slowest to quickest.
3)
..._fcm.csv
    id_act;lenght;time;av_speed;v_max;v_min;cnt;av_accel;a_max;class;p
Description:
    identification, lenght of trajectory, time the trajectory lasted, average_speed, v_min,max, number of points in trajectories, class, and probability of being in that class.

4)
...fcm_new.csv
    id_act;class;0;1;2;3;4
Description:
    The id of traj, the class that is reassigned to, according to the principle, the subnet of the class that contains more points of the trajectory, gives the class.
    So, if a person is moving slowly in the just quick subnet, than, it is reassigned to the quickest class.
    The columns, 0,... are associated to the hierarchical subnets
5)

..._out_features.csv
    id_act;average_speed;v_max;v_min;sinuosity
Description:
    For each trajectory have the informations about the features of the classes
