 	Sorgente esperimento, do il nome di source perchè così non devo 
modificare il make ogni volta.
NOTA:
Gli script sono fatti per analizzare un giorno alla volta. La struttura delle cartelle rispecchia questo. Per ogni giorno analizzato ho una cartella in work_geo/bologna_mdt_detailed e output/bologna_mdt_detailed

Description:
Telecom gives initial dataset day by day with a lot of fields and zipped. I have created mdt_converter.py that essentially takes the dataset, and extract [iD,lat,lon,time] and saves it in a csv that will be given to analysis.cpp



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
