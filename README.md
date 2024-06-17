# city-pro
    This project is a C++,python project whose goal is to analyze mobile phone data in a format that is compatible with Telecom format.
    It is split in a first Cpp part responsible of the computation of the quantities of interest about trajectories and road network and a second part in python responsible for the plotting and analysis of the computed quantities. 
    
# Build & Launch
The project works basing itself on powershell. If you use Linux or Mac make sure to install the powershell.

## Linux
```
# Update the list of packages
sudo apt-get update

# Install pre-requisite packages.
sudo apt-get install -y wget apt-transport-https software-properties-common

# Get the version of Ubuntu
source /etc/os-release

# Download the Microsoft repository keys
wget -q https://packages.microsoft.com/config/ubuntu/$VERSION_ID/packages-microsoft-prod.deb

# Register the Microsoft repository keys
sudo dpkg -i packages-microsoft-prod.deb

# Delete the Microsoft repository keys file
rm packages-microsoft-prod.deb

# Update the list of packages after we added packages.microsoft.com
sudo apt-get update

###################################
# Install PowerShell
sudo apt-get install -y powershell

# Start PowerShell
pwsh
git clone https://github.com/microsoft/vcpkg
```

#### Build
```cd WORKSPACE/city-pro```  
```git submodule update```  
```./ccm/build.ps1 -UseVCPKG -DisableInteractive -DoNotUpdateTOOL -DoNotDeleteBuildFolder```
#### Launch
```/city-pro/bin/city-pro /path/to/configfile/configfile.json```

## MacOs
```brew install powershell/tap/powershell```  
```git submodule update```
#### Build

```./ccm/build.ps1 -UseVCPKG -DisableInteractive -DoNotUpdateTOOL -DoNotDeleteBuildFolder```
#### Launch
```/city-pro/bin/city-pro /path/to/configfile/configfile.json```
# Input:     
REQUIRED:  
    
    1C config.json  
    1I cartography.pnt, cartography.pro
    1D DatiTelecomPreProcessed.csv or DatiTelecomToPreprocess.gzip 


## Description Input:
####    1C 
##### Configuration file   

-   `file_pro`: `/path/to/cartography.pro`
-   `file_pnt`:  `/path/to/cartography.pnt`
-   `file_data`:  [`/path/to/DatiTelecomPreProcessed.csv`] NOTE: It is a list
-   `cartout_basename`: `/path/to/save/output/dir` 
-   `start_time`:  `YY-MM-DD h:m:s`
-   `end_time`:  `YY-MM-DD h:m:s`
-   `bin_time`:  `15.0`
-   `lat_min,lat_max,lon_min,lon_max`: bounding box vertices  
-   `map_resolution`: `60`
-   `grid_resolution`: `150` (m), for searching   algorithms points,poly,arcs ecc...  
-   `l_gauss`: `10`  
-   `min_data_distance`: `50` (m), threshold distance between a `record_base` and a `cluster_base.centroid` to create another `cluster_base` object when filtering trajectories.
-   `max_inst_speed`: `50` (m\s), maximum speed not to consider `record` as an error and not discard it.
-   `min_node_distance`: `10` (m), threshold for two nodes not to be the same. (Not used here, but in other parts of the code base cartography-data, miniedit)
-   `min_poly_distance`: `50` (m), threshold for two poly not to be the same. (Not used here, but in other parts of the code base cartography-data, miniedit) 
-   `enable_threshold`: `true`  
-   `threshold_v` : `50.0`  
-   `threshold_t` : `86400`
-   `threshold_n` : `3`  
-   `enable_multimodality`: `true` Enable Fuzzy algorithm for classification of homogeneous trajectories
-   `enable_slow_classification`: `true` Used to separate the slowest category that usually does not separate walkers and bikers.
-   `num_tm`: `3` number of classes that you want to distinguish.
-   `threshold_p`: `0.5` threshold on the probability for one trajectory to belong to one cluster. If less then 0.5 then it belongs to class `10` (unclassified)
-   `dump_dt`: `60` 
-   `enable_fluxes_print`: `true` Enable output: **{basename}**`.fluxes`  
-   `enable_subnet`: `true` Enable output: **{basename}**`.fluxes.sub`
-   `show_subnet`: `true`
-   `file_subnet`: `/path/to/subnet/{basename}.fluxes.sub`  
-    `multimodality_subnet`: `true` 
-    `num_tm_subnet`: `3`
-    `enable_print`: `true` For `_stats.csv` Deprecated
-    `enable_geojson`: `false`  Uses geojson
-    `enable_gui`: `true`   Activate gui
-    `jump2subnet_analysis`:`false` Does not recalculate the subclass but read them for the construction of the subnetworks 

#### 1I 
##### Cartography in physycom format.  
- (`cartography.pnt`,`cartography.pro`):  
            Contain all informations needed to build the road network in such a way that the program is able
            to read these informations from them.
- `cartography.pnt`:   
            Contains informations about where the points of the road are.
- `cartography.pro`:  
            Contains informations about links.

#### 1D
##### Data
`DatiTelecomToPreprocess.gzip` contains `[iD,lat,lon,time]`, the `DatiTelecomAlreadyPreprocessed.csv` too.




# USAGE: 
    I1: 
    Run the preliminary preprocessing of data to construct the format required for city-pro analysis.

# REQUIRED INPUT

1. Produce .pnt .pro (follow instructions in `$WORKSPACE/cartography-data`)
2.  `cd $WORKSPACE/city-pro`
3. If `DatiTelecomPreProcessed.csv` exists:  
`Do nothing`  
else:  
`python3 ./python/mdt_converter.py`  
NOTE:   insert manuallly the dates in LIST_START, LIST_END depending on the dates you have and ensure that the file directories match the structure in your machine.

#### DESCRIPTION I/O mdt_converter.py
Input:  
    `/path/to/gzip/files` = [`../dir1`,...,`../dirn`] for those who have access are in (`/nas/homes/albertoamaduzzi/dati_mdt_bologna/`)  
Output:
    `/path/to/raw/files` = [`/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data`] [`file_raw_data1,...,file_raw_datan`]   
    Columns:   
    [`id_user,timestamp,lat,lon`]

## CPP
Launch all together:
```
cd WORKSPACE/city-pro
python3 ./python/SetRightDirectoriesConfiguration.py
./config/RunRimini.sh
```
Or 
Input:  
```  
./ccm/build.ps1 -UseVCPKG -DisableInteractive -DoNotUpdateTOOL -DoNotDeleteBuildFolder

./bin/city-pro ./work_geo/bologna_mdt_detailed/date/config_bologna.json
```
# Output:


## Network

1. **{basename}**_class_`i`_velocity_subnet.csv:   
Description:  
Contains informations about the `velocity` and `time percorrence` in time intervals `[start_bin,end_bin]` of poly `poly_id` of the subenetwork of fcm index `i`.   
Columns:  
    `start_bin;end_bin;poly_id;number_people_poly;total_number_people;av_speed;time_percorrence`  

2. **{basename}**...class_`i`.txt 
Description:  
    "Space separated" poly ids of the subnet of class `i`.  
    i.e. 1 2 10 12 16 ...  

3. **{basename}**`i`class_subnet.txt   
Description:  
    "Space separated" poly ids of the subnet of class i that is freed from the network of higher velocity.
    In this way we have a "hierarchy" of subnetwork, that is, if I consider a poly that is contained in multiple subnetwork
    it will be assigned to the quickest subnet. -> This hopefully will help us find traffic via fondamental diagram.

## Trajectories
1. **{basename}**_presence.csv  
    Description:  
    Contains information about all trajectories `id_act` that have just one `stop_point` for the time window  `[timestart,timeend]` at `(lat,lon)`.   
    Columns:  
    `id_act;timestart;timeend;lat;lon`

2. **{basename}**_fcm_centers.csv  
Description:
    Contains informations about the centers in the feature space coming out from the Fuzzy algorithm for clustering of the trajectories.  
    *NO COLUMN NAMES*:  
    `class`;`av_speed`;`vmin`;`vmax`;`sinuosity`  
    Data are ordered by class from slowest (top) to quickest (bottom).  

3. **{basename}**_fcm.csv
Description:
    Contains information about, `lenght` of trajectories `id_act`, duration `time`, average speed `av_speed`, minimum velocity registered `v_min`, maximum velocity registered `v_max`, number of points `cnt`, `class` (output from Fuzzy clustering algorithm), and probability of being in that class `p`,active in the time window `[start_time,end_time]`.  
    Columns:  
        `id_act;lenght;time;av_speed;v_max;v_min;cnt;av_accel;a_max;class;p;start_time;end_time`


4. **{basename}**fcm_new.csv:  
Description:
    Contains information about id of traj `id_act`, the class that is reassigned to `class`, according to the principle, the subnet of the class that contains more points of the trajectory, gives the class.
    So, if a person is moving slowly in the just quick subnet, than, it is reassigned to the quickest class.
    The columns, 0,... are associated to the hierarchical subnets  
Columns:  
`id_act;class;0;1;2;3;4`
5. **{basename}**_out_features.csv  
Description:  
    For each trajectory have the informations about the features of the classes
Columns:  
    `id_act;average_speed;v_max;v_min;sinuosity`
# Structure Program:
    1. READING:  
        1a.   
            Trajectory information:
            Reads .csv files containing information about mobilephone data containing the following columns:
            [iD,lat,lon,time]
            NOTE: Usually, we need to create this dataset. TIM gives another format of data that we need to preprocess and create these columns.
            Further informations about preprocessing needed...  
        1b.   
            Cartography information:
            It creates a graph of the city from cartography.pnt and .pro. Essentially these are used to create objects that are contained
            in carto.h. Whose attributes and functions are initialized in carto.cpp
    2. EXTRACT TRAJECTORIES:
### SUMMARY 
    This script in particular is able to:
    1. generate trajectories from single records, discarding GPS errors by thresholding on the maximum velocity.
    2. Associate the roads they pass by
    3. Cluster them according to a FuzzyKMean
    4.


NOTA:
Gli script sono fatti per analizzare un giorno alla volta. La struttura delle cartelle rispecchia questo. Per ogni giorno analizzato ho una cartella in work_geo/bologna_mdt_detailed e output/bologna_mdt_detailed

Description:
Telecom gives initial dataset day by day with a lot of fields and zipped. I have created mdt_converter.py that essentially takes the dataset, and extract [iD,lat,lon,time] and saves it in a csv that will be given to analysis.cpp

##### COMMENT ON FUZZY ALGORITHM:
Needs to be tuned, try different `num_tm` (3 or 4 for Bologna depending on the day). Increasing the number does not uncover the slow mobility (walkers,bikers), but it finds subgroups on higher velocity group.
This bias is probably due to the sensitivity of the algorithm to the speed, giving more weight in for the separation for classes that have higher velocity.  
# FOR DEVELOPERS
```std::vector<poly_base> poly``` is initialized with a null element in the position 0. Pay attention to that.
Or modify.

In ```make_subnet``` is put by hand the maximum length for a poly extracted from the geojson via geopandas.
For own cartography the parameter needs to be changed. 

# LAUNCH ANALYSIS (WORK IN PROGRESS)

   ``` ./python/config_subnet_create.py ```
(README in the file)  
Output:    
    all_subnets.sh  
    work_geo/bologna_mdt_detailed/date/plot_subnet  

AnalysisPaper.ipynb (non è il top affatto)  
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


#### POSTPROCESSING AGGREGATION PYTHON
COMMAND:
    python3 fondamental_diagram_aggregated.py -c config_fundamental_diagram_aggregated.json
Input:
    class_i_velocity_subnet.csv
    _fcm.csv


