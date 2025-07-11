# city-pro
    This project is a C++ (`processing`), python (`postprocessing`) project whose goal is to analyze mobile phone data in a format that is compatible with Telecom format.
    It is split in a first Cpp part responsible of the computation of the quantities of interest about trajectories and road network and a second part in python responsible for the plotting and analysis of the computed quantities. 
    Notes to set up the project correctly, look at the required input and make sure to have the right directory structure. Use SetRightDirectoriesConfiguration.py to understand how.
# Build & Launch
The project works basing itself on powershell for mantainance and portability pourposes. If you use Linux or Mac make sure to install the powershell.

## Linux
```
sudo apt-get update   
sudo apt-get install -y wget apt-transport-https software-properties-common    
source /etc/os-release    
wget -q https://packages.microsoft.com/config/ubuntu/$VERSION_ID/packages-microsoft-prod.deb    
sudo dpkg -i packages-microsoft-prod.deb   
rm packages-microsoft-prod.deb   
sudo apt-get update   
sudo apt-get install -y powershell   
pwsh   
git clone https://github.com/microsoft/vcpkg   
```

#### Build
```
cd WORKSPACE/city-pro  
git submodule update  --init --recursive
./ccm/build.ps1 -UseVCPKG -DisableInteractive -DoNotUpdateTOOL -DoNotDeleteBuildFolder
```    
NOTE:
Cmake >=3.19


#### Build (if the terminal is opened with conda already activated)
```
rm -rf /home/aamad/codice/city-pro/vcpkg/buildtrees/*   
rm -rf $WORKSPACE/city-pro/vcpkg/packages/*   
rm -rf $WORKSPACE/city-pro/build_release   
rm -rf $WORKSPACE/city-pro/vcpkg_installed   
rm -rf ~/.cache/vcpkg/archives/*    
conda deactivate   
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL _CONDA_ROOT _CONDA_EXE LD_LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS CMAKE_PREFIX_PATH   
./ccm/build.ps1 -UseVCPKG -DisableInteractive -DoNotUpdateTOOL -DoNotDeleteBuildFolder    
```

## MacOs
```
brew install powershell/tap/powershell  
git submodule update   
./ccm/build.ps1 -UseVCPKG -DisableInteractive -DoNotUpdateTOOL -DoNotDeleteBuildFolder
```

#### Setting Python Environment
```
./conda env create -f geostuff.yml
```    


# Launch
As explained in section `city-pro`, there are two stages (`processing`,`postprocessing`) to the project, the Cpp and the python. They can be `launched separately` (WORSE in author's opinion), but also there is a way to launch `Complete analysis`.   
The difference is that the second allows you to launch the cpp in parallel for all the days and the python after that without needing to care aout intermediate steps. 


NOTE: Have a look at the `input` section since there are described the configuration files and input that are needed for each step of the analysis.    
NOTE:  
## Launch Separately 
1a) The Cpp analysis can be launched one day at a time. The command to launch `processing` it is:  

```
./ccm/build.ps1 -UseVCPKG -DisableInteractive -DoNotUpdateTOOL -DoNotDeleteBuildFolder

/city-pro/bin/city-pro /path/to/configfile/configfile.json
```     
1b) Alternatively it is available a bash script to run all the days according to the days at hand:    

```
python3 ./python/work_mdt/SetRightDirectoriesConfiguration.py -c /path/to/config_days_bbox   

./vars/config/RunRimini.sh
```

2) The python analysis, `postprocessing` the :   

```
python3 ./python/work_mdt/AnalysisPaper.py -c ./vars/config
```    
NOTE: `/path/to/configfilePython` = `/WORKSPACE/city-pro/vars/config`     
NOTE: `/path/to/configfile/` is usually in a different folder.       
NOTE: The logic for storing and initialize configuration files is not homogeneous 
## Complete Analysis
```
python3 ./python/LaunchParallelCpp.py -cs ./vars/config/config_days_bbox.json -ca ./vars/config
```   
The script automatically will set the configuration files for each day by calling:    
1) `SetRightDirectoriesConfiguration.py`: This script has hardcoded data that define where the input is and where the output of both the cpp and python will be:     
        1a) Example ../output/bologna_mdt_center -> is the **{basename}** for Cpp    
        1b) Example ../output/bologna_mdt_center/Day -> is the **{basename}** the python     

2) `/city-pro/bin/city-pro /path/to/configfile/configfile.json` for each of the configuration files generated previously, they are run all in parallel exploiting multiprocessing (since datasets outputs are indipendent and no risk of race condition is raised)

3) `python3 ./python/work_mdt/AnalysisPaper.py -c ./vars/config`: Launches the python analysis for traffic patterns, and behavioral patterns.

## Complete Analysis (supplement .ipynb)
`FittingProcedures.ipynb`: is the script used for the fit of time, length, speed distribution. Is messy and not completely standardized with parameters added by hand. This is due to the variability of what you look for.    
`Trajectories.ipynb`: Explore trajectories. Launch just after the simulations of the days are run. It is not in the pipeline as it is conceived as exploratory analysis (Visualizations mainly, speed evolution for single traj) and not completely standardized. This could be expanded to study variance effects.    
`EstimatePenetration.ipynb`: Responsible for penetration.png 

# Input:     
REQUIRED:  
    1Ca `/path/to/configfileCpp/config.json`    
    1Cb `/path/to/configfilePython/AnalysisPython.json`  
    1Ia `/path/to/carto/cartography.pnt`    
    1Ib `/path/to/carto/cartography.pro`    
    1Da `/path/to/data/DatiTelecomPreProcessed.csv`    
    1Db`/path/to/data/DatiTelecomToPreprocess.gzip`

## Cpp
`processing` needs in input:  
    1) `/path/to/carto/cartography.pnt`    
    2) `/path/to/carto/cartography.pro`    
    3) `/path/to/data/DatiTelecomPreProcessed.csv`    
    4) `/path/to/configfileCpp/config.json`     

## Python
`postprocessing` needs in input:  
    1) `/path/to/configfilePython/AnalysisPython.json`    
NOTE: `AnalysisPython.json` is required to be it as it is hardcoded in the complete analysis.



# Description

## Description Input:
##    1C 
### Configuration file  Cpp 

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


### Configuration file Python     
  - `StrDates`: List Dates dataset. Example ["2022-12-30","2022-12-31",...], format %Y-%m-%d
  - `holidays`: List Dates format %Y-%m-%d
  - `not_holidays`: List Dates format %Y-%m-%d
  - `base_name`: **{basename}**
  - `InputBaseDir`: `cartout_basename` 
  - `bounding_box`: Coordinates to cut the cartography and have analysis consistent with Cpp {`lat_min`: 44.487106,`lat_max`: 44.528131,`lon_min`: 11.293156,`lon_max`: 11.378143},
  - `geojson`: Complete name (with directory) in which geojson produced from Cpp of the road network is. Example `../bologna-provincia.geojson`
  - `verbose`: Variable for verbosity (DEPRECATED), keep true
  - `shift_bin`: Ad Hoc for Plots Fit: {"av_speed": 3,"speed_kmh": 0.5,"lenght": 40,"lenght_km": 0.5,"time": 30,"time_hours": 0.5,"av_accel": 0.1}
  - `shift_count`: Ad Hoc for Plots Fit: {"av_speed": 50,"speed_kmh": 50,"lenght": 50,"lenght_km": 50,"time": 50,"time_hours": 50,"av_accel": 50},
  - `interval_bin`: Ad Hoc for Plots Fit: {"av_speed": 10,"speed_kmh": 10,"lenght": 10,"lenght_km": 10,"time": 10,"time_hours": 10,"av_accel": 0.2},
  - `interval_count`: Ad Hoc for Plots Fit: {"av_speed": 300,"speed_kmh": 300,"lenght": 300,"lenght_km": 300,"time": 300,"time_hours": 300,"av_accel": 500},
  - `scale_count`: Ad Hoc for Plots Fit: {"av_speed": "linear","speed_kmh": "linear","lenght": "log","lenght_km": "log","time": "log","time_hours": "log","av_accel": "linear"},
  - `scale_bins`: Ad Hoc for Plots Fit: {"av_speed": "linear","speed_kmh": "linear","lenght": "log","lenght_km": "log","time": "log","time_hours": "log","av_accel": "linear"},
  - `info_fit`: Ad Hoc for Plots Fit: {Computed Automatically}

## 1I 
### Cartography in physycom format.  
- (`cartography.pnt`,`cartography.pro`):  
            Contain all informations needed to build the road network in such a way that the program is able
            to read these informations from them.
- `cartography.pnt`:   
            Contains informations about where the points of the road are.
- `cartography.pro`:  
            Contains informations about links.

To produce them: follow instructions in `$WORKSPACE/cartography-data`

## 1D
### Data
`DatiTelecomToPreprocess.gzip` contains `[iD,lat,lon,time]`, the `DatiTelecomAlreadyPreprocessed.csv` too.    
The first has been preprocessed into the second. `DatiTelecomAlreadyPreprocessed.csv` is the one used.
NOTE:  Use: `python3 ./python/mdt_converter.py` (and change parameters there), to transform the first into the second.
If you have already the Preprocessed.csv, better for you.
NOTE:   insert manuallly the dates in LIST_START, LIST_END depending on the dates you have and ensure that the file directories match the structure in your machine.
NOTE: Since this script was don at the beginning, it should work, but was not thought to be fitting in the pipeline automatically.

###### SUMMARY:    

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







# OUTPUT:
The output of the program is presented here below and briefly explained separately for Cpp and Python.
In both cases we have outputs related to trajectories and network.   
In the case of .cpp the output is relative to 1 day, while in the case of python the output is available either for day and aggregated over many days.
## CPP
### Network

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

### Trajectories
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



## PYTHON (To be done)
The Output of Python is stored in `../output/plots` and it is composed of `single_day` and `aggregated` output.
The `single_day` output consists of quantities computed on the single day and put in the relative `../output/plots/day` directory, while the aggregated files will be held on `../output/plots` directory and will have either averaged or with all the aggregated days.
### single_day   
Found in `../output/plots/day`    
#### Trajectories    
1. `BinTime.csv`:    
Description:    
    Contains the time variables in intervals of `bin_time`    
Columns:     
    `timestamp`,`day_hour`,`hour`    
2. `Class2TimeDeparture2UserId.json`:    
Description:    
    Dictionary {`Class`:{`timestamp`:[`id_act`]}}, for each mobility `Class` computed via Fuzzy algorithm, and each
    `timestamp` separated by `bin_time` contains the list of `id_act` of the trajectories. It is useful to compute
    the fundamental diagram and keep knowledge about fluxes and presences in a road network in some interval of time.

 

#### Network     
1. `Class2TimeInterval2Road2Speed.json`:
Description:    
    Dictionary {`Class`:{`timestamp`:{`poly_id`:`speed`}}}, for each mobility `Class` computed via **Fuzzy** algorithm, and each `timestamp` separated by `bin_time` contains the list of `poly_id` and their respective speeds in km/h. It is useful to compute the speed evolution of the road network over time.        
Goal:   
    Assign speed to each road with `Criterion_1`: average speed of all users that are classified with **Fuzzy** classification.     
2. `Class2TimeInterval2Road2SpeedNew.json`:
Description:    
    Dictionary {`Class`:{`timestamp`:{`poly_id`:`speed`}}}, for each mobility `Class` computed via **Hierarchical** reassignment algorithm, and each `timestamp` separated by `bin_time` contains the list of `poly_id` and their respective speeds in km/h. It is useful to compute the speed evolution of the road network over time.         
Goal:   
    Assign speed to each road with `Criterion_2`: average speed of all users that are classified with **Hierarchical** classification.    

3. `ClassNew2TimeInterval2Road2SpeedActualRoads.json`:   
Description:
    Dictionary {`ClassNew`:{`TimeInt`:{`RoadInClassNew`:`SpeedAllUsers`}} }, for each mobility `ClassNew` computed via  **Hierarchical** reassignment algorithm.    
    for each bin of size `bin_time` we associate all the roads `RoadInClassNew` and the speed computed as the average speed over all the user that have passed in that time interval in that road.    
Goal:   
    Assign speed to each road with `Criterion_3`: average speed of all users that have passed there.    


4. `Class2TotalLengthOrderedSubnet.json`:
Description:    
    Dictionary {`Class`: lenght }, for each mobility `Class` computed via **Hierarchical** reassignment algorithm, `length` of the sub-network as the sum of the length of the roads that form it.

5. `Class2TraffickIndex.json`:
Description:    
    Dictionary {`Class`: [`TraffickIndex`] }, for each mobility `Class` computed via  **Hierarchical** reassignment algorithm.    
    `TraffickIndex` is a vector for each bin of size `bin_time`, and it represents the number of people in the sub-network over the maximum measured there times the difference in speed at that time measured in the  **Fuzzy** -**Hierarchical** over **Fuzzy**

6. `HisteresisInfo_{Day}.csv`:    
Description:    
    pl.DataFrame -> Columns:
     `time`,`population_3`,`speed_kmh_{Class}`,`population_{Class}`,`new_speed_kmh_{Class}`.    
    For each mobility `Class` each column is a vector such that each entrance is separated by `bin_time`, the speed are computed on the subsets generated by **Hierarchical**(new) and **Fuzzy** partitions on trajectories. The population the same.

7. `DfSpeed.parquet`:
    pl.DataFrame -> Columns:
    `Class`,`Day`,`av_speed_kmh_fuzzy`,`av_speed_kmh_hierarchical`,`av_speed_kmh_all`
    For each class I compute the speed of the road.
    0 if not there.
### aggregated    
Found in `../output/plots/`    
#### Trajectories    
1. `aggregated_fit_parameters_`**length_km**`_exponential_new.csv`    
    Description:    
        Contains the parameters of the exponential fit for the trajectories belonging to **Class** of a given **Day**.
    Columns:
        `Day,A,1/x0,<x>,class`
1. `aggregated_fit_parameters_`**length_km**`_powerlaw_new.csv`    
    Description:    
        Contains the parameters of the powerlaw fit for the trajectories belonging to **Class** of a given **Day**.
    Columns:
        `Day,A,alpha,class`
     
#### Network        
1. `LinearCoeff_NewClass.csv`:    
    Description:
        Contains the linear coefficient of the MFD (x = number people in **Class**, y = average speed **Hierarchical** sub-network). They are computed by making the histogram of the vector of speed of the sub-network of a given day aggregated with granularity = `bin_time` (15 min).    
    Columns:    
        `Days,LinearCoeff`



##### COMMENT ON FUZZY ALGORITHM:
Needs to be tuned, try different `num_tm` (3 for Bologna + slow re-classification). Increasing the number does not uncover the slow mobility (walkers,bikers), but it finds subgroups on higher velocity group.
This bias is probably due to the sensitivity of the algorithm to the speed, giving more weight in for the separation for classes that have higher velocity.  
##### COMMENT ON CPU USAGE
`city-pro` utilizes for input file of around 1 GB around 20 GB of RAM.
`Analysis_Paper.py` utilizes for the analysis in parallel of 6 days around 16 GB of RAM.


# FOR DEVELOPERS
```std::vector<poly_base> poly``` is initialized with a null element in the position 0. Pay attention to that.
Or modify.

In ```make_subnet``` is put by hand the maximum length for a poly extracted from the geojson via geopandas (5762 m).
For own cartography the parameter needs to be changed. 

# NOTES
In the case you cannot build with fltk beacouse:
```
-- Running vcpkg install - failed CMake Error at vcpkg/scripts/buildsystems/vcpkg.cmake:904 (message): vcpkg install failed. See logs for more information: /home/aamad/codice/city-pro/build_release/vcpkg-manifest-install.log Call Stack (most recent call first): /usr/share/cmake-3.21/Modules/CMakeDetermineSystem.cmake:124 (include) CMakeLists.txt:36 (project)

CMake Error: CMake was unable to find a build program corresponding to "Ninja". CMAKE_MAKE_PROGRAM is not set. You probably need to select a different build tool. CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage -- Configuring incomplete, errors occurred! Config failed! Exited with error code 1. Exception: ScriptHalted
```
On shell

```export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
cd ${WORKSPACE}/city-pro/vcpkg
./bootstrap-vcpkg.sh
```