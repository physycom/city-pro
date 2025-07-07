"""
    @Description:
        This script initialize the right directories for the configuration files.
        Requirements:
            - WORKSPACE: Path to the ./codice
            - FolderCarto: Path to the carto folder (.pnt,.pro)
            - StrDates: List of dates for which we have DataSet (../BaseName + "_" + StrDate + "_" + StrDate + ".csv")
            - DirConfig2Copy: Path to the configuration file (Given in input to city-pro)
            - Carto: Name of the cartography
    
    NOTE:
        Bounding Box past Study Bologna:
        - bologna_mdt:
            config["lon_min"] = 11.156944 
            config["lon_max"] = 11.490277
            config["lat_min"] = 44.426389 
            config["lat_max"] = 44.586666
        - bologna_mdt_center:
            config["lon_min"] = 11.293156 
            config["lon_max"] =  11.378143
            config["lat_min"] = 44.487106 
            config["lat_max"] = 44.528131
        bologna_mdt_more_center:
            config["lon_min"] = 11.298723 
            config["lon_max"] =  11.370134
            config["lat_min"] =  44.485637
            config["lat_max"] = 44.518693
                        , 
        bologna_mdt_center_2:
            config["lon_min"] = 11.254601
            config["lon_max"] = 11.396050
            config["lat_min"] = 44.479459
            config["lat_max"] = 44.536751




"""
import json
import os
import logging
logger = logging.getLogger(__name__)

def ModifyConfigCpp(WORKSPACE,
                    FolderCarto,
                    BaseFolderData,
                    StrDates,
                    DirConfig2Copy,
                    Carto,
                    BaseName,
                    NewCaseName,
                    lat_min = 44.487106,
                    lat_max = 44.528131,
                    lon_min = 11.293156 ,
                    lon_max = 11.378143,
                    enable_subnet = True,
                    jump2subnet_analysis = False,
                    max_poly_length = 6000,
                    num_tm = 3
                    ):
    """
        @param WORKSPACE: Path to the ./codice
        @param FolderCarto: Path to the carto folder (.pnt,.pro)
        @param BaseFolderData: Path to the folder where we have the DataSet
        @param StrDates: List of dates for which we have DataSet (../BaseName + "_" + StrDate + "_" + StrDate + ".csv")
        @param DirConfig2Copy: Path to the configuration file (Given in input to city-pro)
        @param Carto: Name of the cartography
        @param BaseName: Base Name of the DataSet
        @param NewCaseName: New Case Name for the configuration file
        @param lat_min: Minimum Latitude
        @param lat_max: Maximum Latitude
        @param lon_min: Minimum Longitude
        @param lon_max: Maximum Longitude 
        @Description:
            It saves the configuration for the .cpp in the directory of the given day you want to analyze
    """
    if not os.path.exists(os.path.join(WORKSPACE,"city-pro",NewCaseName)):
        os.mkdir(os.path.join(WORKSPACE,"city-pro",NewCaseName))
    # ----------------- Insert Changes Config Here --------------
    for StrDate in StrDates:
        logger.info(f"Setting right directories for {StrDate}")
        with open(os.path.join(DirConfig2Copy,StrDate,"config_bologna.json"),"r") as f:
            config = json.load(f)
        config["file_pnt"] = os.path.join(FolderCarto,Carto + ".pnt")
        config["file_pro"] = os.path.join(FolderCarto,Carto + ".pro")
        config["file_data"] = [os.path.join(BaseFolderData,StrDate,BaseName + "_" + StrDate + "_" + StrDate + ".csv")]
        # NOTE: Output
        config["cartout_basename"] = os.path.join(WORKSPACE,"city-pro","output",NewCaseName) + "/"
        config["file_subnet"] = os.path.join(WORKSPACE,"city-pro","output",NewCaseName,"weights",NewCaseName + "_" + StrDate + "_" + StrDate + ".fluxes.sub") 
        config["lon_min"] = lon_min
        config["lon_max"] = lon_max
        config["lat_min"] = lat_min 
        config["lat_max"] = lat_max
        config["enable_subnet"] = enable_subnet
        config["jump2subnet_analysis"] = jump2subnet_analysis
        config["max_poly_length"] = max_poly_length
        # NOTE: Status -> Try one more class then the usual analysis to see if the Fcm Clustering separates walkers from bikes.
        config["num_tm"] = num_tm
        # ----------------- Insert Changes Config Here End --------------
        DirNewConfig = os.path.join(WORKSPACE,"city-pro",NewCaseName)
        if not os.path.exists(os.path.join(DirNewConfig,StrDate)):
            os.makedirs(os.path.join(DirNewConfig,StrDate))
        with open(os.path.join(DirNewConfig,StrDate,"config_bologna.json"),"w") as f:
            config = json.dump(config,f,indent=2)
        logger.info(f"Dumped config {StrDate}")
    OutputDir = os.path.join(WORKSPACE,"city-pro","output",NewCaseName,"weights")
    if not os.path.exists(os.path.join(WORKSPACE,"city-pro","output")):
        os.makedirs(os.path.join(WORKSPACE,"city-pro","output"))
    if not os.path.exists(os.path.join(WORKSPACE,"city-pro","output",NewCaseName)):
        os.makedirs(os.path.join(WORKSPACE,"city-pro","output",NewCaseName))
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    return True

def ModifyConfigPy(WORKSPACE,NewCaseName,DirConfig,StrDates,lat_min,lat_max,lon_min,lon_max):
    # NOTE: Load the configuration file for Python Script
    with open(os.path.join(DirConfig,"AnalysisPython.json"),"r") as f:
        ConfigPy = json.load(f)
    ConfigPy["bounding_box"]["lon_min"] = lon_min
    ConfigPy["bounding_box"]["lon_max"] =  lon_max
    ConfigPy["bounding_box"]["lat_min"] = lat_min
    ConfigPy["bounding_box"]["lat_max"] = lat_max
    ConfigPy["InputBaseDir"] = os.path.join(WORKSPACE,"city-pro","output",NewCaseName)
    ConfigPy["StrDates"] = StrDates
    with open(os.path.join(DirConfig,"AnalysisPython.json"),"w") as f:
        json.dump(ConfigPy,f,indent=2)

def InitialSettingProject(FolderCarto,NameCarto,StrDates,BaseFolderData,DirConfig2Copy,NameSeedConfig,NameConfigFile,OutputDir,OutputDirWeights):
    """
        @param FolderCarto: Path to the carto folder (.pnt,.pro)
        @param NameCarto: Name of the cartography
        @param StrDates: List of dates for which we have DataSet (../BaseName + "_" + StrDate + "_" + StrDate + ".csv")
        @param BaseFolderData: Path to the folder where we have the DataSet
        @Description:
            It initializes the directories for the project, checks if all the data required is there, if not it raises an error.

    """
    os.makedirs(FolderCarto,exist_ok=True)
    # Control Cartography
    logger.info(f"Checking Cartography {NameCarto} .pnt .pro Exists")
    if not os.path.isfile(os.path.join(FolderCarto,NameCarto + ".pnt")):
        raise FileNotFoundError(f"File {NameCarto}.pnt not found in {FolderCarto}")
    if not os.path.isfile(os.path.join(FolderCarto,NameCarto + ".pro")):
        raise FileNotFoundError(f"File {NameCarto}.pro not found in {FolderCarto}")
    # Control Data, NOTE: Multiple Dates
    for StrDate in StrDates:
        os.makedirs(os.path.join(BaseFolderData,StrDate),exist_ok=True)
        FileData = os.path.join(BaseFolderData,StrDate,BaseName + "_" + StrDate + "_" + StrDate + ".csv")
        logger.info(f"Checking Trajectory Data for {FileData}")

        if not os.path.isfile(FileData):
            raise FileNotFoundError(f"File Data: {FileData} not found")
        # Check the configuration file exists for each day
        DirectoryDayConfig = os.path.join(DirConfig2Copy,StrDate)
        os.makedirs(DirectoryDayConfig,exist_ok=True)
        FileConfig2Copy = os.path.join(DirConfig2Copy,StrDate,NameConfigFile)
        logger.info(f"Checking Configuration File for {FileConfig2Copy}")
        # If does not exist, copy the seed file that is the same for all the days
        if not os.path.isfile(FileConfig2Copy):
            logger.info(f"{FileConfig2Copy} Does not exist, Copying Seed Configuration File for {FileConfig2Copy}")
            if os.path.isfile(os.path.join(DirConfig2Copy,NameSeedConfig)):
                with open(os.path.join(DirConfig2Copy,NameSeedConfig),"r") as f:
                    config = json.load(f)
                with open(FileConfig2Copy,"w") as f:
                    json.dump(config,f,indent=2)
            else:
                raise FileNotFoundError(f"File Seed Config: {NameSeedConfig} not found in {DirConfig2Copy}")
        else:
            pass
    # Control Output Directory
    os.makedirs(OutputDir,exist_ok=True)
    os.makedirs(OutputDirWeights,exist_ok=True)

    # Control Configuration

# ------------------ INPUT ------------------ #
# NOTE: WORKSPACE is the path to the ./codice 
WORKSPACE = os.environ["WORKSPACE"]
# NOTE: Vars is the common folder for configuration, data and carto
VarsDir = os.path.join(WORKSPACE,"city-pro","vars")
# NOTE: Where you find .pnt,.pro 
FolderCarto = os.path.join(VarsDir,"carto")
# NOTE: Folder where you find the folders of the DataSet (this folder contains a folder for each day in the dataset) NOTE: If you want to divide the dataset into multiple datasets, here it must be changed, adding a name for each sub-dataset. NOTE: It is not automatically handled this case even though it could be useful. NOTE: When analyzed, move the data away not to repeat the analysis.
BaseFolderData = os.path.join(VarsDir,"data")
# NOTE: Dates for which we have DataSet
StrDates = ['2022-12-30','2022-12-31','2023-01-01','2022-05-12','2022-11-11','2022-07-01','2022-08-05','2022-01-31','2023-03-18']
# NOTE: We copy the configuration file from an "original" folder, NOTE: The structure is the same of the DataSet, therefore there is one config_file for each day.
DirConfig2Copy = os.path.join(VarsDir,"config_example")
NameSeedConfig = "config_bologna.json"
NameConfigFile = "config_bologna.json"
# NOTE: Configuration Folder for Python. NOTE: It does not allow to divde your dataset. If the dataset is in VarsDir, it will be considered as a single dataset. Same as BaseFolderData
DirConfigPython = os.path.join(VarsDir,"config")

# NOTE: Name Carotgraphy -> If Download new cartography, change this name
Carto = "bologna_mdt_cleaned"
# NOTE: Base Name DataSet -> If change project, change this name. Do not change it, you really do not care.
BaseName = "bologna_mdt"
# NOTE: Bounding Box, filters both the trajectories (in the cpp) and the sub-network in python
#lat_min = 44.487106
#lat_max = 44.528131
#lon_min = 11.293156
#lon_max = 11.378143
lon_min = 11.254601
lon_max = 11.396050
lat_min = 44.479459
lat_max = 44.536751

# ------------------ CONFIG .cpp ------------------ #
NameCaseAnalysis = "bologna_mdt_center"
# NOTE: Output Directory
OutputDir = os.path.join(WORKSPACE,"city-pro","output",NameCaseAnalysis)
# NOTE: Output Weight Directory
OutputDirWeights = os.path.join(WORKSPACE,"city-pro","output",NameCaseAnalysis,"weights")

# ---------- CONFIG .py ------------------ #

InitialSettingProject(FolderCarto,
                      Carto,
                      StrDates,
                      BaseFolderData,
                      DirConfig2Copy,
                      NameSeedConfig,
                      NameConfigFile,
                      OutputDir,
                      OutputDirWeights)

# Modify the configuration for the .cpp

ModifyConfigCpp(WORKSPACE = WORKSPACE,
                FolderCarto = FolderCarto,
                BaseFolderData = BaseFolderData,
                StrDates = StrDates,
                DirConfig2Copy = DirConfig2Copy,
                Carto = Carto,
                BaseName = BaseName,
                NewCaseName = NameCaseAnalysis,
                lat_min = lat_min,
                lat_max = lat_max,
                lon_min = lon_min,
                lon_max = lon_max,
                enable_subnet = True,
                jump2subnet_analysis = False,
                max_poly_length = 6000,
                num_tm = 3)
# Modify the configuration for the Python Script

ModifyConfigPy(WORKSPACE = WORKSPACE,
                NewCaseName = NameCaseAnalysis,
                DirConfig = DirConfigPython,
                StrDates = StrDates,
                lat_min = lat_min,
                lat_max = lat_max,
                lon_min = lon_min,
                lon_max = lon_max)
