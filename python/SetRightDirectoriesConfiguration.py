import json
import os
import logging
logger = logging.getLogger(__name__)
WORKSPACE = os.environ["WORKSPACE"]
StrDates = ['2022-12-30','2022-12-31','2023-01-01','2022-05-12','2022-11-11','2022-07-01','2022-08-05','2022-01-31','2023-03-18']
BaseName = "bologna_mdt"
Carto = "bologna_mdt_cleaned"
# DETAIL VERSION
#os.mkdir(os.path.join(WORKSPACE,"city-pro","bologna_mdt_detailed"))
if not os.path.exists(os.path.join(WORKSPACE,"city-pro","bologna_mdt")):
     os.mkdir(os.path.join(WORKSPACE,"city-pro","bologna_mdt"))
DirConfig = os.path.join(WORKSPACE,"city-pro","bologna_mdt")
for StrDate in StrDates:
    logger.info(f"Setting right directories for {StrDate}")
    with open(os.path.join(DirConfig,StrDate,"config_bologna.json"),"r") as f:
        config = json.load(f)
    config["file_pnt"] = os.path.join(WORKSPACE,"city-pro","bologna_mdt_detailed","carto",Carto + ".pnt")
    config["file_pro"] = os.path.join(WORKSPACE,"city-pro","bologna_mdt_detailed","carto",Carto + ".pro")
    config["file_data"] = [os.path.join(WORKSPACE,"city-pro","bologna_mdt_detailed",StrDate,BaseName + "_" + StrDate + "_" + StrDate + ".csv")]
# DETAILED VERSION
#    config["cartout_basename"] = os.path.join(WORKSPACE,"city-pro","output","bologna_mdt_detailed") + "/"
#    config["file_subnet"] = os.path.join(WORKSPACE,"city-pro","output","bologna_mdt_detailed","weights",BaseName + "_" + StrDate + "_" + StrDate + ".fluxes.sub") 

# JUST BOLOGNA
    config["cartout_basename"] = os.path.join(WORKSPACE,"city-pro","output","bologna_mdt") + "/"
    config["file_subnet"] = os.path.join(WORKSPACE,"city-pro","output","bologna_mdt","weights",BaseName + "_" + StrDate + "_" + StrDate + ".fluxes.sub") 
    config["lon_min"] = 11.156944 
    config["lon_max"] = 11.490277
    config["lat_min"] = 44.426389 
    config["lat_max"] = 44.586666

    config["enable_subnet"] = True
    config["jump2subnet_analysis"] = False
    config["max_poly_length"] = 6000
    # NOTE: Status -> Try one more class then the usual analysis to see if the Fcm Clustering separates walkers from bikes.
    config["num_tm"] = 3
    config["num_tm_subnet"] = config["num_tm_subnet"]
    if not os.path.exists(os.path.join(DirConfig,StrDate)):
        os.makedirs(os.path.join(DirConfig,StrDate))
    with open(os.path.join(DirConfig,StrDate,"config_bologna.json"),"w") as f:
        config = json.dump(config,f,indent=2)
    logger.info(f"Dumped config {StrDate}")
# DETAIL VERSION
# OutputDir = os.path.join(WORKSPACE,"city-pro","output","bologna_mdt_detailed","weights")
OutputDir = os.path.join(WORKSPACE,"city-pro","output","bologna_mdt","weights")
if not os.path.exists(os.path.join(WORKSPACE,"city-pro","output")):
    os.makedirs(os.path.join(WORKSPACE,"city-pro","output"))
if not os.path.exists(os.path.join(WORKSPACE,"city-pro","output","bologna_mdt")):
    os.makedirs(os.path.join(WORKSPACE,"city-pro","output","bologna_mdt"))
if not os.path.exists(os.path.join(DirConfig,StrDate)):
    os.makedirs(OutputDir)


