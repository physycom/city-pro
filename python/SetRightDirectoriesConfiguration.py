import json
import os
WORKSPACE = os.environ["WORKSPACE"]
StrDates = ['2022-12-30','2022-12-31','2023-01-01','2022-05-12','2022-11-11','2022-07-01','2022-08-05','2022-01-31','2023-03-18']
BaseName = "bologna_mdt"
Carto = "bologna_mdt_cleaned"

for StrDate in StrDates:
    with open(os.path.join(WORKSPACE,"city-pro","bologna_mdt_detailed",StrDate,"config_bologna.json"),"r") as f:
        config = json.load(f)
    config["file_pnt"] = os.path.join(WORKSPACE,"city-pro","bologna_mdt_detailed","carto",Carto + ".pnt")
    config["file_pro"] = os.path.join(WORKSPACE,"city-pro","bologna_mdt_detailed","carto",Carto + ".pro")
    config["file_data"] = [os.path.join(WORKSPACE,"city-pro","bologna_mdt_detailed",StrDate,BaseName + "_" + StrDate + "_" + StrDate + ".csv")]
    config["cartout_basename"] = os.path.join(WORKSPACE,"city-pro","output","bologna_mdt_detailed") + "/"
    config["file_subnet"] = os.path.join(WORKSPACE,"city-pro","output","bologna_mdt_detailed","weights",BaseName + "_" + StrDate + "_" + StrDate + ".fluxes.sub") 
    config["enable_subnet"] = True
    config["jump2subnet_analysis"] = True
    with open(os.path.join(WORKSPACE,"city-pro","bologna_mdt_detailed",StrDate,"config_bologna.json"),"w") as f:
        config = json.dump(config,f,indent=2)

if not os.path.exists(os.path.join(WORKSPACE,"city-pro","output")):
    os.makedirs(os.path.join(WORKSPACE,"city-pro","output"))
if not os.path.exists(os.path.join(WORKSPACE,"city-pro","output","bologna_mdt_detailed")):
    os.makedirs(os.path.join(WORKSPACE,"city-pro","output","bologna_mdt_detailed"))
if not os.path.exists(os.path.join(WORKSPACE,"city-pro","output","bologna_mdt_detailed","weights")):
    os.makedirs(os.path.join(WORKSPACE,"city-pro","output","bologna_mdt_detailed","weights"))



