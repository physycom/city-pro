from AnalysisNetwork1Day import *
#from AnalysisNetworkAllDays import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import sys
sys.path.append(os.path.join(os.environ['TRAFFIC_DIR'],'scripts'))
from FittingProcedures import *
import json

WORKSPACE = os.environ['WORKSPACE']
def parse_arguments():
    '''
        {
            "base_dir": "/home/aamad/Desktop/phd/codice/city-pro/output/bologna_mdt_detailed/",
        }
    '''
    parser = argparse.ArgumentParser(description="Process configuration file.")
    parser.add_argument("config_file","-c", help="Path to the configuration file")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_arguments()
        base_dir = args.base_dir
    except Exception as e:
        print("No Configuration file provided")
        base_dir = os.path.join(WORKSPACE,"output","bologna_mdt_detailed")
    try:
        with open(os.path.join(base_dir,"config.json")) as f:
            config = json.load(f)
    except Exception as e:
        print("No Configuration file provided")
        exit(1)

    # 1 Day Mobility Analysis
    if "StrDates" in config.keys():
        StrDates = config["StrDates"]
    else:
        exit("Missing Dates To Analyze")
    
    ListNetworkDays = []
    for StrDate in StrDates:
        print("Initialiaze Mobility and Network for Date: " + StrDate)
        # Initialize Network
        Network = DailyNetworkStats(config,StrDate)
## +++++++++++++++++ INITIALIZE CLASSES +++++++++++++++++++++++++++
        # Create Classes
        Network.ReadFcmCenters()
        Network.ReadFcm()
        ## Classes associated to inclusion principle
        Network.ReadFcmNew()
        Netwok.AddFcmNew2Fcm()
        Network.GetIncreasinglyIncludedSubnets()
        Network.ReadGeojson()
        # Create Dictionaries
        Network.CreateDictionaryIntClass2StrClass()
        # SAVE SUBNETS IN GEOJSON
        Network.CompleteGeoJsonWithClassInfo()
        # PLOT SUBNETS
        Network.PlotSubnetHTML()
        Network.PlotIncrementSubnetHTML()   
        Network.PlotFluxesHTML()
        Network.PlotTimePercorrenceHTML()     
        # FUNDAMENTAL DIAGRAM 
        Network.ComputeMFDVariables()
        Network.PlotMFD()
        # HYSTERESIS DIAGRAM
        
        Network.ReadFluxes()
## +++++++++++++++ FITTING PROCEDURES +++++++++++++++++++++++++++++
        # ALL CLASSES
        StartingGuessParametersPerLabel = Network.RetrieveGuessParametersPerLabel()
        for label in Network.labels2FitNames2Try.keys():
            for FunctionName in Network.labels2FitNames2Try[label]:
                print("====== FITTING FUNCTION: " + Network.labels2FitNames2Try[label] + " quantity: {} ======".format(label))
                Network.FittingFunctionAllClasses(label,Network.labels2FitNames2Try[label],bins = 100)
        # SUB CLASSES
        StartingGuessParametersPerClassAndLabel = Network.RetrieveGuessParametersPerClassLabel()
        for Class in Network.Classes:
            for label in Network.labels2FitNames2Try.keys():
                for FunctionName in Network.labels2FitNames2Try[label]:
                    print("====== FITTING FUNCTION: " + label + " class {0} {1} ======".format(Class, label))
                    Network.FittingFunctionSubClasses(Class,label,Network.labels2FitNames2Try[label],bins = 100)
    ## ALL DAYS ANALYSIS
## +++++++++++++++++ PLOT SUBNETWORKS +++++++++++++++++++++++++++
        # Add to Lists to Give in Input for the All Days Analysis
        ListNetworkDays.append(Network)
    # 1 Day Network Analysis
    
    # All Days Mobility Analysis
    MobAllDays = MobilityAllDays(ListMobilityDays)
    # All Days Network Analysis


