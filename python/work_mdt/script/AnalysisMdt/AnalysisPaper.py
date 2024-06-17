from AnalysisNetwork1Day import *
from AnalysisNetworkAllDays import *
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
from multiprocessing import Pool
WORKSPACE = os.environ['WORKSPACE']
os.environ["DISPLAY"] = ":0.0"
FittingAnalysis = False
def Main(config,StrDate):
    print("Initialiaze Mobility and Network for Date: " + StrDate)
    # Initialize Network
    Network = DailyNetworkStats(config,StrDate)
## +++++++++++++++++ INITIALIZE CLASSES +++++++++++++++++++++++++++
    # Create Classes
    Network.ReadFcmCenters()
    # Create Dictionaries
    Network.CreateDictionaryIntClass2StrClass()
    Network.ReadFcm()
    ## Classes associated to inclusion principle
    Network.ReadFcmNew()
    Network.AddFcmNew2Fcm()
    # NOTE: Network.FcmCenters -> DataFrame ["class","av_speed","vmin","vmax","sinuosity","count"]
    Network.ReadStats()
    Network.GetIncreasinglyIncludedSubnets()
    Network.ReadGeoJson()
    Network.ReadFluxesSub()
    Network.ComputeMFDVariablesClass()
    # SAVE SUBNETS IN GEOJSON
    Network.CompleteGeoJsonWithClassInfo()
    # TIMED FLUXES
    Network.ReadTimedFluxes()
    # PLOT SUBNETS
    Network.PlotSubnetHTML()
    Network.PlotIncrementSubnetHTML()   
#        Network.PlotFluxesHTML()
#        Network.PlotTimePercorrenceHTML()     
    # FUNDAMENTAL DIAGRAM
    Network.ReadVelocitySubnet()
    Network.PlotMFD()
    # HYSTERESIS DIAGRAM
    Network.ReadFluxes()
## +++++++++++++++++ PLOT TRAJECTORIES STATS +++++++++++++++++++++++++++
    Network.PlotDailySpeedDistr("Aggregated")

## +++++++++++++++ FITTING PROCEDURES +++++++++++++++++++++++++++++
    if FittingAnalysis:
        # ALL CLASSES
        StartingGuessParametersPerLabel = Network.RetrieveGuessParametersPerLabel()
        for label in Network.labels2FitNames2Try.keys():
            for FunctionName in Network.labels2FitNames2Try[label]:
                print("====== FITTING FUNCTION: ",Network.labels2FitNames2Try[label] ," quantity: {} ======".format(label))
                Network.FittingFunctionAllClasses(label,Network.labels2FitNames2Try[label],bins = 100)
        # SUB CLASSES
        StartingGuessParametersPerClassAndLabel = Network.RetrieveGuessParametersPerClassLabel()
        for Class in Network.Classes:
            for label in Network.labels2FitNames2Try.keys():
                for FunctionName in Network.labels2FitNames2Try[label]:
                    print("====== FITTING FUNCTION: " + label + " class {0} {1} ======".format(Class, label))
                    Network.FittingFunctionSubClasses(Class,label,Network.labels2FitNames2Try[label],bins = 100)
## ALL DAYS ANALYSIS
    return Network
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Process configuration file.")
        parser.add_argument("-c","--config_file", help="Path to the configuration file")
        args = parser.parse_args()
        print("Args: ", args)
        base_dir = args.config_file
        print("Base directory: ",base_dir)
    except Exception as e:
        print("No Configuration file provided")
        base_dir = os.path.join(WORKSPACE,"city-pro","config")
        print("Second Directory: ",base_dir)
    try:
        with open(os.path.join(base_dir,"ConfigPythonAnalysis.json")) as f:
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
    parallel = True
    if parallel:
        args = [(config,StrDate) for StrDate in StrDates]
        with Pool(6) as p:
            ListNetworkDays = p.starmap(Main,args)
    else:
        for StrDate in StrDates:
            print("Initialiaze Mobility and Network for Date: " + StrDate)
            # Initialize Network
            Network = DailyNetworkStats(config,StrDate)
    ## +++++++++++++++++ INITIALIZE CLASSES +++++++++++++++++++++++++++
            # Create Classes
            Network.ReadFcmCenters()
            # Create Dictionaries
            Network.CreateDictionaryIntClass2StrClass()
            Network.ReadFcm()
            ## Classes associated to inclusion principle
            Network.ReadFcmNew()
            Network.AddFcmNew2Fcm()
            # NOTE: Network.FcmCenters -> DataFrame ["class","av_speed","vmin","vmax","sinuosity","count"]
            Network.ReadStats()
            Network.GetIncreasinglyIncludedSubnets()
            Network.ReadGeoJson()
            Network.ReadFluxesSub()
            Network.ComputeMFDVariablesClass()
            # SAVE SUBNETS IN GEOJSON
            Network.CompleteGeoJsonWithClassInfo()
            # TIMED FLUXES
            Network.ReadTimedFluxes()
            # PLOT SUBNETS
            Network.PlotSubnetHTML()
            Network.PlotIncrementSubnetHTML()   
            Network.PlotFluxesHTML()
            Network.PlotTimePercorrenceHTML()     
            # FUNDAMENTAL DIAGRAM
            Network.ReadVelocitySubnet()
            Network.PlotMFD()
            # HYSTERESIS DIAGRAM
            Network.ReadFluxes()
    ## +++++++++++++++++ PLOT TRAJECTORIES STATS +++++++++++++++++++++++++++
            Network.PlotDailySpeedDistr("Aggregated")

    ## +++++++++++++++ FITTING PROCEDURES +++++++++++++++++++++++++++++
            if FittingAnalysis:
                # ALL CLASSES
                StartingGuessParametersPerLabel = Network.RetrieveGuessParametersPerLabel()
                for label in Network.labels2FitNames2Try.keys():
                    for FunctionName in Network.labels2FitNames2Try[label]:
                        print("====== FITTING FUNCTION: ",Network.labels2FitNames2Try[label] ," quantity: {} ======".format(label))
                        Network.FittingFunctionAllClasses(label,Network.labels2FitNames2Try[label],bins = 100)
                # SUB CLASSES
                StartingGuessParametersPerClassAndLabel = Network.RetrieveGuessParametersPerClassLabel()
                for Class in Network.Classes:
                    for label in Network.labels2FitNames2Try.keys():
                        for FunctionName in Network.labels2FitNames2Try[label]:
                            print("====== FITTING FUNCTION: " + label + " class {0} {1} ======".format(Class, label))
                            Network.FittingFunctionSubClasses(Class,label,Network.labels2FitNames2Try[label],bins = 100)
        ## ALL DAYS ANALYSIS

            # Add to Lists to Give in Input for the All Days Analysis
            ListNetworkDays.append(Network)
    
    if parallel:
        Network = ListNetworkDays[0]
    # All Days Mobility Analysis
    NetAllDays = NetworkAllDays(ListNetworkDays,Network.PlotDirAggregated,Network.verbose)

    # Map The Classes among different days according to the closest average speed
    NetAllDays.AssociateAvSpeed2StrClass()
    
    # Create Fcm for All -> Distribution lenght and time (Power law )
    NetAllDays.ConcatenatePerClass()
    
    # Create Fcm for All -> Distribution lenght and time (Exponential with all mixed Heterogeneous classes)
    NetAllDays.ConcatenateAllFcms()
    # All Days Plot Distribution Velocity Aggregated
    NetAllDays.PlotDistributionAggregatedAllDays()
    NetAllDays.PlotDistributionAggregatedAllDaysPerClass()
    # Comparison of Distribution Time lenght (Among Days)
    NetAllDays.PlotDistributionComparisonAllDays()
    NetAllDays.PlotDistributionComparisonAllDaysPerClass()
    # All Days Plot Distribution Velocity Comparison


