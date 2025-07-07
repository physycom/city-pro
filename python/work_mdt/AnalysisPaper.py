'''
Example of Configuration File:
{
    "StrDates": ["2022-12-30","2022-12-31","2023-01-01","2022-05-12","2022-11-11","2022-07-01","2022-08-05","2022-01-31","2023-03-18"], # Dates to Analyze
    "base_name": "bologna_mdt",
    "InputBaseDir":"/home/aamad/codice/city-pro/output/bologna_mdt_detailed",   # Useless: Directory where the output of all the main_city-pro is stored
    "bounding_box":{
        "lat_min":"44.463121",      # Same Bounding Box of the city, or of the zone of interest
        "lat_max":"44.518165",
        "lon_min":"11.287085",
        "lon_max":"11.367472"
        },
    "geojson": "/home/aamad/codice/city-pro/bologna-provincia.geojson", # Useless
    "verbose": true,        # Debugging
    "shift_bin":{
        "av_speed": 3,
        "av_speed_kmh": 0.5,
        "lenght": 40,
        "lenght_km": 0.5,               # Parameter For set_xlim for the plot of the distribution 
        "time": 30,
        "time_hours": 0.5,
        "av_accel": 0.1},
    "shift_count":{
        "av_speed": 50,
        "speed_kmh": 50,
        "lenght": 50,           # Parameter For set_ylim for the plot of the distribution
        "lenght_km": 50,
        "time": 50,
        "time_hours": 50,
        "av_accel": 50
    },
    "interval_bin":{
        "av_speed": 10,
        "speed_kmh": 10,
        "lenght": 10,
        "lenght_km": 10,
        "time": 10,                 # Parameter For set_xticks for the plot of the distribution
        "time_hours": 10,
        "av_accel": 0.2
    },
    "interval_count":{
        "av_speed": 300,
        "speed_kmh": 300,               # Parameter For set_yticks for the plot of the distribution
        "lenght": 300,
        "lenght_km": 300,
        "time": 300,
        "time_hours": 300,
        "av_accel": 500
    },
    "scale_count":{
        "av_speed": "linear",
        "speed_kmh": "linear",
        "lenght": "log",
        "lenght_km": "log",             # Parameter For set_yscale for the plot of the distribution
        "time": "log",
        "time_hours": "log",
        "av_accel": "linear"
    },
    "scale_bins":{
        "av_speed": "linear",
        "speed_kmh": "linear",
        "lenght": "linear",
        "lenght_km": "linear",              # Parameter For set_xscale for the plot of the distribution
        "time": "linear",
        "time_hours": "linear",
        "av_accel": "linear"
    }

}
'''

from AnalysisNetwork1Day import *
from AnalysisNetworkAllDays import *
#from AnalysisNetworkAllDays import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import sys
from FittingProcedures import *
import json
from multiprocessing import Pool
import warnings
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ignore all warnings
warnings.filterwarnings("ignore")
WORKSPACE = os.environ['WORKSPACE']
os.environ["DISPLAY"] = ":0.0"
FittingAnalysis = False

import matplotlib as mpl
def setup_mpl():
    mpl.rc('font', size=20)
    mpl.rcParams['legend.fontsize'] = 'small'
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['font.family']='DejaVu Math TeX Gyre'#'Helvetica 45 Light'
    mpl.rcParams['xtick.major.pad']='18'
    mpl.rcParams['ytick.major.pad']='18'
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['xtick.minor.width'] = 2
    mpl.rcParams['ytick.minor.width'] = 2
    mpl.rcParams['xtick.major.size'] = 6
    mpl.rcParams['ytick.major.size'] = 6
    mpl.rcParams['xtick.minor.size'] = 3
    mpl.rcParams['ytick.minor.size'] = 3
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['xtick.top']=True
    mpl.rcParams['ytick.right']=True
    mpl.rcParams['mathtext.default']='regular'
    mpl.rcParams['xtick.major.pad']='4'
    mpl.rcParams['ytick.major.pad']='4'
    mpl.rcParams['axes.labelpad']= 2
    mpl.rcParams['axes.labelsize'] = 20  # Set the font size of the axis labels
    alpha = 0.6
    to_rgba = mpl.colors.ColorConverter().to_rgba

def Main(config,StrDate):
    print("Initialiaze Mobility and Network for Date: " + StrDate)
    # Initialize Network
    Network = DailyNetworkStats(config,StrDate)
## +++++++++++++++++ INITIALIZE CLASSES +++++++++++++++++++++++++++
############ READ #############
    # NOTE: read info about people & clusering
    Network.ReadFcmCenters()                            # NOTE: Read centers from Fuzzy algorithm (coordinates of fcm centers in the feature space)     
    Network.ReadFcm()                                   # NOTE: read classification Fuzzy each user
    Network.ReadFcmNew()                                # NOTE: read hierarchical classification each user
    Network.GetPathFile()                               # NOTE: get the path in terms of roads for each user (succession of cluster_base.path.time, from .cpp)
    Network.GetTrajDataFrame()                          # NOTE: get the path in terms of coordinates for each user (succession of coordinates of cluster_base.lat,cluster_base.lon from .cpp)
    Network.ReadStats()                                 # NOTE: DEPRECATED -> informations present in Fcm already
    Network.ReadTimedFluxes()                           # NOTE: read the fluxes per road of people in the network in time.

    # GeoJson
    Network.ReadGeoJson()                               # NOTE: Read GeoJson -> contains road network
    Network.BoundGeoJsonWithBbox()                      # NOTE: Bound GeoJson with Bounding Box decided already at the .cpp level       
    # Network
    Network.ReadFluxesSub()                             # NOTE: Read fluxes.sub -> contains the list of roads in the sub-network identified by the fuzzy algorithm
    Network.CreateDictionaryIntClass2StrClass()         # NOTE: Create Dictionary IntClass2StrClass
    Network.GetIncreasinglyIncludedSubnets()            # NOTE: Get Increasingly Included Subnets -> hierarchical
    Network.ReadVelocitySubnet()                        # NOTE: Read Velocity Subnet -> contains the velocity of the roads in the sub-network identified by the fuzzy algorithm
    # TIME OBJECTS
    Network.BinTime()                                   # NOTE: Bin Time -> create bins for the time (standard for all plots)
    # Partition by Class
    Network.AddFcmNew2Fcm()                             # NOTE: Add FcmNew2Fcm -> add the new classification to the old one as a column "class_new" to Fcm
############# COMPUTE ###############    
    # MFD
    Network.ComputeHisteresisClass()                    # NOTE: Compute Histeresis Cycle
    Network.ComputeMFD()                                # NOTE: Compute MFD -> Fundamental Diagram
    # GEOJSON 
    Network.CompleteGeoJsonWithClassInfo()              # NOTE: Complete GeoJson With Class Info -> add the class of the road to the GeoJson in the hierarchical case where we compute a real partition of the network
    Network.ComputeTotalLengthSubnet()                  # NOTE: Compute Total Length Subnet -> compute the total length of the roads in the sub-network -> init: Class2TotalLengthOrderedSubnet
    # COMPARE OLD TO NEW CLASS
    Network.CompareOld2NewClass()                       
    Network.SplitPeopleInClasses()
    Network.GetAverageSpeedGeoJson()
#    Network.PlotTrafficVideo() #-> plots the video of the speed
    Network.ComputeDfSpeed()
    # FIT
    Network.CreateDictClass2FitInit()                                   # Create Dictionary Class2FitInit
    Network.CutFcmForFit()                                              # choose trajectories shorter then 2 hours
    # space-time
    Network.ComputeFitPerClassLengthTime()
    Network.ComputeFitLengthTime()
    # speed
    Network.ComputeFitPerClassSpeed()
    Network.ComputeFitDataFrame()                                           # NOTE: not elegant -> but works
    # speed
#    Network.ComputeFitAggregated()
    # EVOLUTION SPEED PER CLASS
    Network.PrepareSpeedEvolutionNewClassConsideringRoadClassification()
    # Detection Traffic Signal
    Network.ComputeCFAR()
    Network.PTestForTraffic()
    Network.ComputeAndPlotDailyTrafficIndicator()
    Network.ComputeAndPlotDailyVarianceTrafficIndicator()
############ PLOT ############### 
    Network.PlotNPeopNRoadsClass()
    Network.PlotTransitionMatrix()
    # Plot Speed Subnets
    #Network.PlotSpeedEvolutionTransitionClasses()
#    Network.PlotTransitionClassesInTime()
    # NOTE: Network.FcmCenters -> DataFrame ["class","av_speed","vmin","vmax","sinuosity","count"]
    # PLOT SUBNETS
#    Network.PlotIncrementSubnetHTML()   
#        Network.PlotFluxesHTML()
#    Network.PlotTimePercorrenceHTML()     
    # FUNDAMENTAL DIAGRAM
#    Network.PlotTimePercorrenceDistributionAllClasses()
#    Network.PlotDistributionLengthRoadPerClass()    # PROBLEMS WITH keys in self.IntClass2StrCLass
    # FIT ALL CLASSES
#    Network.PlotSpeedEvolutionAllSubnets()
    Network.PlotMFD()
    # VIDEO TIME PERCORRENCE (Problems) ValueError: The data range provided to the 'scale' variable is too small for the default scaling function. Normalize your data or provide a custom 'scale_func'.
#    Network.GenerateVideoEvolutionTimePercorrence()
    # HYSTERESIS DIAGRAM
    Network.ReadFluxes()
    
## +++++++++++++++++ PLOT TRAJECTORIES STATS +++++++++++++++++++++++++++
## +++++++++++++++ FITTING PROCEDURES +++++++++++++++++++++++++++++
    return Network
def MainComparison(ListNetworkDays,PlotDirAggregated,config,verbose):
    """
        @param ListNetworkDays: List of Network Objects
        @param PlotDirAggregated: Directory where to save the plots
        @param config: Configuration File
        @param verbose: Verbosity
        @brief: Responsible for the pipeline of all days.
    """
    NetAllDays = NetworkAllDays(ListNetworkDays,PlotDirAggregated,config,verbose)               # NOTE: initialize all days object
    # 
    NetAllDays.ComputeDay2PopulationTime()                                                      # 
    # Create Fcm for All -> Distribution lenght and time (Power law )
    NetAllDays.ConcatenateFcm()
######### CONCATENATION #############
    # MFD
    NetAllDays.PlotMFDPerClassCompared()
    # GeoJson
    NetAllDays.UnifyGeoJsonAllDays()
    NetAllDays.PlotPenetration()
    NetAllDays.CreateClass2SubNetAllDays()
    # FIT
    NetAllDays.ConcatenateplExpFits()                                   # concatenate df_parameters_{fit_type}, df_data_and_fit_{fit_type}
    NetAllDays.Concatenate_gauss_max_speed_fit()                        # concatenate df_parameters_{fit_type}, df_data_and_fit_{fit_type}
    #
    NetAllDays.PlotExponentsGaussianFit()
    NetAllDays.PlotAverageTij()
    NetAllDays.PlotNPeopleOverNRoads()
    # Prepare Fit
    NetAllDays.ComputeAggregatedFit()
    NetAllDays.ComputeAggregatedFitPerClass()
    # People over Roads
     # All Days Plot Distribution Velocity Aggregated
#    NetAllDays.ComputeAggregatedMFDVariablesObj()
    # SIGNAL DETECTION
    NetAllDays.PlotTrafficIndicator()
    NetAllDays.PlotDensity()
    # Plot Distribution Features
    NetAllDays.PlotGridDistrFeat()
    NetAllDays.PlotDistrAggregatedAllDays()
    NetAllDays.PlotAveragePerturbationDistributionSpeed()
    # Plot Comparison among distributions of different days: Will be in the common folder
#    NetAllDays.PlotComparisonDistributionEachFeatureAllDaysRescaledByMean() 
    # Comparison Distribution Features not By class
#    NetAllDays.PlotClass2SubNetAllDays()
#    NetAllDays.PlotClass2SubnetsComparisonAllDays()
    # MFD All Days
    NetAllDays.PlotComparisonPopulationTime()

    # Number Of People Per Day
    NetAllDays.PlotDistributionTotalNumberPeople()
    # Test Heteorgeneity Hp
#    NetAllDays.PlotExponentsFit()
#    NetAllDays.PlotNumberTrajectoriesGivenClass()
    # Compare The Time Percorrence
#    NetAllDays.CompareTimePercorrenceAllDays()
    # Compare sub-nets
#    NetAllDays.PlotComparisonSubnets()
    # Compare sub-nets
    # Aggregated Sub-Networks
    NetAllDays.ComputeOrderedIntersectionsAllDays()
#    NetAllDays.PlotAggregatedSubNetworks()
    NetAllDays.PlotUnionSubnets()

if __name__ == "__main__":
    setup_mpl()
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
    try:
        # Bologna
        ConfigFile = "AnalysisPython.json"
        # Bologna Detailed
#        ConfigFile = "ConfigPythonAnalysis.json"
        with open(os.path.join(base_dir,ConfigFile)) as f:
            config = json.load(f)
    except Exception as e:
        print(e)
        print("No Configuration file provided")
        exit(1)

    # 1 Day Mobility Analysis
    if "StrDates" in config.keys():
        StrDates = config["StrDates"]
    else:
        exit("Missing Dates To Analyze")
    
    ListNetworkDays = []
    parallel = True
    print("Dates to Analyze:\n",StrDates)
    if parallel:
        args = [(config,StrDate) for StrDate in StrDates]
        Ncpu = len(StrDates)
        if Ncpu < os.cpu_count():
            print("Number of CPU: ",Ncpu)
            pass
        else:
            Ncpu = os.cpu_count() - 1
            print("Number of CPU: ",Ncpu)
        with Pool(Ncpu) as p:
            ListNetworkDays = p.starmap(Main,args)
    else:
        for StrDate in StrDates:
            print("Initialiaze Mobility and Network for Date: " + StrDate)
            # Initialize Network
            Network = Main(config,StrDate)        ## ALL DAYS ANALYSIS
            # Add to Lists to Give in Input for the All Days Analysis
            ListNetworkDays.append(Network)
    
    # All Days Mobility Analysis
    if parallel:
        Network = ListNetworkDays[0]
    MainComparison(ListNetworkDays,Network.PlotDirAggregated,config,Network.verbose)
