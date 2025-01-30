'''
    NOTE: stats.csv Is Useless but I keep it for reference.
    SUMMARY:
        NOTE: The choice made to handle plot is to conceive one structure for the aggregated case and then create a mirror dictionary for each class computed with filtered input.
        NOTE: For description look Aggregated case, for reference name at class case.
            Aggregated Case:
                1) MFD = pl.DataFrame({"population":[],"time":[],"speed_kmh":[],"av_speed": []}) Stored in: os.path.join(self.PlotDir,"MFD_{0}.csv".format(self.StrDate))
                    Description: For each interval (self.iterations = 96) compute the average speed and the average speed for the population in the interval.
            
                2) MFD2Plot = {"bins_population":[p0,..,p19],"binned_av_speed":[v0,..,v19],"binned_sqrt_err_speed":[e0,..,e19]} Stored in: os.path.join(self.PlotDir,"MFD_{0}_2Plot.csv".format(self.StrDate))
                    Description: For each bin (15 NOTE: set by hand in this script) (self.iterations = 96) compute the average speed and the average speed for the population in the interval.
            Class Case:
''' 
from collections import defaultdict
import geopandas as gpd
import numpy as np
import os
import polars as pl
import pandas as pd
from shapely.geometry import box
import folium
import datetime
import matplotlib.pyplot as plt
import json
import warnings
from FittingProcedures import *
from MFDAnalysis import *
from CastVariables import *
from analysisPlot import *
from InfoFitMdtDataset import *
from JsonFunctions import *
from LoggingInfo import *
from UsefulStructures import *
from GeoPlot import *
import matplotlib.ticker as ticker
from FunctionsOnTrajectories import *
from ReadFiles import *
from GeographyFunctions import *
from NewAnalysis import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ignore all warnings
warnings.filterwarnings("ignore")
if os.path.isfile(os.path.join(os.environ["WORKSPACE"],"city-pro","custom_style.mplstyle")):
    plt.style.use(os.path.join(os.environ["WORKSPACE"],"city-pro","custom_style.mplstyle"))
else:
    try:
        import PlotSettings
    except Exception as e:
        print("No Plot Settings File Found")


def GetLengthPartitionInGeojSon(GeoJson,Lengths):
    Length2Roads = {Length: [] for Length in Lengths}
    for i in range(len(Lengths)):
        for idx,road in GeoJson.iterrows():
            if road["poly_length"] > Lengths[i] and road["poly_length"] < Lengths[i+1]:
                Length2Roads[Lengths[i]].append(int(road["poly_lid"]))
    return Length2Roads





class DailyNetworkStats:
    '''
        This Class is Used to Contain Informations about The Daily info About the Network.
        NOTE: Directory for Cpp files: WORKSPACE//city-pro/output/bolgona_mdt_files
    '''
    def __init__(self,config,StrDate):
        # BASE NAME (Is the base name of the Cpp-Output-files)
        if "base_name" in config.keys():
            self.BaseFileName = config["base_name"]
        else:
            self.BaseFileName = "bologna_mdt"
        # DATE
        if StrDate in config["StrDates"]:
            self.StrDate = StrDate
        else:
            print("StrDate not found in config")
            exit(1)
        if "verbose" in config.keys():
            self.verbose = config["verbose"]
        else:
            self.verbose = False
        
        # INPUT DIR 
        if "InputBaseDir" in config.keys():
            self.InputBaseDir = config["InputBaseDir"]
        else:
            print("No input directory found in config setting default...")
            print(os.path.join(os.environ['WORKSPACE'],"city-pro","output","bologna_mdt_detailed"))
            self.InputBaseDir = os.path.join(os.environ['WORKSPACE'],"city-pro","output","bologna_mdt_detailed")
        # FILES
        self.DictDirInput = {"fcm": os.path.join(self.InputBaseDir,self.BaseFileName + '_' + self.StrDate + '_' + self.StrDate + '_fcm.csv'),
                        "fcm_centers": os.path.join(self.InputBaseDir,self.BaseFileName + '_' + self.StrDate + '_' + self.StrDate + '_fcm_centers.csv'),
                        "fcm_new":os.path.join(self.InputBaseDir,self.BaseFileName + '_' + self.StrDate + '_' + self.StrDate + '_fcm_new.csv'),
                        "stats":os.path.join(self.InputBaseDir,self.BaseFileName + '_' + self.StrDate + '_' + self.StrDate + '_stats.csv'),
                        "timed_fluxes": os.path.join(self.InputBaseDir,self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + '_timed_fluxes.csv'),
                        "fluxes": os.path.join(self.InputBaseDir,"weights",self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + '.fluxes'),
                        "fluxes_sub": os.path.join(self.InputBaseDir,"weights",self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + '.fluxes.sub')}        
        self.PlotDir = os.path.join(self.InputBaseDir,"plots",self.StrDate)
        self.PlotDirAggregated = os.path.join(self.InputBaseDir,"plots")
        if not os.path.exists(self.PlotDir):
            os.makedirs(self.PlotDir)
        if "geojson" in config.keys():
            if os.path.exists(os.path.join(self.PlotDir,"GeoJson_{0}.geojson".format(self.StrDate))):
                self.GeoJsonFile = os.path.join(self.PlotDir,"GeoJson_{0}.geojson".format(self.StrDate))
            else:
                self.GeoJsonFile = os.path.join(config["geojson"])
        else:
            self.GeoJsonFile = os.path.join(os.environ['WORKSPACE'],"city-pro","bologna-provincia.geojson")
        # BOUNDING BOX
        if "bounding_box" in config.keys():
            try:
                self.bounding_box = [(config["bounding_box"]["lat_min"],config["bounding_box"]["lon_min"]),(config["bounding_box"]["lat_max"],config["bounding_box"]["lon_min"]),(config["bounding_box"]["lat_max"],config["bounding_box"]["lon_max"]),(config["bounding_box"]["lat_min"],config["bounding_box"]["lon_max"])]
                bbox = box(config["bounding_box"]["lat_min"],config["bounding_box"]["lon_min"],config["bounding_box"]["lat_max"],config["bounding_box"]["lon_max"])
                self.centroid = gpd.GeoDataFrame([1], geometry=[bbox], crs="EPSG:4326").centroid
                self.BoxNumeric = [self.bounding_box[0][1], self.bounding_box[0][0], self.bounding_box[2][1], self.bounding_box[2][0]]
            except:
                exit("bounding_box not defined well in config. Should be 'bounding_box': {'lat_min': 44.463121,'lon_min': 11.287085,'lat_max': 44.518165,'lon_max': 11.367472}")
        else:
            # Bologna Mdt Detailed
            self.bounding_box = [(44.463121,11.287085),(44.518165,11.287085),(44.518165,11.367472),(44.463121,11.367472)]
            bbox = box((44.463121,11.287085,44.518165,11.367472))
            self.centroid = gpd.GeoDataFrame([1], geometry=[bbox], crs="EPSG:4326").centroid
            self.BoxNumeric = [self.bounding_box[0][1], self.bounding_box[0][0], self.bounding_box[2][1], self.bounding_box[2][0]]
        ## CONVERSIONS and CONSTANTS
        self.day_in_sec = 24*3600
        self.dt = 15*60
        self.iterations = int(self.day_in_sec/self.dt)
        yy,mm,dd = StrDate2DateFormatLocalProject(self.StrDate)
        self.Date = datetime.datetime(int(yy),int(mm),int(dd),0,0,0)
        self.TimeStampDate = datetime.datetime.timestamp(self.Date)
        self.config = config
        # FLAGS
        self.ReadTime2FluxesBool = False
        self.ReadFluxesBool = False
        self.ReadFluxesSubBool = False
        self.ReadFcmBool = False
        self.ReadFcmNewBool = False
        self.ReadFcmCentersBool = False
        self.ReadGeojsonBool = False
        self.ReadVelocitySubnetBool = False
        self.BoolStrClass2IntClass = False
        self.ComputedMFD = False
        self.ReadStatsBool = False
        self.ReadFluxesSubIncreasinglyIncludedIntersectionBool = False
        self.GeoJsonWithClassBool = False
        self.TimePercorrenceBool = False
        # SETTINGS INFO
        self.ListColors = ['red','blue','green','orange','purple','yellow','cyan','magenta','lime','pink','teal','lavender','brown','beige','maroon','mint','coral','navy','olive','grey']
#        self.Name = BaseName
        self.StrDate = StrDate
        # CLASSES INFO
        self.Class2Color = {"1 slowest": "blue","2 slowest":"green","middle velocity class": "yellow","2 quickest": "orange", "1 quickest":"red"}
        self.IntClass2StrClass = defaultdict() # {0,slowest,...}
        self.StrClass2IntClass = defaultdict() # {slowest: 0,...}
        self.RoadInClass2VelocityDir = defaultdict() # {0: ../.._0velocity_subnet.csv}
        self.Class2DfSpeedAndTimePercorrenceRoads = defaultdict() # {0: [start_bin,end_bin,id,time_percorrence,av_speed]}
        # INPUT FIT INFO
        self.DictConstraintClassLabel = defaultdict(dict)
        self.DictConstraintLabel = defaultdict(dict)
        # INFO FIT
        self.Function2FitInfo = defaultdict(dict)
        self.AllClassLabel2BestFit = defaultdict(dict)
        self.IntClass2BestFit = defaultdict(dict)
        self.labels2FitNames2Try = {"time":["powerlaw","exponential"],
                    "lenght":["powerlaw","exponential"],
                    "av_speed":["gaussian","maxwellian"],
                    "av_accel":["gaussian","maxwellian"]
                    }
        # FIT OUTPUT
        self.InitialGuessPerLabel = defaultdict(dict)
        self.InitialGuessPerClassAndLabel = defaultdict(dict) 
        # FEATURES
        self.Features = ["av_speed","lenght","time","av_accel"]
        self.Features2Fit = ["av_speed","lenght","time","speed_kmh","lenght_km","time_hours"]
        # OUTPUT DICTIONARIES FIT
        self.Feature2Label = {"av_speed":'average speed (m/s)',"speed_kmh":'average speed (km/h)',"av_accel":"average acceleration (m/s^2)","lenght":'lenght (m)',"lenght_km": 'lenght (km)',"time_hours":'time (h)',"time":'time (s)'}
        self.Feature2SaveName = {"av_speed":"average_speed","speed_kmh":'average_speed_kmh',"av_accel":"average_acceleration","lenght":"lenght","lenght_km": 'lenght_km',"time_hours":"time_hours","time":"time"}
        self.Feature2Legend = {"av_speed":"speed (m/s)","speed_kmh":'speed (km/h)',"av_accel":"acceleration (m/s^2)","lenght":"lenght (m)","lenght_km": 'lenght (km)',"time_hours":"time (h)","time":"time (s)"} 
        self.Feature2MaxBins = {"av_speed":{"bins":0,"count":0},"speed_kmh":{"bins":0,"count":0},"av_accel":{"bins":0,"count":0},"lenght":{"bins":0,"count":0},"lenght_km": {"bins":0,"count":0},"time_hours":{"bins":0,"count":0},"time":{"bins":0,"count":0}}
        self.Feature2Function2Fit2InitialGuess = InitFeature2Function2Fit2InitialGuess(self.Features2Fit)
        self.InfoFit = config["info_fit"]
        ## BIN SETTINGS
        if "shift_count" in config.keys():
            self.Feature2ShiftCount = config["shift_count"]
            for feat in self.Features:
                if feat in self.Feature2ShiftCount.keys():
                    pass
                else:
                    raise KeyError(feat + " not in shift_count")
        else:
            self.Feature2ShiftCount = {"av_speed": 50,"speed_kmh": 50,"lenght": 50,"lenght_km": 50,"time": 50,"time_hours": 50,"av_accel": 50},
        if "shift_bin" in config.keys():
            self.Feature2ShiftBin = config["shift_bin"]
            for feat in self.Features:
                if feat in self.Feature2ShiftBin.keys():
                    pass
                else:
                    raise KeyError(feat + " not in shift_bin")
        else:
            self.Feature2ShiftBin = {"av_speed": 3,"speed_kmh": 0.5,"lenght": 40,"lenght_km": 0.5,"time": 30,"time_hours": 0.5,"av_accel": 0.1}
        if "interval_bin" in config.keys():
            self.Feature2IntervalBin = config["interval_bin"]
            for feat in self.Features:
                if feat in self.Feature2IntervalBin.keys():
                    pass
                else:
                    raise KeyError(feat + " not in interval_bin")
        else:
            self.Feature2IntervalBin = {"av_speed": 10,"speed_kmh": 10,"lenght": 10,"lenght_km": 10,"time": 10,"time_hours": 10,"av_accel": 0.1}
        if "interval_count" in config.keys():
            self.Feature2IntervalCount = config["interval_count"]
            for feat in self.Features:
                if feat in self.Feature2IntervalCount.keys():
                    pass
                else:
                    raise KeyError(feat + " not in interval_count")
        else:
            self.Feature2IntervalCount = {"av_speed": 300,"speed_kmh": 300,"lenght": 300,"lenght_km": 300,"time": 300,"time_hours": 300,"av_accel": 500}
        if "scale_count" in config.keys():
            self.Feature2ScaleCount = config["scale_count"]
            for feat in self.Features:
                if feat in self.Feature2ScaleCount.keys():
                    pass
                else:
                    raise KeyError(feat + " not in scale_count")
        else:
            self.Feature2ScaleCount = {"av_speed": "linear","speed_kmh": "linear","lenght": "log","lenght_km": "log","time": "log","time_hours": "log","av_accel": "linear"}
        if "scale_bins" in config.keys():
            self.Feature2ScaleBins = config["scale_bins"]
            for feat in self.Features:
                if feat in self.Feature2ScaleBins.keys():
                    pass
                else:
                    raise KeyError(feat + " not in scale_bins")
        else:
            self.Feature2ScaleBins = {"av_speed": "linear","speed_kmh": "linear","lenght": "linear","lenght_km": "linear","time": "linear","time_hours": "linear","av_accel": "linear"}
        assert self.Feature2ScaleBins.keys() == self.Feature2ScaleCount.keys() == self.Feature2IntervalCount.keys() == self.Feature2IntervalBin.keys() == self.Feature2ShiftBin.keys() == self.Feature2ShiftCount.keys() == self.Feature2Label.keys() == self.Feature2SaveName.keys() == self.Feature2Legend.keys(), "Error: Features not consistent"
        # MINIMUM VALUES FOR (velocity,population,lenght,time) for trajectories of the day
        self.MFD = None
        self.MFD2Plot = None
        self.MinMaxPlot = defaultdict()

        # STATS about TRAJECTORIES
        self.Class2MaxCountSpeed = defaultdict(dict)
        # LOG File
        self.LogFile = os.path.join(self.PlotDir,"{0}.log".format(self.StrDate))
        self.CountFunctionsCalled = 0
        with open(self.LogFile,'w') as f:
            f.write("Log File for {0}\n".format(self.StrDate))
# --------------- Read Files ---------------- #
    def ReadTimedFluxes(self):
        """
            Read ../ouputdir/basename_date_date_timed_fluxes.csv
        """
        self.CountFunctionsCalled += 1
        if os.path.isfile(self.DictDirInput["timed_fluxes"]):
            self.TimedFluxes,self.ReadTime2FluxesBool = ReadTimedFluxes(self.DictDirInput["timed_fluxes"])
        else:   
            pass
    def ReadFluxes(self):
        """
            Read ../ouputdir/weights/basename_date_date.fluxes
        """
        self.CountFunctionsCalled += 1
        if self.verbose:
            print("Reading fluxes")
            print(self.DictDirInput["fluxes"])
        if os.path.isfile(self.DictDirInput["fluxes"]):
            self.Fluxes,self.ReadFluxesBool = ReadFluxes(self.DictDirInput["fluxes"])
        else:
            pass
    def ReadFcm(self):
        """
            Read ../ouputdir/basename_date_date_fcm.csv
        """
        self.CountFunctionsCalled += 1
        if self.verbose:
            print("Reading fcm")
            print(self.DictDirInput["fcm"])
        if os.path.isfile(self.DictDirInput["fcm"]):
            self.Fcm, self.ReadFcmBool = ReadFcm(self.DictDirInput["fcm"])
        else:
            pass
    def ReadStats(self):
        """
            Read ../ouputdir/basename_date_date_stats.csv
        """
        self.CountFunctionsCalled += 1
        if self.verbose:
            print("Reading stats")
            print(self.DictDirInput["stats"])
        if os.path.isfile(self.DictDirInput["stats"]):
            self.Stats,self.ReadStatsBool = ReadStats(self.DictDirInput["stats"])
        else:
            pass
    def ReadFcmNew(self):
        """
            Read ../ouputdir/basename_date_date_fcm_new.csv
        """
        self.CountFunctionsCalled += 1
        if self.verbose:
            print("Reading fcm_new")
            print(self.DictDirInput["fcm_new"])
        if os.path.isfile(self.DictDirInput["fcm_new"]):
            self.FcmNew,self.ReadFcmNewBool = ReadFcmNew(self.DictDirInput["fcm_new"])
        else:
            pass
    def ReadFcmCenters(self):
        """
            Description:
                Read the centers of the FCM
            NOTE: This function will define also what are the classes in any plot since it is used to initialize
            IntClass2StrClass
        """
        self.CountFunctionsCalled += 1
        if os.path.isfile(self.DictDirInput["fcm_centers"]):
            self.FcmCenters,self.ReadFcmCentersBool = ReadFcmCenters(self.DictDirInput["fcm_centers"])
        else:
            pass
    def ReadFluxesSub(self):
        '''
            Input:
                FluxesSubFile: (str) -> FluxesSubFile = '../{basename}_{start}_{start}/fluxes.sub'
                verbose: (bool) -> verbose = False
            Output:
                self.IntClass2Roads: (dict) -> self.IntClass2Roads = {IntClass:[] for IntClass in self.IntClasses}
                self.IntClass2RoadsInit: (bool) -> Boolean value to Say I have stored the SubnetInts For each Class
        '''
        self.CountFunctionsCalled += 1
        # Read Fluxes.sub
        FluxesSub = self.DictDirInput["fluxes_sub"]
        if os.path.isfile(self.DictDirInput["fluxes_sub"]):
            self.IntClass2Roads, self.ReadFluxesSubBool = ReadFluxesSub(self.DictDirInput["fluxes_sub"])
            Message = "{} Read Fluxes Sub: True".format(self.CountFunctionsCalled)
            for Class in self.IntClass2Roads.keys():
                Message += "Class: {0} -> {1} Roads, ".format(Class,len(self.IntClass2Roads[Class]))
            AddMessageToLog(Message,self.LogFile)
        else:
            Message = "{} Read Fluxes Sub: False".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
            print(f"{FluxesSub} not found")
    def ReadGeoJson(self):
        """
            Read the GeoJson File and store it in:
                1) self.GeoJson (Base GeoJson)
                2) self.GeoJsonClassInfo (the GeoJson Used for Plotting the Subnets)
                3) Filter it for the region of interest
        """
        self.CountFunctionsCalled += 1
        # Read GeoJson Without Class Info
        if os.path.isfile(self.GeoJsonFile):
            self.GeoJson, self.ReadGeojsonBool= ReadGeoJson(self.GeoJsonFile)
        if os.path.exists(os.path.join(self.InputBaseDir,"BolognaMDTClassInfo.geojson")):
            DirgeoJsonClassInfo = os.path.join(self.InputBaseDir,"BolognaMDTClassInfo.geojson")
#            self.GeoJsonClassInfo,self.ReadGeojsonClassInfoBool = ReadGeoJsonClassInfo(DirgeoJsonClassInfo)
        else:
            self.ReadGeojsonClassInfoBool = False
            pass
        self.GeoJson = RestrictGeoJsonWithBbox(self.GeoJson,self.BoxNumeric)
    def GetPathFile(self):
        """
            Description:
                Get the Path File
        """
        self.PathFile = os.path.join(self.InputBaseDir,self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + '_paths_on_road.csv')
        if os.path.isfile(self.PathFile):
            self.PathDf,self.ReadPathFileBool = ReadPathFile(self.PathFile)
        else:
            self.ReadPathFileBool = False
    def GetTrajDataFrame(self):
        """
            Description:
                Get the Traj Data Frame
        """
        self.PathTraj = os.path.join(self.InputBaseDir,self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + 'traj_dataframe.csv')
        if os.path.isfile(self.PathTraj):
            self.TrajDf = pl.read_csv(self.PathTraj,separator = ";")
    def GetIncreasinglyIncludedSubnets(self):
        """
            Description:
                Get the Increasingly Included Subnets
        """
        self.CountFunctionsCalled += 1
        if self.BoolStrClass2IntClass:
            self.DictSubnetsTxtDir = GenerateDictSubnetsTxtDir(self.InputBaseDir,self.BaseFileName,self.StrDate,self.IntClass2StrClass)            
            self.IntClass2RoadsIncreasinglyIncludedIntersection,self.ReadFluxesSubIncreasinglyIncludedIntersectionBool = ReadFluxesHierarchicallyOrdered(self.DictSubnetsTxtDir)
        else:
            pass
    def ReadVelocitySubnet(self):
        """
            Description:
                Read the velocity subnetworks.
        """
        self.CountFunctionsCalled += 1
        if self.BoolStrClass2IntClass:
            try:
                for Class in self.IntClass2StrClass.keys():
                    self.RoadInClass2VelocityDir[Class] = os.path.join(os.path.join(self.InputBaseDir,self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + '_class_{}velocity_subnet.csv'.format(Class)))
                    self.Class2DfSpeedAndTimePercorrenceRoads[Class] = pd.read_csv(self.RoadInClass2VelocityDir[Class],delimiter = ';')
                    self.Class2DfSpeedAndTimePercorrenceRoads[Class] = pl.from_pandas(self.Class2DfSpeedAndTimePercorrenceRoads[Class])
                self.ReadVelocitySubnetBool = True
                Message = "{} Read Velocity Subnet: True\n".format(self.CountFunctionsCalled)
                Message += "\tInitialized Class2DfSpeedAndTimePercorrenceRoads: {IntClass:pl.Dataframe[id_poly,time_percorrence,av_speed]}\n"
                AddMessageToLog(Message,self.LogFile)
            except:
                Message = "{} Read Velocity Subnet: False".format(self.CountFunctionsCalled)
                AddMessageToLog(Message,self.LogFile)
                print("VelTimePercorrenceFile not found")
        else:
            print("Warning: No Initialization of Class2DfSpeedAndTimePercorrenceRoads due to lack of definition of IntClass2Str")

### HANDLE TIME
    def BinTime(self):
        """
        @return self.BinTimeTimestamp [self.TimeStampDate,...,self.TimeStampDate+self.iterations*self.dt]
        """
        self.BinTimestamp,self.BinStringDayHour,self.BinStringHour = BinTimeTimestampGivenDay(self.TimeStampDate,self.dt,self.iterations)
        pl.DataFrame({"timestamp":self.BinTimestamp,"day_hour":self.BinStringDayHour,"hour":self.BinStringHour}).write_csv(os.path.join(self.PlotDir,"BinTime.csv"))
#--------- COMPLETE GEOJSON ------- ##
    def BoundGeoJsonWithBbox(self):
        """
            @brief: Bound the GeoJson with the Bounding Box (Whenever you change the area you want to analyze)
        """
        if os.path.exists(os.path.join(self.PlotDir,"GeoJson_{0}.geojson".format(self.StrDate))):
            pass
        else:
            self.GeoJson = RestrictGeoJsonWithBbox(self.GeoJson,self.BoxNumeric)
            self.GeoJson.to_file(os.path.join(self.PlotDir,"GeoJson_{0}.geojson".format(self.StrDate)))
    def CompleteGeoJsonWithClassInfo(self):
        """
            Computes "IntClassOrdered" and "StrClassOrdered" columns for the Geojson.
            NOTE:
                Each road for each day will have the Int, Str, IntOrdered and StrOrdered Class.
        """
        self.CountFunctionsCalled += 1
        if self.ReadGeojsonBool and self.ReadFluxesSubIncreasinglyIncludedIntersectionBool:
            Path2GeoJson = os.path.join(self.InputBaseDir,"GeoJson_{0}.geojson".format(self.StrDate))
            self.GeoJson, self.GeoJsonWithClassBool = AddColumnsAboutClassesRoad2GeoJson(self.GeoJson,self.IntClass2RoadsIncreasinglyIncludedIntersection,self.IntClass2StrClass,self.StrDate)
            self.GeoJson, self.GeoJsonWithClassBool = AddColumnsAboutClassesRoad2GeoJson(self.GeoJson,self.IntClass2Roads,self.IntClass2StrClass,self.StrDate)
            if False:
                self.GeoJson.to_file(Path2GeoJson)
        else:
            self.GeoJsonWithClassBool = False
    def AddFcmNew2Fcm(self):
        """
            Description:
                Convert the class column of the FcmNew to class_new and join it to the Fcm.
                In this way we have in Fcm for each trajectory a new column with the class of the trajectory after having intersected the subnetworks..
        """
        self.CountFunctionsCalled += 1
        if self.ReadFcmBool and self.ReadFcmNewBool:
            self.Fcm = AssociateHierarchicalClass2Users(self.Fcm,self.FcmNew)  
            self.Fcm.write_csv(self.DictDirInput["fcm"])          
        if self.ReadStatsBool and self.ReadFcmNewBool:
            self.Stats = AssociateHierarchicalClass2Users(self.Stats,self.FcmNew)    


    def CompareOld2NewClass(self):
        """
            Description:
                Compare the old class with the new class
            Output:
                DfComparison: (pd.DataFrame) -> DfComparison = pd.DataFrame(["NewNumber","OldNumber","NewClass","OldClass","LengthIntersection"])
                TransitionUsers: (dict) -> TransitionUsers = {(OldClass,NewClass): [ids]}
        """
        self.CountFunctionsCalled += 1
        if self.GeoJsonWithClassBool:
#                self.DfComparison,_ = ReadTransitionClassMatrix(os.path.join(self.PlotDir,"TransitionClassMatrix.csv"))    
#                with open(os.path.join(self.PlotDir,"TransitionUsers.json"),'r') as f:
#                    self.TransitionUsers = json.load(f)
#            else:
            self.DfComparison,self.TransitionUsers = ComputeTransitionClassMatrix(self.Fcm)
            self.DfComparison.write_csv(os.path.join(self.PlotDir,"TransitionClassMatrix.csv"))  
            with open(os.path.join(self.PlotDir,"TransitionUsers.json"),'w') as f:
                json.dump(self.TransitionUsers,f,indent=2)
    def SplitPeopleInClasses(self):
        """
            return:
                self.OrderedClass2TimeDeparture2UserId = {OrderedIntClass: TimeDeparture: [ids]}
                self.Class2TimeDeparture2UserId = {IntClass: TimeDeparture: [ids]}
        """
        self.OrderedClass2TimeDeparture2UserId = GroupByClassAndTimeDeparture(self.Fcm,self.BinTimestamp)
        self.Class2TimeDeparture2UserId = GroupByClassAndTimeDeparture(self.Fcm,self.BinTimestamp,"class")
        self.OrderedClass2UserId = GroupByClass(self.Fcm,'class_new')
        self.Class2UserId = GroupByClass(self.Fcm,'class')
        with open(os.path.join(self.PlotDir,"Class2TimeDeparture2UserId.json"),'w') as f:
            json.dump(self.Class2TimeDeparture2UserId,f,indent=2)
        with open(os.path.join(self.PlotDir,"OrderedClass2TimeDeparture2UserId.json"),'w') as f:
            json.dump(self.OrderedClass2TimeDeparture2UserId,f,indent=2)
        with open(os.path.join(self.PlotDir,"Class2UserId.json"),'w') as f:
            json.dump(self.Class2UserId,f,indent=2)
        with open(os.path.join(self.PlotDir,"OrderedClass2UserId.json"),'w') as f:
            json.dump(self.OrderedClass2UserId,f,indent=2)

    def GetAverageSpeedGeoJson(self):
        """
            self.Class2TimeInterval2Road2SpeedNew {Class:{TimeInterval:{Road:<speed>_{new class}}}}
            self.Class2TimeInterval2Road2Speed {Class:{TimeInterval:<speed>_{class}}}
            self.ColumnsPlotIntervalSpeedNew = ["{0}_Speed_{1}_{2}".format(Type,StrDate,TimeDeparture)]
        """
        # New Classification
        if not os.path.exists(os.path.join(self.PlotDir,"Class2TimeInterval2Road2SpeedNew.json")):
            self.Class2TimeInterval2Road2SpeedNew = ComputeSpeedRoadsPerTimeInterval(self.Fcm,self.OrderedClass2TimeDeparture2UserId,self.IntClass2RoadsIncreasinglyIncludedIntersection)
            with open(os.path.join(self.PlotDir,"Class2TimeInterval2Road2SpeedNew.json"),'w') as f:
                json.dump(self.Class2TimeInterval2Road2SpeedNew,f,indent=2)
            self.GeoJson,self.ColumnsPlotIntervalSpeedNew = AddColumnAverageSpeedGeoJson(self.GeoJson,self.Class2TimeInterval2Road2SpeedNew,self.StrDate,"new_class")
            self.ClassNew2TimeInterval2Road2SpeedActualRoads = ComputeSpeedRoadsPerTimeIntervalByRoadsTravelledByAllUsers(self.Fcm,self.PathDf,self.OrderedClass2TimeDeparture2UserId,self.IntClass2RoadsIncreasinglyIncludedIntersection)
            with open(os.path.join(self.PlotDir,"ClassNew2TimeInterval2Road2SpeedActualRoads.json"),'w') as f:
                json.dump(self.Class2TimeInterval2Road2SpeedNew,f,indent=2)

        else:
            with open(os.path.join(self.PlotDir,"Class2TimeInterval2Road2SpeedNew.json"),'r') as f:
                self.Class2TimeInterval2Road2SpeedNew = json.load(f)
            self.ColumnsPlotIntervalSpeedNew = ReturnColumnPlot(self.Class2TimeInterval2Road2SpeedNew,"new_class",self.StrDate)
            self.ClassNew2TimeInterval2Road2SpeedActualRoads = ComputeSpeedRoadsPerTimeIntervalByRoadsTravelledByAllUsers(self.Fcm,self.PathDf,self.OrderedClass2TimeDeparture2UserId,self.IntClass2RoadsIncreasinglyIncludedIntersection)
            with open(os.path.join(self.PlotDir,"ClassNew2TimeInterval2Road2SpeedActualRoads.json"),'w') as f:
                json.dump(self.Class2TimeInterval2Road2SpeedNew,f,indent=2)

#            self.GeoJson,self.ColumnsPlotIntervalSpeedNew = AddColumnAverageSpeedGeoJson(self.GeoJson,self.Class2TimeInterval2Road2SpeedNew,self.StrDate,"new_class")
        # Old Classification
        if not os.path.exists(os.path.join(self.PlotDir,"Class2TimeInterval2Road2Speed.json")):
            self.Class2TimeInterval2Road2Speed = ComputeSpeedRoadsPerTimeInterval(self.Fcm,self.Class2TimeDeparture2UserId,self.IntClass2RoadsIncreasinglyIncludedIntersection)
            # Compute {Class:{TimeInterval:<speed>_{class}}}
#            self.Class2TimeInterval2Road2SpeedActualRoads = ComputeSpeedRoadsPerTimeIntervalByRoadsTravelledByAllUsers(self.Fcm,self.PathDf,self.Class2TimeDeparture2UserId)
            with open(os.path.join(self.PlotDir,"Class2TimeInterval2Road2Speed.json"),'w') as f:
                json.dump(self.Class2TimeInterval2Road2Speed,f,indent=2)    
            self.GeoJson,self.ColumnsPlotIntervalSpeed = AddColumnAverageSpeedGeoJson(self.GeoJson,self.Class2TimeInterval2Road2Speed,self.StrDate,"class")
            self.GeoJson.to_file(os.path.join(self.PlotDir,"GeoJson_{0}.geojson".format(self.StrDate)))
        else:
            with open(os.path.join(self.PlotDir,"Class2TimeInterval2Road2Speed.json"),'r') as f:
                self.Class2TimeInterval2Road2Speed = json.load(f)
            self.ColumnsPlotIntervalSpeed = ReturnColumnPlot(self.Class2TimeInterval2Road2Speed,"class",self.StrDate)
#            self.GeoJson,self.ColumnsPlotIntervalSpeed = AddColumnAverageSpeedGeoJson(self.GeoJson,self.Class2TimeInterval2Road2Speed,self.StrDate,"class")
#            self.GeoJson.to_file(os.path.join(self.PlotDir,"GeoJson_{0}.geojson".format(self.StrDate)))
        # Get People that change class and their speed on the network
        self.ClassOld2ClassNewTimeInterval2Road2SpeedNew, self.ClassOld2ClassNewTimeInterval2Transition = ComputeSpeedRoadPerTimePeopleChangedClass(self.Fcm,self.OrderedClass2TimeDeparture2UserId,self.Class2TimeDeparture2UserId,self.IntClass2RoadsIncreasinglyIncludedIntersection)
        with open(os.path.join(self.PlotDir,"ClassOld2ClassNewTimeInterval2Road2SpeedNew.json"),'w') as f:
            json.dump(self.ClassOld2ClassNewTimeInterval2Road2SpeedNew,f,indent=2)
        with open(os.path.join(self.PlotDir,"ClassOld2ClassNewTimeInterval2Transition.json"),'w') as f:
            json.dump(self.ClassOld2ClassNewTimeInterval2Transition,f,indent=2)

    def ComputeTotalLengthSubnet(self):
        """
            Description:
                Compute the total length of the subnets
            Output:
                self.Class2TotalLengthSubnet = {Class:TotalLength}
        """
        self.Class2TotalLengthOrderedSubnet = {Class: 0 for Class in self.IntClass2RoadsIncreasinglyIncludedIntersection.keys()}
        for Class in self.IntClass2RoadsIncreasinglyIncludedIntersection.keys():
            self.Class2TotalLengthOrderedSubnet[Class] = self.GeoJson.loc[self.GeoJson[f"IntClassOrdered_{self.StrDate}"] == Class]["poly_length"].sum()
        with open(os.path.join(self.PlotDir,"Class2TotalLengthOrderedSubnet.json"),'w') as f:
            json.dump(self.Class2TotalLengthOrderedSubnet,f,indent=2)

##--------------- Dictionaries --------------##
    def CreateDictionaryIntClass2StrClass(self):
        '''
        Input:
            fcm: dataframe []

        Output: dict: {velocity:'velocity class in words: (slowest,...quickest)]}
        '''
        self.CountFunctionsCalled += 1
        if self.ReadFcmCentersBool:
            self.IntClass2StrClass, self.StrClass2IntClass, self.BoolStrClass2IntClass = CreateIntClass2StrClass(self.FcmCenters,self.IntClass2StrClass,self.StrClass2IntClass)
        else:
            self.BoolStrClass2IntClass = False
    def CreateDictClass2FitInit(self):
        """
            NOTE: DictInitialGuess Must Be Initialized and Have the Form Specified Below
            Output: 
                Class2InitialGuess: {IntClass:DictInitialGuess}
                DictInitialGuess: {"exponential":{"time":{"initial_guess":[],"interval":[]},"time_hours":{"initial_guess":[],"interval":[]}},
            Rule for Initial Guess:
                1) time: 
                    - exponential: [MaxCount,-Avg]
                    - powerlaw: [MaxCount*StartWindowS,-1]
                2) time_hours:
                    - exponential: [MaxCount,-Avg/SecondsInHour]
                    - powerlaw: [MaxCount*StartWindowS,-1]
                3) lenght:
                    - exponential: [MaxCount,-Avg]
                    - powerlaw: [MaxCount*StartWindowS,-1]
            Rule for Interval:
                1) time: [StartWindowS,EndWindowS] (From Experience)
                2) time_hours: [StartWindowS/SecondsInHour,EndWindowS/SecondsInHour] (From Experience)
                3) lenght: [StartWindowS,EndWindowS] (From Experience)
                
            NOTE: 
                The Window for time is more homogeneous among different classes rather then length.
                This suggests that time is better as a perceived cost for people.
                Depending on the velocity we will have distribution of length that are exponential too as the 
                length depend linearly on time via velocity distribution that within a class is homogeneous.
                (Maxwellian -> delta Dirac like distribution)
        """
        self.CountFunctionsCalled += 1
        if self.BoolStrClass2IntClass:
            print("Initialize The Fitting Parameters Initial Guess And Windows")
            self.Feature2Class2Function2Fit2InitialGuess = InitFeature2Class2Function2Fit2InitialGuess(self.Features2Fit,self.IntClass2StrClass,self.StrDate)
            if self.verbose:
                print(self.StrDate)
            for Feature in self.Feature2Class2Function2Fit2InitialGuess:
                for IntClass in self.Feature2Class2Function2Fit2InitialGuess[Feature].keys():
                    StrClass = self.IntClass2StrClass[IntClass]
                    for Function2Test in self.Feature2Class2Function2Fit2InitialGuess[Feature][IntClass].keys(): 
                        NormCount = len(self.Fcm.filter(pl.col("class") == IntClass))
                        if Feature == "av_speed" or Feature == "speed_kmh":
                            pass
                        else:
                            MaxCount = self.InfoFit[Feature][StrClass]["MaxCount"]/NormCount
                            Avg = self.InfoFit[Feature][StrClass]["Avg"]
                            StartWindow = self.InfoFit[Feature][StrClass]["StartWindowS"]
                            EndWindow = self.InfoFit[Feature][StrClass]["EndWindowS"]
                        # Normalization
                        SecondsInHour = 3600
                        MetersinKm = 1000
                        if self.Feature2Class2Function2Fit2InitialGuess[Feature][IntClass][Function2Test] is not None:
                            # Change units for intervals
                            if Feature == "time":
                                pass
                            elif Feature == "time_hours":
                                Avg = Avg/SecondsInHour 
                                StartWindow = StartWindow/SecondsInHour
                                EndWindow = EndWindow/SecondsInHour 
                            elif Feature == "lenght":
                                pass
                            elif Feature == "lenght_km":
                                Avg = Avg/MetersinKm 
                                StartWindow = StartWindow/MetersinKm
                                EndWindow = EndWindow/MetersinKm
                            # Filling the Initial Guess According to the Funtion2Test
                            if "speed" in Feature:
                                self.Feature2Class2Function2Fit2InitialGuess[Feature][IntClass][Function2Test] = FillInitGuessIntervalMxGs(self.Feature2Class2Function2Fit2InitialGuess[Feature][IntClass][Function2Test],
                                                                                                                    self.Fcm,
                                                                                                                    Feature,
                                                                                                                    IntClass)                            
                            else:
                                self.Feature2Class2Function2Fit2InitialGuess[Feature][IntClass][Function2Test] = FillInitGuessIntervalPlExp(self.Feature2Class2Function2Fit2InitialGuess[Feature][IntClass][Function2Test],
                                                                                                                    MaxCount,
                                                                                                                    Avg,
                                                                                                                    StartWindow,
                                                                                                                    EndWindow,
                                                                                                                    Function2Test)

                        else:
                            print("Warning: Initial Guess Not Initialized for Class {0} and Feature {1} Day: {2}".format(IntClass,Feature,self.StrDate))
            Message = "{} Create Dictionary Class2InitialGuess: True".format(self.CountFunctionsCalled)
            if self.verbose:
                pass
#                print("Class2InitialGuess:\n",self.Feature2Class2Function2Fit2InitialGuess)

            AddMessageToLog(Message,self.LogFile)
            # Initialize The Guess Without Classes
            for Feature in self.Feature2Function2Fit2InitialGuess.keys(): 
                for Function2Test in self.Feature2Function2Fit2InitialGuess[Feature].keys():
                    NormCount = len(self.Fcm)
                    if Feature == "av_speed" or Feature == "speed_kmh":
                        pass
                    else:
                        MaxCount = self.InfoFit[Feature]["aggregated"]["MaxCount"]/NormCount
                        Avg = self.InfoFit[Feature]["aggregated"]["Avg"]
                        StartWindow = self.InfoFit[Feature]["aggregated"]["StartWindowS"]
                        EndWindow = self.InfoFit[Feature]["aggregated"]["EndWindowS"]
                    # Normalization
                    SecondsInHour = 3600
                    MetersinKm = 1000
                    if Feature == "time":
                        self.Feature2Function2Fit2InitialGuess[Feature][Function2Test] = FillInitGuessIntervalPlExp(self.Feature2Function2Fit2InitialGuess[Feature][Function2Test],
                                                                                                    MaxCount,
                                                                                                    Avg,
                                                                                                    StartWindow,
                                                                                                    EndWindow,
                                                                                                    Function2Test)
                    elif Feature == "time_hours":
                        Avg = Avg/SecondsInHour 
                        StartWindow = StartWindow/SecondsInHour
                        EndWindow = EndWindow/SecondsInHour 
                        self.Feature2Function2Fit2InitialGuess[Feature][Function2Test] = FillInitGuessIntervalPlExp(self.Feature2Function2Fit2InitialGuess[Feature][Function2Test],
                                                                                                    MaxCount,
                                                                                                    Avg,
                                                                                                    StartWindow,
                                                                                                    EndWindow,
                                                                                                    Function2Test)
                    elif Feature == "lenght":
                        self.Feature2Function2Fit2InitialGuess[Feature][Function2Test] = FillInitGuessIntervalPlExp(self.Feature2Function2Fit2InitialGuess[Feature][Function2Test],
                                                                                                    MaxCount,
                                                                                                    Avg,
                                                                                                    StartWindow,
                                                                                                    EndWindow,
                                                                                                    Function2Test)
                    elif Feature == "lenght_km":
                        Avg = Avg/MetersinKm 
                        StartWindow = StartWindow/MetersinKm
                        EndWindow = EndWindow/MetersinKm
                        self.Feature2Function2Fit2InitialGuess[Feature][Function2Test] = FillInitGuessIntervalPlExp(self.Feature2Function2Fit2InitialGuess[Feature][Function2Test],
                                                                                                    MaxCount,
                                                                                                    Avg,
                                                                                                    StartWindow,
                                                                                                    EndWindow,
                                                                                                    Function2Test)
                    elif Feature == "av_speed":
                        self.Feature2Function2Fit2InitialGuess[Feature][Function2Test] = FillInitGuessIntervalMxGs(self.Feature2Function2Fit2InitialGuess[Feature][Function2Test],
                                                                                                self.Fcm,
                                                                                                Feature,
                                                                                                None)
                    elif Feature == "speed_kmh":
                        self.Feature2Function2Fit2InitialGuess[Feature][Function2Test] = FillInitGuessIntervalMxGs(self.Feature2Function2Fit2InitialGuess[Feature][Function2Test],
                                                                                                    self.Fcm,
                                                                                                    Feature,
                                                                                                    None)
            with open(os.path.join(self.PlotDir,"DictInitialGuess_{0}.json".format(self.StrDate)),'w') as f:
                json.dump(self.Feature2Function2Fit2InitialGuess,f,cls = NumpyArrayEncoder,indent=2)
            Message = "{} Create Class Dictionary DictInitialGuess: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
## DISTRIBUTIONS
    def ComputeFitAggregated(self):
        """
            Computes the fit for the aggregated data
        """
        self.Feature2AllFitTry = InitFeature2AllFitTry(self.Feature2Function2Fit2InitialGuess)
        logger.info(f"{self.StrDate} Fit Data Without Separating by Class")
        for Feature in self.Feature2AllFitTry.keys():
            ObservedData = self.Fcm[Feature].to_list()
            # Compute the Fit for functions you are Undecided from
            if Feature == "av_speed" or Feature == "speed_kmh":
                self.Feature2AllFitTry[Feature] = ReturnFitInfoFromDict(ObservedData,
                                                                        self.Feature2Function2Fit2InitialGuess[Feature],
                                                                        self.Feature2AllFitTry[Feature],
                                                                        False)
                self.Feature2AllFitTry[Feature] = ChooseBestFit(self.Feature2AllFitTry[Feature])
            else:
                self.Feature2AllFitTry[Feature] = ComputeAndChooseBestFit(ObservedData,self.Feature2Function2Fit2InitialGuess[Feature],
                                                                        self.Feature2AllFitTry[Feature],
                                                                        False)
        if self.verbose:
            pass
        with open(os.path.join(self.PlotDir,"Feature2AllFitTry_{0}.json".format(self.StrDate)),'w') as f:
            json.dump(self.Feature2AllFitTry,f,cls = NumpyArrayEncoder,indent=2)
            #print("Dictionary Fit All Features All Functions without aggregation:\n",self.Feature2AllFitTry)
    def ComputeFitPerClass(self):
        # Save All the Tried Fit
        self.Feature2Class2AllFitTry = InitFeature2Class2AllFitTry(self.Feature2Class2Function2Fit2InitialGuess)
        # Returns for each function to try the best fit.
        logger.info(f"{self.StrDate} Fit Data Separated by Class")
        for Feature in self.Feature2Class2AllFitTry.keys():
#            print("Feature: ",Feature)
            for IntClass in self.Feature2Class2AllFitTry[Feature].keys():
#                print("Class {} ".format(IntClass))
                FcmFilteredByClass = self.Fcm.filter(pl.col("class") == IntClass)
                ObservedData = FcmFilteredByClass[Feature].to_list()
                if Feature == "av_speed" or Feature == "speed_kmh":
                    # Compute the Fit for functions you are Undecided from
                    self.Feature2Class2AllFitTry[Feature][IntClass] = ReturnFitInfoFromDict(ObservedData,
                                                                                            self.Feature2Class2Function2Fit2InitialGuess[Feature][IntClass],
                                                                                            self.Feature2Class2AllFitTry[Feature][IntClass],
                                                                                            False)
                    # Choose the Best Fit among all the tried feature
                    self.Feature2Class2AllFitTry[Feature][IntClass] = ChooseBestFit(self.Feature2Class2AllFitTry[Feature][IntClass])
                else:
                    self.Feature2Class2AllFitTry[Feature][IntClass] = ComputeAndChooseBestFit(ObservedData,
                                                                                            self.Feature2Class2Function2Fit2InitialGuess[Feature][IntClass],
                                                                                            self.Feature2Class2AllFitTry[Feature][IntClass],
                                                                                            False)
        with open(os.path.join(self.PlotDir,"Feature2Class2AllFitTry_{0}.json".format(self.StrDate)),'w') as f:
            json.dump(self.Feature2Class2AllFitTry,f,cls = NumpyArrayEncoder,indent=2)

#                if self.verbose:
#                    print("Feature: ",Feature)
#                    print("Class: ",IntClass)
        if self.verbose:
            pass
#            print("Dictionary Fit All Features All Functions per Class:\n",self.Feature2Class2AllFitTry)

    def ComputeFitDataFrame(self):
        self.Feature2IntClass2FcmDistr = defaultdict()
        self.Feature2IntClass2Feat2AvgVar = defaultdict()
        self.Freature2IntClass2FitInfo = defaultdict()
        for Feature in self.Feature2AllFitTry.keys():
            Class2FcmDistr,IntClass2Feat2AvgVar = SplitFcmByClass(self.Fcm,Feature,self.IntClass2StrClass) 
            self.Feature2IntClass2FcmDistr[Feature] = Class2FcmDistr
            self.Feature2IntClass2Feat2AvgVar[Feature] = IntClass2Feat2AvgVar
#            print("Feature: ",Feature)
#            print("Feature2IntClass2Feat2AvgVar:\n",self.Feature2IntClass2Feat2AvgVar)
        InfoPlotDistrFeat = {"figsize":(10,10),"minx":0,"miny":0,"maxx":0,"maxy":0}
        # Compute the MinMax for the Plot
        self.Feature2InfoPlotDistrFeat = {Feature: ComputeMinMaxPlotGivenFeature(self.Feature2IntClass2FcmDistr[Feature],InfoPlotDistrFeat) for Feature in self.Feature2AllFitTry.keys()}
        self.Feature2DistributionPlot = {Feature: {"fig":None,"ax":None} for Feature in self.Feature2AllFitTry.keys()}
        self.Feature2FitDf = FitDataFrame(self.Feature2Class2AllFitTry,self.PlotDir)        


## ------- FUNDAMENTAL DIAGRAM ------ ##
    def ComputeMFDVariablesClass(self):
        '''
            Description:
                Computes the MFD variables (t,population,speed) -> and the hysteresis diagram:
                    1) Aggregated data for the day
                    2) Conditional to class
            Save them in two dictionaries 
                1) self.MFD = {time:[],population:[],speed:[]}
        '''
        if self.ReadFcmBool:
            if "start_time" in self.Fcm.columns:
                if os.path.isfile(os.path.join(self.PlotDir,"MFD_{}.csv".format(self.StrDate))):
                    self.MFD = pd.read_csv(os.path.join(self.PlotDir,"MFD_{}.csv".format(self.StrDate)))
                    self.MFD = pl.from_pandas(self.MFD)
                else:
                    self.MFD = pl.DataFrame({"time":[]})
                    for Class,FcmClass in self.Fcm.groupby("class"): 
                        self.MFD = AddColumns2MFD(self.MFD,FcmClass,Class,self.BinTimestamp,False)
                    for NewClass,FcmNewClass in self.Fcm.groupby("class_new"):
                        self.MFD = AddColumns2MFD(self.MFD,FcmNewClass,NewClass,self.BinTimestamp,True)
                    self.MFD.write_csv(os.path.join(self.PlotDir,"MFD_{}.csv".format(self.StrDate)))    
            self.ComputedMFD = True

    def ComputeVariablesForMFDPlot(self):
        """
        Description:
            Computes the variables for the MFD Plot.
        """
        self.MFDPlotDir = os.path.join(self.PlotDir,"MFDPlot_{0}.csv".format(self.StrDate))
        if os.path.exists(self.MFDPlotDir):
            Classes = [Class for Class,_ in self.Fcm.groupby("class")]
            self.MFD2Plot = ReadFilePlotMFD(self.PlotDir,self.StrDate)
            self.Class2RelativeChange = GetRelativeChange(self.MFD2Plot,Classes,False)
            self.Class2NewRelativeChange = GetRelativeChange(self.MFD2Plot,Classes,True)
        else:
            self.MinMaxPlotPerClass = {Class: defaultdict() for Class,_ in self.Fcm.groupby("class")}
            self.MinMaxPlotPerClassNew = {Class: defaultdict() for Class,_ in self.Fcm.groupby("class")}  
            self.Class2RelativeChange = {Class: defaultdict() for Class,_ in self.Fcm.groupby("class")}
            self.Class2NewRelativeChange = {Class: defaultdict() for Class,_ in self.Fcm.groupby("class_new")}     
            for Class,_ in self.Fcm.groupby("class"):
                # OLD CLASSIFICATION
                self.MFD2Plot, self.MinMaxPlotPerClass,self.Class2RelativeChange[Class] = GetMFDForPlot(MFD = self.MFD,
                                                                                        MFD2Plot = self.MFD2Plot,
                                                                                        MinMaxPlot = self.MinMaxPlotPerClass,
                                                                                        Class = Class,
                                                                                        case = None,
                                                                                        NewClass= False,
                                                                                        bins_ = 20)
            for Class,_ in self.Fcm.groupby("class_new"):
                # NEW CLASSIFICATION
                self.MFD2Plot, self.MinMaxPlotPerClassNew,self.Class2NewRelativeChange[Class] = GetMFDForPlot(MFD = self.MFD,
                                                                                                    MFD2Plot = self.MFD2Plot,
                                                                                                    MinMaxPlot = self.MinMaxPlotPerClassNew,
                                                                                                    Class = Class,
                                                                                                    case = None,
                                                                                                    NewClass= True,
                                                                                                    bins_ = 20)
            self.MFD2Plot = pl.DataFrame(pd.DataFrame(self.MFD2Plot))
            self.MFD2Plot.write_csv(self.MFDPlotDir)
            

    def PrepareSpeedEvolutionNewClassConsideringRoadClassification(self):
        """
            @brief: Prepare the Speed Evolution considering the Road Classification
            Class2Speed: {Class:{TimeInterval:{Road:<speed>_{class}}}}
            Class2SpeedH: {Class:{TimeInterval:{Road:<speed>_{hierarchical class}}}}
            Class2SpeedO: {Class:{TimeInterval:<speed>_{class}}}
            TimeIntervalsDt: [datetime]
        """
        self.Class2Speed,self.Class2SpeedH,self.Class2SpeedO,self.TimeIntervalsDt = PrepareSpeedEvolutionNewClassConsideringRoadClassification(self.ClassNew2TimeInterval2Road2SpeedActualRoads,self.Class2TimeInterval2Road2SpeedNew,self.Class2TimeInterval2Road2Speed)        

    def PlotSpeedEvolutionAllSubnets(self):
        """
            @brief: Plot the Speed Evolution for all the subnets
        """
        PlotSpeedEvolutionFromGeoJson(self.Class2TimeInterval2Road2Speed,self.Class2TimeInterval2Road2SpeedNew,self.BinStringHour,self.PlotDir)        

    def ComputeCFAR(self):
        """
            @brief: Compute the CFAR for the Speed Evolution
            In particular sees if Speed Original - Speed Hierarchical is >> 0
            and characterizes the signal of Jam
        """
        self.Class2Signal = defaultdict()
        self.Class2Cut = defaultdict()
        self.Class2CFARClassification = defaultdict()
        self.Class2IsSignalPTest = defaultdict()
        # With 96 bins Pfa < 1
        Pfa = 0.001
        # 1 hour interval
        half_train = 4 
        half_guard = 5
        N = 2*(half_train)
        alpha = N*(Pfa**(-1/N) - 1)
        for Class in self.Class2SpeedO.keys():
            mu = np.nanmean(np.array(self.Class2SpeedO[Class]) - np.array(self.Class2SpeedH[Class]))
            sigma = np.nanstd(np.array(self.Class2SpeedO[Class]) - np.array(self.Class2SpeedH[Class]))
            self.Class2Signal[Class] = np.array([np.array(self.Class2SpeedO[Class])[i] - np.array(self.Class2SpeedH[Class])[i] if not (np.isnan(np.array(self.Class2SpeedH[Class])[i]) or np.array(self.Class2SpeedH[Class])[i] is None or np.array(self.Class2SpeedH[Class])[i] == 0) else mu for i in range(len(self.Class2SpeedO[Class]))])
            # Consider Signal if it is Outside 95% Confidence Interval if it was a Gaussian
            Count,Speed = np.histogram(self.Class2Signal[Class],bins = 30)
            mask_is_signal_p_test,idces_percentile = Ptest(Speed,self.Class2Signal[Class],mu,sigma,percentile = 0.95)
            assert len(mask_is_signal_p_test) == len(self.Class2Signal[Class])
            self.Class2IsSignalPTest[Class] = [1 if mask_is_signal_p_test[i] else 0 for i in range(len(mask_is_signal_p_test))]
            # Now I have that I have Jam, just if the signal is outside the 95% Confidence Interval
            fig,ax = plt.subplots()
            ax.plot(self.TimeIntervalsDt,self.Class2Signal[Class],label = r"$\langle v_o - v_h \rangle$")

#            ax.scatter(np.array(self.TimeIntervalsDt)[self.Class2IsSignalPTest[Class]],np.array(self.Class2Signal[Class])[self.Class2IsSignalPTest[Class]],label = "Jam")
            ax.axhline(y = mu, label = r"$\langle v_o - v_h \rangle$",color = 'r', linestyle = '-')
            try:
                SpeedThreshold = np.array(self.Class2Signal[Class][mask_is_signal_p_test]).argmin()
                ax.axhline(y = SpeedThreshold, label = r"$\langle v_o - v_h \rangle$ 95% Confidence Interval",color = 'g', linestyle = '--')
            except:
                pass
            ax.plot(self.TimeIntervalsDt,self.Class2SpeedO[Class],label = r"$\langle v_o \rangle$")
            ax.plot(self.TimeIntervalsDt,self.Class2SpeedH[Class],label = r"$\langle v_h \rangle$")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
            tick_locations = np.arange(0, len(self.TimeIntervalsDt), 8)
            tick_labels = self.BinStringHour[::8]
            ax.set_xticks(tick_locations)
            ax.set_xticklabels(tick_labels, rotation=90)
            plt.savefig(os.path.join(self.PlotDir,"SignalClass_{}.png".format(Class)))
            plt.close()
#            self.Class2Signal[Class][np.where(self.Class2Signal[Class] < 0)] = 0
            self.Class2CFARClassification[Class], self.Class2Cut[Class] = soca_cfar(half_guard, half_train, alpha, self.Class2Signal[Class])

    def ComputeAndPlotDailyTrafficIndicator(self):
        """
            @brief: Compute and Plot the Daily Traffic Indicator
            The formula we use is:
            Index = [(Speed Original - Speed Hierarchical)/Speed Natural]*FractionPeoplePerClassInTime
            In this way we have that traffic is more intense if:
                - Speed Original - Speed Hierarchical is high
                - Speed Natural is low
                - FractionPeoplePerClassInTime is high
            Class2TrafficIndex: {Class:[Index]} where Index is the traffic indicator    
            It is 1 super trafficked, 0 not trafficked
        """
        self.Class2traffickIndex = defaultdict()
        self.Class2CriticalTraffic = defaultdict()
        for Class in self.Class2SpeedO.keys():
            NaturalSpeedSubnet = np.nanmean(np.array(self.Class2SpeedO[Class]))
#            SpeedNewSubnet = np.nanmean(np.array(self.Class2SpeedH[Class]))
            NaturalSpeedSubnetV = np.ones(len(self.Class2SpeedO[Class]))*NaturalSpeedSubnet
            self.Class2traffickIndex[Class] = np.array(self.Class2SpeedO[Class] - np.array(self.Class2SpeedH[Class]))/NaturalSpeedSubnetV
            MaxPeopleClass = max([len(self.OrderedClass2TimeDeparture2UserId[Class][t]) for t in self.OrderedClass2TimeDeparture2UserId[Class].keys()])
            FractionPeoplePerClassInTime = np.array([len(self.OrderedClass2TimeDeparture2UserId[Class][t]) for t in self.OrderedClass2TimeDeparture2UserId[Class].keys()])/MaxPeopleClass
            self.Class2traffickIndex[Class] = self.Class2traffickIndex[Class]*FractionPeoplePerClassInTime
            self.Class2traffickIndex[Class] = [self.Class2traffickIndex[Class][i] if self.Class2traffickIndex[Class][i] > 0 else 0 for i in range(len(self.Class2traffickIndex[Class]))]
            Mask = np.array([True if self.Class2IsSignalPTest[Class][i] > 0 else False for i in range(len(self.Class2IsSignalPTest[Class]))])
            IndexMinimumError = None
            MinimumError = 1e10
            for i in range(len(Mask)):
                if Mask[i]:
                    if self.Class2Signal[Class][i] < MinimumError:
                        MinimumError = self.Class2Signal[Class][i]
                        IndexMinimumError = i
            self.Class2CriticalTraffic[Class] = IndexMinimumError        
        fig,ax = plt.subplots(2,2,figsize = (20,20))
        Class2Idx = {0:(0,0),1:(0,1),2:(1,0),3:(1,1)}
        for Class in self.Class2traffickIndex.keys():
            ax0 = Class2Idx[Class][0]
            ax1 = Class2Idx[Class][1]
            
            ax[ax0,ax1].plot(self.TimeIntervalsDt,self.Class2traffickIndex[Class],label = "Class {}".format(Class))
            ax[ax0,ax1].hlines(self.Class2traffickIndex[Class][IndexMinimumError],self.TimeIntervalsDt[0],self.TimeIntervalsDt[-1],label = "Critical Traffic")
            ax[ax0,ax1].set_xlabel("Time")
            ax[ax0,ax1].set_ylabel(r"$\frac{(v_o(t) - v_h(t))}{\langle v_n \rangle_t}\frac{N_{class}}{N_{max}}$")
            tick_locations = np.arange(0, len(self.TimeIntervalsDt), 8)
            tick_labels = self.BinStringHour[::8]
            ax[ax0,ax1].set_xticks(tick_locations)
            ax[ax0,ax1].set_xticklabels(tick_labels, rotation=90)

        plt.savefig(os.path.join(self.PlotDir,"DailyTrafficIndicator.png"))
        with open(os.path.join(self.PlotDir,"Class2traffickIndex.json"),'w') as f:
            json.dump(self.Class2traffickIndex,f,indent=2)
        plt.close()

    def PTestForTraffic(self):
        """
            @brief: Perform the PTest for Traffic
            Computes vector:
              self.Class2IsSignalPTest[Class]: 1 if the signal is a jam, 0 otherwise
              According to the prinple of ptet
        """
        self.Class2IsSignalPTest = defaultdict()
        for Class in self.Class2Signal.keys():
            y_measured,x = np.histogram(self.Class2Signal[Class],bins = 30)
            mu = np.nanmean(self.Class2Signal[Class])
            sigma = np.nanstd(self.Class2Signal[Class])
            fit,StdError,ConvergenceSuccess,FittedData,_,_ = FitAndStdErrorFromXY(x[:-1],y_measured,"gaussian",[1,mu,sigma],maxfev = 50000,interval = [])
            if ConvergenceSuccess:
                A = fit[0][0]
                mu = fit[0][1]
                sigma = fit[0][2]
                logger.info(f"Day: {self.StrDate} Class {Class} mu: {mu} sigma: {sigma}")
                fig, ax = plt.subplots()
                ax.hist(self.Class2Signal[Class],bins = 30)
                ax.plot(x[1:],np.exp(-0.5*((x[1:]-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi)))
                plt.savefig(os.path.join(self.PlotDir,"HistogramClass_{}.png".format(Class)))
                plt.close()

            mask_is_signal_p_test,_ = Ptest(x,self.Class2Signal[Class],mu,sigma,percentile = 0.95)
            assert len(mask_is_signal_p_test) == len(self.Class2Signal[Class])
            self.Class2IsSignalPTest[Class] = [1 if mask_is_signal_p_test[i] else 0 for i in range(len(mask_is_signal_p_test))]

    def FromGeoJson2Grid(self):
        """
            @brief: Convert the GeoJson to a Grid
        """
        if os.path.isfile(os.path.join(self.PlotDir,"Grid.geojson")):
            self.Grid = gpd.read_file(os.path.join(self.PlotDir,"Grid.geojson"))
        from GeographyFunctions import FromGeoJson2Grid
        self.Grid = FromGeoJson2Grid(self.GeoJson,500,500)



    def PlotPtest(self):
        PlotPtest(self.Class2IsSignalPTest,self.TimeIntervalsDt,self.PlotDir)

##   PLOT CFAR
    def PlotCFAR(self):
        """
            @brief: Plot the CFAR for the Speed Evolution
        """
        PlotCFAR(self.Class2Signal,self.Class2Cut,self.Class2CFARClassification,self.TimeIntervalsDt,self.PlotDir)
    
    def PlotMFD(self):
        """
        Description:
            Plots the Fundamental Diagram for the calculated MFD (Mobility Fundamental Diagram) and per class.
        <<<
            This function plots the Fundamental Diagram for the calculated MFD (Mobility Fundamental Diagram) and per class. 
            The Fundamental Diagram shows the average speed and the standard deviation of the speed for each population bin.
            The population bins are determined by the number of vehicles in each bin.
        >>>
        Parameters:
            self (object): The instance of the class.
        
        Returns:
            NOTE: Aggregated Important Variables Initialized
            self.MFD2Plot (dict): {"binned_av_speed": [], "binned_sqrt_err_speed": [], "bins_population": []}
            self.MinMaxPlot (dict): {"aggregated": {"population": {"min": int, "max": int}, "speed_kmh": {"min": int, "max": int}}}
            NOTE: Per Class Important Variables Initialized
            self.Class2MFD2Plot (dict): {Class: {"binned_av_speed": [], "binned_sqrt_err_speed": [], "bins_population": []}}
            self.MinMaxPlotPerClass (dict): {Class: {"population": {"min": int, "max": int}, "speed_kmh": {"min": int, "max": int}}}
        
        Raises:
            None
        """
        if self.ComputedMFD: 
            for Class,_ in self.Fcm.groupby("class"):
                ColSpeed = f"speed_kmh_{Class}"
                ColPop = f"population_{Class}"
                PlotHysteresis(MFD = self.MFD,
                            Title = "Hysteresis Cycle Class {}".format(self.IntClass2StrClass[Class]),
                            ColPop= ColPop,
                            ColSpeed= ColSpeed,
                            SaveDir = self.PlotDir,
                            NameFile = "HysteresysClass_{}.png".format(self.IntClass2StrClass[Class]))
                ColNewSpeed = f"new_speed_kmh_{Class}"
                ColNewPop = f"new_population_{Class}"
                PlotHysteresis(MFD = self.MFD,
                            Title = "Hysteresis Cycle Class {}".format(self.IntClass2StrClass[Class]),
                            ColPop= ColNewPop,
                            ColSpeed= ColNewSpeed,
                            SaveDir = self.PlotDir,
                            NameFile = "New_HysteresysClass_{}.png".format(self.IntClass2StrClass[Class]))                
                if self.BoolStrClass2IntClass:
                    # OLD CLASSIFICATION
                    ColumnSpeed = f"bin_{ColSpeed}"
                    ColumnPop = f"bins_{ColPop}"
                    VarianceSpeed = f'binned_sqrt_err_{ColSpeed}'
                    PlotMFD(self.MFD2Plot[ColumnPop],
                                self.MFD2Plot[ColumnSpeed],
                                self.MFD2Plot[VarianceSpeed],
                                RelativeChange = self.Class2RelativeChange[Class],
                                SaveDir = self.PlotDir,
                                Title = "{}".format(self.IntClass2StrClass[Class]),
                                NameFile = "MFD_{}.png".format(Class))
                    ColumnSpeed = f"bin_{ColNewSpeed}"
                    ColumnPop = f"bins_{ColNewPop}"
                    VarianceSpeed = f'binned_sqrt_err_{ColNewSpeed}'                    
                    # NEW CLASSIFICATION
                    PlotMFD(self.MFD2Plot[ColumnPop],
                                self.MFD2Plot[ColumnSpeed],
                                self.MFD2Plot[VarianceSpeed],
                                RelativeChange = self.Class2NewRelativeChange[Class],
                                SaveDir = self.PlotDir,
                                Title = "{} (hierarchical)".format(self.IntClass2StrClass[Class]),
                                NameFile = "MFDNew_{}.png".format(Class))
                else:
                    print("Warning: Fondamental Diagram Not Computed for Class Since IntClass2Str is Not Initialized")

## SPARSENESS DATA TIME  ##
    def PlotNPeopNRoadsClass(self): 
        PlotNPeopNRoadsClass(self.OrderedClass2TimeDeparture2UserId,self.IntClass2RoadsIncreasinglyIncludedIntersection,"Hierarchical",self.PlotDir)
        PlotNPeopNRoadsClass(self.Class2TimeDeparture2UserId,self.IntClass2Roads,"Not-Hierarchical",self.PlotDir)

# --- TIME PERCORRENCE PLOT --- #
    def PlotTimePercorrenceDistributionAllClasses(self):
        """
            NOTE: We compute for 96 bins (15 minutes each), the time of percorrence estimated as the lenght of the road divided by the average speed.
            NOTE: We consider the distribution for the network indipendently of the class.
                and the subnetwork associated to the class. (This second is the one we look at to look for traffic)
        """
        self.CountFunctionsCalled += 1
        if self.ReadVelocitySubnetBool:
            print("Plotting TimePercorrence Distribution")
            ListNames = ["Class2Time2Distr_{0}.json".format(self.StrDate),"Class2AvgTimePercorrence_{0}.json".format(self.StrDate)]
            ListFileNames = GenerateListFilesCommonBaseDir(self.PlotDir,ListNames)
            ListDict,Upload = UploadDictsFromListFilesJson(ListFileNames)
            if Upload:
                self.Class2Time2Distr = ListDict[0]
                self.Class2AvgTimePercorrence = ListDict[1]
                pass
            else:
                pass
            if True:
                self.Class2Time2Distr = {IntClass:[] for IntClass in self.IntClass2StrClass.keys()} # For key Shape (96,Number of Roads)
                self.Class2AvgTimePercorrence = {IntClass:[] for IntClass in self.IntClass2StrClass.keys()} # For key Shape (96,)
                # Per Class
                for IntClass in self.IntClass2StrClass.keys():
                    File2Save = os.path.join(self.PlotDir,"TimePercorrenceDistribution_Class_{0}_{1}.png".format(IntClass,self.StrDate))
                    StrTimesLabel = []
                    self.Class2Time2Distr[IntClass],self.Class2AvgTimePercorrence[IntClass] = PlotTimePercorrenceDistribution(self.Class2DfSpeedAndTimePercorrenceRoads[IntClass],
                                                                                                                              self.Class2Time2Distr[IntClass],
                                                                                                                              self.Class2AvgTimePercorrence[IntClass],
                                                                                                                              StrTimesLabel,
                                                                                                                              File2Save)
                    if not os.path.isfile(os.path.join(self.PlotDir,"GeoJson_{0}.geojson".format(self.StrDate))): 
                        self.GeoJson = ComputeTimePercorrence(self.GeoJson,self.Class2DfSpeedAndTimePercorrenceRoads[IntClass],IntClass,self.StrDate)
                        self.GeoJson = BuildListStepsGivenDay(self.GeoJson,self.StrDate,"AvSpeed_")
                        self.GeoJson = BuildListStepsGivenDay(self.GeoJson,self.StrDate,"TimePercorrence_")
                    else:
                        pass

            self.TimePercorrenceBool = True
            self.PlotTimePercorrenceConditionalLengthRoad()
                
                
    def GenerateVideoEvolutionTimePercorrence(self):
            if not os.path.isfile(os.path.join(self.PlotDir,"GeoJson_{0}.geojson".format(self.StrDate))): 
                self.GeoJson.to_file(os.path.join(self.PlotDir,"GeoJson_{0}.geojson".format(self.StrDate)))
            else:
                self.GeoJson = gpd.read_file(os.path.join(self.PlotDir,"GeoJson_{0}.geojson".format(self.StrDate)))
#            VideoEvolutionTimePercorrence(self.GeoJson,"TimePercorrence_",self.StrDate,self.PlotDir)
            VideoEvolutionTimePercorrence(self.GeoJson,"AvSpeed_",self.StrDate,self.PlotDir)

            print("GeoJson")
            print(self.GeoJson)

            SaveProcedure(BaseDir=self.PlotDir,
                        ListKeys = ["Class2Time2Distr","Class2AvgTimePercorrence"],
                            ListDicts = [self.Class2Time2Distr,self.Class2AvgTimePercorrence],
                            ListFormats = [self.StrDate],
                            Extension = ".json")
            Upload = False
            MessagePlotTimePercorrenceDistributionAllClasses(self.CountFunctionsCalled,self.LogFile,Upload)



    def PlotTimePercorrenceConditionalLengthRoad(self):
        """
            Description:
                1) Draws the distribution of road lengths
                2) Draws the time percorrence distribution conditioned to the length of the road.

        """
        #self.PlotDistributionLengthRoadPerClass()
        self.CountFunctionsCalled += 1
        
        # Drop rows with NaN values in the 'poly_length' column
        self.GeoJson = self.GeoJson.dropna(subset=['poly_length'])
        # Calculate the histogram
        CountLengths,Lengths = np.histogram(self.GeoJson["poly_length"][~np.isnan(self.GeoJson["poly_length"])],bins = 10)
        self.Lenght2Roads = GetLengthPartitionInGeojSon(self.GeoJson,Lengths)
        self.Length2Class2Time2Distr = {Length:{IntClass:[] for IntClass in self.IntClass2StrClass.keys()} for Length in Lengths}
        self.Length2Class2AvgTimePercorrence = {Length:{IntClass:[] for IntClass in self.IntClass2StrClass.keys()} for Length in Lengths}
        for IntClass in self.IntClass2StrClass.keys():
            for Length,Roads in self.Lenght2Roads.items():
                File2Json = os.path.join(self.PlotDir,"Length2Class2Time2Distr_{0}.json".format(self.StrDate))
                if not os.path.isfile(File2Json):
                    StrTimesLabel = []
                    File2Save = os.path.join(self.PlotDir,"TimePercorrenceDistribution_Class_{0}_{1}_Length_{2}.png".format(IntClass,self.StrDate,round(Length,2)))
                    Length2VelTimePerccorenceClass = self.Class2DfSpeedAndTimePercorrenceRoads[IntClass].filter(pl.col("poly_id").is_in(Roads))
                    if len(Roads)>0:
                        self.Length2Class2Time2Distr[Length][IntClass],self.Length2Class2AvgTimePercorrence[Length][IntClass] = PlotTimePercorrenceDistribution(Length2VelTimePerccorenceClass,
                                                    self.Length2Class2Time2Distr[Length][IntClass],
                                                    self.Length2Class2AvgTimePercorrence[Length][IntClass],
                                                    StrTimesLabel,
                                                    File2Save)

                        Upload = False
                    else:
                        # Do not need to add any road inside the length partition
                        pass
                else:
                    Upload = True
                    with open(os.path.join(self.PlotDir,"Length2Class2Time2Distr_{0}.json".format(self.StrDate)),'w') as f:
                        json.dump(self.Length2Class2Time2Distr,f,cls = NumpyArrayEncoder,indent=2)
                    with open(os.path.join(self.PlotDir,"Length2Class2AvgTimePercorrence_{0}.json".format(self.StrDate)),'w') as f:
                        json.dump(self.Length2Class2AvgTimePercorrence,f,cls =NumpyArrayEncoder,indent=2)
        if Upload:
            Message = "{} Plotting TimePercorrence Distribution Conditioned to Length Road: True\n".format(self.CountFunctionsCalled)
            Message += "\tUpload Length2Class2Time2Distr, Length2Class2AvgTimePercorrence"
            AddMessageToLog(Message,self.LogFile)
        else:
            Message = "{} Plotting TimePercorrence Distribution Conditioned to Length Road: True\n".format(self.CountFunctionsCalled)
            Message += "\tComputed Length2Class2Time2Distr, Length2Class2AvgTimePercorrence"
            AddMessageToLog(Message,self.LogFile)

    def PlotDistributionLengthRoadPerClass(self):
        """
          Compute the distribution of roads length for each class.
          NOTE: The idea is to try to understand if there is some bias in the length of the roads for each class.
          NOTE: By bias I mean that the distributions are peaked around some length, and the centroid for different
          classes are different, maybe in the sense of sigmas in a gaussian distribution. 
        """
        plt.subplots(1,1,figsize = (10,10))
        self.IntClass2RoadLengthDistr = {IntClass:[] for IntClass in self.IntClass2StrClass.keys()}
        for IntClass in self.IntClass2StrClass.keys():
            Lengths = self.Class2DfSpeedAndTimePercorrenceRoads[IntClass]["poly_length"]
            n,bins = np.histogram(Lengths,bins = 100)
            self.IntClass2RoadLengthDistr[IntClass] = {"n":n,"bins":bins}
            sns.histplot(Lengths,bins = 100,label = self.IntClass2StrClass[IntClass],kde = True)
        plt.legend()
        plt.xlabel("Length [m]")
        plt.ylabel("Counts")
        plt.title("Distribution of Roads Length for Each Class")
        plt.savefig(os.path.join(self.PlotDir,"DistributionRoadLengthPerClass_{0}.png".format(self.StrDate)))
        plt.close()

##--------------- Plot Network --------------## 
    def PlotIncrementSubnetHTML(self):
        """
            NOTE: Informations about the subnet are taken from Subnet Files
            Description:
                Plots the subnetwork. Considers the case of intersections
        """
        self.CountFunctionsCalled += 1
        Message = PlotIncrementSubnetHTML(self.GeoJson,
                                self.IntClass2StrClass,
                                self.centroid,
                                self.PlotDir,
                                self.StrDate,
                                self.ReadFluxesSubIncreasinglyIncludedIntersectionBool,
                                self.ReadGeojsonBool,
                                self.Class2Color,
                                "SubnetsIncrementalInclusion",
                                self.verbose)
        Message = "{} ".format(self.CountFunctionsCalled) + Message
        AddMessageToLog(Message,self.LogFile)

    def PlotSubnetHTML(self):
        """
            Description:
                Plots in HTML the road network with its subnets.
                NOTE: 
                    Does not consider the intersection
        """
        self.CountFunctionsCalled += 1
        Message = PlotSubnetHTML(self.GeoJson,
                                 self.IntClass2StrClass,
                                 self.centroid,
                                 self.PlotDir,
                                 self.StrDate,
                                 self.ReadFluxesSubBool,
                                 self.ReadGeojsonBool,
                                 self.BoolStrClass2IntClass,
                                 self.Class2Color,
                                 self.verbose)
        Message = "{} ".format(self.CountFunctionsCalled) + Message
        AddMessageToLog(Message,self.LogFile)
        
    def PlotFluxesHTML(self):
        '''
            Description:
                Plots in .html the map of the bounding box considered.
                For each road color with the fluxes.
                    1) FT
                    2) TF
                    3) TF + FT
        '''
        self.CountFunctionsCalled += 1
        Message = PlotFluxesHTML(self.GeoJson,
                       self.TimedFluxes,
                       self.centroid,
                       self.StrDate,
                       self.PlotDir,
                       self.ReadTime2FluxesBool,
                       "Fluxes",
                       "TailFrontFluxes",
                       "FrontTailFluxes")
        Message = "{} ".format(self.CountFunctionsCalled) + Message
        AddMessageToLog(Message,self.LogFile)
    
    def PlotTimePercorrenceHTML(self):
        """
            Description:
                Plots in .html the map of the bounding box considered.
                For each class color the road subnetwork according to time of percorrence.
        """
        self.CountFunctionsCalled += 1
        Message = PlotTimePercorrenceHTML(self.GeoJson,
                                self.Class2DfSpeedAndTimePercorrenceRoads,
                                self.IntClass2BestFit,
                                self.ReadGeojsonBool,
                                self.ReadVelocitySubnetBool,
                                self.centroid,
                                self.PlotDir,
                                self.StrDate,
                                self.Class2Color,
                                "AvSpeed","TimePercorrence",
                                self.verbose)
        Message = "{} ".format(self.CountFunctionsCalled) + Message
        AddMessageToLog(Message,self.LogFile)




    def PlotDistrFeature(self):
        PlotComparisonDistributionSpeedNewOld(self.Fcm,self.PlotDir)
        for Feature in self.Feature2AllFitTry.keys():
            if "speed" in Feature:
                pass
            else:
                y,x = np.histogram(self.Fcm[Feature],bins = 50)  
                y = y/np.sum(y)
                x_mean = np.mean(x)
                fig,ax = plt.subplots(1,1,figsize = (10,10))
                fit = pwl.Fit(np.array(self.Fcm[Feature]),
                            xmin = min(np.array(self.Fcm[Feature])),
                            xmax = max(np.array(self.Fcm[Feature]))
                            )
                ax.plot(x[:-1],y)
                ax.plot(x[:-1],x[:-1]**[-fit.alpha])
                ax.vlines(x_mean,0,1,linestyles = "dashed")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(self.Feature2Label[Feature])
                ax.set_title("Feature: {0} Power Law Fit Alpha: {1} Sigma: {2}".format(Feature,fit.alpha,fit.sigma))
                plt.savefig(os.path.join(self.PlotDir,"Feature_{0}_PowerLawFit.png".format(Feature)),dpi = 200)
                plt.close()


    def PlotDistrPerClass(self):
        """
            Description:
                Computes the fit for each Feature each class and aggregated.
                Plots them and save the variables in:
            Return:
                InfoDayFit: dict -> {IntClass: {Feature: {Function: [A,b]}}}

        """
        if self.verbose:
            print("++++++ Aggregated Fit ++++++")

        self.CountFunctionsCalled += 1
        self.InfoDayFit = {IntClass: {} for IntClass in self.IntClass2StrClass.keys()}
        # Compute the Fcm Partition For Each Feature

        self.CountFunctionsCalled += 1
        fig,ax = PlotFeatureDistrSeparatedByClass(self.Feature2IntClass2FcmDistr,
                                            self.Feature2InfoPlotDistrFeat,
                                            self.IntClass2StrClass,
                                            self.Feature2Class2AllFitTry,
                                            self.Feature2Legend,
                                            self.Feature2IntervalBin,
                                            self.Feature2IntervalCount,
                                            self.Feature2Label,
                                            self.Feature2ShiftBin,
                                            self.Feature2ShiftCount,
                                            self.Feature2ScaleBins,
                                            self.Feature2ScaleCount,
                                            self.Feature2DistributionPlot,
                                            self.PlotDir,
                                            self.Feature2SaveName)

        for Feature in self.Feature2Class2AllFitTry.keys():
            for IntClass in self.Feature2Class2AllFitTry[Feature].keys():
                fig,ax = PlotFeatureSingleClass(self.Feature2IntClass2FcmDistr[Feature][IntClass],
                                        self.Feature2Class2AllFitTry[Feature][IntClass],
                                        self.Feature2IntervalBin[Feature],
                                        self.Feature2IntervalCount[Feature],
                                        self.Feature2Label[Feature],
                                        self.Feature2ShiftBin[Feature],
                                        self.Feature2ShiftCount[Feature],
                                        self.Feature2ScaleBins[Feature],
                                        self.Feature2ScaleCount[Feature])
                if not os.path.exists(os.path.join(self.PlotDir,"Fit")):
                    os.makedirs(os.path.join(self.PlotDir,"Fit"),exist_ok = True)
                if self.Feature2Class2AllFitTry[Feature][IntClass]["best_fit"] == "":
                    fig.savefig(os.path.join(self.PlotDir,"Fit",'{0}_Class_{1}_{2}_{3}.png'.format("NonConvergedFit",IntClass,self.Feature2SaveName[Feature],self.StrDate)),dpi = 200)
                else:
                    fig.savefig(os.path.join(self.PlotDir,"Fit",'{0}_Class_{1}_{2}_{3}.png'.format(self.Feature2Class2AllFitTry[Feature][IntClass]["best_fit"],IntClass,self.Feature2SaveName[Feature],self.StrDate)),dpi = 200)
                plt.close()

    def PlotSpeedEvolutionTransitionClasses(self):
        """
            Description:
                Computes the speed evolution for each class and the transition between classes.
                N_{on} (t) = N_{old} \intersection N_{new} (t)
                N_{on} (t) := number of trajectories that are re-assigned old -> new,
                             that started in the time interval [t,t +dt]
                N_{on} (t) = ClassOld2ClassNewTimeInterval2Transition[ClassOld][ClassNew][TimeInterval]
        """
        PlotComparisonSpeedClassesAndNPeopleTogether(self.Class2TimeInterval2Road2Speed,self.Class2TimeInterval2Road2SpeedNew,self.ClassNew2TimeInterval2Road2SpeedActualRoads,self.ClassOld2ClassNewTimeInterval2Road2SpeedNew,self.ClassOld2ClassNewTimeInterval2Transition,self.OrderedClass2TimeDeparture2UserId,self.PlotDir)
#        PlotSpeedEvolutionTransitionClasses(self.ClassOld2ClassNewTimeInterval2Road2SpeedNew,self.Class2TimeInterval2Road2SpeedNew,self.ClassOld2ClassNewTimeInterval2Transition,self.OrderedClass2TimeDeparture2UserId,self.Class2TimeDeparture2UserId,self.PlotDir)
#        PlotSpeedEvolutionNewClassConsideringRoadClassification(self.ClassNew2TimeInterval2Road2SpeedActualRoads,self.Class2TimeInterval2Road2SpeedNew,self.Class2TimeInterval2Road2Speed,self.PlotDir)
        

    def PlotTransitionMatrix(self):
        self.Tij = PlotTransitionClass2ClassNew(self.DfComparison,self.PlotDir)

    def PlotTransitionClassesInTime(self):
        PlotTransitionClassesInTime(self.ClassOld2ClassNewTimeInterval2Transition,self.PlotDir)
## ------------------- PRINT UTILITIES ---------------- #
    def PrintTimeInfo(self):
        print("StrDate: ", self.StrDate, "Type: ", type(self.StrDate))
        print("TimeStampDate: ", str(self.TimeStampDate), "Type: ", type(self.TimeStampDate))
        print("str(Timestamp2Datetime(self.TimeStampDate)): ", str(Timestamp2Datetime(self.TimeStampDate)), "Type: ", type(self.TimeStampDate))    
        print("Iterations: ", str(self.iterations), "Type: ", type(self.iterations))
        print("Day in seconds: ", str(self.day_in_sec), "Type: ", type(self.day_in_sec))
        print("Date: ", self.Date, "Type: ", type(self.Date))

    def PrintInputDirectories(self):
        print("Input Directories: ")
        print(self.DictDirInput)
        print(self.GeoJsonFile)
    def PrintBool(self):
        print("Read Fcm: ",self.ReadFcmBool)
        print("Read FcmCenters: ",self.ReadFcmCentersBool)
        print("Read FcmNew: ",self.ReadFcmNewBool)
        print("Read GeoJson: ",self.ReadGeojsonBool)
        print("Read Fluxes: ",self.ReadFluxesBool)
        print("Time2Fluxes: ",self.ReadTime2FluxesBool)
        print("Read FluxesSub: ",self.ReadFluxesSubBool)
        print("Read Velocity Subnet: ",self.ReadVelocitySubnetBool)
        print("StrClass2IntClass: ",self.BoolStrClass2IntClass)
        print("MFD: ",self.ComputedMFD)
        print("Stats: ",self.ReadStatsBool)
        print("Incremental subnet: ",self.ReadFluxesSubIncreasinglyIncludedIntersectionBool)


