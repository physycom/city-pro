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
                1) Class2MFD = {Class:pl.DataFrame{"population":[],"time":[],"speed_kmh":[],"av_speed"}:[]} Stored in: os.path.join(self.PlotDir,"MFD_{0}_{1}.csv".format(self.StrDate,Class))
                2) Class2MFD2Plot = {Class:{"bins_population":[p0,..,p19],"binned_av_speed":[v0,..,v19],"binned_sqrt_err_speed":[e0,..,e19]}} Stored in: os.path.join(self.PlotDir,"MFD_{0}_{1}_2Plot.csv".format(self.StrDate,Class))
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
import matplotlib.ticker as ticker

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
        if "geojson" in config.keys():
            self.GeoJsonFile = os.path.join(config["geojson"])
        else:
            self.GeoJsonFile = os.path.join(os.environ['WORKSPACE'],"city-pro","bologna-provincia.geojson")
        self.PlotDir = os.path.join(os.environ['WORKSPACE'],"city-pro","output","bologna_mdt_detailed","plots",self.StrDate)
        self.PlotDirAggregated = os.path.join(os.environ['WORKSPACE'],"city-pro","output","bologna_mdt_detailed","plots")
        if not os.path.exists(self.PlotDir):
            os.makedirs(self.PlotDir)
        # BOUNDING BOX
        if "bounding_box" in config.keys():
            try:
                self.bounding_box = [(config["bounding_box"]["lat_min"],config["bounding_box"]["lon_min"]),(config["bounding_box"]["lat_max"],config["bounding_box"]["lon_min"]),(config["bounding_box"]["lat_max"],config["bounding_box"]["lon_max"]),(config["bounding_box"]["lat_min"],config["bounding_box"]["lon_max"])]
                bbox = box(config["bounding_box"]["lat_min"],config["bounding_box"]["lon_min"],config["bounding_box"]["lat_max"],config["bounding_box"]["lon_max"])
                self.centroid = gpd.GeoDataFrame([1], geometry=[bbox], crs="EPSG:4326").centroid
            except:
                exit("bounding_box not defined well in config. Should be 'bounding_box': {'lat_min': 44.463121,'lon_min': 11.287085,'lat_max': 44.518165,'lon_max': 11.367472}")
        else:
            self.bounding_box = [(44.463121,11.287085),(44.518165,11.287085),(44.518165,11.367472),(44.463121,11.367472)]
            bbox = box((44.463121,11.287085,44.518165,11.367472))
            self.centroid = gpd.GeoDataFrame([1], geometry=[bbox], crs="EPSG:4326").centroid
        ## CONVERSIONS and CONSTANTS
        self.day_in_sec = 24*3600
        self.dt = 15*60
        self.iterations = self.day_in_sec/self.dt
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
        self.VelTimePercorrenceClass = defaultdict() # {0: [start_bin,end_bin,id,time_percorrence,av_speed]}
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
        # OUTPUT DICTIONARIES
        self.Feature2Label = {"av_speed":'average speed (m/s)',"speed_kmh":'average speed (km/h)',"av_accel":"average acceleration (m/s^2)","lenght":'lenght (m)',"lenght_km": 'lenght (km)',"time_hours":'time (h)',"time":'time (s)'}
        self.Column2SaveName = {"av_speed":"average_speed","speed_kmh":'average_speed_kmh',"av_accel":"average_acceleration","lenght":"lenght","lenght_km": 'lenght_km',"time_hours":"time_hours","time":"time"}
        self.Column2Legend = {"av_speed":"speed (m/s)","speed_kmh":'speed (km/h)',"av_accel":"acceleration (m/s^2)","lenght":"lenght (m)","lenght_km": 'lenght (km)',"time_hours":"time (h)","time":"time (s)"} 
        self.Feature2MaxBins = {"av_speed":{"bins":0,"count":0},"speed_kmh":{"bins":0,"count":0},"av_accel":{"bins":0,"count":0},"lenght":{"bins":0,"count":0},"lenght_km": {"bins":0,"count":0},"time_hours":{"bins":0,"count":0},"time":{"bins":0,"count":0}}
        self.DictInitialGuess = {"maxwellian":{
                                    "av_speed":{"initial_guess":[0,0],"interval":[]},
                                    "speed_kmh":{"initial_guess":[0,0],"interval":[]}},
                                "gaussian":{
                                    "av_speed":{"initial_guess":[0,0],"interval":[]},
                                    "speed_kmh":{"initial_guess":[0,0],"interval":[]}
                                    },      
                                "powerlaw":{                          
                                    "lenght":{"initial_guess":[0,0],"interval":[]},
                                    "lenght_km":{"initial_guess":[0,0],"interval":[]},
                                    "time_hours":{"initial_guess":[0,0],"interval":[]},
                                    "time":{"initial_guess":[0,0],"interval":[]}
                                    },
                                "exponential":{
                                    "lenght":{"initial_guess":[0,0],"interval":[]},
                                    "lenght_km":{"initial_guess":[0,0],"interval":[]},
                                    "time_hours":{"initial_guess":[0,0],"interval":[]},
                                    "time":{"initial_guess":[0,0],"interval":[]}
                                    }
                                }

        self.DictFittedData = {Feature: {"best_fit":[], "fitted_data":[],"parameters":[],"start_window":None,"end_window":None} for Feature in list(self.Features2Fit)}
        self.InfoFittedParameters =  {Function2Fit: {Feature:{"fit":None,"StdError":None,"success":None} for Feature in self.DictInitialGuess[Function2Fit].keys()} for Function2Fit in self.DictInitialGuess.keys()}
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
        assert self.Feature2ScaleBins.keys() == self.Feature2ScaleCount.keys() == self.Feature2IntervalCount.keys() == self.Feature2IntervalBin.keys() == self.Feature2ShiftBin.keys() == self.Feature2ShiftCount.keys() == self.Feature2Label.keys() == self.Column2SaveName.keys() == self.Column2Legend.keys(), "Error: Features not consistent"
        # FUNDAMENTAL DIAGRAM
        self.MFD = Dict2PolarsDF({"time":[],"population":[],"speed_kmh":[],"av_speed":[]},schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed_kmh":pl.Float64,"av_speed":pl.Float64})
        self.MFDNew = Dict2PolarsDF({"time":[],"population":[],"speed_kmh":[],"av_speed":[]},schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed_kmh":pl.Float64,"av_speed":pl.Float64})
        self.MFD2Plot = {"binned_av_speed":[],"binned_sqrt_err_speed":[],"bins_population":[]}
        self.MFDNew2Plot = {"binned_av_speed":[],"binned_sqrt_err_speed":[],"bins_population":[]}
        if self.BoolStrClass2IntClass:
            self.Class2MFD = {class_:Dict2PolarsDF({"time":[],"population":[],"speed_kmh":[],"av_speed":[]},schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed_kmh":pl.Float64,"av_speed":pl.Float64}) for class_ in self.IntClass2StrClass.keys()}
            self.Class2MFDNew = {class_:Dict2PolarsDF({"time":[],"population":[],"speed_kmh":[],"av_speed":[]},schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed_kmh":pl.Float64,"av_speed":pl.Float64}) for class_ in self.IntClass2StrClass.keys()}
        else:
            self.Class2MFD = defaultdict(dict)
            self.Class2MFDNew = defaultdict(dict)
            print("Warning: Not Initialized Class2MFD")
        # MINIMUM VALUES FOR (velocity,population,lenght,time) for trajectories of the day
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
        if self.verbose:
            print("Reading timed_fluxes")
            print(self.DictDirInput["timed_fluxes"])
        if os.path.isfile(self.DictDirInput["timed_fluxes"]):
            self.TimedFluxes = pd.read_csv(self.DictDirInput["timed_fluxes"],delimiter = ';')
            self.TimedFluxes = pl.from_pandas(self.TimedFluxes)
            self.ReadTime2FluxesBool = True
            Message = "{} Read Timed Fluxes: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
        else:   
            Message = "{} Read Timed Fluxes: False".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
            print("No timed_fluxes")    

    def ReadFluxes(self):
        """
            Read ../ouputdir/weights/basename_date_date.fluxes
        """
        self.CountFunctionsCalled += 1
        if self.verbose:
            print("Reading fluxes")
            print(self.DictDirInput["fluxes"])
        if os.path.isfile(self.DictDirInput["fluxes"]):
            self.Fluxes = pd.read_csv(self.DictDirInput["fluxes"],delimiter = ';')
            self.Fluxes = pl.from_pandas(self.Fluxes)
            self.ReadFluxesBool = True     
            Message = "{} Read Fluxes: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)   
        else:
            Message = "{} Read Fluxes: False".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
            print("No fluxes")    

    def ReadFcm(self):
        """
            Read ../ouputdir/basename_date_date_fcm.csv
        """
        self.CountFunctionsCalled += 1
        if self.verbose:
            print("Reading fcm")
            print(self.DictDirInput["fcm"])
        if os.path.isfile(self.DictDirInput["fcm"]):
            self.Fcm = pd.read_csv(self.DictDirInput["fcm"],delimiter = ';')
            self.Fcm = pl.from_pandas(self.Fcm)
            self.Fcm = self.Fcm.filter(pl.col("av_speed")<43.0)
            self.Fcm = self.Fcm.with_columns(pl.col("av_speed").apply(lambda x: ms2kmh(x), return_dtype=pl.Float64).alias("speed_kmh"))
            self.Fcm = self.Fcm.with_columns(pl.col("lenght").apply(lambda x: m2km(x), return_dtype=pl.Float64).alias("lenght_km"))
            self.Fcm = self.Fcm.with_columns(pl.col("time").apply(lambda x: s2h(x), return_dtype=pl.Float64).alias("time_hours"))
            self.ReadFcmBool = True
            Message = "{} Read Fcm: True".format(self.CountFunctionsCalled)
            Message += "\n\tInitialize self.Fcm"
            AddMessageToLog(Message,self.LogFile)
        else:
            Message = "{} Read Fcm: False".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
            print("No fcm")

    def ReadStats(self):
        """
            Read ../ouputdir/basename_date_date_stats.csv
        """
        self.CountFunctionsCalled += 1
        if self.verbose:
            print("Reading stats")
            print(self.DictDirInput["stats"])
        if os.path.isfile(self.DictDirInput["stats"]):
            self.Stats = pd.read_csv(self.DictDirInput["stats"],delimiter = ';')
            self.Stats = pl.from_pandas(self.Stats)
            self.Stats = self.Stats.filter(pl.col("av_speed")<43.0)
            self.Stats = self.Stats.with_columns(pl.col("av_speed").apply(lambda x: ms2kmh(x), return_dtype=pl.Float64).alias("speed_kmh"))
            self.Stats = self.Stats.with_columns(pl.col("lenght").apply(lambda x: m2km(x), return_dtype=pl.Float64).alias("lenght_km"))
            self.Stats = self.Stats.with_columns(pl.col("time").apply(lambda x: s2h(x), return_dtype=pl.Float64).alias("time_hours"))
            self.ReadStatsBool = True
            Message = "{} Read Stats: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
        else:
            Message = "{} Read Stats: False".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
            print("No stats")    

    def ReadFcmNew(self):
        """
            Read ../ouputdir/basename_date_date_fcm_new.csv
        """
        self.CountFunctionsCalled += 1
        if self.verbose:
            print("Reading fcm_new")
            print(self.DictDirInput["fcm_new"])
        if os.path.isfile(self.DictDirInput["fcm_new"]):
            self.FcmNew = pd.read_csv(self.DictDirInput["fcm_new"],delimiter = ';')
            self.FcmNew = pl.from_pandas(self.FcmNew)
            self.ReadFcmNewBool = True
            Message = "{} Read Fcm New: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
        else:
            Message = "{} Read Fcm New: False".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
            print("No fcm_new")    

    def ReadFcmCenters(self,verbose=False):
        """
            Description:
                Read the centers of the FCM
            NOTE: This function will define also what are the classes in any plot since it is used to initialize
            IntClass2StrClass
        """
        self.CountFunctionsCalled += 1
        if self.verbose:
            print("Reading fcm_centers")
            print(self.DictDirInput["fcm_centers"])
        Features = {"class":[],"av_speed":[],"v_max":[],"v_min":[],"sinuosity":[],"people":[]}
        if os.path.isfile(self.DictDirInput["fcm_centers"]):
            FcmCenters = pd.read_csv(self.DictDirInput["fcm_centers"],delimiter = ';') 
            FcmCenters = pl.from_pandas(FcmCenters)
            FlattenedFcmCenters = FcmCenters.to_numpy().flatten()    
            Row2Jump = False   
            idxcol = 0
            # Initialize Features by Reading the the columns as 1st row.
            for col in FcmCenters.columns:
                if idxcol == 0 or idxcol == len(FcmCenters.columns)-1:
                    Features[list(Features.keys())[idxcol]].append(int(col))
                else:
                    Features[list(Features.keys())[idxcol]].append(float(col))
                idxcol += 1
            # Fill the other rows
            for val in range(len(FlattenedFcmCenters)):
                # Check that is the first row
                if int(val/len(FcmCenters.columns)) == 0: 
                    if verbose:
                        print("Iteration: ", val," case Row {}".format(int(val/len(FcmCenters.columns)))," Col: " ,list(Features.keys())[val])
                    # Check that is the first element of the row
                    if val%len(FcmCenters.columns) == 0: 
                        if verbose:
                            print("\tIteration: ", val," case Row {}".format(int(val/len(FcmCenters.columns)))," Col: " ,list(Features.keys())[val])
                        Row2Jump = False
                        # Check that the velocity is not too high
                        if float(FlattenedFcmCenters[val + 1]) > 50:
                            if verbose:                
                                print("\t\tDiscarded Row: ", val/len(FcmCenters.columns))
                            Row2Jump = True
                            pass
                        else:
                            if verbose:
                                print("\t\tRow2Jump: ",Row2Jump)
                            if not Row2Jump:
                                if verbose:
                                    print("\t\tIteration: ", val," not Jump:")
                                keyidx = val
                                if list(Features.keys())[keyidx] == "class" or list(Features.keys())[keyidx] == "people":
                                    Features[list(Features.keys())[keyidx]].append(int(FlattenedFcmCenters[val]))
                                    if verbose:
                                        print("\t\t\tIteration: ", val," Col: " ,list(Features.keys())[keyidx]," Appending: ", int(FlattenedFcmCenters[val]))
                                else:
                                    Features[list(Features.keys())[keyidx]].append(float(FlattenedFcmCenters[val]))
                                    if verbose:
                                        print("\t\t\tIteration: ", val," Col: " ,list(Features.keys())[keyidx]," Appending: ", float(FlattenedFcmCenters[val]))
                            else:
                                if verbose:
                                    print("\t\tJump: ", "Iteration: ", val,"Row: {}".format(int(val/len(FcmCenters.columns)))," Col: " ,list(Features.keys())[keyidx],"Value FCM info: ", float(FlattenedFcmCenters[val]))
                    else:
                        if verbose:
                            print("\tIteration: ",  val,"Row: {}".format(int(val/len(FcmCenters.columns)))," Col: " ,list(Features.keys())[keyidx],"Value FCM info: ", float(FlattenedFcmCenters[val]))
                        if not Row2Jump:
                            if verbose:
                                print("\t\tIteration: ", val," not Jump:")
                            keyidx = val
                            if list(Features.keys())[keyidx] == "class" or list(Features.keys())[keyidx] == "people":
                                Features[list(Features.keys())[keyidx]].append(int(FlattenedFcmCenters[val]))
                                if verbose:
                                    print("\t\t\tIteration: ", val," Col: " ,list(Features.keys())[keyidx]," Appending: ", int(FlattenedFcmCenters[val]))
                            else:
                                Features[list(Features.keys())[keyidx]].append(float(FlattenedFcmCenters[val]))
                                if verbose:
                                    print("\t\t\tIteration: ", val," Col: " ,list(Features.keys())[keyidx]," Appending: ", float(FlattenedFcmCenters[val]))
                        else:
                            if verbose:
                                print("\t\tJump: ", "Iteration: ", val,"Row: {}".format(int(val/len(FcmCenters.columns)))," Col: " ,list(Features.keys())[keyidx],"Value FCM info: ", float(FlattenedFcmCenters[val]))
                # Not first row
                else:
                    if verbose:        
                        print("Iteration: ", val," case Row {}".format(int(val/len(FcmCenters.columns)))," Col: " ,list(Features.keys())[val%len(FcmCenters.columns)])                    
                    # Check that is the first element of the row
                    if val%len(FcmCenters.columns) == 0:
                        if verbose:                
                            print("\tIteration: ", val," case Row {}".format(int(val/len(FcmCenters.columns)))," Col: " ,list(Features.keys())[val%len(FcmCenters.columns)])                    
                        # Check that the velocity is not too high
                        Row2Jump = False
                        if float(FlattenedFcmCenters[val + 1]) > 50:
                            Row2Jump = True
                            if verbose:                
                                print("\t\tDiscarded Row: ", val/len(FcmCenters.columns))
                            pass
                        else:
                            if not Row2Jump:
                                if verbose:                
                                    print("\t\tIteration: ", val," not Jump:")
                                keyidx = int(val%len(FcmCenters.columns))
                                if list(Features.keys())[keyidx] == "class" or list(Features.keys())[keyidx] == "people":
                                    Features[list(Features.keys())[keyidx]].append(int(FlattenedFcmCenters[val]))
                                    if verbose:
                                        print("\t\t\tIteration: ", val," Col: " ,list(Features.keys())[keyidx]," Appending: ", int(FlattenedFcmCenters[val]))
                                else:
                                    Features[list(Features.keys())[keyidx]].append(float(FlattenedFcmCenters[val]))
                                    if verbose:
                                        print("\t\t\tIteration: ", val," Col: " ,list(Features.keys())[keyidx]," Appending: ", float(FlattenedFcmCenters[val]))
                            else:
                                if verbose:
                                    print("\t\tJump: ", "Iteration: ", val,"Row: {}".format(int(val/len(FcmCenters.columns)))," Col: " ,list(Features.keys())[keyidx],"Value FCM info: ", float(FlattenedFcmCenters[val]))
                    else:
                        if verbose:
                            print("\tIteration: ",  val,"Row: {}".format(int(val/len(FcmCenters.columns)))," Col: " ,list(Features.keys())[keyidx],"Value FCM info: ", float(FlattenedFcmCenters[val]))
                        if not Row2Jump:
                            if verbose:
                                print("\t\tIteration: ", val," not Jump:")
                            keyidx = val%len(FcmCenters.columns)
                            if list(Features.keys())[keyidx] == "class" or list(Features.keys())[keyidx] == "people":
                                Features[list(Features.keys())[keyidx]].append(int(FlattenedFcmCenters[val]))
                                if verbose:
                                    print("\t\t\tIteration: ", val," Col: " ,list(Features.keys())[keyidx]," Appending: ", int(FlattenedFcmCenters[val]))
                            else:
                                Features[list(Features.keys())[keyidx]].append(float(FlattenedFcmCenters[val]))
                                if verbose:
                                    print("\t\t\tIteration: ", val," Col: " ,list(Features.keys())[keyidx]," Appending: ", float(FlattenedFcmCenters[val]))
                        else:
                            if verbose:
                                print("\t\tJump: ", "Iteration: ", val,"Row: {}".format(int(val/len(FcmCenters.columns)))," Col: " ,list(Features.keys())[keyidx],"Value FCM info: ", float(FlattenedFcmCenters[val]))

                
            self.FcmCenters = pl.DataFrame(Features)
            self.ReadFcmCentersBool = True    
            Message = "{} Read Fcm Centers: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
        else:
            Message = "{} Read Fcm Centers: False".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
            print("No fcm_centers")

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
        if self.verbose:
            print("Reading fluxes_sub")
            print(self.DictDirInput["fluxes_sub"])
        DoNothing = False
        self.IntClass2Roads = defaultdict(list)
        # Read Fluxes.sub
        if os.path.isfile(self.DictDirInput["fluxes_sub"]):
            with open(self.DictDirInput["fluxes_sub"],'r') as f:
                FluxesSub = f.readlines()
            for ClassLines in FluxesSub:
                ClassandID = ClassLines.split('\t')
                ClassId  = ClassandID[0].split('_')[1]
                if self.verbose:
                    print("Class: ",ClassId)
                try:
                    ClassFractionRoadsConsidered = ClassandID[0].split('_')[2]
                    if self.verbose:
                        print("Fraction of roads considered: ",ClassFractionRoadsConsidered)
                except IndexError:
                    DoNothing = True
                    if self.verbose:
                        print("Considering the Total Subnetwork indipendent on the Subclass")
                if DoNothing:
                    pass
                else:
                    IdRoads = [int(RoadId) for RoadId in ClassandID[1:] if RoadId != '\n']
                    self.IntClass2Roads[int(ClassId)] = IdRoads
                    if self.verbose:
                        print("Number of Roads SubNetwork: ",len(IdRoads))     
            self.ReadFluxesSubBool = True
            Message = "{} Read Fluxes Sub: True".format(self.CountFunctionsCalled)
            for Class in self.IntClass2Roads.keys():
                Message += "Class: {0} -> {1} Roads, ".format(Class,len(self.IntClass2Roads[Class]))
            AddMessageToLog(Message,self.LogFile)
        else:
            Message = "{} Read Fluxes Sub: False".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
            print("FluxesSubFile not found")

    def ReadGeoJson(self):
        """
            Read the GeoJson File and store it in self.GeoJson
        """
        self.CountFunctionsCalled += 1
        if self.verbose:
            print("Reading GeoJson")
        if not os.path.isfile(self.GeoJsonFile) and not os.path.isfile(os.path.join(self.InputBaseDir,"BolognaMDTClassInfo.geojson")):
            Message = "{} Read GeoJson: False".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
            exit("GeoJsonFile not found")
        if os.path.exists(os.path.join(self.InputBaseDir,"BolognaMDTClassInfo.geojson")):
            self.GeoJson = gpd.read_file(os.path.join(self.InputBaseDir,"BolognaMDTClassInfo.geojson"))
            self.ReadGeojsonBool = True
            Message = "{} Read GeoJson: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
        else:
            self.GeoJson = gpd.read_file(self.GeoJsonFile)
            self.ReadGeojsonBool = True
            Message = "{} Read GeoJson: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)

    def GetIncreasinglyIncludedSubnets(self):
        """
            Description:
                Get the Increasingly Included Subnets
        """
        self.CountFunctionsCalled += 1
        Message = "{} Get Increasingly Included Subnets: True".format(self.CountFunctionsCalled)
        AddMessageToLog(Message,self.LogFile)
        self.DictSubnetsTxtDir = defaultdict(dict)
        if self.BoolStrClass2IntClass:
            for Class in self.IntClass2StrClass.keys():
                self.DictSubnetsTxtDir[Class] = os.path.join(self.InputBaseDir,self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + '{}_class_subnet.txt'.format(Class))
            self.ReadFluxesSubIncreasinglyIncludedIntersection()
            if self.verbose:
                print("Get increasingly included subnets")
                for class_ in self.DictSubnetsTxtDir:
                    print(self.DictSubnetsTxtDir[class_])
        else:
            Message = "{} Get Increasingly Included Subnets: False".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
            print("Warning: Not Initialized DictSubnetsTxtDir -> Will not have Increasingly Included SubNetworks")

    def ReadFluxesSubIncreasinglyIncludedIntersection(self):
        '''
            Input:
                FluxesSubFile: (str) -> FluxesSubFile = '../{basename}_{start}_{start}/fluxes.sub'
                verbose: (bool) -> verbose = False
            Output:
                self.IntClass2RoadsIncreasinglyIncludedIntersection: (dict) -> {IntClass:[] for IntClass in self.IntClasses}
        '''
        self.CountFunctionsCalled += 1
        DoNothing = False
        self.IntClass2RoadsIncreasinglyIncludedIntersection = defaultdict(list)
        try:
            for Class in self.DictSubnetsTxtDir.keys():
                with open(self.DictSubnetsTxtDir[Class],'r') as f:
                    FluxesSub = f.readlines()
                for Road in FluxesSub[0].split(" "):
                    intRoad,BoolInt = CastString2Int(Road)
                    if BoolInt:
                        self.IntClass2RoadsIncreasinglyIncludedIntersection[Class].append(intRoad)
                    else:
                        pass
    #            self.IntClass2RoadsIncreasinglyIncludedIntersection[Class] = [CastString2Int(Road) for Road in FluxesSub[0].split(" ")] 
            self.ReadFluxesSubIncreasinglyIncludedIntersectionBool = True
            Message = "{} Read Fluxes Sub Increasingly Included Intersection: True".format(self.CountFunctionsCalled)
            for Class in self.IntClass2RoadsIncreasinglyIncludedIntersection.keys():
                Message += "Class: {0} -> {1} Roads, ".format(Class,len(self.IntClass2RoadsIncreasinglyIncludedIntersection[Class]))
            AddMessageToLog(Message,self.LogFile)
        except:
            Message = "{} Read Fluxes Sub Increasingly Included Intersection: False".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
            print("FluxesSubFile not found")

#--------- COMPLETE GEOJSON ------- ##
    def CompleteGeoJsonWithClassInfo(self):
        """
            Computes "IntClassOrdered" and "StrClassOrdered" columns for the Geojson.
            NOTE:
                Each road for each day will have the Int, Str, IntOrdered and StrOrdered Class.
        """
        self.CountFunctionsCalled += 1
        if self.ReadGeojsonBool and self.ReadFluxesSubIncreasinglyIncludedIntersectionBool:
            print("Completing GeoJson with Class Info" )
            ClassOrderedForGeojsonRoads = np.zeros(len(self.GeoJson),dtype = int)
            for Class in self.IntClass2RoadsIncreasinglyIncludedIntersection.keys():
                for Road in self.IntClass2RoadsIncreasinglyIncludedIntersection[Class]: 
                    ClassOrderedForGeojsonRoads[np.where(self.GeoJson["poly_lid"] == Road)[0]] = Class 
            self.GeoJson["IntClassOrdered_{}".format(self.StrDate)] = ClassOrderedForGeojsonRoads
            self.GeoJson["StrClassOrdered_{}".format(self.StrDate)] = [self.IntClass2StrClass[intclass] for intclass in ClassOrderedForGeojsonRoads]
            # Normal Subnet
            ClassOrderedForGeojsonRoads = np.zeros(len(self.GeoJson),dtype = int)
            for Class in self.IntClass2Roads.keys():
                for Road in self.IntClass2Roads[Class]: 
                    ClassOrderedForGeojsonRoads[np.where(self.GeoJson["poly_lid"] == Road)[0]] = Class 
            self.GeoJson["IntClass_{}".format(self.StrDate)] = ClassOrderedForGeojsonRoads
            self.GeoJson.to_file(os.path.join(self.InputBaseDir,"BolognaMDTClassInfo.geojson"))
            self.GeoJsonWithClassBool = True
            Message = "{} GeoJson with Class Info: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
        else:
            Message = "{} GeoJson with Class Info: False".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)

    def CompareOld2NewClass(self):
        """NOTE: Computing the intersection Matrix
                The Intersection Diagonal is the fraction that do not change from N -> I
                Upper Triangular Matrix is the fraction that change from N -> I (Is the flux we are interested in)
            NOTE:
            The Iterative inclusion of subnetworks is done in the following way:
                Let's call Road2Subnets -> Vector that contains the list of subnetworks that a road belongs to.
                Example:
                Road2Subnet[R1] = [0,1,3]
                    The transition seen is 0 -> 3
                Making intersection among different classes we can see the transition from one class to another.
                Taking the set of roads I consider the characterization in Old and New Class.
                |N_i| = sum_{j \diff i} |Intersection(N_i,O_j)| + |Intersection(N_i,O_i)|
            Goal:    
                Look at the fraction of roads that change from one class to another. -> |Intersection(N_j,O_i)|/|O_i|
        """
        self.CountFunctionsCalled += 1
        if self.GeoJsonWithClassBool:
            print("Comparison of trajectories from Old to New Class {}".format(self.StrDate))
            # Compare from Subnetworks
            KeysNormal = ["NormalSubnet_{}".format(IntClass) for IntClass in self.IntClass2Roads.keys()]
            KeysIncr = ["IncrSubnet_{}".format(IntClass) for IntClass in self.IntClass2RoadsIncreasinglyIncludedIntersection.keys()]
            DfComparison = pd.DataFrame(index = KeysNormal,columns = KeysIncr)     
            self.Normal2Incr = {Key_Norm:{Key_Incr: (set(self.IntClass2Roads[int(Key_Norm.split("_")[1])]),set(self.IntClass2RoadsIncreasinglyIncludedIntersection[int(Key_Incr.split("_")[1])])) for Key_Incr in KeysIncr} for Key_Norm in KeysNormal}
            for Key_Norm in self.Normal2Incr.keys():
                for Key_Incr,(VecI,VecN) in self.Normal2Incr[Key_Norm].items():
                    # Elements in I and N
                    Intersection = VecN.intersection(VecI)          # Intersection(N_i,O_j)
                    FractionTransition = len(Intersection)/len(VecN) # |Intersection(N_i,O_j)|/|O_j|
                    DfComparison[Key_Incr].loc[Key_Norm] = FractionTransition
            DfComparison.to_csv(os.path.join(self.PlotDir,"Comparison_Network_Old2NewClass_FromRoads.csv"))
            Message = "{} Comparison of trajectories from Old to New Class: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
        if self.ReadFcmBool and self.ReadFcmNewBool:
            print("Comparison of trajectories from Old to New Class {}".format(self.StrDate))
            # Compare from Fcm 
            KeysNormal = ["Class_{}".format(IntClass) for IntClass in np.unique(self.Fcm["class"])]
            KeysIncr = ["New_Class_{}".format(IntClass) for IntClass in np.unique(self.Fcm["class_new"])] 
            DfComparison = pd.DataFrame(index = KeysNormal,columns = KeysIncr)
            self.Normal2Incr = {Key_Norm:{Key_Incr: (set(self.Fcm.filter(pl.col("class") == int(Key_Norm.split("_")[1]))["id_act"].to_list()),set(self.Fcm.filter(pl.col("class") == int(Key_Norm.split("_")[1]))["id_act"].to_list())) for Key_Incr in KeysIncr} for Key_Norm in KeysNormal}               
            for Key_Norm in self.Normal2Incr.keys():
                for Key_Incr,(VecI,VecN) in self.Normal2Incr[Key_Norm].items():
                    # Elements in I and N
                    Intersection = VecN.intersection(VecI)          # Intersection(N_i,O_j)
                    FractionTransition = len(Intersection)/len(VecN) # |Intersection(N_i,O_j)|/|O_j|
                    DfComparison[Key_Incr].loc[Key_Norm] = FractionTransition
            DfComparison.to_csv(os.path.join(self.PlotDir,"Comparison_Network_Old2NewClass_FromFcm.csv"))
            Message = "{} Comparison of trajectories from Old to New Class: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
    
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
                    self.VelTimePercorrenceClass[Class] = pd.read_csv(self.RoadInClass2VelocityDir[Class],delimiter = ';')
                    self.VelTimePercorrenceClass[Class] = pl.from_pandas(self.VelTimePercorrenceClass[Class])
                self.ReadVelocitySubnetBool = True
                Message = "{} Read Velocity Subnet: True\n".format(self.CountFunctionsCalled)
                Message += "\tInitialized VelTimePercorrenceClass: {IntClass:pl.Dataframe[id_poly,time_percorrence,av_speed]}\n"
                AddMessageToLog(Message,self.LogFile)
            except:
                Message = "{} Read Velocity Subnet: False".format(self.CountFunctionsCalled)
                AddMessageToLog(Message,self.LogFile)
                print("VelTimePercorrenceFile not found")
        else:
            print("Warning: No Initialization of VelTimePercorrenceClass due to lack of definition of IntClass2Str")

    def AddFcmNew2Fcm(self,verbose = True):
        """
            Description:
                Convert the class column of the FcmNew to class_new and join it to the Fcm.
                In this way we have in Fcm for each trajectory a new column with the class of the trajectory after having intersected the subnetworks..
        """
        self.CountFunctionsCalled += 1
        if self.ReadFcmBool and self.ReadFcmNewBool:
            FcmNew = self.FcmNew.with_columns([self.FcmNew['class'].alias('class_new')])
            self.Fcm = self.Fcm.join(FcmNew[['id_act', 'class_new']], on='id_act', how='left')
            if verbose:
                print("1st join Fcm: ",self.Fcm.columns)
                print("Date: ",self.StrDate)
            self.Fcm =self.Fcm.with_columns([self.Fcm['class'].alias('class_new')])
            Message = "{} Add Fcm New to Fcm: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
            if verbose:
                print("renamed: ",self.Fcm.columns)
        if self.ReadStatsBool and self.ReadFcmNewBool:
            FcmNew = self.FcmNew.with_columns([self.FcmNew['class'].alias('class_new')])
            self.Stats["class_new"] = self.Stats.join(self.FcmNew[['id_act', 'class_new']], on='id_act', how='left')
            if verbose:
                print("1st join Stats: ",self.Stats.columns)
                print("Date: ",self.StrDate)
            self.Stats =self.Stats.with_columns([self.Stats['class'].alias('class_new')])
            if verbose:
                print("renamed: ",self.Stats.columns)

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
                                self.VelTimePercorrenceClass,
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
## ------- FUNDAMENTAL DIAGRAM ------ ##
    def ComputeMFDVariablesClass(self):
        '''
            Description:
                Computes the MFD variables (t,population,speed) -> and the hysteresis diagram:
                    1) Aggregated data for the day
                    2) Conditional to class
            Save them in two dictionaries 
                1) self.MFD = {time:[],population:[],speed:[]}
                2) self.Class2MFD = {Class:pl.DataFrame{"time":[],"population":[],"speed_kmh":[]}}
        '''
        if self.ReadFcmBool:
            print("Computing MFD Variables from Fcm")
            if "start_time" in self.Fcm.columns:
                if os.path.isfile(os.path.join(self.PlotDir,"MFD_{}.csv".format(self.StrDate))):
                    self.MFD = pd.read_csv(os.path.join(self.PlotDir,"MFD_{}.csv".format(self.StrDate)))
                    self.MFD = pl.from_pandas(self.MFD)
                    self.CountFunctionsCalled += 1
                    Message = "{} Compute MFD Variables: True\n".format(self.CountFunctionsCalled)
                    Message += "\tUpload self.MFD from CSV"
                    AddMessageToLog(Message,self.LogFile)
                else:
                    # ALL TOGETHER MFD
                    self.MFD,self.Fcm = ComputeMFDVariables(self.Fcm,self.MFD,self.TimeStampDate,self.dt,self.iterations)
                    self.MFD.write_csv(os.path.join(self.PlotDir,"MFD_{}.csv".format(self.StrDate)))
                    self.CountFunctionsCalled += 1
                    Message = "{} Compute MFD Variables: True\n".format(self.CountFunctionsCalled)
                    Message += "\tComputed self.MFD "
                    AddMessageToLog(Message,self.LogFile)
                # PER CLASS
                self.Class2MFD = {class_:Dict2PolarsDF({"time":[],"population":[],"speed_kmh":[],"av_speed":[]},schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed_kmh":pl.Float64,"av_speed":pl.Float64}) for class_ in self.IntClass2StrClass.keys()}
                self.Class2MFDNew = {class_:Dict2PolarsDF({"time":[],"population":[],"speed_kmh":[],"av_speed":[]},schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed_kmh":pl.Float64,"av_speed":pl.Float64}) for class_ in self.IntClass2StrClass.keys()}
                if self.verbose:
                    print(self.Class2MFD.keys())
                for Class in self.Class2MFD.keys():
                    if os.path.isfile(os.path.join(self.PlotDir,"Class2MFD_{0}_{1}.csv".format(Class,self.StrDate))):
                        print("Upload ClassMFD, ClassMFDNew from CSV")
                        self.Class2MFD[Class] = pd.read_csv(os.path.join(self.PlotDir,"Class2MFD_{0}_{1}.csv".format(Class,self.StrDate)))
                        self.Class2MFD[Class] = pl.from_pandas(self.Class2MFD[Class])
                        self.Class2MFDNew[Class] = pd.read_csv(os.path.join(self.PlotDir,"Class2MFDNew_{0}_{1}.csv".format(Class,self.StrDate)))
                        self.Class2MFDNew[Class] = pl.from_pandas(self.Class2MFDNew[Class])
                        # Log
                        Upload = True
                    else:
                        print("Computing Class2MFD, Class2MFDNew")
                        Fcm2Class = self.Fcm.filter(pl.col("class") == Class)
                        self.Class2MFD[Class],Fcm2Class = ComputeMFDVariables(Fcm2Class,self.Class2MFD[Class],self.TimeStampDate,self.dt,self.iterations)
                        FcmNew2Class = self.Fcm.filter(pl.col("class_new") == Class)
                        self.Class2MFDNew[Class],FcmNew2Class = ComputeMFDVariables(FcmNew2Class,self.Class2MFDNew[Class],self.TimeStampDate,self.dt,self.iterations)
                        if self.verbose:
                            PrintMFDDictInfo(self.Class2MFD[Class],StartingString = "Class 2 MFD: ")
                            PrintMFDDictInfo(self.Class2MFDNew[Class],StartingString = "Class 2 MFD New: ")
                        self.Class2MFD[Class].write_csv(os.path.join(self.PlotDir,"Class2MFD_{0}_{1}.csv".format(Class,self.StrDate)))
                        self.Class2MFDNew[Class].write_csv(os.path.join(self.PlotDir,"Class2MFDNew_{0}_{1}.csv".format(Class,self.StrDate)))
                        # Log
                        Upload = False
                if Upload:
                    self.CountFunctionsCalled += 1
                    Message = "{} Compute MFD Variables: True\n".format(self.CountFunctionsCalled)
                    Message += "\tUpload ClassMFD, ClassMFDNew from CSV"
                    AddMessageToLog(Message,self.LogFile)
                else:
                    self.CountFunctionsCalled += 1
                    Message = "{} Compute MFD Variables: True\n".format(self.CountFunctionsCalled)
                    Message += "\tComputed ClassMFD, ClassMFDNew"
                    AddMessageToLog(Message,self.LogFile)
                self.ComputedMFD = True    
            else:
                self.CountFunctionsCalled += 1
                Message = "{} Compute MFD Variables: False".format(self.CountFunctionsCalled)
                AddMessageToLog(Message,self.LogFile)
                pass
                
        if self.ReadStatsBool:
            print("Computing MFD Variables from Stats")
            # ALL TOGETHER MFD
            self.Stats = self.Stats.join(self.Fcm[['id_act', 'class']], on='id_act', how='left')
            self.MFD,self.Stats = ComputeMFDVariables(self.Stats,self.MFD,self.TimeStampDate,self.dt,self.iterations)
            # PER CLASS
            if self.verbose:
                PrintMFDDictInfo(self.MFD,StartingString = "MFD: ")
            if self.BoolStrClass2IntClass:
                print("Filling Class2MFD: ")
                for class_ in self.IntClass2StrClass.keys():
                    if self.verbose:
                        print("Class: ",class_)
                    self.Class2MFD[class_] = Dict2PolarsDF({"time":[],"population":[],"speed_kmh":[]},schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed_kmh":pl.Float64,"av_speed":pl.Float64})
                    self.Class2MFDNew[class_] = Dict2PolarsDF({"time":[],"population":[],"speed_kmh":[]},schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed_kmh":pl.Float64,"av_speed":pl.Float64})
            else:
                print("Warning: No Plotting MFD due to not Definition of StrInt2Class")
            for Class in self.Class2MFD.keys():
                Stats2Class = self.Stats.filter(pl.col("class") == Class)
                self.Class2MFD[Class],Stats2Class = ComputeMFDVariables(Stats2Class,self.Class2MFD[Class],self.TimeStampDate,self.dt,self.iterations,self.verbose)
                StatsNew2Class = self.Stats.filter(pl.col("class_new") == Class)
                self.Class2MFDNew[Class],StatsNew2Class = ComputeMFDVariables(StatsNew2Class,self.Class2MFDNew[Class],self.TimeStampDate,self.dt,self.iterations,self.verbose)
#                if self.verbose:
#                    PrintMFDDictInfo(self.Class2MFD[Class],StartingString = "Class 2 MFD: ")
            self.ComputedMFD = True

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
            if os.path.isfile(os.path.join(self.PlotDir,"MFD2Plot_{0}.png".format(self.StrDate))):
                print("Upload MFD2Plot")
                with open(os.path.join(self.PlotDir,"MFD2Plot_{0}.json".format(self.StrDate)),'r') as f:
                    self.MFD2Plot = json.load(f)
                with open(os.path.join(self.PlotDir,"MinMaxPlot_{0}.json".format(self.StrDate)),'r') as f:
                    self.MinMaxPlot = json.load(f)
                self.CountFunctionsCalled += 1
                Message = "{} Plot MFD: True\n".format(self.CountFunctionsCalled)
                Message += "\tUpload MFD2Plot"
                AddMessageToLog(Message,self.LogFile)
            else:
                print("Computing MFD Plot")
                    
                # AGGREGATED 
                self.MFD2Plot, self.MinMaxPlot,RelativeChange = GetMFDForPlot(MFD = self.MFD,
                                                                            MFD2Plot = self.MFD2Plot,
                                                                            MinMaxPlot = self.MinMaxPlot,
                                                                            Class = None,
                                                                            case = "no-classes",
                                                                            verbose = self.verbose,
                                                                            bins_ = 20)
                
                self.CountFunctionsCalled += 1
                Message = "{} Plot MFD: True\n".format(self.CountFunctionsCalled)
                Message += "\tComputed MFD2Plot"
                AddMessageToLog(Message,self.LogFile)
                PlotHysteresis(MFD = self.MFD,
                            Title = "Hysteresis Cycle Aggregated",
                            SaveDir = self.PlotDir,
                            NameFile = "Hysteresys.png")
                self.CountFunctionsCalled += 1
                Message = "\tComputed Hysteresis"
                AddMessageToLog(Message,self.LogFile)
                with open(os.path.join(self.PlotDir,"MFD2Plot_{0}.json".format(self.StrDate)),'w') as f:
                    json.dump(self.MFD2Plot,f,cls = NumpyArrayEncoder,indent=2)
                with open(os.path.join(self.PlotDir,"MinMaxPlot_{0}.json".format(self.StrDate)),'w') as f:
                    json.dump(self.MinMaxPlot,f,cls = NumpyArrayEncoder,indent=2)
                if self.verbose:
                    print("After GetMFDForPlot:\n")
                SaveMFDPlot(self.MFD2Plot["bins_population"],
                            self.MFD2Plot["binned_av_speed"],
                            self.MFD2Plot["binned_sqrt_err_speed"],
                            RelativeChange,
                            self.PlotDir,
                            NameFile = "MFD.png") 
                Message = "\tSaved MFD Plot"
                AddMessageToLog(Message,self.LogFile)           
            if os.path.exists(os.path.join(self.PlotDir,"Class2MFD2Plot_{0}.json".format(self.StrDate))) and os.path.exists(os.path.join(self.PlotDir,"Class2MFDNew2Plot_{0}.json".format(self.StrDate))):
                print("Loading Class2MFD2Plot and Class2MFDNew2Plot")
                with open(os.path.join(self.PlotDir,"Class2MFD2Plot_{0}.json".format(self.StrDate)),'r') as f:
                    self.Class2MFD2Plot = json.load(f)
                with open(os.path.join(self.PlotDir,"Class2MFDNew2Plot_{0}.json".format(self.StrDate)),'r') as f:
                    self.Class2MFDNew2Plot = json.load(f)
                self.CountFunctionsCalled += 1
                Message = "{} Plot Class2MFD: True\n".format(self.CountFunctionsCalled)
                Message += "\tUpload Class2MFD2Plot and Class2MFDNew2Plot in csv"
                AddMessageToLog(Message,self.LogFile)
            else:
                print("Computing MFD Variables for Classes")
                # PER CLASS 
                self.MinMaxPlotPerClass = {Class: defaultdict() for Class in self.Class2MFD.keys()}
                self.MinMaxPlotPerClassNew = {Class: defaultdict() for Class in self.Class2MFDNew.keys()}       
                self.Class2MFD2Plot = {Class:{"binned_av_speed": [], "binned_sqrt_err_speed": [], "bins_population": []} for Class in self.Class2MFD.keys()}
                self.Class2MFDNew2Plot = {Class:{"binned_av_speed": [], "binned_sqrt_err_speed": [], "bins_population": []} for Class in self.Class2MFD.keys()}
                Message = "{} Plot Class2MFD: True\n".format(self.CountFunctionsCalled)
                for Class in self.Class2MFD.keys():
                    # Fill Average/Std Speed (to plot)
                    # OLD CLASSIFICATION
                    self.Class2MFD2Plot[Class], self.MinMaxPlotPerClass,RelativeChange = GetMFDForPlot(MFD = self.Class2MFD[Class],
                                                                                                        MFD2Plot = self.Class2MFD2Plot[Class],
                                                                                                        MinMaxPlot = self.MinMaxPlotPerClass,
                                                                                                        Class = Class,
                                                                                                        case = None,
                                                                                                        verbose = self.verbose,
                                                                                                        bins_ = 20)

                    # NEW CLASSIFICATION
                    self.Class2MFDNew2Plot[Class], self.MinMaxPlotPerClassNew,RelativeChangeNew = GetMFDForPlot(MFD = self.Class2MFDNew[Class],
                                                                                                        MFD2Plot = self.Class2MFDNew2Plot[Class],
                                                                                                        MinMaxPlot = self.MinMaxPlotPerClassNew,
                                                                                                        Class = Class,
                                                                                                        case = None,
                                                                                                        verbose = self.verbose,
                                                                                                        bins_ = 20)
                    Message += "\tComputed Class2MFD2Plot and Class2MFDNew2Plot Class {}\n".format(Class)
                    AddMessageToLog(Message,self.LogFile)
                    
                    PlotHysteresis(MFD = self.Class2MFD[Class],
                                Title = "Hysteresis Cycle Class {}".format(self.IntClass2StrClass[Class]),
                                SaveDir = self.PlotDir,
                                NameFile = "HysteresysClass_{}.png".format(self.IntClass2StrClass[Class]))
                    
                    PlotHysteresis(MFD = self.Class2MFDNew[Class],
                                Title = "Hysteresis Cycle Class New {}".format(self.IntClass2StrClass[Class]),
                                SaveDir = self.PlotDir,
                                NameFile = "HysteresysClassNew_{}.png".format(self.IntClass2StrClass[Class]))
                    Message += "\tPlot Hysteresis {}\n".format(Class)
                    AddMessageToLog(Message,self.LogFile)
                    
                    with open(os.path.join(self.PlotDir,"Class2MFD2Plot_{0}.json".format(self.StrDate)),'w') as f:
                        json.dump(self.Class2MFD2Plot,f,cls = NumpyArrayEncoder,indent=2)
                    with open(os.path.join(self.PlotDir,"Class2MFDNew2Plot_{0}.json".format(self.StrDate)),'w') as f:
                        json.dump(self.Class2MFDNew2Plot,f,cls = NumpyArrayEncoder,indent=2)
                    
                    if self.verbose:
                        print("After GetMFDForPlot Class {}:\n".format(Class))
    #                    print("\nClass2MFD2Plot:\n",self.Class2MFD2Plot)
    #                    print("\nMinMaxPlotPerClass:\n",self.MinMaxPlotPerClass)

                    if self.BoolStrClass2IntClass:
                        # Plotting and Save Per Class
                        # OLD CLASSIFICATION
                        SaveMFDPlot(self.Class2MFD2Plot[Class]["bins_population"],
                                    self.Class2MFD2Plot[Class]["binned_av_speed"],
                                    self.Class2MFD2Plot[Class]["binned_sqrt_err_speed"],
                                    RelativeChange = RelativeChange,
                                    SaveDir = self.PlotDir,
                                    Title = "Fondamental Diagram {}".format(self.IntClass2StrClass[Class]),
                                    NameFile = "MFD_{}.png".format(Class))
                        # NEW CLASSIFICATION
                        SaveMFDPlot(self.Class2MFDNew2Plot[Class]["bins_population"],
                                    self.Class2MFDNew2Plot[Class]["binned_av_speed"],
                                    self.Class2MFDNew2Plot[Class]["binned_sqrt_err_speed"],
                                    RelativeChange = RelativeChangeNew,
                                    SaveDir = self.PlotDir,
                                    Title = "Fondamental Diagram New {}".format(self.IntClass2StrClass[Class]),
                                    NameFile = "MFDNew_{}.png".format(Class))
                        Message += "\tSaved MFD Plot Class {}\n".format(Class)
                        AddMessageToLog(Message,self.LogFile)
                else:
                    print("Warning: Fondamental Diagram Not Computed for Class Since IntClass2Str is Not Initialized")

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
                self.Class2Time2Distr = {IntClass:[] for IntClass in self.IntClass2StrClass.keys()} # For key Shape (96,Number of Roads)
                self.Class2AvgTimePercorrence = {IntClass:[] for IntClass in self.IntClass2StrClass.keys()} # For key Shape (96,)
                # Per Class
                for IntClass in self.IntClass2StrClass.keys():
                    File2Save = os.path.join(self.PlotDir,"TimePercorrenceDistribution_Class_{0}_{1}.png".format(IntClass,self.StrDate))
                    StrTimesLabel = []
                    self.Class2Time2Distr[IntClass],self.Class2AvgTimePercorrence[IntClass] = PlotTimePercorrenceDistribution(self.VelTimePercorrenceClass[IntClass],
                                                                                                                              self.Class2Time2Distr[IntClass],
                                                                                                                              self.Class2AvgTimePercorrence[IntClass],
                                                                                                                              StrTimesLabel,
                                                                                                                              File2Save)
                
                SaveProcedure(ListKeys = ["Class2Time2Distr","Class2AvgTimePercorrence"],
                             ListDicts = [self.Class2Time2Distr,self.Class2AvgTimePercorrence],
                             ListFormats = [self.StrDate],
                             Extension = ".json")
                Upload = False
            MessagePlotTimePercorrenceDistributionAllClasses(self.CountFunctionsCalled,self.LogFile,Upload)
            self.TimePercorrenceBool = True
            self.PlotTimePercorrenceConditionalLengthRoad()

    def PlotTimePercorrenceConditionalLengthRoad(self):
        self.CountFunctionsCalled += 1
        CountLengths,Lengths = np.histogram(self.GeoJson["poly_length"],bins = 10)
        self.Lenght2Roads = GetLengthPartitionInGeojSon(self.GeoJson,Lengths)
        self.Length2Class2Time2Distr = {Length:{IntClass:[] for IntClass in self.IntClass2StrClass.keys()} for Length in Lengths}
        self.Length2Class2AvgTimePercorrence = {Length:{IntClass:[] for IntClass in self.IntClass2StrClass.keys()} for Length in Lengths}
        for IntClass in self.IntClass2StrClass.keys():
            for Length,Roads in self.Lenght2Roads.items():
                File2Json = os.path.join(self.PlotDir,"Length2Class2Time2Distr_{0}.json".format(self.StrDate))
                if not os.path.isfile(File2Json):
                    StrTimesLabel = []
                    File2Save = os.path.join(self.PlotDir,"TimePercorrenceDistribution_Class_{0}_{1}_Length_{2}.png".format(IntClass,self.StrDate,round(Length,2)))
                    Length2VelTimePerccorenceClass = self.VelTimePercorrenceClass[IntClass].filter(pl.col("poly_id").is_in(Roads))
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


##--------------- Dictionaries --------------##
    def CreateDictionaryIntClass2StrClass(self):
        '''
        Input:
            fcm: dataframe []

        Output: dict: {velocity:'velocity class in words: (slowest,...quickest)]}
        '''
        self.CountFunctionsCalled += 1
        if self.ReadFcmCentersBool:
            number_classes = len(self.FcmCenters["class"]) 
            for i in range(number_classes):
                if self.FcmCenters.filter(pl.col("class") == i)["av_speed"].to_list()[0] > 130:
                    pass
                else:
                    if i<number_classes/2:
                        self.IntClass2StrClass[list(self.FcmCenters["class"])[i]] = '{} slowest'.format(i+1)
                        self.StrClass2IntClass['{} slowest'.format(i+1)] = list(self.FcmCenters["class"])[i]
                    elif i == number_classes/2:
                        self.IntClass2StrClass[list(self.FcmCenters["class"])[i]] = 'middle velocity class'
                        self.StrClass2IntClass['middle velocity class'] = list(self.FcmCenters["class"])[i]
                    else:
                        self.IntClass2StrClass[list(self.FcmCenters["class"])[i]] = '{} quickest'.format(number_classes - i)             
                        self.StrClass2IntClass['{} quickest'.format(number_classes - i)] = list(self.FcmCenters["class"])[i]
            self.BoolStrClass2IntClass = True
            Message = "{} Create Dictionary IntClass2StrClass: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
        else:
            self.BoolStrClass2IntClass = False
            Message = "{} Create Dictionary IntClass2StrClass: False".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
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
            self.Class2InitialGuess = {IntClass: self.DictInitialGuess for IntClass in self.IntClass2StrClass.keys()}
            if self.verbose:
                print(self.StrDate)
                print("Class2InitialGuess:\n",self.Class2InitialGuess)
            for IntClass in self.IntClass2StrClass.keys():
                StrClass = self.IntClass2StrClass[IntClass]
                for Function2Test in self.Class2InitialGuess[IntClass]: 
                    for Feature in self.Class2InitialGuess[IntClass][Function2Test]:
                        if Feature == "av_speed" or Feature == "speed_kmh":
                            pass
                        else:
                            MaxCount = self.InfoFit[Feature][StrClass]["MaxCount"]
                            Avg = self.InfoFit[Feature][StrClass]["Avg"]
                            StartWindow = self.InfoFit[Feature][StrClass]["StartWindowS"]
                            EndWindow = self.InfoFit[Feature][StrClass]["EndWindowS"]
                        # Normalization
                        SecondsInHour = 3600
                        MetersinKm = 1000
                        if self.verbose:
                            print(self.StrDate)
                            print("Feature: ",Feature)
                            print("IntClass: ",IntClass)
                            print("Function2Test: ",Function2Test)
                            print("self.Class2InitialGuess[IntClass][Function2Test][Feature]:\n",self.Class2InitialGuess[IntClass][Function2Test][Feature])
                        if self.Class2InitialGuess[IntClass][Function2Test][Feature] is not None:
                            if Feature == "time":
                                self.Class2InitialGuess[IntClass][Function2Test][Feature] = FillInitGuessIntervalPlExp(self.Class2InitialGuess[IntClass][Function2Test][Feature],
                                                                                                                        MaxCount,
                                                                                                                        Avg,
                                                                                                                        StartWindow,
                                                                                                                        EndWindow,
                                                                                                                        Function2Test)
                            elif Feature == "time_hours":
                                Avg = Avg/SecondsInHour 
                                StartWindow = StartWindow/SecondsInHour
                                EndWindow = EndWindow/SecondsInHour 
                                self.Class2InitialGuess[IntClass][Function2Test][Feature] = FillInitGuessIntervalPlExp(self.Class2InitialGuess[IntClass][Function2Test][Feature],
                                                                                                                        MaxCount,
                                                                                                                        Avg,
                                                                                                                        StartWindow,
                                                                                                                        EndWindow,
                                                                                                                        Function2Test)
                            elif Feature == "lenght":
                                self.Class2InitialGuess[IntClass][Function2Test][Feature] = FillInitGuessIntervalPlExp(self.Class2InitialGuess[IntClass][Function2Test][Feature],
                                                            MaxCount,
                                                            Avg,
                                                            StartWindow,
                                                            EndWindow,
                                                            Function2Test)
                            elif Feature == "lenght_km":
                                Avg = Avg/MetersinKm 
                                StartWindow = StartWindow/MetersinKm
                                EndWindow = EndWindow/MetersinKm
                                self.Class2InitialGuess[IntClass][Function2Test][Feature] = FillInitGuessIntervalPlExp(self.Class2InitialGuess[IntClass][Function2Test][Feature],
                                                                                                                    MaxCount,
                                                                                                                    Avg,
                                                                                                                    StartWindow,
                                                                                                                    EndWindow,
                                                                                                                    Function2Test)
                            elif Feature == "av_speed":
                                self.Class2InitialGuess[IntClass][Function2Test][Feature] = FillInitGuessIntervalMxGs(self.Class2InitialGuess[IntClass][Function2Test][Feature],
                                                                                                                    self.Fcm,
                                                                                                                    Feature,
                                                                                                                    IntClass)
                            elif Feature == "speed_kmh":
                                self.Class2InitialGuess[IntClass][Function2Test][Feature] = FillInitGuessIntervalMxGs(self.Class2InitialGuess[IntClass][Function2Test][Feature],
                                                                                                                    self.Fcm,
                                                                                                                    Feature,
                                                                                                                    IntClass)
                        else:
                            print("Warning: Initial Guess Not Initialized for Class {0} and Feature {1} Day: {2}".format(IntClass,Feature,self.StrDate))
            Message = "{} Create Dictionary Class2InitialGuess: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
            # Initialize The Guess Without Classes
            for Function2Test in self.DictInitialGuess: 
                for Feature in self.DictInitialGuess[Function2Test]:
                    if Feature == "av_speed" or Feature == "speed_kmh":
                        pass
                    else:
                        MaxCount = self.InfoFit[Feature]["aggregated"]["MaxCount"]
                        Avg = self.InfoFit[Feature]["aggregated"]["Avg"]
                        StartWindow = self.InfoFit[Feature]["aggregated"]["StartWindowS"]
                        EndWindow = self.InfoFit[Feature]["aggregated"]["EndWindowS"]
                    # Normalization
                    SecondsInHour = 3600
                    MetersinKm = 1000
                    if Feature == "time":
                        self.DictInitialGuess[Function2Test][Feature] = FillInitGuessIntervalPlExp(self.DictInitialGuess[Function2Test][Feature],
                                                                                                    MaxCount,
                                                                                                    Avg,
                                                                                                    StartWindow,
                                                                                                    EndWindow,
                                                                                                    Function2Test)
                    elif Feature == "time_hours":
                        Avg = Avg/SecondsInHour 
                        StartWindow = StartWindow/SecondsInHour
                        EndWindow = EndWindow/SecondsInHour 
                        self.DictInitialGuess[Function2Test][Feature] = FillInitGuessIntervalPlExp(self.DictInitialGuess[Function2Test][Feature],
                                                                                                    MaxCount,
                                                                                                    Avg,
                                                                                                    StartWindow,
                                                                                                    EndWindow,
                                                                                                    Function2Test)
                    elif Feature == "lenght":
                        self.DictInitialGuess[Function2Test][Feature] = FillInitGuessIntervalPlExp(self.DictInitialGuess[Function2Test][Feature],
                                                                                                    MaxCount,
                                                                                                    Avg,
                                                                                                    StartWindow,
                                                                                                    EndWindow,
                                                                                                    Function2Test)
                    elif Feature == "lenght_km":
                        Avg = Avg/MetersinKm 
                        StartWindow = StartWindow/MetersinKm
                        EndWindow = EndWindow/MetersinKm
                        self.DictInitialGuess[Function2Test][Feature] = FillInitGuessIntervalPlExp(self.DictInitialGuess[Function2Test][Feature],
                                                                                                    MaxCount,
                                                                                                    Avg,
                                                                                                    StartWindow,
                                                                                                    EndWindow,
                                                                                                    Function2Test)
                    elif Feature == "av_speed":
                        self.DictInitialGuess[Function2Test][Feature] = FillInitGuessIntervalMxGs(self.DictInitialGuess[Function2Test][Feature],
                                                                                                self.Fcm,
                                                                                                Feature,
                                                                                                None)
                    elif Feature == "speed_kmh":
                        self.DictInitialGuess[Function2Test][Feature] = FillInitGuessIntervalMxGs(self.DictInitialGuess[Function2Test][Feature],
                                                                                                    self.Fcm,
                                                                                                    Feature,
                                                                                                    None)
            with open(os.path.join(self.PlotDir,"DictInitialGuess_{0}.json".format(self.StrDate)),'w') as f:
                json.dump(self.DictInitialGuess,f,cls = NumpyArrayEncoder,indent=2)
            Message = "{} Create Class Dictionary DictInitialGuess: True".format(self.CountFunctionsCalled)
            AddMessageToLog(Message,self.LogFile)
## DISTRIBUTIONS

    def PlotDailyDistr(self,LableSave = "Aggregated"):
        """
            Input:
                LabelSave: str -> 'Aggregated' or 'Class_i'
            Description:
                self.Feature2MaxBins = {
                    "time": {"bins": int, "count": int},
                    "lenght": {"bins": int, "count": int},
                    "av_speed": {"bins": int, "count": int}} -> Contains informations about the feature of interest for trajectories
        """
        print('all different groups colored differently')
        # Inititialize Fit for all different classes
        self.CreateDictClass2FitInit()
        if self.verbose:
            print("++++++ Aggregated Fit ++++++")
        self.InfoFittedParameters,self.DictFittedData,Upload,SuccessFit = ReturnFitInfoFromDict(Fcm = self.Fcm,
                                                                            InitialGuess = self.DictInitialGuess,
                                                                            DictFittedData = self.DictFittedData,
                                                                            InfoFittedParameters = self.InfoFittedParameters,
                                                                            Feature2Label = self.Feature2Label,
                                                                            FitFile = os.path.join(self.PlotDir,'Fit_Aggregated'),
                                                                            FittedDataFile = os.path.join(self.PlotDir,'FittedData_Aggregated'))
        self.CountFunctionsCalled += 1
        MessagePlotDailyDistr(self.CountFunctionsCalled,self.LogFile,Upload)
        # Compute the Fcm Partition For Each Feature
        self.Feature2Class2FcmDistr = {Feature: SplitFcmByClass(self.Fcm,Feature,self.IntClass2StrClass) for Feature in self.DictFittedData.keys()}
        InfoPlotDistrFeat = {"figsize":(4,4),"minx":0,"miny":0,"maxx":0,"maxy":0}
        # Compute the MinMax for the Plot
        self.InfoPlotDistrFeat = {Feature: {ComputeMinMaxPlotGivenFeature(self.Feature2Class2FcmDistr[Feature],InfoPlotDistrFeat)} for Feature in self.DictFittedData.keys()}
        for Feature in self.DictFittedData.keys():
            fig,ax = PlotFeatureDistrSeparatedByClass(self.Feature2Class2FcmDistr[Feature],
                                                Feature,
                                                self.InfoPlotDistrFeat[Feature],
                                                self.IntClass2StrClass,
                                                self.DictFittedData,
                                                self.Column2Legend,
                                                self.Feature2IntervalBin,
                                                self.Feature2IntervalCount,
                                                self.Feature2Label,
                                                self.Feature2ShiftBin,
                                                self.Feature2ShiftCount,
                                                self.Feature2ScaleBins,
                                                self.Feature2ScaleCount)
            fig.savefig(os.path.join(self.PlotDir,'{0}_{1}.png'.format(LableSave,self.Column2SaveName[Feature])),dpi = 200)
            plt.close()
            Message = "\tPlot {} Distribution: True\n".format(Feature)
            Message += "\t\tFitting Function {0}\n".format(self.DictFittedData[Feature]["best_fit"])
            AddMessageToLog(Message,self.LogFile)




    def PlotDistrPerClass(self):
        """
            Description:
                For each Feature of ColumnLabel: ['time','lenght','av_speed','time_hours','lenght_km','speed_kmh']
                Plot the distribution of the feature for each class. 
                With a set of guesses for the fitting functions.
            Return:
                InfoDayFit: dict -> {IntClass: {Feature: {Function: [A,b]}}}

        """
        self.InfoDayFit = {IntClass: {} for IntClass in self.IntClass2StrClass.keys()}
        self.Class2DictFittedData = {IntClass: {Feature: {"best_fit":[], "fitted_data":[],"parameters:":[]} for Feature in list(self.Features2Fit)} for IntClass in self.IntClass2StrClass.keys()}
        self.Class2InfoFittedParameters = {IntClass: {Function2Fit: {Feature:{"fit":None,"StdError":None} for Feature in self.DictInitialGuess[Function2Fit].keys()} for Function2Fit in self.DictInitialGuess.keys()} for IntClass in self.IntClass2StrClass.keys()}
        for IntClass in self.IntClass2StrClass:
            if self.verbose:
                print("++++++ Class {} Fit ++++++".format(IntClass))
            self.Class2InfoFittedParameters[IntClass],self.Class2DictFittedData[IntClass],Upload,SuccessFit = ReturnFitInfoFromDict(Fcm = self.Fcm.filter(pl.col("class") == IntClass),
                                                                                                                InitialGuess = self.Class2InitialGuess[IntClass],
                                                                                                                DictFittedData = self.Class2DictFittedData[IntClass],
                                                                                                                InfoFittedParameters = self.Class2InfoFittedParameters[IntClass],
                                                                                                                Feature2Label = self.Feature2Label,
                                                                                                                FitFile = os.path.join(self.PlotDir,'Fit_Class_{0}'.format(IntClass)),
                                                                                                                FittedDataFile = os.path.join(self.PlotDir,'FittedData_Class_{0}'.format(IntClass)))

            if self.verbose:
                print("++++++++++++++++++++")
        if Upload:
            self.CountFunctionsCalled += 1
            Message = "{} Plot Distr Per Class: True\n".format(self.CountFunctionsCalled)
            Message += "\tUpload Class2DictFittedData, Class2InfoFittedParameters"
            AddMessageToLog(Message,self.LogFile)
        else:
            self.CountFunctionsCalled += 1
            Message = "{} Plot Distr Per Class: True\n".format(self.CountFunctionsCalled)
            Message += "\tComputed Class2DictFittedData, Class2InfoFittedParameters"
            AddMessageToLog(Message,self.LogFile)
        for Feature in self.DictFittedData.keys():
            for IntClass in self.IntClass2StrClass:
                fig,ax = plt.subplots(1,1,figsize= (15,12))
                df = self.Fcm.filter(pl.col("class") == IntClass)
                y,x = np.histogram(df[Feature].to_list(),bins = 50)
                if Feature == "av_speed" or Feature == "speed_kmh":
                    y = y/np.sum(y)
# ALREADY COMPUTED WITH ReturnFitInfoFromDict
#                self.Class2DictFittedData[IntClass],self.Class2InfoFittedParameters[IntClass] = FitAndPlot(x[1:],y,self.DictInitialGuess,Feature,self.Class2DictFittedData[IntClass],self.Class2InfoFittedParameters[IntClass])                
                if IntClass!=10 and IntClass!=11:
                    LocalBoolScatPlot = False
                    LocalBoolPlot = False
                    LocalFittedData = np.array(self.Class2DictFittedData[IntClass][Feature]["fitted_data"])
                    LocalMask = LocalFittedData > 0
                    if "speed" in Feature:
                        LocalMask1 = y > 0
                        ScatterY2Plot = y[LocalMask1]
                        ScatterX2Plot = x[1:][LocalMask1]
                        print("{0} {1} ScatterY2Plot: ".format(Feature,IntClass),ScatterY2Plot)
                        if len(ScatterY2Plot)>0:
                            ax.scatter(ScatterX2Plot,ScatterY2Plot)
                            LocalBoolScatPlot = True
                    if len(x[:1]) == len(self.Class2DictFittedData[IntClass][Feature]["fitted_data"]):
                        # Fit                        
                        Y2Plot = LocalFittedData[LocalMask]
                        X2Plot = x[1:][np.array(self.Class2DictFittedData[IntClass][Feature]["fitted_data"]) > 0.9]
                        if len(Y2Plot)>0:
                            ax.plot(X2Plot,np.array(Y2Plot),label = self.Class2DictFittedData[IntClass][Feature]["best_fit"])
                            LocalBoolPlot = True
                    if isinstance(self.Class2DictFittedData[IntClass][Feature]["best_fit"],str): 
                        if LocalBoolScatPlot or LocalBoolPlot:
                            ax.set_xticks(np.arange(x[0],x[-1],self.Feature2IntervalBin[Feature]))
                            ax.set_yticks(np.arange(min(y),max(y),self.Feature2IntervalCount[Feature]))
                            ax.set_xlabel(self.Feature2Label[Feature])
                            ax.set_ylabel('Count')
                            ax.set_xlim(left = 0,right = max(x) + self.Feature2ShiftBin[Feature])
                            ax.set_ylim(bottom = 1,top = max(y) + self.Feature2ShiftCount[Feature])
                            ax.set_xscale(self.Feature2ScaleBins[Feature])
                            ax.set_yscale(self.Feature2ScaleCount[Feature])
                            if len(LocalMask) == len(LocalFittedData):
                                print("Class: ",IntClass," Feature: ",Feature," Day: ",self.StrDate)
                                print("Number of fitted values less than 0: ",len(LocalFittedData) - len(LocalFittedData[LocalMask]))
                            plt.savefig(os.path.join(self.PlotDir,'{0}_Class_{1}_{2}.png'.format(self.Class2DictFittedData[IntClass][Feature]["best_fit"],IntClass,self.Column2SaveName[Feature])),dpi = 200)
                            plt.close()
                        Message = "\tPlot {0} Distribution Class {1}: True\n".format(Feature,IntClass)
                        Message = "\t\tFitting Function {0}".format(self.Class2DictFittedData[IntClass][Feature]["best_fit"])
                        AddMessageToLog(Message,self.LogFile)  

                    else:
                        print("Warning: Best Fit Not Found for Class {0} and Feature {1} Day: {2}".format(IntClass,Feature,self.StrDate))
                        Message = "\tPlot {0} Distribution Class {1}: False\n".format(Feature,IntClass)
                        Message = "\t\tBest Fit Not Found"
                        AddMessageToLog(Message,self.LogFile)        

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


def GetDistributionPerClass(fcm,Feature,class_):
    """
        Input:
            Feature: str -> time, lenght, av_speed, p, a_max
        Returns:
            n, bins of velocity distribution
    """
    n, bins = np.histogram(fcm.filter(pl.col("class") == class_)[Feature].to_list(), bins = bins)



def PlotSubnetHTML(ListDailyNetwork,Daily = True):
    for DailyNetwork in ListDailyNetwork:
        list_of_lists = DailyNetwork.IntClass2Roads
        # Create a base map
        m = folium.Map()

        # Iterate through the list of lists
        for class_, index_list in list_of_lists.items():
            # Filter GeoDataFrame for roads with indices in the current list
            filtered_gdf = DailyNetwork.GeoJson[DailyNetwork.GeoJson['index'].isin(index_list)]
            
            # Create a feature group for the current layer
            layer_group = folium.FeatureGroup(name=f"Layer {class_}").add_to(m)
            
            # Add roads to the feature group with a unique color
            for _, road in filtered_gdf.iterrows():
                color = 'blue'  # Choose a color for the road based on index or any other criterion
                folium.GeoJson(road.geometry, style_function=lambda x: {'color': color}).add_to(layer_group)
            
            # Add the feature group to the map
            layer_group.add_to(m)

        # Add layer control to the map
        folium.LayerControl().add_to(m)

        # Save or display the map
        m.save("map_with_layers.html")




"""resolution = 100
n_bins_std = 100
bin_width = 5
rescaling_factor_pdf = resolution/n_bins_std
i=0
for fcm_data in fcm:
    plot_distribution_velocity_all_class_together_per_day(fcm_data,list_dict_name,i)
    plot_aggregated_velocity(fcm_data,list_dict_name,i)
    for cl,df in fcm_data.groupby('class'):
        if cl!=10 and len(list_dict_name[i][cl])!=0:
            n,bins = np.histogram(df['av_speed'].to_numpy(),bins = n_bins_std,range = [0,n_bins_std-bin_width])            
            scaling_factor_data = np.sum(n)
            initial_guess_sigma = np.std(df['av_speed'].to_numpy())
            initial_guess_mu = np.mean(df['av_speed'].to_numpy())
            params, pcov = curve_fit(maxwellian,xdata = bins[:-1],ydata = np.array(n)/scaling_factor_data, p0=[initial_guess_sigma, initial_guess_mu])
            a_maxwell,b_maxwell = params
            print("covariance matrix a,b:\n",pcov)
            print("a_maxwell,b_maxwell:\n",a_maxwell,b_maxwell)
#            a_maxwell,b_maxwell = maxwell.fit(df['av_speed'].to_numpy(),floc = np.mean(df['av_speed']))
            a_gauss,b_gauss = norm.fit(df['av_speed'].to_numpy(),floc = np.mean(df['av_speed']))
            fig,ax = plt.subplots(1,1,figsize= (15,12))
            plt.hist(df['av_speed'].to_numpy(),bins = n_bins_std,range = [0,n_bins_std-bin_width],density = True)
            av_speed = np.mean(df['av_speed'].to_numpy())       
            ax.set_xlabel('average speed (m/s)')
            ax.set_ylabel('Count')
            ax.set_title(list_dict_name[i][cl] + ' vel: ' + str(round(av_speed,3)) +' m/s')
#            print('maxwell pdf:\n ',maxwellian(np.linspace(min(bins),max(bins),resolution),a_maxwell,b_maxwell))
#            print('gaussian pdf rescaled:\n ',norm.pdf(np.linspace(min(bins),max(bins),resolution),a_gauss,b_gauss))
            plt.plot(np.linspace(min(bins),max(bins),resolution),maxwellian(np.linspace(min(bins),max(bins),resolution),a_maxwell,b_maxwell),label = 'maxwell',color = 'violet')
            plt.plot(np.linspace(min(bins),max(bins),resolution),norm.pdf(np.linspace(min(bins),max(bins),resolution),a_gauss,b_gauss),label = 'gauss',color = 'red')
            plt.legend(['maxwell','gauss'])
            plt.savefig(os.path.join(s_dir[i],'average_speed_{}.png'.format(list_dict_name[i][cl])),dpi = 200)
            plt.show()
    i+=1

"""