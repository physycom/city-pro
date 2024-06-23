'''
    NOTE: stats.csv Is Useless but I keep it for reference.
    NOTE: The Organization of the script is around DailyNetworkStats.
        This class contains all the informations about trrajectories and Network in one day.
    The motivation is to simplify the analysis for multiple days.
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

# Ignore all warnings
warnings.filterwarnings("ignore")
if os.path.isfile(os.path.join(os.environ["WORKSPACE"],"city-pro","custom_style.mplstyle")):
    plt.style.use(os.path.join(os.environ["WORKSPACE"],"city-pro","custom_style.mplstyle"))
else:
    try:
        import PlotSettings
    except Exception as e:
        print("No Plot Settings File Found")

        

def Dict2PolarsDF(Dict,schema):
    return pl.DataFrame(Dict,schema=schema)

def ComputeMFDVariables(Df,MFD,TimeStampDate,dt,iterations,verbose = False):
    """
        NOTE: The bins in time that have 0 trajectories have 0 average speed
        NOTE: Speed in MFD in km/h
    """
    print("Compute MFD Variables:")
    TmpDict = {"time":[],"population":[],"speed_kmh":[],"av_speed":[]}
    for t in range(int(iterations)):
        StartInterval = datetime.datetime.fromtimestamp(int(TimeStampDate)+t*dt)
        EndInterval = datetime.datetime.fromtimestamp(int(TimeStampDate)+(t+1)*dt)                    
        TmpDf = Df.with_columns(pl.col('start_time').apply(lambda x: datetime.datetime.fromtimestamp(x), return_dtype=pl.Datetime).alias("start_time_datetime"),
                                    pl.col('end_time').apply(lambda x: datetime.datetime.fromtimestamp(x), return_dtype=pl.Datetime).alias("end_time_datetime"))

        TmpFcm = TmpDf.filter(pl.col('start_time_datetime').is_between(StartInterval,EndInterval))
        Hstr = StartInterval.strftime("%Y-%m-%d %H:%M:%S").split(" ")[1]
        TmpDict["time"].append(Hstr)
        TmpDict["population"].append(len(TmpFcm))

        if len(TmpFcm) > 0:
            AvSpeed = TmpFcm.select(pl.col("speed_kmh").mean()).to_pandas().iloc[0]["speed_kmh"]
            TmpDict["speed_kmh"].append(AvSpeed)
            AvSpeed = TmpFcm.select(pl.col("av_speed").mean()).to_pandas().iloc[0]["av_speed"]
            TmpDict["av_speed"].append(AvSpeed)
            MoreThan0Traj = True
        else:
            TmpDict["speed_kmh"].append(0)
            TmpDict["av_speed"].append(0)
            MoreThan0Traj = False
#        if verbose:
#            print("Iteration: ",t)
#            print("Considered Hour: ",Hstr)
#            print("Population: ",len(TmpFcm))
#            print("Size dict: ",len(TmpDict["time"]))
#            if MoreThan0Traj:
#                print("Speed: ",AvSpeed)
#    if verbose:
#        print("Dict: ",TmpDict)
    MFD = Dict2PolarsDF(TmpDict,schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed_kmh":pl.Float64,"av_speed":pl.Float64})
    return MFD,Df

def GetAverageConditional(Df,ConditioningLabel,ConditionedLabel,binsi,binsi1):
    """
        Df: pl.DataFrame 
        ConditioningLabel: str -> Conditioning Variable name in the DataFrame.
        ConditionedLabel: str -> The Column from which extracting the average.
        bini: int
        binsi1: int
        Return: float -> Average of the ConditionedLabel in the interval [bini,binsi1]

    """
    assert ConditioningLabel in Df.columns,  "Error: ConditioningLabel -> {} Column in the DataFrame".format(ConditioningLabel)
    assert ConditionedLabel in Df.columns,  "Error: ConditionedLabel -> {} Column in the DataFrame".format(ConditionedLabel)
    DfTmp = Df.filter((pl.col(ConditioningLabel) >= binsi) & 
                (pl.col(ConditioningLabel)<=binsi1))
    if len(DfTmp)>0:
        return DfTmp.with_columns(pl.col(ConditionedLabel).mean()).to_pandas().iloc[0][ConditionedLabel]
    else:
        return 0

def GetStdErrorConditional(Df,ConditioningLabel,ConditionedLabel,binsi,binsi1):
    """
        Df: pl.DataFrame 
        ConditioningLabel: str -> Conditioning Variable name in the DataFrame.
        ConditionedLabel: str -> The Column from which extracting the average.
        bini: int
        binsi1: int
        Return: float -> Standard Error of the ConditionedLabel in the interval [bini,binsi1]

    """
    assert ConditioningLabel in Df.columns,  "Error: ConditioningLabel -> {} Column in the DataFrame".format(ConditioningLabel)
    assert ConditionedLabel in Df.columns,  "Error: ConditionedLabel -> {} Column in the DataFrame".format(ConditionedLabel)
    DfTmp = Df.filter((pl.col(ConditioningLabel) >= binsi) & 
                (pl.col(ConditioningLabel)<=binsi1))
    if len(DfTmp)>1:
        return DfTmp.with_columns(pl.col(ConditionedLabel).std()).to_pandas().iloc[0][ConditionedLabel]
    else:
        return 0

def GetLowerBoundsFromBins(bins,label,MinMaxPlot,Class,case):
    if case == "no-classes":
        print("Get Lower Bounds From Bins: {}".format(label))
        MinMaxPlot[label] = {"min":bins[0],"max":bins[-1]}  
        return MinMaxPlot
    else:
        print("Get Lower Bounds From Bins: {0} Class {1}".format(label,Class))
        MinMaxPlot[Class][label] = {"min":bins[0],"max":bins[-1]}
        return MinMaxPlot

def GetMFDForPlot(MFD,MFD2Plot,MinMaxPlot,Class,case,verbose = False,bins_ = 20):
    """
        Input:
            MFD: {"population":[],"time":[],"speed_kmh":[]} or {Class:pl.DataFrame{"population":[],"time":[],"speed_kmh":[]}}
        NOTE: Used in self.PlotMFD()
        NOTE: Modifies MDF2Plot = {"bins_population":[p0,..,p19],"binned_av_speed":[v0,..,v19],"binned_sqrt_err_speed":[e0,..,e19]}
        NOTE: Modifies MinMaxPlot = {"speed_kmh":{"min":v0,"max":v19},"population":{"min":p0,"max":p19}}    
    """
    assert "population" in MFD.columns, "population not in MFD"
    assert "speed_kmh" in MFD.columns, "speed not in MFD"
#    assert "bins_population" in MFD2Plot.columns, "bins_population not in MFD2Plot"
#    assert "binned_av_speed" in MFD2Plot.columns, "binned_av_speed not in MFD2Plot"
#    assert "binned_sqrt_err_speed" in MFD2Plot.columns, "binned_sqrt_err_speed not in MFD2Plot"
    print("Get MFD For Plot: {}".format(Class))
    n, bins = np.histogram(MFD["population"],bins = bins_)
    labels = range(len(bins) - 1)
    for i in range(len(labels)):
        # Fill Average/Std Speed (to plot)
        BinnedAvSpeed = GetAverageConditional(MFD,"population","speed_kmh",bins[i],bins[i+1])
        MFD2Plot['binned_av_speed'].append(BinnedAvSpeed)
        BinnedSqrtSpeed = GetStdErrorConditional(MFD,"population","speed_kmh",bins[i],bins[i+1])
        MFD2Plot['binned_sqrt_err_speed'].append(BinnedSqrtSpeed)
#        if verbose:
#            print("Bin [",bins[i],',',bins[i+1],']')
#            print("Av Speed: ",BinnedAvSpeed)
#            print("SqrtError: ",BinnedSqrtSpeed)
    MFD2Plot["bins_population"] = bins
    if verbose:
        print("MFD Features Aggregated: ")
#        print("Bins Population:\n",MFD2Plot['bins_population'])
#        print("\nBins Average Speed:\n",MFD2Plot['binned_av_speed'])
#        print("\nBins Standard Deviation:\n",MFD2Plot['binned_sqrt_err_speed'])
    MinMaxPlot = GetLowerBoundsFromBins(bins = bins,label = "population",MinMaxPlot = MinMaxPlot,Class = Class,case = case)
    MinMaxPlot = GetLowerBoundsFromBins(bins = MFD2Plot['binned_av_speed'],label = "speed_kmh",MinMaxPlot = MinMaxPlot, Class = Class,case = case)
    Y_Interval = max(MFD2Plot['binned_av_speed']) - min(MFD2Plot['binned_av_speed'])
    RelativeChange = Y_Interval/max(MFD2Plot['binned_av_speed'])/100
    if verbose:
#        print("\nMinMaxPlot:\n",MinMaxPlot)
        print("\nInterval Error: ",Y_Interval)            
    return MFD2Plot,MinMaxPlot,RelativeChange

def SaveMFDPlot(binsPop,binsAvSpeed,binsSqrt,RelativeChange,SaveDir,Title = "Fondamental Diagram Aggregated",NameFile = "MFD.png"):
    """
        
    """
#    assert "bins_population" in MFD2Plot.columns, "bins_population not in MFD2Plot"
#    assert "binned_av_speed" in MFD2Plot.columns, "binned_av_speed not in MFD2Plot"
#    assert "binned_sqrt_err_speed" in MFD2Plot.columns, "binned_sqrt_err_speed not in MFD2Plot"
    print("Plotting MFD:\n")
    fig, ax = plt.subplots(1,1,figsize = (10,8))
    text = "Relative change : {}%".format(round(RelativeChange,2))
    ax.plot(binsPop[1:],binsAvSpeed)
    ax.fill_between(np.array(binsPop[1:]),
                        np.array(binsAvSpeed) - np.array(binsSqrt), 
                        np.array(binsAvSpeed) + np.array(binsSqrt), color='gray', alpha=0.2, label='Std')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10,
    verticalalignment='top', bbox=props)
    ax.set_title(Title)
    ax.set_xlabel("number people")
    ax.set_ylabel("speed (km/h)")
    plt.savefig(os.path.join(SaveDir,NameFile),dpi = 200)
    plt.close()

def PlotHysteresis(MFD,Title,SaveDir,NameFile):
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    x = MFD['population'].to_list()
    y = MFD['speed_kmh'].to_list()
    u = [x[i+1]-x[i] for i in range(len(x)-1)]
    v = [y[i+1]-y[i] for i in range(len(y)-1)]
    u.append(x[len(x)-1] -x[0])
    v.append(y[len(y)-1] -y[0])
    plt.quiver(x,y,u,v,angles='xy', scale_units='xy', scale=1,width = 0.0025)
    plt.xlabel('Number People')
    plt.ylabel('Speed (km/h)')
    plt.title(Title)
    plt.savefig(os.path.join(SaveDir,NameFile),dpi = 200)
    plt.close()

def NormalizeWidthForPlot(arr,min_val,max_val,min_width = 1, max_width = 10):
    '''
        Description:
            Normalizes the width for road fluxes
    '''
    if not isinstance(arr,np.ndarray):
        arr = np.array(arr)
    else:
        pass
    if not np.isnan(min_val):
        pass
    else:
        min_val = 0
    if not np.isnan(max_val):
        pass
    else:
        max_val = 130
    if (max_val - min_val) == 0:
        print("Max value {0} Is equal to the minimum {1}.".format(max_val,min_val))
        return 1
    return (arr - min_val) / (max_val - min_val) * (max_width - min_width) + min_width
# CAST
def CastString2Int(Road):
    try:
        int(Road)
        return int(Road),True
    except:
        print("Road exception: ",Road)
        return Road,False

# CONVERSIONE
def ms2kmh(v):
    return v*3.6
def kmh2ms(v):
    return v/3.6
def s2h(t):
    return t/3600
def h2s(t):
    return t*3600
def m2km(x):
    return x/1000
def km2m(x):
    return x*1000
# TIME
def StrDate2DateFormatLocalProject(StrDate):
    return StrDate.split("-")[0],StrDate.split("-")[1],StrDate.split("-")[2]

def Timestamp2Datetime(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)

def Timestamp2Date(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).date()

def Datetime2Timestamp(datetime_):
    return datetime_.timestamp()

def InInterval(start_time,end_time,TimeStampDate,t,dt):
    if (int(start_time)> int(TimeStampDate)+t*dt and int(start_time)<int(TimeStampDate)+(t+1)*dt) and (int(end_time)> int(TimeStampDate)+t*dt and int(end_time)<int(TimeStampDate)+(t+1)*dt):
        return True
    else:
        return False
# PRINT
def PrintMFDDictInfo(MFD,StartingString = "Class 2 MFD: "):
    print(StartingString)
    print(MFD)

# FIT AND DICT

def FillInitGuessIntervalPlExp(DictInitGuessInterval,MaxCount,Avg,StartWindow,EndWindow,Function2Test):
    """
        Input:
            - DictInitGuessInterval: dict -> Dictionary with initial_guess and interval
                NOTE: Usage In Program:
                    1) Class2InitialGuess[IntClass][Function2Test][Feature]
                    2) DictInitialGuess[Function2Test][Feature]
            - MaxCount: int -> Maximum Count of the Class
            - Avg: float -> Average of the Class
            - StartWindow: int -> Start Window in Seconds
            - EndWindow: int -> End Window in Seconds
                NOTE: Usage in Program:
                    For (time,lenght):
                        1) LocalDict2Params[StrClass]["MaxCount"]
                        2) LocalDict2Params[StrClass]["Avg"]
                        3) LocalDict2Params[StrClass]["StartWindowS"]
                        4) LocalDict2Params[StrClass]["EndWindowS"]
                    For (time_hours,lenght_km):
                        1) LocalDict2Params[StrClass]["MaxCount"]
                        2) LocalDict2Params[StrClass]["Avg"]/(SecondsInHour-MetersInKm)
                        3) LocalDict2Params[StrClass]["StartWindowS"]/(SecondsInHour-MetersInKm)
                        4) LocalDict2Params[StrClass]["EndWindowS"]/(SecondsInHour-MetersInKm)
            - Function2Test: str -> Function to Test (exponential,powerlaw)

            LocalDict2Params: dict -> Dictionary with parameters for the class

    """
    if Function2Test == "exponential":
        DictInitGuessInterval["initial_guess"] = [MaxCount,Avg]
        DictInitGuessInterval["interval"] = [StartWindow,EndWindow]        
    elif Function2Test == "powerlaw":
        DictInitGuessInterval["initial_guess"] = [MaxCount*StartWindow,-1]
        DictInitGuessInterval["interval"] = [StartWindow,EndWindow]    
    return DictInitGuessInterval

def FillInitGuessIntervalMxGs(DictInitGuessInterval,Fcm,Feature,IntClass):
    """
        Input:
            - DictInitGuessInterval: dict -> Dictionary with initial_guess and interval
                NOTE: Usage In Program:
                    1) Class2InitialGuess[IntClass][Function2Test][Feature]
                    2) DictInitialGuess[Function2Test][Feature]
            - Fcm: pl.DataFrame -> DataFrame with the FCM
            - Feature: str -> Feature to Test
    """
    if IntClass is None:
        LocalMeanMs = np.mean(Fcm[Feature].to_list())
        LocalStdMs = np.std(Fcm[Feature].to_list())
    else:
        LocalMeanMs = Fcm.filter(pl.col("class") == IntClass)[Feature].mean()
        LocalStdMs = Fcm.filter(pl.col("class") == IntClass)[Feature].std()
    DictInitGuessInterval["initial_guess"] = [LocalMeanMs,LocalStdMs]
    return DictInitGuessInterval

def ReturnFitInfoFromDict(Fcm,InitialGuess,DictFittedData,Feature2Label,FitFile,FittedDataFile):
    """
        Input:
            Fcm: pl.DataFrame -> Fcm Dataframe
            InfoFittedParameters: dict -> {Feature: {Function: [A,b]}}
            DictFittedData: dict -> {Feature: {"fitted_data": [],"best_fit": str}}
            InitialGuess: dict -> {"exponential":{"time":{"initial_guess":[],"interval":[]},"time_hours":{"initial_guess":[],"interval":[]}},
            Feature2Label: dict -> {Feature: Label}
            FitFile: str -> Path to the Fit File
            FittedDataFile: str -> Path to the Fitted Data File
        Description:
            For each Feature of ColumnLabel: ['time','lenght','av_speed','time_hours','lenght_km','speed_kmh']
            Fit the distribution of the feature and plot the distribution.
        Output:
            InfoFittedParameters: dict -> {Feature: {Function: [A,b]}}
            DictFittedData: dict -> {Feature: {"fitted_data": [],"best_fit": str}}
    """
    print("Return Fit Info From Dict:")
    InfoFittedParameters =  {Function2Fit: {Feature:{"fit":None,"StdError":None} for Feature in Feature2Label.keys()} for Function2Fit in InitialGuess.keys()}
    DictFittedData = {Feature: {"best_fit":[], "fitted_data":[]} for Feature in list(Feature2Label.keys())}
    if os.path.isfile(FitFile) and os.path.isfile(FittedDataFile):
        with open(FitFile,'r') as f:
            InfoFittedParameters = json.load(f)
        with open(FittedDataFile,'r') as f:
            DictFittedData = json.load(f)
    else:
        for Feature in Feature2Label.keys():
            y,x = np.histogram(Fcm[Feature].to_list(),bins = 50)
            InfoFittedParameters, DictFittedData = FitAndPlot(x[1:],y,InitialGuess,Feature,InfoFittedParameters,DictFittedData)
        with open(FitFile,'w') as f:
            json.dump(InfoFittedParameters,f)
        with open(FittedDataFile,'w') as f:
            json.dump(DictFittedData,f)

    return InfoFittedParameters,DictFittedData

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
# --------------- Read Files ---------------- #
    def ReadTimedFluxes(self):
        if self.verbose:
            print("Reading timed_fluxes")
            print(self.DictDirInput["timed_fluxes"])
        if os.path.isfile(self.DictDirInput["timed_fluxes"]):
            self.TimedFluxes = pd.read_csv(self.DictDirInput["timed_fluxes"],delimiter = ';')
            self.TimedFluxes = pl.from_pandas(self.TimedFluxes)
            self.ReadTime2FluxesBool = True
        else:   
            print("No timed_fluxes")    

    def ReadFluxes(self):
        if self.verbose:
            print("Reading fluxes")
            print(self.DictDirInput["fluxes"])
        if os.path.isfile(self.DictDirInput["fluxes"]):
            self.Fluxes = pd.read_csv(self.DictDirInput["fluxes"],delimiter = ';')
            self.Fluxes = pl.from_pandas(self.Fluxes)
            self.ReadFluxesBool = True        
        else:
            print("No fluxes")    

    def ReadFcm(self):
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
        else:
            print("No fcm")

    def ReadStats(self):
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
        else:
            print("No stats")    

    def ReadFcmNew(self):
        if self.verbose:
            print("Reading fcm_new")
            print(self.DictDirInput["fcm_new"])
        if os.path.isfile(self.DictDirInput["fcm_new"]):
            self.FcmNew = pd.read_csv(self.DictDirInput["fcm_new"],delimiter = ';')
            self.FcmNew = pl.from_pandas(self.FcmNew)
            self.ReadFcmNewBool = True
        else:
            print("No fcm_new")    

    def ReadFcmCenters(self,verbose=False):
        """
            Description:
                Read the centers of the FCM
            NOTE: This function will define also what are the classes in any plot since it is used to initialize
            IntClass2StrClass
        """
        if self.verbose:
            print("Reading fcm_centers")
            print(self.DictDirInput["fcm_centers"])
        Features = {"class":[],"av_speed":[],"v_max":[],"v_min":[],"sinuosity":[],"people":[]}
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

    def ReadFluxesSub(self):
        '''
            Input:
                FluxesSubFile: (str) -> FluxesSubFile = '../{basename}_{start}_{start}/fluxes.sub'
                verbose: (bool) -> verbose = False
            Output:
                self.IntClass2Roads: (dict) -> self.IntClass2Roads = {IntClass:[] for IntClass in self.IntClasses}
                self.IntClass2RoadsInit: (bool) -> Boolean value to Say I have stored the SubnetInts For each Class
        '''
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
        else:
            print("FluxesSubFile not found")

    def ReadGeoJson(self):
        if self.verbose:
            print("Reading GeoJson")
        if not os.path.isfile(self.GeoJsonFile):
            exit("GeoJsonFile not found")
        self.GeoJson = gpd.read_file(self.GeoJsonFile)
        self.ReadGeoJsonBool = True

    def GetIncreasinglyIncludedSubnets(self):
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
            print("Warning: Not Initialized DictSubnetsTxtDir -> Will not have Increasingly Included SubNetworks")

    def ReadFluxesSubIncreasinglyIncludedIntersection(self):
        '''
            Input:
                FluxesSubFile: (str) -> FluxesSubFile = '../{basename}_{start}_{start}/fluxes.sub'
                verbose: (bool) -> verbose = False
            Output:
                self.IntClass2RoadsIncreasinglyIncludedIntersection: (dict) -> {IntClass:[] for IntClass in self.IntClasses}
        '''
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
        except:
            print("FluxesSubFile not found")

#--------- COMPLETE GEOJSON ------- ##
    def CompleteGeoJsonWithClassInfo(self):
        """
            Computes "IntClassOrdered" and "StrClassOrdered" columns for the Geojson.
            Useful when I want to reconstruct the road network for all the days.
        """
        if self.ReadGeojsonBool and self.ReadFluxesSubIncreasinglyIncludedIntersectionBool:
            ClassOrderedForGeojsonRoads = np.zeros(len(self.GeoJson),dtype = int)
            for Class in self.IntClass2RoadsIncreasinglyIncludedIntersection.keys():
                for Road in self.IntClass2RoadsIncreasinglyIncludedIntersection[Class]: 
                    ClassOrderedForGeojsonRoads[np.where(self.Geojson["poly_lid"] == Road)] = Class 
            self.GeoJson["IntClassOrdered_{}".format(self.StrDate)] = ClassOrderedForGeojsonRoads
            self.GeoJson["StrClassOrdered_{}".format(self.StrDate)] = [self.IntClass2StrClass[intclass] for intclass in ClassOrderedForGeojsonRoads]
        self.GeoJson.to_file(os.path.join(self.InputBaseDir,"BolognaMDTClassInfo.geojson"))

    def ReadVelocitySubnet(self):
        if self.BoolStrClass2IntClass:
            try:
                for Class in self.IntClass2StrClass.keys():
                    self.RoadInClass2VelocityDir[Class] = os.path.join(os.path.join(self.InputBaseDir,self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + '_class_{}velocity_subnet.csv'.format(Class)))
                    self.VelTimePercorrenceClass[Class] = pd.read_csv(self.RoadInClass2VelocityDir[Class],delimiter = ';')
                    self.VelTimePercorrenceClass[Class] = pl.from_pandas(self.VelTimePercorrenceClass[Class])
                self.ReadVelocitySubnetBool = True
            except:
                print("VelTimePercorrenceFile not found")
        else:
            print("Warning: No Initialization of VelTimePercorrenceClass due to lack of definition of IntClass2Str")

    def AddFcmNew2Fcm(self,verbose = True):
        """
            Description:
                Convert the class column of the FcmNew to class_new and join it to the Fcm.
                In this way we have in Fcm for each trajectory a new column with the class of the trajectory after having intersected the subnetworks..
        """
        if self.ReadFcmBool and self.ReadFcmNewBool:
            FcmNew = self.FcmNew.with_columns([self.FcmNew['class'].alias('class_new')])
            self.Fcm = self.Fcm.join(FcmNew[['id_act', 'class_new']], on='id_act', how='left')
            if verbose:
                print("1st join Fcm: ",self.Fcm.columns)
                print("Date: ",self.StrDate)
            self.Fcm =self.Fcm.with_columns([self.Fcm['class'].alias('class_new')])
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
        if self.ReadFluxesSubIncreasinglyIncludedIntersectionBool and self.ReadGeoJsonBool and self.BoolStrClass2IntClass:
            print("Plotting Daily Incremental Subnetworks in HTML")
            if not os.path.isfile(os.path.join(self.PlotDir,"Subnets_{}.html".format(self.StrDate))):
                print("Save in: ",os.path.join(self.PlotDir,"SubnetsIncrementalInclusion_{}.html".format(self.StrDate)))
                # Create a base map
                m = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
                # Iterate through the Dictionary of list of poly_lid
                if self.verbose:
                    for class_ in self.IntClass2RoadsIncreasinglyIncludedIntersection.keys(): 
                        print(self.IntClass2RoadsIncreasinglyIncludedIntersection[class_][:10])
                for IntClass in self.IntClass2StrClass.keys():
                    for index_list in self.IntClass2RoadsIncreasinglyIncludedIntersection[IntClass]:
                        if self.verbose:
                            print("Plotting Class ",IntClass)
                            if isinstance(index_list,int):
                                index_list = [index_list]                        
                            print("Number of Roads: ",len(index_list))
                        if isinstance(index_list,int):
                            index_list = [index_list]                        
                        # Filter GeoDataFrame for roads with indices in the current list
                        filtered_gdf = self.GeoJson[self.GeoJson['poly_lid'].isin(index_list)]
                        # Create a feature group for the current layer
                        layer_group = folium.FeatureGroup(name="Layer {}".format(IntClass)).add_to(m)
                        # Add roads to the feature group with a unique color
                        if self.verbose:
                            print("Class: ",IntClass," Number of Roads: ",len(filtered_gdf),"Color: ",self.Class2Color[self.IntClass2StrClass[IntClass]])
                        for _, road in filtered_gdf.iterrows():
                            folium.GeoJson(road.geometry, style_function=lambda x: {'color': self.Class2Color[self.IntClass2StrClass[IntClass]]}).add_to(layer_group)
                        
                        # Add the feature group to the map
                        layer_group.add_to(m)
                        # Add layer control to the map
                        folium.LayerControl().add_to(m)

                # Save or display the map
                m.save(os.path.join(self.PlotDir,"SubnetsIncrementalInclusion_{}.html".format(self.StrDate)))
            else:
                print("Subnets Increasingly already Plotted in HTML")
        else:
            print("No Subnetworks to Plot")
            return False

    def PlotSubnetHTML(self):
        """
            Description:
                Plots in HTML the road network with its subnets.
                NOTE: 
                    Does not consider the intersection
        """
        if self.ReadFluxesSubBool and self.ReadGeoJsonBool and self.BoolStrClass2IntClass:

            print("Plotting Daily Subnetworks in HTML")
            if not os.path.isfile(os.path.join(self.PlotDir,"Subnets_{}.html".format(self.StrDate))):
                # Create a base map
                m = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
                # Iterate through the Dictionary of list of poly_lid
                for IntClass in self.IntClass2StrClass.keys():
                    for index_list in self.IntClass2Roads[IntClass]:
                        if isinstance(index_list,int):
                            index_list = [index_list]
                        # Filter GeoDataFrame for roads with indices in the current list
                        filtered_gdf = self.GeoJson[self.GeoJson['poly_lid'].isin(index_list)]
                        # Create a feature group for the current layer
                        layer_group = folium.FeatureGroup(name=f"Layer {IntClass}").add_to(m)
                        
                        # Add roads to the feature group with a unique color
                        for _, road in filtered_gdf.iterrows():
                            color = 'blue'  # Choose a color for the road based on index or any other criterion
                            folium.GeoJson(road.geometry, style_function=lambda x: {'color': self.Class2Color[self.IntClass2StrClass[IntClass]]}).add_to(layer_group)
                        
                        # Add the feature group to the map
                        layer_group.add_to(m)

                        # Add layer control to the map
                        folium.LayerControl().add_to(m)

                # Save or display the map
                m.save(os.path.join(self.PlotDir,"Subnets_{}.html".format(self.StrDate)))
            else:
                print("Subnets already Plotted in HTML")
        else:
            print("No Subnetworks to Plot")
            return False

    def PlotFluxesHTML(self):
        '''
            Description:
                Plots in .html the map of the bounding box considered.
                For each road color with the fluxes.
                    1) FT
                    2) TF
                    3) TF + FT
        '''
        if self.ReadTime2FluxesBool:
            print("Plotting Daily Fluxes in HTML")
            if not os.path.isfile(os.path.join(self.PlotDir,"Fluxes_{}.html".format(self.StrDate))):
                # Create a base map
                m = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
                mFT = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
                mTF = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
                TF = self.TimedFluxes
                min_val = min(TF["total_fluxes"])
                max_val = max(TF["total_fluxes"])
                TF = TF.with_columns(pl.col("total_fluxes").apply(lambda x: NormalizeWidthForPlot(x,min_val,max_val), return_dtype=pl.Int64).alias("width_total_fluxes"))
                TF = TF.with_columns(pl.col("n_traj_FT").apply(lambda x: NormalizeWidthForPlot(x,min_val,max_val), return_dtype=pl.Int64).alias("width_n_traj_FT"))
                TF = TF.with_columns(pl.col("n_traj_TF").apply(lambda x: NormalizeWidthForPlot(x,min_val,max_val), return_dtype=pl.Int64).alias("width_n_traj_TF"))
                CopyGdf = self.GeoJson
                CopyGdf = CopyGdf.merge(TF.to_pandas(),how = 'left',left_on = 'poly_lid',right_on = 'id_local')
                # Iterate through the Dictionary of list of poly_lid            
                for t,tdf in TF.group_by("time"):
                    # Filter GeoDataFrame for roads with indices in the current list
                    # Create a feature group for the current layer
                    layer_group = folium.FeatureGroup(name=f"Layer {t}").add_to(m)
                    layer_groupFT = folium.FeatureGroup(name=f"Layer {t}").add_to(m)
                    layer_groupTF = folium.FeatureGroup(name=f"Layer {t}").add_to(m)
                    # Add roads to the feature group with a unique color
                    color = 'blue'  # Choose a color for the road based on index or any other criterion
                    print(CopyGdf.columns)
                    for idx, row in CopyGdf.iterrows(): 
                        folium.GeoJson(row.geometry, style_function=lambda x: {'color': color,"weight":row["width_total_fluxes"]}).add_to(layer_group)
                        folium.GeoJson(row.geometry, style_function=lambda x: {'color': color,"weight":row["width_n_traj_TF"]}).add_to(layer_groupTF)
                        folium.GeoJson(row.geometry, style_function=lambda x: {'color': color,"weight":row["width_n_traj_FT"]}).add_to(layer_groupFT)
                    
                    # Add the feature group to the map
                    layer_group.add_to(m)
                    layer_group.add_to(mTF)
                    layer_group.add_to(mFT)


                # Add layer control to the map
                folium.LayerControl().add_to(m)
                folium.LayerControl().add_to(mTF)
                folium.LayerControl().add_to(mFT)

                # Save or display the map
                m.save(os.path.join(self.PlotDir,"Fluxes_{}.html".format(self.StrDate)))
                mTF.save(os.path.join(self.PlotDir,"TailFrontFluxes_{}.html".format(self.StrDate)))
                mFT.save(os.path.join(self.PlotDir,"FrontTailFluxes_{}.html".format(self.StrDate)))
            else:
                print("Fluxes already Plotted in HTML")
    def PlotTimePercorrenceHTML(self):
        if self.ReadGeojsonBool and self.ReadVelocitySubnetBool:
            print("Plotting Daily Fluxes in HTML")
            if not os.path.isfile(os.path.join(self.PlotDir,"AvSpeed_{}.html".format(self.StrDate))):            
                # Create a base map
                m = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
                m1 = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
                for time,RTV in RoadsTimeVel.groupby("start_bin"):
                    layer_group = folium.FeatureGroup(name=f"Layer {time}").add_to(m)
                    layer_group1 = folium.FeatureGroup(name=f"Layer {time}").add_to(m)
                    for Class in self.IntClass2BestFit.keys():
                        RoadsTimeVel = self.VelTimePercorrenceClass[Class]
                        RoadsTimeVel["av_speed"] = [x if x!=-1 else 0 for x in RoadsTimeVel["av_speed"]]
                        RoadsTimeVel["time_percorrence"] = [x if x!=-1 else 0 for x in RoadsTimeVel["time_percorrence"]]
                        min_val = min(RoadsTimeVel["av_speed"])
                        max_val = max(RoadsTimeVel["av_speed"])
                        RoadsTimeVel = RoadsTimeVel.with_columns(pl.col("av_speed").apply(lambda x: NormalizeWidthForPlot(x,min_val,max_val), return_dtype=pl.Int64).alias("width_speed"))
                        min_val = min(RoadsTimeVel["time_percorrence"])
                        max_val = max(RoadsTimeVel["time_percorrence"])
                        RoadsTimeVel = RoadsTimeVel.with_columns(pl.col("time_percorrence").apply(lambda x: NormalizeWidthForPlot(x,min_val,max_val), return_dtype=pl.Int64).alias("width_time"))
                        # Hey, cool, I wrote this bug!
                        # Add roads to the feature group with a unique color
                        list_colored_roads_speed = RTV.loc[RTV["av_speed"]!=0]["poly_id"]
                        filtered_gdf = self.GeoJson[self.GeoJson['poly_lid'].isin(list_colored_roads_speed)]
                        filtered_gdf["width_speed"] = RoadsTimeVel["width_speed"]
                        filtered_gdf["width_time"] = RoadsTimeVel["width_time"]
                        for idx, row in filtered_gdf.iterrows(): 
                            folium.GeoJson(row.geometry,style_function=lambda x: {
                                            'color': self.Class2Color[Class],
                                            'weight': row['width_speed']}).add_to(layer_group)                    
                            folium.GeoJson(row.geometry,style_function=lambda x: {
                                            'color': self.Class2Color[Class],
                                            'weight': row['width_time']}).add_to(layer_group1)                    

                        # Add the feature group to the map
                        layer_group.add_to(m)
                        layer_group.add_to(m1)

                    # Add layer control to the map
                    folium.LayerControl().add_to(m)
                    folium.LayerControl().add_to(m1)

                # Save or display the map
                m.save(os.path.join(self.PlotDir,"AvSpeed_{}.html".format(self.StrDate)))
                m1.save(os.path.join(self.PlotDir,"TimePercorrence_{}.html".format(self.StrDate)))
            else:
                print("AvSpeed already Plotted in HTML")

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
                # ALL TOGETHER MFD
                self.MFD,self.Fcm = ComputeMFDVariables(self.Fcm,self.MFD,self.TimeStampDate,self.dt,self.iterations)
#                if self.verbose:
#                    PrintMFDDictInfo(self.MFD,StartingString = "MFD: ")            
                # PER CLASS
                self.Class2MFD = {class_:Dict2PolarsDF({"time":[],"population":[],"speed_kmh":[],"av_speed":[]},schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed_kmh":pl.Float64,"av_speed":pl.Float64}) for class_ in self.IntClass2StrClass.keys()}
                self.Class2MFDNew = {class_:Dict2PolarsDF({"time":[],"population":[],"speed_kmh":[],"av_speed":[]},schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed_kmh":pl.Float64,"av_speed":pl.Float64}) for class_ in self.IntClass2StrClass.keys()}
                if self.verbose:
                    print(self.Class2MFD.keys())
                for Class in self.Class2MFD.keys():
                    Fcm2Class = self.Fcm.filter(pl.col("class") == Class)
                    self.Class2MFD[Class],Fcm2Class = ComputeMFDVariables(Fcm2Class,self.Class2MFD[Class],self.TimeStampDate,self.dt,self.iterations)
                    FcmNew2Class = self.Fcm.filter(pl.col("class_new") == Class)
                    self.Class2MFDNew[Class],FcmNew2Class = ComputeMFDVariables(FcmNew2Class,self.Class2MFDNew[Class],self.TimeStampDate,self.dt,self.iterations)
                    if self.verbose:
                        PrintMFDDictInfo(self.Class2MFD[Class],StartingString = "Class 2 MFD: ")
                        PrintMFDDictInfo(self.Class2MFDNew[Class],StartingString = "Class 2 MFD New: ")
                self.ComputedMFD = True    
            else:
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
            # AGGREGATED 
            self.MFD2Plot, self.MinMaxPlot,RelativeChange = GetMFDForPlot(MFD = self.MFD,
                                                                        MFD2Plot = self.MFD2Plot,
                                                                        MinMaxPlot = self.MinMaxPlot,
                                                                        Class = None,
                                                                        case = "no-classes",
                                                                        verbose = self.verbose,
                                                                        bins_ = 20)
            
            PlotHysteresis(MFD = self.MFD,
                           Title = "Hysteresis Cycle Aggregated",
                           SaveDir = self.PlotDir,
                           NameFile = "Hysteresys.png")
            if self.verbose:
                print("After GetMFDForPlot:\n")
#                print("\nMFD2Plot:\n",self.MFD2Plot)
#                print("\nMinMaxPlot:\n",self.MinMaxPlot)
            # Plotting and Save Aggregated
            SaveMFDPlot(self.MFD2Plot["bins_population"],
                        self.MFD2Plot["binned_av_speed"],
                        self.MFD2Plot["binned_sqrt_err_speed"],
                        RelativeChange,
                        self.PlotDir,
                        NameFile = "MFD.png")            
            # PER CLASS 
            self.MinMaxPlotPerClass = {Class: defaultdict() for Class in self.Class2MFD.keys()}
            self.MinMaxPlotPerClassNew = {Class: defaultdict() for Class in self.Class2MFDNew.keys()}       
            self.Class2MFD2Plot = {Class:{"binned_av_speed": [], "binned_sqrt_err_speed": [], "bins_population": []} for Class in self.Class2MFD.keys()}
            self.Class2MFDNew2Plot = {Class:{"binned_av_speed": [], "binned_sqrt_err_speed": [], "bins_population": []} for Class in self.Class2MFD.keys()}
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
                
                PlotHysteresis(MFD = self.Class2MFD[Class],
                            Title = "Hysteresis Cycle Class {}".format(self.IntClass2StrClass[Class]),
                            SaveDir = self.PlotDir,
                            NameFile = "HysteresysClass_{}.png".format(self.IntClass2StrClass[Class]))
                
                PlotHysteresis(MFD = self.Class2MFDNew[Class],
                            Title = "Hysteresis Cycle Class New {}".format(self.IntClass2StrClass[Class]),
                            SaveDir = self.PlotDir,
                            NameFile = "HysteresysClassNew_{}.png".format(self.IntClass2StrClass[Class]))
                
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
                else:
                    print("Warning: Fondamental Diagram Not Computed for Class Since IntClass2Str is Not Initialized")
##--------------- Dictionaries --------------##
    def CreateDictionaryIntClass2StrClass(self):
        '''
        Input:
            fcm: dataframe []

        Output: dict: {velocity:'velocity class in words: (slowest,...quickest)]}
        '''
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
            
## DISTRIBUTIONS
    def PlotDailySpeedDistr(self,LableSave = "Aggregated"):
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
        self.DictFittedData = {IntClass: {} for IntClass in self.IntClass2StrClass.keys()}
        self.InfoFittedParameters = {IntClass: {} for IntClass in self.IntClass2StrClass.keys()}
        self.InfoFittedParameters,self.DictFittedData = ReturnFitInfoFromDict(Fcm = self.Fcm,
                                                                            InitialGuess = self.DictInitialGuess,
                                                                            DictFittedData = self.DictFittedData,
                                                                            Feature2Label = self.Feature2Label,
                                                                            FitFile = os.path.join(self.PlotDir,'Fit_Aggregated.json'),
                                                                            FittedDataFile = os.path.join(self.PlotDir,'FittedData_Aggregated.json'))
        for Feature in self.Feature2Label.keys():
            # Plot each feature separately
            fig,ax = plt.subplots(1,1,figsize= (15,12))
            legend = []
            aggregation = False
            n,bins = np.histogram(self.Fcm[Feature],bins = 50)
            maxCount = 0
            maxSpeed = bins[-1]
            self.Feature2MaxBins[Feature]["bins"] = maxSpeed
            self.Feature2MaxBins[Feature]["count"] = maxCount
            y,x = np.histogram(self.Fcm[Feature].to_list(),bins = 50)
            if max(y)>maxCount:
                maxCount = max(y)
                # Data
            ax.scatter(x[1:],y)
            # Fit
            ax.plot(x[1:],np.array(self.DictFittedData["fitted_data"]),label = self.DictFittedData[IntClass]["best_fit"])
            av_speed = np.mean(self.Fcm[Feature].to_list())
            legend.append(str(self.IntClass2StrClass[IntClass]) + " " + self.Column2Legend[Feature] + " " + str(round(av_speed,3)))
            ax.set_xticks(np.arange(bins[0],bins[-1],self.Feature2IntervalBin[Feature]))
            ax.set_yticks(np.arange(min(n),max(n),self.Feature2IntervalCount[Feature]))
            ax.set_xlabel(self.Feature2Label[Feature])
            ax.set_ylabel('Count')
            ax.set_xlim(right = maxSpeed + self.Feature2ShiftBin[Feature])
            ax.set_ylim(top = maxCount + self.Feature2ShiftCount[Feature])
            ax.set_xscale(self.Feature2ScaleBins[Feature])
            ax.set_yscale(self.Feature2ScaleCount[Feature])
            legend_ = plt.legend(legend)
            frame = legend_.get_frame()
            frame.set_facecolor('white')
            plt.savefig(os.path.join(self.PlotDir,'{0}_{1}.png'.format(LableSave,self.Column2SaveName[Feature])),dpi = 200)
            plt.close()



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
        self.Class2DictFittedData = {IntClass: {Feature: {"best_fit":[], "fitted_data":[]} for Feature in list(self.Features2Fit)} for IntClass in self.IntClass2StrClass.keys()}
        self.Class2InfoFittedParameters = {IntClass: {Function2Fit: {Feature:{"fit":None,"StdError":None} for Feature in self.DictInitialGuess[Function2Fit].keys()} for Function2Fit in self.DictInitialGuess.keys()} for IntClass in self.IntClass2StrClass.keys()}
        for IntClass in self.IntClass2StrClass:
            self.Class2InfoFittedParameters[IntClass],self.Class2DictFittedData[IntClass] = ReturnFitInfoFromDict(Fcm = self.Fcm.filter(pl.col("class") == IntClass),
                                                                                                                InitialGuess = self.Class2InitialGuess[IntClass],
                                                                                                                DictFittedData = self.Class2DictFittedData[IntClass],
                                                                                                                Feature2Label = self.Feature2Label,
                                                                                                                FitFile = os.path.join(self.PlotDir,'Fit_Class_{0}.json'.format(IntClass)),
                                                                                                                FittedDataFile = os.path.join(self.PlotDir,'FittedData_Class_{0}.json'.format(IntClass)))

        for Feature in self.Feature2Label.keys():
            for IntClass in self.IntClass2StrClass:
                fig,ax = plt.subplots(1,1,figsize= (15,12))
                df = self.Fcm.filter(pl.col("class") == IntClass)
                y,x = np.histogram(df[Feature].to_list(),bins = 50)
                self.Class2DictFittedData[IntClass],self.Class2InfoFittedParameters[IntClass] = FitAndPlot(x[1:],y,self.DictInitialGuess,Feature,self.Class2DictFittedData[IntClass],self.Class2InfoFittedParameters[IntClass])                
                if IntClass!=10 and IntClass!=11:
                    if "speed" in Feature:
                        ax.scatter(x[1:],y)
                        # Fit
                        ax.plot(x[1:],np.array(self.Class2DictFittedData[IntClass][Feature]["fitted_data"]),label = self.Class2DictFittedData[IntClass][Feature]["best_fit"])
                    av_speed = np.mean(df[Feature].to_list())
                    ax.set_xticks(np.arange(x[0],x[-1],self.Feature2IntervalBin[Feature]))
                    ax.set_yticks(np.arange(min(y),max(y),self.Feature2IntervalCount[Feature]))
                    ax.set_xlabel(self.Feature2Label[Feature])
                    ax.set_ylabel('Count')
                    ax.set_xlim(right = max(x) + self.Feature2ShiftBin[Feature])
                    ax.set_ylim(top = max(y) + self.Feature2ShiftCount[Feature])
                    ax.set_xscale(self.Feature2ScaleBins[Feature])
                    ax.set_yscale(self.Feature2ScaleCount[Feature])
                    plt.savefig(os.path.join(self.PlotDir,'{0}_Class_{1}_{2}.png'.format(BestLabel,IntClass,self.Column2SaveName[Feature])),dpi = 200)
                    plt.close()
        

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
        print("Read GeoJson: ",self.ReadGeoJsonBool)
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
            layer_group = folium.FeatureGroup(name=f"Layer {Class_}").add_to(m)
            
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