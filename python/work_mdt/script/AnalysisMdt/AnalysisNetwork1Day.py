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
if os.path.isfile(os.path.join(os.environ["WORKSPACE"],"city-pro","custom_style.mplstyle")):
    plt.style.use(os.path.join(os.environ["WORKSPACE"],"city-pro","custom_style.mplstyle"))
else:
    try:
        import PlotSettings
    except Exception as e:
        print("No Plot Settings File Found")

def Dict2PolarsDF(Dict,schema):
    return pl.DataFrame(Dict,schema=schema)

def ComputeMFDVariables(Df,DictMFD,TimeStampDate,dt,iterations,verbose):
    """
        NOTE: The bins in time that have 0 trajectories have 0 average speed
    """
    TmpDict = {"time":[],"population":[],"speed":[]}
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
            AvSpeed = TmpFcm.select(pl.col("av_speed").mean()).to_pandas().iloc[0]["av_speed"]
            TmpDict["speed"].append(AvSpeed)
            MoreThan0Traj = True
        else:
            TmpDict["speed"].append(0)
            MoreThan0Traj = False
        if verbose:
            print("Iteration: ",t)
            print("Considered Hour: ",Hstr)
            print("Population: ",len(TmpFcm))
            print("Size dict: ",len(TmpDict["time"]))
            if MoreThan0Traj:
                print("Speed: ",AvSpeed)
    if verbose:
        print("Dict: ",TmpDict)
    DictMFD = Dict2PolarsDF(TmpDict,schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed":pl.Float64})
    return DictMFD,Df


def NormalizeWidthForPlot(arr,min_width = 1, max_width = 10):
    '''
        Description:
            Normalizes the width for road fluxes
    '''
    if not isinstance(arr,np.ndarray):
        arr = np.array(arr)
    else:
        pass
    min_val = arr.min()
    max_val = arr.max()
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
    return v/3.6
def kmh2ms(v):
    return v*3.6
def s2h(t):
    return t/3600
def h2s(t):
    return t*3600
def m2km(x):
    return m/1000
def km2m(x):
    return m*1000
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
        self.colors = ['red','blue','green','orange','purple','yellow','cyan','magenta','lime','pink','teal','lavender','brown','beige','maroon','mint','coral','navy','olive','grey']
#        self.Name = BaseName
        self.StrDate = StrDate
        # CLASSES INFO
        self.Class2Color = {"1 slowest": "blue","2 slowest":"green","middle velocity class": "yellow","2 quickest": "orange", "1 quickest":"red"}
        self.IntClass2StrClass = defaultdict(dict) # {0,slowest,...}
        self.StrClass2IntClass = defaultdict(dict) # {slowest: 0,...}
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
                    "length":["powerlaw","exponential"],
                    "av_speed":["gaussian","maxwellian"],
                    "av_accel":["gaussian","maxwellian"]
                    }
        # FIT OUTPUT
        self.InitialGuessPerLabel = defaultdict(dict)
        self.InitialGuessPerClassAndLabel = defaultdict(dict) 
        # FUNDAMENTAL DIAGRAM
        self.MFD = Dict2PolarsDF({"time":[],"population":[],"speed":[]},schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed":pl.Float64})
        self.MFD2Plot = defaultdict()
        self.Class2MFD = {class_:Dict2PolarsDF({"time":[],"population":[],"speed":[]},schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed":pl.Float64}) for class_ in self.IntClass2StrClass.keys()}
        # MINIMUM VALUES FOR (velocity,population,length,time) for trajectories of the day
        self.MinMaxPlot = defaultdict()
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
                    if verbose:
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
            print(self.GeoJsonFile)
        if not os.path.isfile(self.GeoJsonFile):
            exit("GeoJsonFile not found")
        self.GeoJson = gpd.read_file(self.GeoJsonFile)
        self.ReadGeoJsonBool = True

    def GetIncreasinglyIncludedSubnets(self):
        self.DictSubnetsTxtDir = defaultdict(dict)
        for Class in self.IntClass2StrClass.keys():
            self.DictSubnetsTxtDir[Class] = os.path.join(self.InputBaseDir,self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + '{}_class_subnet.txt'.format(Class))
        self.ReadFluxesSubIncreasinglyIncludedIntersection()
        if self.verbose:
            print("Get increasingly included subnets")
            for class_ in self.DictSubnetsTxtDir:
                print(self.DictSubnetsTxtDir[class_])
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
        # Read Fluxes.sub
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
        for Class in self.IntClass2StrClass.keys():
            self.RoadInClass2VelocityDir[Class] = os.path.join(os.path.join(self.InputBaseDir,self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + '_class_{}velocity_subnet.csv'.format(Class)))
            self.VelTimePercorrenceClass[Class] = pd.read_csv(self.RoadInClass2VelocityDir[Class],delimiter = ';')
            self.VelTimePercorrenceClass[Class] = pl.from_pandas(self.VelTimePercorrenceClass[Class])
        self.ReadVelocitySubnetBool = True

    def AddFcmNew2Fcm(self):
        if self.ReadFcmBool and self.ReadFcmNewBool:
            FcmNew = self.FcmNew.with_columns([self.FcmNew['class'].alias('class_new')])
            self.Fcm = self.Fcm.join(FcmNew[['id_act', 'class_new']], on='id_act', how='left')
            print("1st join Fcm: ",self.Fcm.head())
            self.Fcm =self.Fcm.with_columns([self.Fcm['class'].alias('class_new')])
            print("renamed: ",self.Fcm.head())
        if self.ReadStatsBool and self.ReadFcmNewBool:
            FcmNew = self.FcmNew.with_columns([self.FcmNew['class'].alias('class_new')])
            self.Stats["class_new"] = self.Stats.join(self.FcmNew[['id_act', 'class_new']], on='id_act', how='left')
            print("1st join Stats: ",self.Stats.head())
            self.Stats =self.Stats.with_columns([self.Stats['class'].alias('class_new')])
            print("renamed: ",self.Stats.head())
##--------------- Plot Network --------------## 
    def PlotIncrementSubnetHTML(self):
        """
            NOTE: Informations about the subnet are taken from Subnet Files
            Description:
                Plots the subnetwork. Considers the case of intersections
        """
        if self.ReadFluxesSubIncreasinglyIncludedIntersectionBool and self.ReadGeoJsonBool:
            print("Plotting Daily Incremental Subnetworks in HTML")
            print("Save in: ",os.path.join(self.PlotDir,"SubnetsIncrementalInclusion_{}.html".format(self.StrDate)))
            # Create a base map
            m = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
            # Iterate through the Dictionary of list of poly_lid
            if self.verbose:
                for class_ in self.IntClass2RoadsIncreasinglyIncludedIntersection.keys(): 
                    print(self.IntClass2RoadsIncreasinglyIncludedIntersection[class_][:10])
            for class_, index_list in self.IntClass2RoadsIncreasinglyIncludedIntersection.items():
                print("Plotting Class ",class_)
                print("Number of Roads: ",len(index_list))
                # Filter GeoDataFrame for roads with indices in the current list
                filtered_gdf = self.GeoJson[self.GeoJson['poly_lid'].isin(index_list)]
                # Create a feature group for the current layer
                layer_group = folium.FeatureGroup(name="Layer {}".format(class_))
                # Add roads to the feature group with a unique color
                print("Class: ",class_," Number of Roads: ",len(filtered_gdf),"Color: ",self.Class2Color[self.IntClass2StrClass[class_]])
                for _, road in filtered_gdf.iterrows():
                    folium.GeoJson(road.geometry, style_function=lambda x: {'color': self.Class2Color[self.IntClass2StrClass[class_]]}).add_to(layer_group)
                
                # Add the feature group to the map
                layer_group.add_to(m)
                # Add layer control to the map
                folium.LayerControl().add_to(m)

            # Save or display the map
            m.save(os.path.join(self.PlotDir,"SubnetsIncrementalInclusion_{}.html".format(self.StrDate)))

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
        if self.ReadFluxesSubBool and self.ReadGeoJsonBool:
            print("Plotting Daily Subnetworks in HTML")
            # Create a base map
            m = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
            # Iterate through the Dictionary of list of poly_lid
            for class_, index_list in self.IntClass2Roads.items():
                # Filter GeoDataFrame for roads with indices in the current list
                filtered_gdf = self.GeoJson[self.GeoJson['poly_lid'].isin(index_list)]
                # Create a feature group for the current layer
                layer_group = folium.FeatureGroup(name=f"Layer {class_}")
                
                # Add roads to the feature group with a unique color
                for _, road in filtered_gdf.iterrows():
                    color = 'blue'  # Choose a color for the road based on index or any other criterion
                    folium.GeoJson(road.geometry, style_function=lambda x: {'color': self.Class2Color[self.IntClass2StrClass[class_]]}).add_to(layer_group)
                
                # Add the feature group to the map
                layer_group.add_to(m)

                # Add layer control to the map
                folium.LayerControl().add_to(m)

            # Save or display the map
            m.save(os.path.join(self.PlotDir,"Subnets_{}.html".format(self.StrDate)))

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
            # Create a base map
            m = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
            mFT = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
            mTF = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
            TF = self.TimedFluxes.copy()
            TF["width_total_fluxes"] = NormalizeWidthForPlot(TF["total_fluxes"])
            TF["width_n_traj_FT"] = NormalizeWidthForPlot(TF["n_traj_FT"])
            TF["width_n_traj_TF"] = NormalizeWidthForPlot(TF["n_traj_TF"])
            CopyGdf = self.GeoJson.copy()
            CopyGdf["width_total_fluxes"] = TF["width_total_fluxes"]
            CopyGdf["width_n_traj_FT"] = TF["width_n_traj_FT"]
            CopyGdf["width_n_traj_TF"] = TF["width_n_traj_TF"]
            # Iterate through the Dictionary of list of poly_lid
            for t,tdf in TF.groupby("time"):
                # Filter GeoDataFrame for roads with indices in the current list
                # Create a feature group for the current layer
                layer_group = folium.FeatureGroup(name=f"Layer {t}")
                layer_groupFT = folium.FeatureGroup(name=f"Layer {t}")
                layer_groupTF = folium.FeatureGroup(name=f"Layer {t}")
                # Add roads to the feature group with a unique color
                color = 'blue'  # Choose a color for the road based on index or any other criterion
                for idx, row in filtered_gdf.iterrows(): 
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

    def PlotTimePercorrenceHTML(self):
        if self.ReadGeojsonBool and self.ReadVelocitySubnetBool:
            print("Plotting Daily Fluxes in HTML")
            # Create a base map
            m = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
            m1 = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
            for time,RTV in RoadsTimeVel.groupby("start_bin"):
                layer_group = folium.FeatureGroup(name=f"Layer {time}")
                layer_group1 = folium.FeatureGroup(name=f"Layer {time}")
                for Class in self.IntClass2BestFit.keys():
                    RoadsTimeVel = self.VelTimePercorrenceClass[Class].copy()
                    RoadsTimeVel["av_speed"] = [x if x!=-1 else 0 for x in RoadsTimeVel["av_speed"]]
                    RoadsTimeVel["time_percorrence"] = [x if x!=-1 else 0 for x in RoadsTimeVel["time_percorrence"]]
                    RoadsTimeVel["width_speed"] = NormalizeWidthForPlot(RoadsTimeVel["av_speed"])
                    RoadsTimeVel["width_time"] = NormalizeWidthForPlot(RoadsTimeVel["time_percorrence"])                
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

## ------- FUNDAMENTAL DIAGRAM ------ ##
    def ComputeMFDVariables(self):
        '''
            Description:
                Computes the MFD variables (t,population,speed) -> and the hysteresis diagram:
                    1) Aggregated data for the day
                    2) Conditional to class
            Save them in two dictionaries 
                1) self.MFD = {time:[],population:[],speed:[]}
                2) self.Class2MFD = {Class:{"time":[],"population":[],"speed":[]}}
        '''
        if self.ReadFcmBool:
            if self.verbose:
                print("Computing MFD Variables from Fcm")

            if "start_time" in self.Fcm.columns:
                # ALL TOGETHER MFD
                self.MFD,self.Fcm = ComputeMFDVariables(self.Fcm,self.MFD,self.TimeStampDate,self.dt,self.iterations,self.verbose)
                # PER CLASS
                for Class in self.Class2MFD.keys():
                    self.Class2MFD[Class],self.Fcm = ComputeMFDVariables(self.Fcm,self.Class2MFD[Class],self.TimeStampDate,self.dt,self.iterations,self.verbose)
                if self.verbose:
                    print("Class 2 MFD: ")
                    print("Keys: ",self.Class2MFD.keys())
                    print("Lenght of values: ")
                    for key in self.Class2MFD.keys():
                        print("Values: ",len(self.Class2MFD[key]))
                self.ComputedMFD = True    
            else:
                pass
                
        if self.ReadStatsBool:
            if self.verbose:
                print("Computing MFD Variables from Stats")
                # ALL TOGETHER MFD
                self.MFD,self.Stats = ComputeMFDVariables(self.Stats,self.MFD,self.TimeStampDate,self.dt,self.iterations,self.verbose)
                # PER CLASS
                for Class in self.Class2MFD.keys():
                    self.Class2MFD[Class],self.Stats = ComputeMFDVariables(self.Stats,self.Class2MFD[Class],self.TimeStampDate,self.dt,self.iterations,self.verbose)

                if self.verbose:
                    print("Class 2 MFD: ")
                    print("Keys: ",self.Class2MFD.keys())
                    print("Lenght of values: ")
                    for key in self.Class2MFD.keys():
                        print("Values: ",len(self.Class2MFD[key]))

                self.ComputedMFD = True

    def GetLowerBoundsFromBins(self,bins,label):
        if len(self.MinMaxPlot.keys())==0:
            self.MinMaxPlot["aggregated"] = defaultdict()
        else:
            self.MinMaxPlot["aggregated"][label] = {"min":bins[0],"max":bins[-1]}  

    def GetLowerBoundsFromBinsPerClass(self,Class,bins,label):
        if len(self.MinMaxPlotPerClass[Class].keys())==0:
            self.MinMaxPlotPerClass[Class] = defaultdict()
        else:
            self.MinMaxPlotPerClass[Class][label] = {"min":bins[0],"max":bins[-1]}  

    def PlotMFD(self):
        """
        Plots the Fundamental Diagram for the calculated MFD (Mobility Fundamental Diagram) and per class.
        
        This function plots the Fundamental Diagram for the calculated MFD (Mobility Fundamental Diagram) and per class. 
        The Fundamental Diagram shows the average speed and the standard deviation of the speed for each population bin.
        The population bins are determined by the number of vehicles in each bin.
        
        Parameters:
            self (object): The instance of the class.
        
        Returns:
            None
        
        Raises:
            None
        """
        if self.ComputedMFD: 
            # AGGREGATED
            fig, ax = plt.subplots(1,1,figsize = (10,8))
            n, bins = np.histogram(self.MFD["population"],bins = 20)
            labels = range(len(bins) - 1)
            self.MFD["bins_population"] = pd.cut(self.MFD["population"],bins=bins,labels=labels,right=False)
            self.MFD2Plot['binned_av_speed'] = self.MFD.groupby('bins_population', observed=False)['speed'].mean().reset_index()
            self.MFD2Plot['binned_av_speed'] = self.MFD2Plot['binned_av_speed'][::-1].interpolate(method='pad')[::-1]
            self.MFD2Plot['binned_sqrt_err_speed'] = self.MFD.groupby('bins_population', observed=False)['speed'].std().reset_index()    
            if self.verbose:
                print("MFD Features Aggregated: ")
                print("Bins Population:\n",self.MFD['bins_population'])
                print("Bins Average Speed:\n",self.MFD2Plot['binned_av_speed'])
                print("Bins Standard Deviation:\n",self.MFD2Plot['binned_sqrt_err_speed'])
            self.GetLowerBoundsFromBins(bins,"population")
            self.GetLowerBoundsFromBins(self.MFD2Plot['binned_av_speed'],"speed")
            Y_Interval = max(self.MFD2Plot['binned_av_speed']) - min(self.MFD2Plot['binned_av_speed'])
            RelativeChange = Y_Interval/max(self.MFD2Plot['binned_av_speed'])/100
            if self.verbose:
                print("min(bins): ",self.MinMaxPlot["aggregated"]["population"]["min"]," max(bins): ",self.MinMaxPlot["aggregated"]["population"]["max"])                
                print("Interval Error: ",Y_Interval)            
            text = "Relative change : {}%".format(RelativeChange)
            ax.plot(bins,self.MFD2Plot['binned_av_speed'])
            ax.fill_between(bins, self.MFD2Plot['binned_av_speed'] - self.MFD2Plot['binned_sqrt_err_speed'], self.MFD2Plot['binned_av_speed'] + self.MFD2Plot['binned_sqrt_err_speed'], color='gray', alpha=0.2, label='Std')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            ax.set_title("Fondamental Diagram")
            ax.set_xlabel("time")
            ax.set_ylabel("speed")
            plt.savefig(os.path.join(self.SaveDir,"MFD.png"),dpi = 200)

            # PER CLASS
            if self.Class2MFD.keys():
                self.MinMaxPlotPerClass = {Class: defaultdict() for Class in self.Class2MFD.keys()}
            else:
                self.MinMaxPlotPerClass = defaultdict(dict)
            for Class in self.Class2MFD.keys():
                fig, ax = plt.subplots(1,1,figsize = (10,8))
                n, bins = np.histogram(self.Class2MFD[Class]["population"],bins = 20)
                self.Class2MFD[Class]["bins_population"] = pd.cut(self.Class2MFD[Class]["population"],bins,labels=bins,right=False)
                self.Class2MFD2Plot[Class]['binned_av_speed'] = self.MFD.groupby('bins_population', observed=True)['speed'].mean().reset_index()
                self.Class2MFD2Plot[Class]['binned_av_speed'] = self.Class2MFD2Plot[Class]['binned_av_speed'].fillna(method='bfill')
                self.Class2MFD2Plot[Class]['binned_sqrt_err_speed'] = self.MFD.groupby('bins_population', observed=True)['speed'].std().reset_index()                
                if self.verbose:
                    print("MFD Features Class: ",Class)
                    print("Bins Average Speed:\n",self.Class2MFD2Plot[Class]['binned_av_speed'])
                    print("Bins Standard Deviation:\n",self.Class2MFD2Plot[Class]['binned_sqrt_err_speed'])
                self.GetLowerBoundsFromBinsPerClass(Class,bins,"population")
                self.GetLowerBoundsFromBinsPerClass(Class,self.Class2MFD2Plot[Class]['binned_av_speed'],"speed")
                Y_Interval = max(self.Class2MFD2Plot[Class]['binned_av_speed']) - min(self.Class2MFD2Plot[Class]['binned_av_speed'])
                RelativeChange = Y_Interval/max(self.Class2MFD2Plot[Class]['binned_av_speed'])/100
                if self.verbose:
                    print("min(bins): ",self.MinMaxPlotPerClass[Class]["min"]," max(bins): ",self.MinMaxPlotPerClass[Class]["max"])                
                    print("Interval Error: ",Y_Interval)
                text = "Relative change : {}%".format(RelativeChange)
                ax.plot(bins,self.Class2MFD[Class]['binned_av_speed'])
                ax.fill_between(bins, self.Class2MFD2Plot[Class]['binned_av_speed'] - self.Class2MFD2Plot[Class]['binned_sqrt_err_speed'], self.Class2MFD2Plot[Class]['binned_av_speed'] + self.Class2MFD2Plot[Class]['binned_sqrt_err_speed'], color='gray', alpha=0.2, label='Std')
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
                ax.set_title(self.IntClass2StrClass[Class])
                ax.set_xlabel("time")
                ax.set_ylabel("speed")
                plt.savefig(os.path.join(self.SaveDir,"MFD_{}.png".format(Class)),dpi = 200)
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
                if i<number_classes/2:
                    self.IntClass2StrClass[list(self.FcmCenters["class"])[i]] = '{} slowest'.format(i+1)
                    self.StrClass2IntClass['{} slowest'.format(i+1)] = list(self.FcmCenters["class"])[i]
                elif i<number_classes==2:
                    self.IntClass2StrClass[list(self.FcmCenters["class"])[i]] = 'middle velocity class'
                    self.StrClass2IntClass['middle velocity class'] = list(self.FcmCenters["class"])[i]
                else:
                    self.IntClass2StrClass[list(self.FcmCenters["class"])[i]] = '{} quickest'.format(number_classes - i)             
                    self.StrClass2IntClass['{} quickest'.format(number_classes - i)] = list(self.FcmCenters["class"])[i]
        self.BoolStrClass2IntClass = True
    def CreateDictConstraintsClass(self):
        """
            Output: dict: {'velocity class in words': {function: {length: {xmin,xmax},time: {tmin,tmax}}}}
        """
        if self.BoolStrClass2IntClass:
            for Strclass_ in self.StrClass2IntClass.keys():
                self.DictConstraintClassLabel[Strclass_] = defaultdict(dict)
                for label in self.labels2FitNames2Try.keys():
                    self.DictConstraintClassLabel[Strclass_][label] = defaultdict(dict)
                    if label == "time":
                            self.DictConstraintClassLabel[Strclass_]["time"] = {"tmin":6000,"tmax":10000}
                    elif label == "length":
                        if Strclass_ == "1 slowest":
                            self.DictConstraintClassLabel[Strclass_][label]["length"] = {"xmin":10,"xmax":500}
                        elif Strclass_ == "2 slowest":
                            self.DictConstraintClassLabel[Strclass_][label]["length"] = {"xmin":500,"xmax":5000}
                        elif Strclass_ == "2 quickest":
                            self.DictConstraintClassLabel[Strclass_][label]["length"] = {"xmin":2000,"xmax":9000}
                        elif Strclass_ == "1 quickest":
                            self.DictConstraintClassLabel[Strclass_][label]["length"] = {"xmin":1000,"xmax":25000}
   
   
    def CreateDictConstraintsAll(self):
        pass



# --------------- FITTING  ---------------- #   
# --------------- ALAGGREGATED CLASSES  ---------------- #
    def RetrieveGuessParametersPerLabel(self):
        """
            Output: dict: {'velocity class in words': {function: {length: {xmin,xmax},time: {tmin,tmax}}}}
        """
        if self.BoolStrClass2IntClass:
            for Strclass_ in self.StrClass2IntClass.keys():
                self.DictConstraintClassLabel[Strclass_] = defaultdict(dict)
                for label in self.labels2FitNames2Try.keys():
                    self.DictConstraintClassLabel[Strclass_][label] = defaultdict(dict)
                    if label == "time":
                            self.DictConstraintClassLabel[Strclass_]["time"] = {"tmin":6000,"tmax":10000}
                    elif label == "length":
                        if Strclass_ == "1 slowest":
                            self.DictConstraintClassLabel[Strclass_][label]["length"] = {"xmin":10,"xmax":500}
                        elif Strclass_ == "2 slowest":
                            self.DictConstraintClassLabel[Strclass_][label]["length"] = {"xmin":500,"xmax":5000}
                        elif Strclass_ == "2 quickest":
                            self.DictConstraintClassLabel[Strclass_][label]["length"] = {"xmin":2000,"xmax":9000}
                        elif Strclass_ == "1 quickest":
                            self.DictConstraintClassLabel[Strclass_][label]["length"] = {"xmin":1000,"xmax":25000}
    def FittingFunctionAllClasses(self,label,FunctionName,initial_guess,bins = 100):
        
        n, bins = np.histogram(self.Fcm[label],bins = bins)
        fit = Fitting(bins[1:],n,label = FunctionName,initial_guess = (6000,0.3),maxfev = 10000)
        ErrorFittedCurve = L2Error2ParamsFunctions(Bins[1:],n,FunctionName)
        self.Function2FitInfo[FunctionName] = {"A":fit[0][0],"b":fit[0][1],"Error":ErrorFittedCurve} 
        self.Fitting = True
    def GetBestFitsAndParameters(self):
        """
            Description:
                Compares all the fitting procedures and selects the one with the lowest L2 error.
        """
        if self.Fitting:
            Min_Error = 100000
            for FunctionName in self.Function2FitInfo.keys():      
                if self.Function2FitInfo[FunctionName]["Error"] is not None:
                    if self.Function2FitInfo[FunctionName]["Error"] < Min_Error:
                        Min_Error = self.Function2FitInfo[FunctionName]["Error"]
                        self.FunctionName = FunctionName
                        self.A = self.Function2FitInfo[FunctionName]["A"]
                        self.b = self.Function2FitInfo[FunctionName]["b"]
                    else:
                        pass
                else:
                    pass            

# --------------- PER EACH CLASS  ---------------- #
    def RetrieveGuessParametersPerClassLabel(self):
        """
            TODO
            Description:
                Retrieves the initial guess parameters for each
                label: (time,length,av_speed) 
                    Class: (0,1,2,3)
                        corresponnding functions to try.
            Return:
                InitialGuessPerClassAndLabel: dict -> {class:{label:{function:(A,b)}}}
        """
        if self.ReadFcmCentersBool:
            for class_ in self.FcmCenters["class"]:
                self.InitialGuessPerClassAndLabel[class_] = defaultdict(dict)
                for label in self.labels2FitNames2Try.keys():
                    self.InitialGuessPerClassAndLabel[class_][label] = defaultdict(dict)
                    for function in self.labels2FitNames2Try[label]:
                        if "powerlaw" in function:
                            self.InitialGuessPerClassAndLabel[class_][label][function] = (6000,-1)
                        elif "exponential" in function:
                            self.InitialGuessPerClassAndLabel[class_][label][function] = (6000,np.mean(self.Fcm.groupby("class").get_group(class_)[label]))
                        elif "gaussian" in function:
                            self.InitialGuessPerClassAndLabel[class_][label][function] = (6000,self.Fcm.groupby("class").get_group(class_)[label])
                        else:
                            self.InitialGuessPerClassAndLabel[class_][label][function] = (6000,self.Fcm.groupby("class").get_group(class_)[label])
        else:
            print("FcmCenters not read Not retrieving parameters")

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

def GetDistributionPerClass(fcm,label,class_):
    """
        Input:
            label: str -> time, length, av_speed, p, a_max
        Returns:
            n, bins of velocity distribution
    """
    n, bins = np.histogram(fcm.groupby("class").get_group(label), bins = bins)



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
            layer_group = folium.FeatureGroup(name=f"Layer {Class_}")
            
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



