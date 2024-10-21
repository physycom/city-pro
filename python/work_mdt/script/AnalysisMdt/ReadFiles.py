import polars as pl
import logging
from CastVariables import *
from collections import defaultdict
import geopandas as gpd
import pandas as pd
import os

logger = logging.getLogger(__name__)

def ReadTimedFluxes(DirTimedFluxes):
    """
        @brief: Read the TimedFluxes
        @param DirTimedFluxes: Path to the TimedFluxes
    """
    logger.info("Read TimedFluxes from %s",DirTimedFluxes)
    TimedFluxes = pd.read_csv(DirTimedFluxes,delimiter = ';')
    TimedFluxes = pl.from_pandas(TimedFluxes)
    ReadTime2FluxesBool = True
    return TimedFluxes,ReadTime2FluxesBool

def ReadFluxes(DirFluxes):
    """
        @brief: Read the Fluxes
        @param DirFluxes: Path to the Fluxes
    """
    logger.info("Read Fluxes from %s",DirFluxes)
    Fluxes = pd.read_csv(DirFluxes,delimiter = ';')
    Fluxes = pl.from_pandas(Fluxes)
    ReadFluxesBool = True
    return Fluxes,ReadFluxesBool


def ReadFcm(DirFcm):
    """
        @brief: Read the Fcm
        @param DirFcm: Path to the Fcm
    """
    logger.info("Read Fcm from %s",DirFcm)
    Fcm = pd.read_csv(DirFcm,delimiter = ';')
    Fcm = pl.from_pandas(Fcm)
    try:
        Fcm = Fcm.filter(pl.col("av_speed")<43.0)
    except:
        Fcm = pd.read_csv(DirFcm,delimiter = ',')
        Fcm = pl.from_pandas(Fcm)
        Fcm = Fcm.filter(pl.col("av_speed")<43.0)

    Fcm = Fcm.with_columns(pl.col("av_speed").apply(lambda x: ms2kmh(x), return_dtype=pl.Float64).alias("speed_kmh"))
    Fcm = Fcm.with_columns(pl.col("lenght").apply(lambda x: m2km(x), return_dtype=pl.Float64).alias("lenght_km"))
    Fcm = Fcm.with_columns(pl.col("time").apply(lambda x: s2h(x), return_dtype=pl.Float64).alias("time_hours"))
    ReadFcmBool = True
    return Fcm,ReadFcmBool

def ReadFcmNew(DirFcmNew):
    """
        @brief: Read the FcmNew
        @param DirFcmNew: Path to the FcmNew
    """
    logger.info("Read FcmNew from %s",DirFcmNew)
    FcmNew = pd.read_csv(DirFcmNew,delimiter = ';')
    FcmNew = pl.from_pandas(FcmNew)
    ReadFcmNewBool = True
    return FcmNew,ReadFcmNewBool

def ReadStats(DirStats):
    """
        @brief: Read the Stats
        @param DirStats: Path to the Stats
    """
    logger.info("Read Stats from %s",DirStats)
    Stats = pd.read_csv(DirStats,delimiter = ';')
    Stats = pl.from_pandas(Stats)
    Stats = Stats.filter(pl.col("av_speed")<43.0)
    Stats = Stats.with_columns(pl.col("av_speed").apply(lambda x: ms2kmh(x), return_dtype=pl.Float64).alias("speed_kmh"))
    Stats = Stats.with_columns(pl.col("lenght").apply(lambda x: m2km(x), return_dtype=pl.Float64).alias("lenght_km"))
    Stats = Stats.with_columns(pl.col("time").apply(lambda x: s2h(x), return_dtype=pl.Float64).alias("time_hours"))
    ReadStatsBool = True
    return Stats,ReadStatsBool


def ReadFcmCenters(DirFcmCenters):
    """
        @brief: Read the FcmCenters
        @param DirFcmCenters: Path to the FcmCenters

    """
    logger.info("Read FcmCenters from %s",DirFcmCenters)
    FcmCenters = pd.read_csv(DirFcmCenters,delimiter = ';') 
    FcmCenters = pl.from_pandas(FcmCenters)
    FlattenedFcmCenters = FcmCenters.to_numpy().flatten()    
    Row2Jump = False   
    idxcol = 0
    Features = {"class":[],"av_speed":[],"v_max":[],"v_min":[],"sinuosity":[],"people":[]}
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
            # Check that is the first element of the row
            if val%len(FcmCenters.columns) == 0: 
                Row2Jump = False
                # Check that the velocity is not too high
                if float(FlattenedFcmCenters[val + 1]) > 50:
                    Row2Jump = True
                    pass
                else:
                    if not Row2Jump:
                        keyidx = val
                        if list(Features.keys())[keyidx] == "class" or list(Features.keys())[keyidx] == "people":
                            Features[list(Features.keys())[keyidx]].append(int(FlattenedFcmCenters[val]))
                        else:
                            Features[list(Features.keys())[keyidx]].append(float(FlattenedFcmCenters[val]))
                    else:
                        pass
            else:
                if not Row2Jump:
                    keyidx = val
                    if list(Features.keys())[keyidx] == "class" or list(Features.keys())[keyidx] == "people":
                        Features[list(Features.keys())[keyidx]].append(int(FlattenedFcmCenters[val]))
                    else:
                        Features[list(Features.keys())[keyidx]].append(float(FlattenedFcmCenters[val]))
                else:
                    pass
        # Not first row
        else:
            # Check that is the first element of the row
            if val%len(FcmCenters.columns) == 0:
                # Check that the velocity is not too high
                Row2Jump = False
                if float(FlattenedFcmCenters[val + 1]) > 50:
                    Row2Jump = True
                    pass
                else:
                    if not Row2Jump:
                        keyidx = int(val%len(FcmCenters.columns))
                        if list(Features.keys())[keyidx] == "class" or list(Features.keys())[keyidx] == "people":
                            Features[list(Features.keys())[keyidx]].append(int(FlattenedFcmCenters[val]))
                        else:
                            Features[list(Features.keys())[keyidx]].append(float(FlattenedFcmCenters[val]))
                    else:
                        pass
            else:
                if not Row2Jump:
                    keyidx = val%len(FcmCenters.columns)
                    if list(Features.keys())[keyidx] == "class" or list(Features.keys())[keyidx] == "people":
                        Features[list(Features.keys())[keyidx]].append(int(FlattenedFcmCenters[val]))
                    else:
                        Features[list(Features.keys())[keyidx]].append(float(FlattenedFcmCenters[val]))
                else:
                    pass
    FcmCenters = pl.DataFrame(Features)
    ReadFcmCentersBool = True    
    return FcmCenters,ReadFcmCentersBool

def ReadFluxesSub(DirFluxesSub):
    """
        @brief: Read the FluxesSub
        @param DirFluxesSub: Path to the FluxesSub
        @return IntClass2Roads: Dictionary with the Class as key and the Roads as values
        IntClass2Roads: {Class: [Lid Roads]}
    """
    logger.info("Read FluxesSub from %s",DirFluxesSub)
    DoNothing = False
    IntClass2Roads = defaultdict(list)
    with open(DirFluxesSub,'r') as f:
        FluxesSub = f.readlines()
    for ClassLines in FluxesSub:
        ClassandID = ClassLines.split('\t')
        ClassId  = ClassandID[0].split('_')[1]
        try:
            ClassFractionRoadsConsidered = ClassandID[0].split('_')[2]
        except IndexError:
            DoNothing = True
        if DoNothing:
            pass
        else:
            IdRoads = [int(RoadId) for RoadId in ClassandID[1:] if RoadId != '\n']
            IntClass2Roads[int(ClassId)] = IdRoads
    ReadFluxesSubBool = True
    return IntClass2Roads,ReadFluxesSubBool

def GenerateDictSubnetsTxtDir(InputBaseDir,BaseFileName,StrDate,IntClass2StrClass):
    """
        @brief: Generate the dictionary with the subnets txt directory
        @param InputBaseDir: Base directory
        @param BaseFileName: Base file name
        @param StrDate: Date
        @param IntClass2StrClass: Dictionary
    """
    logger.info("Generate the dictionary with the hierarchical subnets (.txt files)")
    DictSubnetsTxtDir = defaultdict()
    for Class in IntClass2StrClass.keys():
        DictSubnetsTxtDir[Class] = InputBaseDir + "/"+ BaseFileName+'_'+ StrDate+'_'+ StrDate + '{}_class_subnet.txt'.format(Class)
    return DictSubnetsTxtDir

def ReadFluxesHierarchicallyOrdered(DictSubnetsTxtDir):
    """
        @brief: Read the hierarchical subnets
        @param DictSubnetsTxtDir: Dictionary with the subnets txt directory
        @return IntClass2RoadsIncreasinglyIncludedIntersection: Dictionary with the Class as key and the Roads as values
        IntClass2RoadsIncreasinglyIncludedIntersection: {Class: [Lid Roads]}
    """
    logger.info("Generate {IntClass: [Lid Roads Hierarchically Composed]}")
    IntClass2RoadsIncreasinglyIncludedIntersection = defaultdict(list)
    for Class in DictSubnetsTxtDir.keys():
        with open(DictSubnetsTxtDir[Class],'r') as f:
            FluxesSub = f.readlines()
        for Road in FluxesSub[0].split(" "):
            intRoad,BoolInt = CastString2Int(Road)
            if BoolInt:
                IntClass2RoadsIncreasinglyIncludedIntersection[Class].append(intRoad)
            else:
                pass
#            self.IntClass2RoadsIncreasinglyIncludedIntersection[Class] = [CastString2Int(Road) for Road in FluxesSub[0].split(" ")] 
    ReadFluxesSubIncreasinglyIncludedIntersectionBool = True
    return IntClass2RoadsIncreasinglyIncludedIntersection,ReadFluxesSubIncreasinglyIncludedIntersectionBool

def ReadGeoJson(DirGeoJson):
    """
        @brief: Read the GeoJson
        @param DirGeoJsonClassInfo: Path to the GeoJson with informations about classes
        @param DirGeoJson: Path to the GeoJson

    """
    logger.info("Read GeoJson from %s",DirGeoJson)
    GeoJson = gpd.read_file(DirGeoJson)
    ReadGeoJsonBool = True
    return GeoJson,ReadGeoJsonBool

def ReadGeoJsonClassInfo(DirGeoJsonClassInfo):
    """
        @brief: Read the GeoJsonClassInfo
        @param DirGeoJsonClassInfo: Path to the GeoJsonClassInfo
    """
    logger.info("Read GeoJsonClassInfo from %s",DirGeoJsonClassInfo)
    GeoJsonClassInfo = gpd.read_file(DirGeoJsonClassInfo)
    ReadGeoJsonClassInfoBool = True
    return GeoJsonClassInfo,ReadGeoJsonClassInfoBool

def ReadPathFile(DirPathFile):
    """
        @brief: Read the PathFile
        @param DirPathFile: Path to the PathFile
    """
    logger.info("Read PathFile from %s",DirPathFile)
    PathFile = pd.read_csv(DirPathFile,delimiter = ';')
    PathFile = pl.from_pandas(PathFile)
    ReadPathFileBool = True
    return PathFile,ReadPathFileBool


# Computed Objects In Python

def ReadTransitionClassMatrix(DirTransitionClassMatrix):
    """
        @brief: Read the TransitionClassMatrix
        @param DirTransitionClassMatrix: Path to the TransitionClassMatrix
    """
    logger.info("Read TransitionClassMatrix from %s",DirTransitionClassMatrix)
    TransitionClassMatrix = pd.read_csv(DirTransitionClassMatrix,delimiter = ';')
    TransitionClassMatrix = pl.from_pandas(TransitionClassMatrix)
    ReadTransitionClassMatrixBool = True
    return TransitionClassMatrix,ReadTransitionClassMatrixBool

def ReadFilePlotMFD(PlotDir,StrDate):
    MFD2Plot = pl.read_csv(os.path.join(PlotDir,"MFDPlot_{0}.csv".format(StrDate)))
    return MFD2Plot