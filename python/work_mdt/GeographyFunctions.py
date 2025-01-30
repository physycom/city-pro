import numpy as np
from collections import defaultdict
import logging
import polars as pl
import time
from shapely.geometry import Polygon, Point, MultiPolygon,LineString
import geopandas as gpd
from MFDAnalysis import fill_zeros_with_average

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def AddColumnsAboutClassesRoad2GeoJson(GeoJson,IntClass2RoadsIncreasinglyIncludedIntersection,IntClass2StrClass,StrDate):
    """
    @param GeoJson: GeoDataFrame
    @param IntClass2RoadsIncreasinglyIncludedIntersection: Dictionary {IntClass: [ PolyLid Roads]}
    @param IntClass2StrClass: Dictionary {IntClass: StrClass}
    @param StrDate: String (Day of Interest)
    @details: It is used for Analysis Network 1 Day
        Add for each day:
            - IntClassOrdered_YYYY-MM-DD
            - StrClassOrdered_YYYY-MM-DD
        These are the columns that contain information about the hierarchical class the road belongs to. 
    """
    logger.info("Add Columns about Classes Road to GeoJson")
    ClassOrderedForGeojsonRoads = np.zeros(len(GeoJson),dtype = int)
    # Increasingly Included Subnet
    for Class in IntClass2RoadsIncreasinglyIncludedIntersection.keys():
        for Road in IntClass2RoadsIncreasinglyIncludedIntersection[Class]: 
            # Poly 2 Class
            ClassOrderedForGeojsonRoads[np.where(GeoJson["poly_lid"] == Road)[0]] = Class 
    GeoJson["IntClassOrdered_{}".format(StrDate)] = ClassOrderedForGeojsonRoads
    GeoJson["StrClassOrdered_{}".format(StrDate)] = [IntClass2StrClass[intclass] for intclass in ClassOrderedForGeojsonRoads]

    GeoJsonWithClassBool = True
    return GeoJson,GeoJsonWithClassBool


def RestrictGeoJsonWithBbox(GeoJson,Bbox):
    """
    @param GeoJson: GeoDataFrame
    @param Bbox: List [minx,miny,maxx,maxy]
    @details: It is used for Analysis Network 1 Day
        Restrict the GeoJson with the Bbox
    """
    logger.info("Restrict GeoJson with Bbox")
    GeoJson = GeoJson.cx[Bbox[0]:Bbox[2],Bbox[1]:Bbox[3]]
    return GeoJson

def ComputeTimeInterval2UserId(OrderedClass2TimeDeparture2UserId):
    TimeInterval2UserId = defaultdict()
    for Class in OrderedClass2TimeDeparture2UserId.keys():
        for TimeDeparture in OrderedClass2TimeDeparture2UserId[Class].keys():
            if TimeDeparture not in TimeInterval2UserId.keys():
                TimeInterval2UserId[TimeDeparture] = []
            else:
                pass
                TimeInterval2UserId[TimeDeparture].extend(OrderedClass2TimeDeparture2UserId[Class][TimeDeparture])
    return TimeInterval2UserId
def ComputeSpeedRoadsPerTimeIntervalByRoadsTravelledByAllUsers(Fcm,DfPathOnRoad,OrderedClass2TimeDeparture2UserId,IntClass2RoadsIncreasinglyIncludedIntersection):
    """
    @param GeoJson: GeoDataFrame
    @param Fcm: DataFrame with the Fcm
    @param OrderedClass2TimeDeparture2UserId: Dictionary {Class: {TimeDeparture: [UserId]}}
    @param StrDate: String (Day of Interest)
    @param StrClassCol: String (Column with the hierarchical class)
    @details: It is used for Analysis Network 1 Day
        Compute the Speed for each road in the GeoJson
    return:
        TimeInterval2SpeedNew: Dictionary {TimeInterval: {Road:<Speed>_{User in FcmNew Class}}}

    NOTE:
        I am considering for each road the speed that user that have passed there have.
        So I am considering all the traffic.
        If people are passed in the road quickly (have a point there) then I explain why I see higher speeds for the roads.
    """
    Class2TimeInterval2Road2SpeedNew = defaultdict()
    logger.info("Compute Speed Subnet")
    TimeInterval2UserId = ComputeTimeInterval2UserId(OrderedClass2TimeDeparture2UserId)
    # Choose The User of Class
    for Class in OrderedClass2TimeDeparture2UserId.keys():
        RoadsPerClass = IntClass2RoadsIncreasinglyIncludedIntersection[Class]
        if Class not in Class2TimeInterval2Road2SpeedNew.keys():
            Class2TimeInterval2Road2SpeedNew[Class] = defaultdict()
        # Pick the People That are present in the interval
        for TimeDeparture in OrderedClass2TimeDeparture2UserId[Class].keys():
            if TimeDeparture not in Class2TimeInterval2Road2SpeedNew[Class].keys():
                Class2TimeInterval2Road2SpeedNew[Class][TimeDeparture] = defaultdict()
            # Choose among all users
            Users = TimeInterval2UserId[TimeDeparture]
            # Take the absolute value of the 'poly_id' column in DfPathOnRoad
            DfPathOnRoad = DfPathOnRoad.with_columns(pl.col("poly_id").abs().alias("poly_id"))
            MaskRoads = DfPathOnRoad["poly_id"].is_in(RoadsPerClass)
            RoadsInClass = DfPathOnRoad.filter(MaskRoads)            # Pick Roads Used by People in Class in Time Interval
            # Create a mask for DfPathOnRoad
            MaskPath = RoadsInClass["user_id"].is_in(Users)
            UsersSubnet = RoadsInClass.filter(MaskPath)            # Pick Roads Used by People in Class in Time Interval
            MaskFcm = Fcm["id_act"].is_in(UsersSubnet["user_id"])            
            TmpFcm = Fcm.filter(MaskFcm)
            # Compute Speed
            if len(TmpFcm) > 0:
                # Scalar Speed Associated to the Subnet
                AvSpeedSubnet = TmpFcm["speed_kmh"].mean()
                # Associate Speed to the Roads
                for Road in RoadsPerClass:
                    Class2TimeInterval2Road2SpeedNew[Class][TimeDeparture][Road] = AvSpeedSubnet
            else:
                pass
    return Class2TimeInterval2Road2SpeedNew

def ComputeSpeedRoadsPerTimeInterval(Fcm,OrderedClass2TimeDeparture2UserId,IntClass2RoadsIncreasinglyIncludedIntersection):
    """
    @param GeoJson: GeoDataFrame
    @param Fcm: DataFrame with the Fcm
    @param OrderedClass2TimeDeparture2UserId: Dictionary {Class: {TimeDeparture: [UserId]}}
    @param StrDate: String (Day of Interest)
    @param StrClassCol: String (Column with the hierarchical class)
    @details: It is used for Analysis Network 1 Day
        Compute the Speed for each road in the GeoJson
    return:
        TimeInterval2SpeedNew: Dictionary {TimeInterval: {Road:<Speed>_{User in FcmNew Class}}}

    NOTE:
        I am considering for each road the speed that user that have passed there have.
        So I am considering all the traffic.
        If people are passed in the road quickly (have a point there) then I explain why I see higher speeds for the roads.
    """
    Class2TimeInterval2Road2SpeedNew = defaultdict()
    logger.info("Compute Speed Subnet")
    # Choose The User of Class
    for Class in OrderedClass2TimeDeparture2UserId.keys():
        Roads = IntClass2RoadsIncreasinglyIncludedIntersection[Class]
        if Class not in Class2TimeInterval2Road2SpeedNew.keys():
            Class2TimeInterval2Road2SpeedNew[Class] = defaultdict()
        # Pick the People That are present in the interval
        for TimeDeparture in OrderedClass2TimeDeparture2UserId[Class].keys():
            if TimeDeparture not in Class2TimeInterval2Road2SpeedNew[Class].keys():
                Class2TimeInterval2Road2SpeedNew[Class][TimeDeparture] = defaultdict()
            Users = OrderedClass2TimeDeparture2UserId[Class][TimeDeparture]
            # Take the absolute value of the 'poly_id' column in DfPathOnRoad
            MaskFcm = Fcm["id_act"].is_in(Users)            
            TmpFcm = Fcm.filter(MaskFcm)
            # Compute Speed
            if len(TmpFcm) > 0:
                # Scalar Speed Associated to the Subnet
                AvSpeedSubnet = TmpFcm["speed_kmh"].mean()
                # Associate Speed to the Roads
                for Road in Roads:
                    Class2TimeInterval2Road2SpeedNew[Class][TimeDeparture][Road] = AvSpeedSubnet
            else:
                pass
    return Class2TimeInterval2Road2SpeedNew

def ComputeSpeedRoadPerTimePeopleChangedClass(Fcm,OrderedClass2TimeDeparture2UserId,Class2TimeDeparture2UserId,IntClass2RoadsIncreasinglyIncludedIntersection):
    """
    @param Fcm: DataFrame with the Fcm
    @param OrderedClass2TimeDeparture2UserId: Dictionary {NewClass: {TimeDeparture: [UserId]}}
    @param Class2TimeDeparture2UserId: Dictionary {OldClass: {TimeDeparture: [UserId]}}
    @param IntClass2RoadsIncreasinglyIncludedIntersection: Dictionary {NewClass: [ PolyLid Roads]}
    @return ClassOld2ClassNewTimeInterval2Road2SpeedNew: Dictionary {OldClass: {NewClass: {TimeDeparture: {Road: Speed}}}}

    Picks people from the old classification, that are in the new classification and computes the speed of the roads that are in the new classification.
    """
    # Compute the speed for each transition group in the subnet of interest for each time interval
    ClassOld2ClassNewTimeInterval2Road2SpeedNew = defaultdict()
    # Count the people that transition to the new class
    ClassOld2ClassNewTimeInterval2Transition = defaultdict()
    logger.info("Compute Speed Transitions")
    # Choose The User of Class Old
    for OldClass in Class2TimeDeparture2UserId.keys():
        if OldClass not in ClassOld2ClassNewTimeInterval2Road2SpeedNew.keys():
            ClassOld2ClassNewTimeInterval2Road2SpeedNew[OldClass] = defaultdict()
            ClassOld2ClassNewTimeInterval2Transition[OldClass] = defaultdict()
        # Choose People In Class New
        for ClassNew in OrderedClass2TimeDeparture2UserId.keys():
            # Choose Network That belong to the new Class (Save the speed that is associated to the trajectory that are newly added to the road network)            
            Roads = IntClass2RoadsIncreasinglyIncludedIntersection[ClassNew]
            if ClassNew not in ClassOld2ClassNewTimeInterval2Road2SpeedNew[OldClass].keys():
                # Transition from OldClass to ClassNew due to hierarchy
                ClassOld2ClassNewTimeInterval2Road2SpeedNew[OldClass][ClassNew] = defaultdict()            
                ClassOld2ClassNewTimeInterval2Transition[OldClass][ClassNew] = defaultdict()
                # Pick the People That are present in the interval
                for TimeDeparture in OrderedClass2TimeDeparture2UserId[ClassNew].keys():
                    if TimeDeparture not in ClassOld2ClassNewTimeInterval2Road2SpeedNew[OldClass][ClassNew].keys():
                        ClassOld2ClassNewTimeInterval2Road2SpeedNew[OldClass][ClassNew][TimeDeparture] = defaultdict()            
                        ClassOld2ClassNewTimeInterval2Transition[OldClass][ClassNew][TimeDeparture] = 0
                    # The old classification now is considered in the new
                    Users = list(set(Class2TimeDeparture2UserId[OldClass][TimeDeparture]).intersection(OrderedClass2TimeDeparture2UserId[ClassNew][TimeDeparture]))
                    # Take the absolute value of the 'poly_id' column in DfPathOnRoad
                    MaskFcm = Fcm["id_act"].is_in(Users)            
                    TmpFcm = Fcm.filter(MaskFcm)
                    # Compute Speed
                    if len(TmpFcm) > 0:
                        # Scalar Speed Associated to the Subnet
                        AvSpeedSubnet = TmpFcm["speed_kmh"].mean()
                        # Associate Speed to the Roads
                        ClassOld2ClassNewTimeInterval2Transition[OldClass][ClassNew][TimeDeparture] = len(TmpFcm)
                        for Road in Roads:
                            ClassOld2ClassNewTimeInterval2Road2SpeedNew[OldClass][ClassNew][TimeDeparture][Road] = AvSpeedSubnet
                    else:
                        pass
    return ClassOld2ClassNewTimeInterval2Road2SpeedNew,ClassOld2ClassNewTimeInterval2Transition
                        
def ComputeAvSpeedPerClass(GeoJson,PlotColumn):
    """
    NOTE: ColumnName = "{0}_Speed_{1}_{2}".format(Type,StrDate,TimeDeparture)
    @param GeoJson: GeoDataFrame
    @param PlotColumn: List [Column]
    @param StrDate: String (Day of Interest)
    @param Type: String (Hierarchical or not hierarchical subnet) [new,'']
    @details: It is used for Analysis Network 1 Day
        Compute the Average Speed for each Class
    """
    logger.info("Compute Average Speed Per Class")
    Class2Speed = defaultdict()
    Class2Count = defaultdict()
    for Column in PlotColumn:
        Type = Column.split("_")[0]
        TimeDeparture = Column.split("_")[2]
        StrDate = Column.split("_")[1]
        Classes = GeoJson["IntClassOrdered_{}".format(StrDate)].unique()
        if Type == "new":
            # Compute Average Speed for Each Class
            for Class in Classes:
                Mask = GeoJson["IntClassOrdered_{}".format(StrDate)] == Class
                if Class not in Class2Speed.keys():
                    Class2Speed[Class] = GeoJson.loc[Mask,Column].mean()
                    Class2Count[Class] = 1
                else:
                    Class2Speed[Class] += GeoJson.loc[Mask,Column].mean()
                    Class2Count[Class] += 1
    for Class in Class2Speed.keys():
        Class2Speed[Class] /= Class2Count[Class]
    return Class2Speed



    


def AddColumnAverageSpeedGeoJson(GeoJson,Class2TimeInterval2Speed,StrDate,Type):
    """
    @param GeoJson: GeoDataFrame
    @param Class2TimeInterval2Speed: Dictionary {Class: {TimeDeparture: {Road: Speed}}}
    @param StrDate: String (Day of Interest)
    @param Type: String (Hierarchical or not hierarchical subnet) [new,'']
    """
    logger.info("Add Column Average Speed to GeoJson")
    ColumnPlot = []
    Classes = list(Class2TimeInterval2Speed.keys())
    # Time Interval is Equal To All Classes
    for TimeDeparture in Class2TimeInterval2Speed[Classes[0]].keys():
        if "{0}_Speed_{1}_{2}".format(Type,StrDate,TimeDeparture) in GeoJson.columns:
            ColumnPlot = ["{0}_Speed_{1}_{2}".format(Type,StrDate,TimeDeparture) for TimeDeparture in Class2TimeInterval2Speed[Classes[0]]]
            break
        else:
            ColumnSpeed = np.zeros(len(GeoJson),dtype = float)
    #        CountClass = np.zeros(len(GeoJson),dtype = float)
            for Class in Class2TimeInterval2Speed.keys():
                Speed = Class2TimeInterval2Speed[Class][TimeDeparture]
                # Associate the Speed to Roads in GeoJson
                for Road in Speed.keys():
                    ColumnSpeed[np.where(GeoJson["poly_lid"] == int(Road))[0]] = Speed[Road]
    #                CountClass[np.where(GeoJson["poly_lid"] == Road)[0]] += 1
    #        ColumnSpeed[np.where(ColumnSpeed!=0)[0]]/=CountClass[np.where(ColumnSpeed!=0)[0]]
            ColumnSpeed = fill_zeros_with_average(ColumnSpeed)
            ColumnName = "{0}_Speed_{1}_{2}".format(Type,StrDate,TimeDeparture)
            if ColumnName not in GeoJson.columns:
                ColumnSpeed = list(ColumnSpeed)
                GeoJson[ColumnName] = list(ColumnSpeed)
                ColumnPlot.append(ColumnName)

    return GeoJson,ColumnPlot

def ReturnColumnPlot(Class2TimeInterval2Speed,Type,StrDate):
    """
    @param StrDate: String (Day of Interest)
    @param TimeDeparture: String (Time of Interest)
    """
    ColumnPlot = []
    for Class in Class2TimeInterval2Speed.keys():
        for TimeDeparture in Class2TimeInterval2Speed[Class].keys():
            if "{0}_Speed_{1}_{2}".format(Type,StrDate,TimeDeparture) not in ColumnPlot:
                ColumnPlot.append("{0}_Speed_{1}_{2}".format(Type,StrDate,TimeDeparture))
    return ColumnPlot

def GetIncrementalIntersectionAllDaysClasses(GpdClasses,StrClassesOrderedColsDate,UniqueClassesOrdered):
    Class2Road2VecBoolBelongDay = {Class:{Road:[] for Road in GpdClasses.index} for Class in UniqueClassesOrdered}  
    for Class in UniqueClassesOrdered:
        print(Class)
        tRoad0 = time.time()
        CountRoad = 0
        for Road in GpdClasses.index:
            for DayCol in StrClassesOrderedColsDate.keys():
                if GpdClasses.at[Road,DayCol] == Class:
                    Class2Road2VecBoolBelongDay[Class][Road].append(True)
                else:
                    Class2Road2VecBoolBelongDay[Class][Road].append(False)
            tRoad1 = time.time()
            if CountRoad == 0:
                print(f"Time Spent to Classify the Belonging of {Road}",tRoad1-tRoad0)
                print("Estimated Time Process: ",len(UniqueClassesOrdered)*len(GpdClasses.index)*(tRoad1-tRoad0)/60," minutes")
                print("Number Days: ",np.shape(Class2Road2VecBoolBelongDay[Class][Road]))
                CountRoad += 1
    Class2Road2Intersection = {Class: {Road: np.logical_and.reduce(Class2Road2VecBoolBelongDay[Class][Road]) for Road in Class2Road2VecBoolBelongDay[Class].keys()} for Class in UniqueClassesOrdered}
    Class2Road2Union = {Class: {Road: np.logical_or.reduce(Class2Road2VecBoolBelongDay[Class][Road]) for Road in Class2Road2VecBoolBelongDay[Class].keys()} for Class in UniqueClassesOrdered}
    return Class2Road2Intersection,Class2Road2Union

def GetIntersectionAllDaysClasses(GpdClasses,StrClassesColsDate,UniqueClasses):
    Class2Road2VecBoolBelongDay = {Class:{Road:[] for Road in GpdClasses.index} for Class in UniqueClasses}  
    for Class in UniqueClasses:
        print(Class)
        tRoad0 = time.time()
        CountRoad = 0
        for Road in GpdClasses.index:
            for DayCol in StrClassesColsDate.keys():
                if GpdClasses.at[Road,DayCol] == Class:
                    Class2Road2VecBoolBelongDay[Class][Road].append(True)
                else:
                    Class2Road2VecBoolBelongDay[Class][Road].append(False)
            tRoad1 = time.time()
            if CountRoad == 0:
                print(f"Time Spent to Classify the Belonging of {Road}",tRoad1-tRoad0)
                print("Estimated Time Process: ",len(UniqueClasses)*len(GpdClasses.index)*(tRoad1-tRoad0)/60," minutes")
                print("Number Days: ",np.shape(Class2Road2VecBoolBelongDay[Class][Road]))
                CountRoad += 1
    Class2Road2Intersection = {Class: {Road: np.logical_and.reduce(Class2Road2VecBoolBelongDay[Class][Road]) for Road in Class2Road2VecBoolBelongDay[Class].keys()} for Class in UniqueClasses}
    Class2Road2Union = {Class: {Road: np.logical_or.reduce(Class2Road2VecBoolBelongDay[Class][Road]) for Road in Class2Road2VecBoolBelongDay[Class].keys()} for Class in UniqueClasses}
    return Class2Road2Intersection,Class2Road2Union



def UpdateGeoJsonWithUnionAndIntersectionColumns(GpdClasses,Class2Road2Intersection,Class2Road2Union,StrUnion = "Union_",StrIntersection = "Intersection_"):
    """
        @param GpdClasses: GeoDataFrame
        @param Class2Road2Intersection: Dictionary {Class: {Road: Bool}}
        @param Class2Road2Union: Dictionary {Class: {Road: Bool}}
        @param StrUnion: String (Prefix for the Union Column)   
        @param StrIntersection: String (Prefix for the Intersection Column)
        @details: It is used for Analysis Network 1 Day
            Add the Intersection and Union Columns to the GeoDataFrame
    """
    for Class in Class2Road2Intersection.keys():
        Intersection = []
        Union = []
        for Road in GpdClasses.index:
            Intersection.append(Class2Road2Intersection[Class][Road])
            Union.append(Class2Road2Union[Class][Road])
        GpdClasses[StrIntersection + Class] = Intersection
        GpdClasses[StrUnion + Class] = Union    
    return GpdClasses



def FromGeoJsonRoads2Grid(GeoJson,Lx,Ly):
    """
    @param GeoJson: GeoDataFrame containing carography informations.
    @param Lx: Float (Length of the square in the x direction in meters)
    @param Ly: Float (Length of the square in the y direction in meters)
    @return grid: GeoDataFrame containing the grid
    NOTE: Needed for Plots About OD Matrix
    """
    angle_crs = GeoJson.crs 
    # project the GeoJson to UTM
    GeoJson = GeoJson.to_crs(GeoJson.estimate_utm_crs())
    minx, miny, maxx, maxy = GeoJson.total_bounds
    squares = []
    Is,Js = [], []
    # 
    coord_x = minx
    coord_y = miny
    i = 0
    while (coord_x < maxx):
        j = 0
        while (coord_y < maxy):
            square = Polygon([(coord_x, coord_y), (coord_x + Lx, coord_y), (coord_x + Lx, coord_y + Ly), (coord_x, coord_y + Ly)])
            squares.append(square)
            Is.append(int(i))
            Js.append(int(j))
            j += 1
            coord_y += Ly
        i += 1
        coord_x += Lx
        coord_y = miny
    squares_gdf = gpd.GeoDataFrame(geometry=squares) # crs= "EPSG:32632"
    squares_gdf["i"] = Is
    squares_gdf["j"] = Js   
    grid = gpd.GeoDataFrame(geometry=squares_gdf.geometry,crs = GeoJson.crs) # crs= "EPSG:32632"
    grid["i"] = squares_gdf["i"]
    grid["j"] = squares_gdf["j"]   
    grid = grid.to_crs(angle_crs)
    return grid

def FilterGeoDataFrameWithBoundingBox(gdf,Bbox):
    """
    @param gdf: GeoDataFrame
    @param lat_min: Float (Latitude Minimum)
    @param lat_max: Float (Latitude Maximum)
    @param lon_min: Float (Longitude Minimum)
    @param lon_max: Float (Longitude Maximum)
    @return gdf: GeoDataFrame
    """
    minx, miny, maxx, maxy = Bbox
    if isinstance(gdf.geometry[0],Point):
        filtered_gdf = gdf.cx[minx:maxx, miny:maxy]
    elif isinstance(gdf.geometry[0],LineString):
        filtered_gdf = gdf.cx[minx:maxx, miny:maxy]
    elif isinstance(gdf.geometry[0],Polygon):
        filtered_gdf = gdf.cx[minx:maxx, miny:maxy]
    return filtered_gdf