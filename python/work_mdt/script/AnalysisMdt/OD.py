import polars as pl
from collections import defaultdict
from shapely.geometry import Point
import geopandas as gpd
def ODfromFcm(Fcm,Grid,ClassCol = 'class_new',TimeDepartureCol = 'start_time'):
    """
        @brief: Compute the Origin-Destination matrix from the Fcm
        @param Fcm: DataFrame with the Fcm
        @param Grid: DataFrame with the Grid
        
    """
    NewClass2Users = defaultdict(list)
    OrderedClass2TimeDeparture2UserId = defaultdict()
    for Class,FcmClass in Fcm.groupby(ClassCol):
        Grid[f"O_{Class}"] = [0 for i in range(len(Grid))]
        Grid[f"D_{Class}"] = [0 for i in range(len(Grid))]
        UserGeomO = gpd.GeoDataFrame(FcmClass,geometry = gpd.points_from_xy(FcmClass["origin_lon"],FcmClass["origin_lat"]))
        UserGeomD = gpd.GeoDataFrame(FcmClass,geometry = gpd.points_from_xy(FcmClass["destination_lon"],FcmClass["destination_lat"]))
        O = Point(FcmClassTimeDeparture["origin_lat"],FcmClassTimeDeparture["origin_lon"])
        D = Point(FcmClassTimeDeparture["destination_lat"],FcmClassTimeDeparture["destination_lon"])
        for i in range(len(Grid)):
            for j in range(len(Grid)):
                if Grid["geometry"][i].contains(O) and Grid["geometry"][j].contains(D):
                    Grid.loc[i][f"O_{Class}"] += 1
                    Grid.loc[j][f"D_{Class}"] += 1
        for TimeDeparture,FcmClassTimeDeparture in FcmClass.groupby(TimeDepartureCol):
            O = Point(FcmClassTimeDeparture["origin_lat"],FcmClassTimeDeparture["origin_lon"])
            D = Point(FcmClassTimeDeparture["destination_lat"],FcmClassTimeDeparture["destination_lon"])

    return Grid