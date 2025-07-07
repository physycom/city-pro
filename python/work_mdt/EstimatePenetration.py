from collections import defaultdict
from datetime import datetime
import os
import polars as pl
import pandas as pd
import geopandas as gpd
from GeographyFunctions import FilterGeoDataFrameWithBoundingBox
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
import numpy as np

def GetTimeRanges(time_ranges, base_date):
    time_range_to_datetime = defaultdict(tuple)
    for time_range in time_ranges:
        start_time, end_time = time_range.split('-')
        if end_time != '24:00':
            start_datetime = datetime.strptime(f"{base_date} {start_time}", "%Y-%m-%d %H:%M")
            time_range_to_datetime[time_range] = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return time_range_to_datetime
def PolyCid2NameFromFilePro(FilePro):#poly_cid  node_front_cid  node_tail_cid  length  lvl_ps  ?  ?  ?  speed  ?  name
    PolyCid2Name = defaultdict(str)
    with open(FilePro, 'r') as file:
        header = file.readline().strip().split()
    
        # Iterate over each line in the file
        for line in file:
            # Split the line into values
            values = line.strip().split()
            
            # Assign the values to respective variables
            poly_cid = int(values[0])
            node_front_cid = int(values[1])
            node_tail_cid = int(values[2])
            length = float(values[3])
            lvl_ps = float(values[4])
            unknown1 = int(values[5])
            unknown2 = int(values[6])
            unknown3 = int(values[7])
            speed = float(values[8])
            unknown4 = int(values[9])
            name = ' '.join(values[10:])  # Join the remaining values as the name
            PolyCid2Name[poly_cid] = name
    PolyCid2Name = Substitute_WithSpacePolyCid2Name(PolyCid2Name)

    return PolyCid2Name 
def Substitute_WithSpacePolyCid2Name(PolyCid2Name):
    PolyCid2Name_Space = defaultdict(str)
    for key in PolyCid2Name:
        PolyCid2Name_Space[key] = PolyCid2Name[key].replace('_', ' ')
    return PolyCid2Name_Space

def GetTimeRangesFromDfOpenData(DfTrafficOpenData):
    TimeRanges = []
    for Col in DfTrafficOpenData.columns:
        if ":" in Col:
            TimeRanges.append(Col)
    return TimeRanges

def DiscardColumnsSpeedGdfRoads(GdfRoads):
    Columns2Accept = []
    for Col in GdfRoads.columns:
        if 'Speed' not in Col:
            Columns2Accept.append(Col)
    GdfRoads = GdfRoads[Columns2Accept]
    return GdfRoads

def EstimatePenetrationAndPlot(GdfRoads,DfTrafficOpenData,Bbox,Days,DirTimedFluxes,CutIndex,PlotDir):
    """
        @param GdfRoads: GeoDataFrame with the roads
        @param DfTrafficOpenData: DataFrame with the traffic data
        @param Bbox: Bounding box for the city
        @param Days: List of days to consider
        @param PlotDir: Directory to save the plots

    """
    DfTrafficOpenData_pd = DfTrafficOpenData.to_pandas()
    GdfTrafficOpenData = gpd.GeoDataFrame(
        DfTrafficOpenData_pd, geometry=gpd.points_from_xy(DfTrafficOpenData_pd["longitudine"], DfTrafficOpenData_pd["latitudine"]))
    GdfTrafficOpenData.set_crs(epsg=4326, inplace=True)
    TimeRanges = GetTimeRangesFromDfOpenData(DfTrafficOpenData)
    Column2ConsiderGdfTrafficOpenData = TimeRanges.copy()
    Column2ConsiderGdfTrafficOpenData.append("data")
    Column2ConsiderGdfTrafficOpenData.append("geometry")
    GdfTrafficOpenData = GdfTrafficOpenData[Column2ConsiderGdfTrafficOpenData]
    GdfRoads = FilterGeoDataFrameWithBoundingBox(GdfRoads,Bbox)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    
    Colors = ["red","blue","green","black","yellow","orange","purple","pink"]
    if len(Days) > len(Colors):
        raise ValueError("Too many days to plot, change EstimatePenetrationAndPlot:Colors")
    Day2Color = dict(zip(Days,Colors))
    CountDay = 0
    for Day in Days:
        TimeRange2Time = GetTimeRanges(TimeRanges,Day)
        TimedFluxes = pl.read_csv(DirTimedFluxes[CountDay],separator=';')#,f"bologna_mdt_{Day}_{Day}_timed_fluxes.csv"
        GdfTrafficOpenData = GdfTrafficOpenData.to_crs(epsg=3857)
        GdfRoads = GdfRoads.to_crs(epsg=3857)    
        GdfJoin = gpd.sjoin_nearest(GdfTrafficOpenData, GdfRoads, how='left', distance_col='distance')
        CountDay += 1
        GdfJoin = GdfJoin.loc[GdfJoin["distance"]<2]
        GdfJoin.set_crs(epsg=4326, inplace=True,allow_override=True)
        GdfJoin = DiscardColumnsSpeedGdfRoads(GdfJoin)
        PolyLid2VectorFluxes = {"average_penetration_tot": [], "average_penetration_FT": [], "average_penetration_TF": []}
        TimeHours = [Time.split(" ")[1] for Time in TimeRange2Time.values()]
        for TimeRange in TimeRange2Time.keys():
            Time = TimeRange2Time[TimeRange]
            FluxesPolysAtTime = TimedFluxes.filter(pl.col("time") == Time)
            DfJoinAtTime = pl.DataFrame(GdfJoin[[TimeRange, "poly_lid"]])
            DfJoinAtTime = DfJoinAtTime.filter(pl.col(TimeRange).is_not_null(),
                                pl.col(TimeRange).is_not_nan(),
                                pl.col(TimeRange) > 0)
            # Cast
            FluxesPolysAtTime = FluxesPolysAtTime.with_columns(pl.col("id").cast(pl.Int32))
            DfJoinAtTime = DfJoinAtTime.with_columns(pl.col("poly_lid").cast(pl.Int32))            

            FluxesPolysAtTime = FluxesPolysAtTime.join(DfJoinAtTime,left_on="id",right_on="poly_lid")
            
            FluxesPolysAtTime = FluxesPolysAtTime.with_columns((pl.col("total_fluxes")/pl.col(TimeRange)).alias("penetration_total"),
                                                            (pl.col("n_traj_FT")/pl.col(TimeRange)).alias("penetration_FT"),
                                                            (pl.col("n_traj_TF")/pl.col(TimeRange)).alias("penetration_TF")
                                                            )
            FluxesPolysAtTime = FluxesPolysAtTime.filter(pl.col("penetration_total").is_not_null(),
                                                        pl.col("penetration_FT").is_not_null(),
                                                        pl.col("penetration_TF").is_not_null(),
                                                        pl.col("penetration_total").is_not_nan(),
                                                        pl.col("penetration_FT").is_not_nan(),
                                                        pl.col("penetration_TF").is_not_nan(),
                                                        pl.col("penetration_total") > 0,
                                                        pl.col("penetration_TF") < 1,
                                                        pl.col("penetration_FT") < 1,
                                                        pl.col("penetration_total") < 1,
    #                                                    pl.col("length") >100,
                                                        pl.col("id")!=18952)
    #        print(FluxesPolysAtTime[["total_fluxes","id_local",TimeRange,"penetration_total","penetration_TF","penetration_FT"]].head())
    #        plt.hist(FluxesPolysAtTime["penetration_FT"].to_numpy())
    #        plt.title("Penetration_FT" + TimeRange)
    #        plt.show()
    #        plt.hist(FluxesPolysAtTime["penetration_TF"].to_numpy())
    #        plt.title("Penetration_TF" + TimeRange)
    #        plt.show()
    #        plt.hist(FluxesPolysAtTime["penetration_total"].to_numpy())
    #        plt.title("Penetration_total" + TimeRange)
    #        plt.show()
            PenetrationTotal = np.nanmedian(FluxesPolysAtTime["penetration_total"].to_numpy())
            PenetrationFT = np.nanmedian(FluxesPolysAtTime["penetration_FT"].to_numpy())
            PenetrationTF = np.nanmedian(FluxesPolysAtTime["penetration_TF"].to_numpy())
            PolyLid2VectorFluxes["average_penetration_tot"].append(PenetrationTotal) 
            PolyLid2VectorFluxes["average_penetration_FT"].append(PenetrationFT)
            PolyLid2VectorFluxes["average_penetration_TF"].append(PenetrationTF) 
    #    ax.plot(TimeHours,PolyLid2VectorFluxes["average_penetration_tot"],label=Day,color=Day2Color[Day])
        #    ax.plot(TimeHours,PolyLid2VectorFluxes["average_penetration_tot"],label=Day,color=Day2Color[Day])
        ax.plot(TimeHours[CutIndex:],PolyLid2VectorFluxes["average_penetration_FT"][CutIndex:],label=Day,color=Day2Color[Day])
#    ax.plot(TimeHours,PolyLid2VectorFluxes["average_penetration_TF"],label=Day,color=Day2Color[Day])
    ax.legend()
    ax.set_xticks(range(len(TimeHours[CutIndex:]))[::4])  # Set the ticks to correspond to the labels
    ax.set_xticklabels(TimeHours[CutIndex::4], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
    ax.set_xlabel("Time")
    ax.set_ylabel("Penetration")
    plt.savefig(os.path.join(PlotDir,"penetration.png"))
    plt.show()