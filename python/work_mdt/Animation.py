
from collections import defaultdict

import datetime
import geopandas as gpd 
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation, FFMpegWriter

import numpy as np

import os

def GetInfoFrameEvolutionSpeed(gdf):
    """
        @param gdf: GeoDataFrame with the traffic information
        @return x_label: Label of the x-axis
        @return y_label: Label of the y-axis
        @return time2column: Dict with the time (str) and the column ["new_class_Speed_" + Day + "_" + str(timestamp)]
    """
    x_label = "time"
    y_label = "v (km/h)"
    timestamps = []
    time2column = defaultdict()
    for Col in gdf.columns:
        if "new_class" in Col:
            Day = Col.split("_")[-2]
            timestamp = int(Col.split("_")[-1])
            timestamps.append(timestamp)
    timestamps = np.sort(timestamps)
    for timestamp in timestamps:
        t_hour = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S').split(" ")[1]
        time2column[str(t_hour)] = "new_class_Speed_" + Day + "_" + str(timestamp)
    return x_label,y_label,time2column,Day

def GetClass2TimeSpeed(gdf,time2column,Day):
    Class2TimeSpeed = defaultdict()
    for Class, gdfClass in gdf.groupby("IntClassOrdered_"+Day): 
        if Class not in Class2TimeSpeed.keys():
            Class2TimeSpeed[Class] = {"time":[],"speed":[]}
        for time_h,col in time2column.items():
            Class2TimeSpeed[Class]["time"].append(time_h)
            Class2TimeSpeed[Class]["speed"].append(gdfClass[col].mean())
    return Class2TimeSpeed

def AnimateEvolutionTraffic(FrameN, gdfClass, Time, Speed, Columns, TimeKey, linewidth_col=0.5, cmap='inferno'):
    """
        @param gdfClass: GeoDataFrame with the Class
        @param Class2TimeSpeed: Dict with the time and speed of the class
        @param FrameN: Frame Number
        @param linewidth_col: Linewidth of the column (set the width of the street with traffic)
    """
    
    Column = Columns[FrameN]
#    Column = Time2Column[TimeKey]
    for coll in ax_geo.collections[1:]:
        coll.remove()
    # Extract Column To Plot
    line.set_xdata(Time[:FrameN])
    line.set_ydata(Speed[:FrameN])    
    # Geo
    GeoPlt = gdf.plot(color='gray', linewidth=linewidth, ax=ax_geo,alpha = 0.8)
    gdfClass.plot(column=Column, linewidth=linewidth_col, ax=ax_geo,alpha = 1)
    ax_geo.set_title(f"Class {Class} {Day} {TimeKey}")    
    FrameN += 1
    return ax_geo.collections

def EvolutionRoadSpeed(PlotDir,Day):
    gdf = gpd.read_file(os.path.join(PlotDir,Day,f"GeoJson_{Day}.geojson"))
    x_label,y_label,Time2Column,Day = GetInfoFrameEvolutionSpeed(gdf)
    Class2TimeSpeed = GetClass2TimeSpeed(gdf,time2column,Day)
    for Class, gdfClass in gdf.groupby(f"IntClassOrdered_{Day}"):
        FrameN = 0
        fig, (ax_curve,ax_geo) = plt.subplots(1,2,figsize=(10,5),width_ratios=[1,3])
        # Global Variables
        Time = Class2TimeSpeed[Class]["time"]
        
        # BaseMap
        line = ax_curve.plot(Time[0], np.mean(gdfClass[Columns[0]]))[0]
        ax_curve.set_xticks(range(len(Time))[::8])
        ax_curve.set_xticklabels(Time[::8], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
        ax_curve.set(xlabel='Time', ylabel='v km/h')

        # Base Geo-Map
        linewidth = 0.1
        GeoPlt = gdf.plot(color='gray', linewidth=linewidth, ax=ax_geo,alpha = 0.8)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=gdfClass[column].min(), vmax=gdfClass[column].max()))
        sm._A = []  # Dummy array for the ScalarMappable
        cbar = fig.colorbar(sm, ax=ax_geo)
        cbar.set_label(y_label)    
        # ANIMATION
        NumberFrames = np.arange(len(list(time2column.keys())))
        TimeSpeed = Class2TimeSpeed[Class]
        Time2Column = dict(Time2Column)
        TimeKey = list(Time2Column.keys())[FrameN] 
        Columns = list(Time2Column.values())
        # List Time
        Time = list(TimeSpeed["time"])
        # List Speed
        Speed = list(TimeSpeed["speed"])
        ax_geo.set_title(f"Class {Class} {Day} {TimeKey}")
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, AnimateEvolutionTraffic, NumberFrames, fargs=(gdfClass, Time, Speed, Columns, TimeKey),
                                    interval=100, blit=True, repeat=False)
        writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        output_path = os.path.join("/home/aamad/codice/city-pro/output/bologna_mdt_detailed/plots", Day, f'Class_{Class}_{Day}.mp4')
        anim.save(output_path, writer=writer, dpi=200)
        anim.save(os.path.join("/home/aamad/codice/city-pro/output/bologna_mdt_detailed/plots",Day,f'Class_{Class}_{Day}.mp4'), dpi=200)