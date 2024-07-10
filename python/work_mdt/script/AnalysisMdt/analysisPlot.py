import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import folium
import polars as pl
from collections import defaultdict
from MFDAnalysis import *
from CastVariables import *
##----------------------------------- PLOT VELOCITIES -----------------------------------##

def QuiverPopulationVelocityClass(population,velocity,save_dir,day,idx,dict_name,average_all_days = False):
    '''
        Input:
            population: (np.array 1D) -> population 
            velocity: (np.array 1D) -> velocity 
            dict_idx: (dict) -> dict_idx = {'population':[],'velocity':[]}
            save_dir: (str) -> save_dir = '/home/aamad/Desktop/phd/berkeley/traffic_phase_transition/data/carto/BOS'
            day: (str) -> day = 'day_1'
            idx: (int) -> idx = 0
            dict_name: (dict) -> dict_name = {0:'1 slowest',1:'2 slowest'
    '''
    assert population is not None, 'population must be provided'
    assert velocity is not None, 'velocity must be provided'
    assert len(population) == len(velocity), 'population and velocity must have the same length'
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    u = [population[i+1]-population[i] for i in range(len(population)-1)]
    v = [velocity[i+1]-velocity[i] for i in range(len(velocity)-1)]
    u.append(population[len(population)-1] -population[0])
    v.append(velocity[len(velocity)-1] -velocity[0])
    ax.quiver(population,velocity,u,v,angles='xy', scale_units='xy', scale=1,width = 0.0025)
    ax.set_xlabel('number people')
    ax.set_ylabel('velocity')
    ax.set_title(str(dict_name[idx]))
    if average_all_days:
        plt.savefig(os.path.join(save_dir,'Hysteresis_Average_{0}_Class_{1}.png'.format(day,dict_name[idx])),dpi = 200)
    else:
        plt.savefig(os.path.join(save_dir,'Hysteresis_{0}_Class_{1}.png'.format(day,dict_name[idx])),dpi = 200)
    plt.show()

def MFDByClass(population,velocity,dict_name,idx,save_dir,day,verbose = False): 

    nx,binsPop = np.histogram(population,range = (min(population),max(population)))
    y_avg = np.zeros(len(binsPop))
    y_dev = np.zeros(len(binsPop))
    for dx in range(len(binsPop)-1):
        idx_ = np.array([True if xi>=binsPop[dx] and xi<=binsPop[dx+1] else False  for xi in x])
        y_avg[dx] += np.mean(velocity[idx_])
        y_dev[dx] = np.std(velocity[idx_])
    print('mean:\t',y_avg[:-1],'\nstd-dev:\t',y_dev[:-1],'\ndev/mean:\t',y_dev[:-1]/y_avg[:-1])
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    ax.plot(binsPop[:-1],y_avg[:-1])
    ax.plot(binsPop[:-1],y_avg[:-1]+y_dev[:-1])
    ax.plot(binsPop[:-1],y_avg[:-1]-y_dev[:-1])
    ax.set_xlabel('number people')
    ax.set_ylabel('velocity')
    ax.set_title(str(dict_name[idx]))
    ax.legend(['mean','mean+std','mean-std'])
    plt.savefig(os.path.join(save_dir,'{0}_class_averageV_per_D_{1}.png'.format(dict_name[idx],day)),dpi = 200)
    plt.show()



##----------------------------------- PLOT TIMES -----------------------------------##
def PlotTimePercorrenceDistribution(RoadsTimeVel,Time2Distr,AvgTimePercorrence,StrTimesLabel,File2Save):    
    """
        Input:
            RoadsTimeVel: pl.DataFrame -> DataFrame with the Roads Time Velocities 
        Description:
            Time2Distr: array -> Array with the Time Percorrence Distribution (should be list of 96 values, Number of roads available)
            AvgTimePercorrence: array -> Array with the Average Time Percorrence (should be list of 96 values)
        NOTE: In the case I do not have information about a street of the subnet I will not have any information about the time_percorrence
    """
    Slicing = 8
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    RoadsTimeVel = RoadsTimeVel.sort("start_bin")
    for time,RTV in RoadsTimeVel.groupby("start_bin"):
        ValidTime = RTV.filter(pl.col("time_percorrence")>0)
        Time2Distr.append(ValidTime["time_percorrence"].to_list())
        AvgTimePercorrence.append(np.mean(ValidTime["time_percorrence"].to_list()))
        StartInterval = datetime.datetime.fromtimestamp(time)
        StrTimesLabel.append(StartInterval.strftime("%Y-%m-%d %H:%M:%S").split(" ")[1])
    ax.plot(StrTimesLabel, AvgTimePercorrence)
    ax.boxplot(Time2Distr,sym='')
    ax.set_xticks(range(len(StrTimesLabel))[::Slicing])  # Set the ticks to correspond to the labels
    ax.set_xticklabels(StrTimesLabel[::Slicing], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
    ax.set_xlabel("Time")
    ax.set_ylabel("Time Percorrence")
    plt.savefig(File2Save,dpi = 200)
    return Time2Distr,AvgTimePercorrence




##----------------------------------- PLOT DISTRIBUTIONS -----------------------------------##
def SplitFcmByClass(Fcm,Feature,IntClass2StrClass):
    Class2FcmDistr = defaultdict()
    for IntClass in IntClass2StrClass:
        y,x = np.histogram(Fcm.filter(pl.col("class") == IntClass)[Feature].to_list(),bins = 50)
        if Feature == "av_speed" or Feature == "speed_kmh":
            y = y/np.sum(y)
        Class2FcmDistr[IntClass] = {"x":x,"y":y,"maxx":max(x),"maxy":max(y),"minx":min(x),"miny":min(y),"mean":np.mean(Fcm.filter(pl.col("class") == IntClass)[Feature].to_list())}
    return Class2FcmDistr

def ComputeMinMaxPlotGivenFeature(Class2FcmDistr,InfoPlotDistrFeat):
    maxx = 0
    maxy = 0
    minx = 10000000
    miny = 10000000
    for IntClass in Class2FcmDistr.keys():
        if maxx < Class2FcmDistr[IntClass]["maxx"]:
            maxx = Class2FcmDistr[IntClass]["maxx"]
        if maxy < Class2FcmDistr[IntClass]["maxy"]:
            maxy = Class2FcmDistr[IntClass]["maxy"]
        if minx > Class2FcmDistr[IntClass]["minx"]:
            minx = Class2FcmDistr[IntClass]["minx"]
        if miny > Class2FcmDistr[IntClass]["miny"]:
            miny = Class2FcmDistr[IntClass]["miny"]
    InfoPlotDistrFeat["maxx"] = maxx
    InfoPlotDistrFeat["maxy"] = maxy
    InfoPlotDistrFeat["minx"] = minx
    InfoPlotDistrFeat["miny"] = miny
    return InfoPlotDistrFeat

def PlotFeatureDistrSeparatedByClass(Class2FcmDistr,InfoPlotDistrFeat,Feature,IntClass2StrClass,DictFittedData,Column2Legend,Feature2IntervalBin,Feature2IntervalCount,Feature2Label,Feature2ShiftBin,Feature2ShiftCount,Feature2ScaleBins,Feature2ScaleCount):
    """
        Class2FcmDistr: dict -> {IntClass: {"x":x,"y":y,"maxx":max(x),"maxy":max(y),"minx":min(x),"miny":min(y),"mean":np.mean(Fcm.filter(pl.col("class") == IntClass)[Feature].to_list())}}
        InfoPlotDistrFeat: dict -> {"maxx":0,"maxy":0,"minx":10000000,"miny":10000000}
        Feature: str -> Feature to Plot
        IntClass2StrClass: dict -> {IntClass: StrClass}
        DictFittedData: dict -> {Feature: {"fitted_data": [],"best_fit": str}}
        Column2Legend: dict -> {Feature: Legend}
        Feature2IntervalBin: dict -> {Feature: IntervalBin}
        Feature2IntervalCount: dict -> {Feature: IntervalCount}
        Feature2Label: dict -> {Feature: Label}
        Feature2ShiftBin: dict -> {Feature: ShiftBin}
        Feature2ShiftCount: dict -> {Feature: ShiftCount}
        Feature2ScaleBins: dict -> {Feature: ScaleBins}
        Feature2ScaleCount: dict -> {Feature: ScaleCount}
        PlotDir: str -> Path to Save the Plot
    """
    fig,ax = plt.subplots(1,1,InfoPlotDistrFeat["figsize"])
    legend = []
#    Class2FcmDistr = SplitFcmByClass(Fcm,Feature,IntClass2StrClass)
    for IntClass in Class2FcmDistr.keys():
        # Scatter Points
        ax.scatter(Class2FcmDistr[IntClass]["x"][1:],Class2FcmDistr[IntClass]["y"])
        legend.append(str(IntClass2StrClass[IntClass]) + " " + Column2Legend[Feature] + " " + str(round(Class2FcmDistr[IntClass]["mean"],3)))
        # Fit
        if len(Class2FcmDistr[IntClass]["x"][1:]) == len(DictFittedData[Feature]["fitted_data"]):
            ax.plot(Class2FcmDistr[IntClass]["x"][1:],np.array(DictFittedData[Feature]["fitted_data"]),label = DictFittedData[Feature]["best_fit"])
            legend.append(str(IntClass2StrClass[IntClass]) + " " + Column2Legend[Feature] + " " + str(round(Class2FcmDistr[IntClass]["mean"],3)))
        ax.set_xticks(np.arange(InfoPlotDistrFeat["minx"],InfoPlotDistrFeat["maxx"],Feature2IntervalBin[Feature]))
        ax.set_yticks(np.arange(InfoPlotDistrFeat["miny"],InfoPlotDistrFeat["maxy"],Feature2IntervalCount[Feature]))
        ax.set_xlabel(Feature2Label[Feature])
        ax.set_ylabel('Count')
        ax.set_xlim(right = InfoPlotDistrFeat["maxx"] + Feature2ShiftBin[Feature])
        ax.set_ylim(bottom = 1,top = InfoPlotDistrFeat["maxy"] + Feature2ShiftCount[Feature])
        ax.set_xscale(Feature2ScaleBins[Feature])
        ax.set_yscale(Feature2ScaleCount[Feature])
    legend_ = plt.legend(legend)
    frame = legend_.get_frame()
    frame.set_facecolor('white')
    return fig,ax

        



##----------------------------------- PLOT FLUXES -----------------------------------##



## --------------------------------- PLOT NETWORKS ---------------------------------- ##
def PlotIncrementSubnetHTML(GeoJson,IntClass2StrClass,centroid,PlotDir,StrDate,ReadFluxesSubIncreasinglyIncludedIntersectionBool,ReadGeojsonBool,Class2Color,BaseNameFile = "SubnetsIncrementalInclusion",verbose = False):
    if ReadFluxesSubIncreasinglyIncludedIntersectionBool and ReadGeojsonBool and ReadGeojsonBool:
        print("Plotting Daily Incremental Subnetworks in HTML")
        if not os.path.isfile(os.path.join(PlotDir,"Subnets_{}.html".format(StrDate))) or True:
            print("Save in: ",os.path.join(PlotDir,"SubnetsIncrementalInclusion_{}.html".format(StrDate)))
            # Create a base map
            m = folium.Map(location=[centroid.x, centroid.y], zoom_start=12)
            # Iterate through the Dictionary of list of poly_lid
            for IntClass in np.unique(GeoJson["IntClassOrdered_{}".format(StrDate)]):
                mclass = folium.Map(location=[centroid.x, centroid.y], zoom_start=12)
#                    for index_list in self.IntClass2RoadsIncreasinglyIncludedIntersection[IntClass]:
                # Filter GeoDataFrame for roads with indices in the current list
                print("IntClass: ",IntClass)
                print("Available IntClassOrdered_{}".format(StrDate))
                print(np.unique(GeoJson["IntClassOrdered_{}".format(StrDate)]))
                filtered_gdf = GeoJson.groupby("IntClassOrdered_{}".format(StrDate)).get_group(IntClass)
#                    index_list = self.IntClass2RoadsIncreasinglyIncludedIntersection[IntClass]
#                    filtered_gdf = self.GeoJson[self.GeoJson['poly_lid'].isin(index_list)]
                # Create a feature group for the current layer
                layer_group = folium.FeatureGroup(name="Layer {}".format(IntClass)).add_to(m)
                layer_group_class = folium.FeatureGroup(name="Layer {}".format(IntClass)).add_to(mclass)
                # Add roads to the feature group with a unique color
                if verbose:
                    print("Class: ",IntClass," Number of Roads: ",len(filtered_gdf),"Color: ",Class2Color[IntClass2StrClass[IntClass]])
                    print("filtered_gdf:",filtered_gdf.head())
                
                for _, road in filtered_gdf.iterrows():
                    if road.geometry is not None:
                        folium.GeoJson(road.geometry, style_function=lambda x: {'color': Class2Color[IntClass2StrClass[IntClass]]}).add_to(layer_group)
                        folium.GeoJson(road.geometry, style_function=lambda x: {'color': Class2Color[IntClass2StrClass[IntClass]]}).add_to(layer_group_class)
                
                # Add the feature group to the map
                layer_group.add_to(m)
                layer_group_class.add_to(mclass)
                # Add layer control to the map
                folium.LayerControl().add_to(m)
                folium.LayerControl().add_to(mclass)
                # Save or display the map
                mclass.save(os.path.join(PlotDir,BaseNameFile + "_{0}_{1}.html".format(StrDate,IntClass)))
            m.save(os.path.join(PlotDir,BaseNameFile + "_{}.html".format(StrDate)))
            Message = "Plotting Daily Incremental Subnetworks in HTML: True"
        else:
            Message = "Plotting Daily Incremental Subnetworks in HTML: Already Plotted"
            print("Subnets Increasingly already Plotted in HTML")
    else:
        Message = "Plotting Daily Incremental Subnetworks in HTML: False"
        print("No Subnetworks to Plot")
    return Message

def PlotSubnetHTML(GeoJson,IntClass2StrClass,centroid,PlotDir,StrDate,ReadFluxesSubBool,ReadGeojsonBool,BoolStrClass2IntClass,Class2Color,verbose = False):
    if ReadFluxesSubBool and ReadGeojsonBool and BoolStrClass2IntClass:
        print("Plotting Daily Subnetworks in HTML")
        if not os.path.isfile(os.path.join(PlotDir,"Subnets_{}.html".format(StrDate))) or True:
            # Create a base map
            m = folium.Map(location=[centroid.x, centroid.y], zoom_start=12)
            # Iterate through the Dictionary of list of poly_lid
            for IntClass in np.unique(GeoJson["IntClassOrdered_{}".format(StrDate)]):
                mclass = folium.Map(location=[centroid.x, centroid.y], zoom_start=12)
#                    for index_list in self.IntClass2Roads[IntClass]:
                if isinstance(index_list,int):
                    index_list = [index_list]
                # Filter GeoDataFrame for roads with indices in the current list
                filtered_gdf = GeoJson.groupby("IntClass_{}".format(StrDate)).get_group(IntClass)
#                    index_list = self.IntClass2Roads[IntClass]
#                    filtered_gdf = self.GeoJson[self.GeoJson['poly_lid'].isin(index_list)]
                # Create a feature group for the current layer
                layer_group = folium.FeatureGroup(name=f"Layer {IntClass}").add_to(m)
                layer_group_class = folium.FeatureGroup(name=f"Layer {IntClass}").add_to(mclass)
                # Add roads to the feature group with a unique color
                for _, road in filtered_gdf.iterrows():
                    color = 'blue'  # Choose a color for the road based on index or any other criterion
                    if road.geometry is not None:
                        folium.GeoJson(road.geometry, style_function=lambda x: {'color': Class2Color[IntClass2StrClass[IntClass]]}).add_to(layer_group)
                
                # Add the feature group to the map
                layer_group.add_to(m)
                layer_group_class.add_to(mclass)
                # Add layer control to the map
                folium.LayerControl().add_to(m)
                folium.LayerControl().add_to(mclass)
                mclass.save(os.path.join(PlotDir,"Subnets_{0}_{1}.html".format(StrDate,IntClass)))
            # Save or display the map
            m.save(os.path.join(PlotDir,"Subnets_{}.html".format(StrDate)))
            Message = "Plotting Daily Subnetworks in HTML: True"
        else:
            Message = "Plotting Daily Subnetworks in HTML: Already Plotted"
            print("Subnets already Plotted in HTML")
    else:
        Message = "Plotting Daily Subnetworks in HTML: False"
        print("No Subnetworks to Plot")
    return Message

def PlotFluxesHTML(GeoJson,TimedFluxes,centroid,StrDate,PlotDir,ReadTime2FluxesBool,NameFluxesFile = "Fluxes",NameTFFile = "TailFrontFluxes",NameFTFile = "FrontTailFluxes"):
    '''
        Description:
            Plots in .html the map of the bounding box considered.
            For each road color with the fluxes.
                1) FT
                2) TF
                3) TF + FT
    '''
    if ReadTime2FluxesBool:
        print("Plotting Daily Fluxes in HTML")
        if not os.path.isfile(os.path.join(PlotDir,"Fluxes_{}.html".format(StrDate))):
            # Create a base map
            m = folium.Map(location=[centroid.x, centroid.y], zoom_start=12)
            mFT = folium.Map(location=[centroid.x, centroid.y], zoom_start=12)
            mTF = folium.Map(location=[centroid.x, centroid.y], zoom_start=12)
            TF = TimedFluxes
            min_val = min(TF["total_fluxes"])
            max_val = max(TF["total_fluxes"])
            TF = TF.with_columns(pl.col("total_fluxes").apply(lambda x: NormalizeWidthForPlot(x,min_val,max_val), return_dtype=pl.Int64).alias("width_total_fluxes"))
            TF = TF.with_columns(pl.col("n_traj_FT").apply(lambda x: NormalizeWidthForPlot(x,min_val,max_val), return_dtype=pl.Int64).alias("width_n_traj_FT"))
            TF = TF.with_columns(pl.col("n_traj_TF").apply(lambda x: NormalizeWidthForPlot(x,min_val,max_val), return_dtype=pl.Int64).alias("width_n_traj_TF"))
            CopyGdf = GeoJson
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
            m.save(os.path.join(PlotDir,NameFluxesFile + "_{}.html".format(StrDate)))
            mTF.save(os.path.join(PlotDir,NameTFFile +"_{}.html".format(StrDate)))
            mFT.save(os.path.join(PlotDir,NameFTFile + "_{}.html".format(StrDate)))
            Message = "Plotting Daily Fluxes in HTML: True"
        else:
            Message = "Plotting Daily Fluxes in HTML: Already Plotted"
            print("Fluxes already Plotted in HTML")
    else:
        Message = "Plotting Daily Fluxes in HTML: False"
        print("No Fluxes to Plot")
    return Message

def PlotTimePercorrenceHTML(GeoJson,VelTimePercorrenceClass,IntClass2BestFit,ReadGeojsonBool,ReadVelocitySubnetBool,centroid,PlotDir,StrDate,Class2Color,NameAvSpeed = "AvSpeed",NameTimePercorrenceFile = "TimePercorrence",verbose = False):
    """
        Description:
            Plots in .html the map of the bounding box considered.
            For each class color the road subnetwork according to time of percorrence.
    """
    if ReadGeojsonBool and ReadVelocitySubnetBool:
        print("Plotting Daily Fluxes in HTML")
        if not os.path.isfile(os.path.join(PlotDir,"AvSpeed_{}.html".format(StrDate))):            
            # Create a base map
            m = folium.Map(location=[centroid.x, centroid.y], zoom_start=12)
            m1 = folium.Map(location=[centroid.x, centroid.y], zoom_start=12)
            for time,RTV in RoadsTimeVel.groupby("start_bin"):
                layer_group = folium.FeatureGroup(name=f"Layer {time}").add_to(m)
                layer_group1 = folium.FeatureGroup(name=f"Layer {time}").add_to(m)
                for Class in IntClass2BestFit.keys():
                    RoadsTimeVel = VelTimePercorrenceClass[Class]
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
                    filtered_gdf = GeoJson[GeoJson['poly_lid'].isin(list_colored_roads_speed)]
                    filtered_gdf["width_speed"] = RoadsTimeVel["width_speed"]
                    filtered_gdf["width_time"] = RoadsTimeVel["width_time"]
                    for idx, row in filtered_gdf.iterrows(): 
                        folium.GeoJson(row.geometry,style_function=lambda x: {
                                        'color': Class2Color[Class],
                                        'weight': row['width_speed']}).add_to(layer_group)                    
                        folium.GeoJson(row.geometry,style_function=lambda x: {
                                        'color': Class2Color[Class],
                                        'weight': row['width_time']}).add_to(layer_group1)                    

                    # Add the feature group to the map
                    layer_group.add_to(m)
                    layer_group.add_to(m1)

                # Add layer control to the map
                folium.LayerControl().add_to(m)
                folium.LayerControl().add_to(m1)

            # Save or display the map
            m.save(os.path.join(PlotDir,NameAvSpeed + "_{}.html".format(StrDate)))
            m1.save(os.path.join(PlotDir,NameTimePercorrenceFile + "_{}.html".format(StrDate)))
            Message = "Plotting Daily Fluxes in HTML: True"
        else:
            Message = "Plotting Daily Fluxes in HTML: Already Plotted"
            print("AvSpeed already Plotted in HTML")
    else:
        Message = "Plotting Daily Fluxes in HTML: False"
        print("No AvSpeed to Plot")
    return Message


# -------------------------- SPECIFIC ALL DAYS ----------------------------#
def ComputeAggregatedMFDVariables(ListDailyNetwork,MFDAggregated):
    """
        Description:
            Every Day I count for each hour, how many people and the speed of the 
            1. Network -> MFDAggregated = {"population":[],"time":[],"speed_kmh":[]}
            2. SubNetwork -> Class2MFDAggregated = {StrClass: {"population":[sum_i pop_{t0,dayi},...,sum_i pop_{iteration,dayi}],"time":[t0,...,iteration],"speed_kmh":[sum_i speed_{t0,dayi},...,sum_i speed_{iteration,dayi}]}}
            NOTE: time is pl.DateTime
        NOTE: Each Time interval has its own average speed and population. For 15 minutes,
            since iteration in 1 Day Analysis is set in that way. 
        NOTE: If at time t there is no population, the speed is set to 0.
    """
    LocalDayCount = 0
    # AGGREGATE MFD FOR ALL DAYS
    for MobDate in ListDailyNetwork:
        if LocalDayCount == 0:
            MFDAggregated = MobDate.MFD
            MFDAggregated["count_days"] = list(np.zeros(len(MFDAggregated["time"])))
            MFDAggregated["total_number_people"] = list(np.zeros(len(MFDAggregated["time"])))
            LocalDayCount += 1
        else:            
            for t in range(len(MobDate.MFD["time"])):
                WeightedSpeedAtTime = MobDate.MFD["speed_kmh"][t]*MobDate.MFD["population"][t]
                PopulationAtTime = MobDate.MFD["population"][t]
                WeightedSpeedAtTimeS = MobDate.MFD["av_speed"][t]*MobDate.MFD["population"][t]
                if PopulationAtTime != 0 and WeightedSpeedAtTime !=0:
                    MFDAggregated["speed_kmh"][t] += WeightedSpeedAtTime
                    MFDAggregated["population"][t] += PopulationAtTime
                    MFDAggregated["count_days"][t] += 1
                    MFDAggregated["av_speed"][t] += WeightedSpeedAtTimeS
                    MFDAggregated["total_number_people"][t] += PopulationAtTime
                else:
                    pass
    for t in range(len(MFDAggregated["time"])):
        if MFDAggregated["count_days"][t] != 0:
            MFDAggregated["speed_kmh"][t] = MFDAggregated["speed_kmh"][t]/(MFDAggregated["count_days"][t]*MFDAggregated["total_number_people"][t])
            MFDAggregated["population"][t] = MFDAggregated["population"][t]/(MFDAggregated["count_days"][t]*MFDAggregated["total_number_people"][t])
            MFDAggregated["av_speed"][t] = MFDAggregated["av_speed"][t]/(MFDAggregated["count_days"][t]*MFDAggregated["total_number_people"][t])
        else:
            pass
    MFDAggregated = Dict2PolarsDF(MFDAggregated,schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed_kmh":pl.Float64,"av_speed":pl.Float64})
    return MFDAggregated
