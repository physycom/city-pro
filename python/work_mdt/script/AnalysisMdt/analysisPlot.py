import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import folium
import polars as pl
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

def MFDByClass(population,velocity,dict_name,idx,save_dir,verbose = False): 

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




##----------------------------------- PLOT DISTANCES -----------------------------------##




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
            for IntClass in IntClass2StrClass.keys():
                mclass = folium.Map(location=[centroid.x, centroid.y], zoom_start=12)
#                    for index_list in self.IntClass2RoadsIncreasinglyIncludedIntersection[IntClass]:
                # Filter GeoDataFrame for roads with indices in the current list
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
            for IntClass in IntClass2StrClass.keys():
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