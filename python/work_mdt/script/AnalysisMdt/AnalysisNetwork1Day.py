'''
    NOTE: Stats is Useless
'''
from collections import defaultdict
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from shapely.geometry import box
import folium
import datetime

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

def StrDate2DateFormatLocalProject(StrDate):
    return StrDate.split("_")[0],StrDate.split("_")[1],StrDate.split("_")[2]
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
                        "timed_fluxes": os.path.join(self.InputBaseDir,self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + '_timed_fluxes.csv'),
                        "fluxes": os.path.join(self.InputBaseDir,"weights",self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + '.fluxes'),
                        "fluxes_sub": os.path.join(self.InputBaseDir,"weights",self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + '.fluxes.sub')}
        if "geojson_file" in config.keys():
            self.GeojsonDirFile = os.path.join(config["geojson_file"])
        else:
            self.GeojsonDirFile = os.path.join(os.environ['WORKSPACE'],"city-pro","city-pro-carto.geojson")
        
        self.PlotDir = os.path.join(os.environ['WORKSPACE'],"city-pro","output","bologna_mdt_detailed","plots",self.StrDate)
        if not os.path.exists(self.PlotDir):
            os.makedirs(self.PlotDir)
        # BOUNDING BOX
        if "bounding_box" in config.keys():
            try:
                self.bounding_box = [(config["bounding_box"]["lat_min"],config["bounding_box"]["lon_min"]),(config["bounding_box"]["lat_max"],config["bounding_box"]["lon_min"]),(config["bounding_box"]["lat_max"],config["bounding_box"]["lon_max"]),(config["bounding_box"]["lat_min"],config["bounding_box"]["lon_max"])]
                bbox = box((config["bounding_box"]["lat_min"],config["bounding_box"]["lon_min"],config["bounding_box"]["lat_max"],config["bounding_box"]["lon_max"]))
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
        self.iterations = day_in_sec/dt
        yy,mm,dd = StrDate2DateFormatLocalProject(self.StrDate)
        self.Date = datetime.datetime(yy,mm,dd,0,0,0)
        self.TimeStampDate = datetime.datetime.timestamp(self.Date)
        
        # FLAGS
        self.ReadTime2Fluxes = False
        self.ReadFluxes = False
        self.ReadFluxesSub = False
        self.ReadFcm = False
        self.ReadFcmCenters = False
        self.ReadGeojson = False
        self.ReadVelocitySubnet = False
        self.BoolStrClass2IntClass = False
        # SETTINGS INFO
        self.colors = ['red','blue','green','orange','purple','yellow','cyan','magenta','lime','pink','teal','lavender','brown','beige','maroon','mint','coral','navy','olive','grey']
        self.Name = BaseName
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
        self.MFD = pd.DataFrame({"time":[],"population":[],"speed":[]})
        self.Class2MFD = {class_:pd.DataFrame({"time":[],"population":[],"speed":[]}) for class_ in self.IntClass2StrClass.keys()}



# --------------- Read Files ---------------- #
    def ReadTimedFluxes(self):
        self.TimedFluxes = pd.read_csv(self.InputBaseDir["timed_fluxes"],delimiter = ';')
        self.ReadTime2Fluxes = True
    
    def ReadFluxes(self):
        self.Fluxes = pd.read_csv(self.InputBaseDir["fluxes"],delimiter = ';')
        self.ReadFluxes = True
    
    
    def ReadFcm(self):
        self.Fcm = pd.read_csv(self.InputBaseDir["fcm"],delimiter = ';')
        self.ReadFcm = True
    def ReadFcmCenters(self):
        Features = {"class":[],"av_speed":[],"v_min":[],"v_max":[],"sinuosity":[]}
        FcmCenters = pd.read_csv(self.InputBaseDir["fcm_centers"],delimiter = ';')
        keyidx = 0
        for feat in FcmCenters.columns:
            if keyidx == 0:
                Features[Features.keys()[keyidx]] = int(feat)
            else:
                Features[Features.keys()[keyidx]] = float(feat)
            keyidx += 1
        self.FcmCenters = pd.DataFrame(Features)
        self.ReadFcmCenters = True
    def ReadFluxesSub(self,verbose = False):
        '''
            Input:
                FluxesSubFile: (str) -> FluxesSubFile = '../{basename}_{start}_{start}/fluxes.sub'
                verbose: (bool) -> verbose = False
            Output:
                self.IntClass2Roads: (dict) -> self.IntClass2Roads = {IntClass:[] for IntClass in self.IntClasses}
                self.IntClass2RoadsInit: (bool) -> Boolean value to Say I have stored the SubnetInts For each Class
        '''
        DoNothing = False
        self.IntClass2Roads = defaultdict(list)
        # Read Fluxes.sub
        with open(self.DictDirInput["fluxes_sub"],'r') as f:
            FluxesSub = f.readlines()
        for ClassLines in FluxesSub:
            ClassandID = ClassLines.split('\t')
            ClassId  = ClassandID[0].split('_')[1]
            if verbose:
                print("Class: ",ClassId)
            try:
                ClassFractionRoadsConsidered = ClassandID[0].split('_')[2]
                if verbose:
                    print("Fraction of roads considered: ",ClassFractionRoadsConsidered)
            except IndexError:
                DoNothing = True
                if verbose:
                    print("Considering the Total Subnetwork indipendent on the Subclass")
            if DoNothing:
                pass
            else:
                IdRoads = [int(RoadId) for RoadId in ClassandID[1:] if RoadId != '\n']
                self.IntClass2Roads[int(ClassId)] = IdRoads
                if verbose:
                    print("Number of Roads SubNetwork: ",len(IdRoads))     
        self.ReadFluxesSub = True

    def ReadGeoJson(self):
        if not os.path.exists(GeoJsonFile):
            exit("GeoJsonFile not found")
        self.GeoJson = gpd.read_file(GeoJsonFile)
        self.ReadGeoJson = True

    def GetIncreasinglyIncludedSubnets(self):
        self.DictSubnetsTxtDir = defaultdict(dict)
        for Class in self.IntClass2StrClass.keys():
            self.DictSubnetsTxtDir[Class] = os.path.join(self.InputBaseDir,self.BaseFileName+'_'+ self.StrDate+'_'+ self.StrDate + '{}_class_subnet.txt'.format(Class))
        self.ReadFluxesSubIncreasinglyIncludedIntersection()

    def ReadFluxesSubIncreasinglyIncludedIntersection(self,verbose = False):
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
            self.IntClass2RoadsIncreasinglyIncludedIntersection[Class] = [int(Road) for Road in FluxesSub.split(" ")]

#--------- COMPLETE GEOJSON ------- ##
    def CompleteGeoJsonWithClassInfo(self):
        """
            Computes "IntClassOrdered" and "StrClassOrdered" columns for the Geojson.
            Useful when I want to reconstruct the road network for all the days.
        """
        if self.ReadGeojson and self.ReadFluxesSubIncreasinglyIncludedIntersection:
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
        self.ReadVelocitySubnet = True

##--------------- Plot Network --------------## 
    def PlotIncrementSubnetHTML(self):
        """
            Description:
                Plots the subnetwork. Considers the case of intersections
        """
        if self.ReadFluxesSub and self.ReadGeoJson:
            print("Plotting Daily Subnetworks in HTML")
            # Create a base map
            m = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
            # Iterate through the Dictionary of list of poly_lid
            for class_, index_list in self.IntClass2RoadsIncreasinglyIncludedIntersection.items():
                # Filter GeoDataFrame for roads with indices in the current list
                filtered_gdf = self.GeoJson[self.GeoJson['poly_lid'].isin(index_list)]
                # Create a feature group for the current layer
                layer_group = folium.FeatureGroup(name=f"Layer {class_}")
                # Add roads to the feature group with a unique color
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
        if self.ReadFluxesSub and self.ReadGeoJson:
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
        if self.ReadTime2Fluxes:
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
        if self.ReadGeojson and self.ReadVelocitySubnet:
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
def FilterStatsByClass(fcm,i,idx,stats):
    fcm_idx = fcm[i].groupby('class').get_group(idx)['id_act'].to_numpy()
    mask_idx = [True if x in fcm_idx else False for x in stats[i]['id_act'].to_numpy()]
    f_idx = stats[i].loc[mask_idx]
    f_idx = f_idx.sort_values(by = 'start_time')
    return f_idx

    def PlotMFD(self):
        if self.ReadFcm:
            if "start_time" in self.Fcm.columns:
                # ALL TOGETHER MFD
                for t in range(int(self.iterations)):
                    mask_idx = [True if (int(x['start_time'])>int(self.TimeStampDate)+t*self.dt and int(x['start_time'])<int(self.TimeStampDate)+(t+1)*self.dt) or (int(x['end_time'])>int(self.TimeStampDate)+t*self.dt and int(x['end_time'])<int(self.TimeStampDate)+(t+1)*self.dt) else False for _,x in self.Fcm.iterrows()]
                    TmpFcm = self.Fcm[mask_idx]
                    # TODO timstamp2datetime
                    self.MFD["time"].append()
                    self.MFD["population"].append(len(TmpFcm))
                    self.MFD["speed"].append(np.mean(TmpFcm["av_speed"]))
                # PER CLASS
                for t in range(int(self.iterations)):
                    for Class in self.Class2MFD.keys():
                        TmpClassFCM = self.FCM.groupby("class").get_group(Class)[mask_idx]
                        mask_idx = [True if (int(x['start_time'])>int(self.TimeStampDate)+t*self.dt and int(x['start_time'])<int(self.TimeStampDate)+(t+1)*self.dt) or (int(x['end_time'])>int(self.TimeStampDate)+t*self.dt and int(x['end_time'])<int(self.TimeStampDate)+(t+1)*self.dt) else False for _,x in TmpClassFCM.iterrows()]
                        TmpClassFCM = TmpClassFCM[mask_idx]
                        # TODO timstamp2datetime
                        self.Class2MFD[Class]["time"].append() 
                        self.Class2MFD[Class]["population"].append(len(TmpClassFCM))
                        self.Class2MFD[Class]["speed"].append(np.mean(TmpClassFcm["av_speed"]))
                    
            else:
                pass
        elif self.ReadStats:
            if "start_time" in self.Stats.columns:
                mask_idx = [True if (int(x['start_time'])>int(self.TimeStampDate)+t*self.dt and int(x['start_time'])<int(self.TimeStampDate)+(t+1)*self.dt) or (int(x['end_time'])>int(self.TimeStampDate)+t*self.dt and int(x['end_time'])<int(self.TimeStampDate)+(t+1)*self.dt) else False for _,x in self.Stats.iterrows()]

        

##--------------- Dictionaries --------------##
    def CreateDictionaryIntClass2StrClass(self):
        '''
        Input:
            fcm: dataframe []

        Output: dict: {velocity:'velocity class in words: (slowest,...quickest)]}
        '''
        if self.ReadFcmCenters:
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
        if self.ReadFcmCenters:
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
                            self.InitialGuessPerClassAndLabel[class_][label][function] = (,self.Fcm.groupby("class").get_group(class_)[label])
                        else:
                            self.InitialGuessPerClassAndLabel[class_][label][function] = (,self.Fcm.groupby("class").get_group(class_)[label])
        else:
            print("FcmCenters not read Not retrieving parameters")
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



