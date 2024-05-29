from collections import defaultdict
import geopandas as gpd
import numpy as np
import os
import pandas as pd
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
                        "stats": os.path.join(self.InputBaseDir, self.BaseFileName + '_' + self.StrDate + '_' + self.StrDate + '_stats.csv'),
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
            except:
                exit("bounding_box not defined well in config. Should be 'bounding_box': {'lat_min': 44.463121,'lon_min': 11.287085,'lat_max': 44.518165,'lon_max': 11.367472}")
        else:
            self.bounding_box = [(44.463121,11.287085),(44.518165,11.287085),(44.518165,11.367472),(44.463121,11.367472)]
        # FLAGS
        self.ReadTime2Fluxes = False
        self.ReadFluxes = False
        self.ReadFluxesSub = False
        self.ReadStats = False
        self.ReadFcm = False
        self.ReadFcmCenters = False
        self.ReadGeojson = False
        # SETTINGS INFO
        self.colors = ['red','blue','green','orange','purple','yellow','cyan','magenta','lime','pink','teal','lavender','brown','beige','maroon','mint','coral','navy','olive','grey']
        self.Name = BaseName
        self.StrDate = StrDate
        # CLASSES INFO
        self.IntClass2StrClass = defaultdict(dict)
        self.StrClass2IntClass = defaultdict(dict)
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


# --------------- Read Files ---------------- #
    def ReadTimedFluxes(self):
        self.TimedFluxes = pd.read_csv(self.InputBaseDir["timed_fluxes"],delimiter = ';')
        self.ReadTime2Fluxes = True
    
    def ReadFluxes(self):
        self.Fluxes = pd.read_csv(self.InputBaseDir["fluxes"],delimiter = ';')
        self.ReadFluxes = True
    
    def ReadStats(self):
        self.Stats = pd.read_csv(self.InputBaseDir["stats"],delimiter = ';')
        self.ReadStats = True
    
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



##--------------- Plot Network --------------## 

    def PlotSubnetHTML(self):
        if self.ReadFluxesSub and self.ReadGeoJson:
            print("Plotting Daily Subnetworks in HTML")
            # Create a base map
            m = folium.Map()
            # Iterate through the Dictionary of list of poly_lid
            for class_, index_list in self.IntClass2Roads.items():
                # Filter GeoDataFrame for roads with indices in the current list
                filtered_gdf = self.GeoJson[self.GeoJson['poly_lid'].isin(index_list)]
                # Create a feature group for the current layer
                layer_group = folium.FeatureGroup(name=f"Layer {class_}")
                
                # Add roads to the feature group with a unique color
                for _, road in filtered_gdf.iterrows():
                    color = 'blue'  # Choose a color for the road based on index or any other criterion
                    folium.GeoJson(road.geometry, style_function=lambda x: {'color': color}).add_to(layer_group)
                
                # Add the feature group to the map
                layer_group.add_to(m)

            # Add layer control to the map
            folium.LayerControl().add_to(m)

            # Save or display the map
            m.save(os.path.join(self.PlotDir,"Subnets_{}.html".format(self.StrDate)))

        else:
            print("No Subnetworks to Plot")
            return False


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



# --------------- FITTING  ---------------- #   
# --------------- ALAGGREGATED CLASSES  ---------------- #
    def RetrieveGuessParametersPerLabel(self):
        """
            TODO
            Description:
                Retrieves the initial guess parameters for each label (time,length,av_speed) and the 
                corresponnding functions to try.
        """
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
        """

def GetDistributionPerClass(fcm,label,class_):
    """
        Input:
            label: str -> time, length, av_speed, p, a_max
        Returns:
            n, bins of velocity distribution
    """
    n, bins = np.histogram(fcm.groupby("class").get_group(label), bins = bins)




        
day_in_sec = 24*3600
dt = 15*60
iterations = day_in_sec/dt



