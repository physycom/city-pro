"""
Multi-Day Network Analysis for Urban Mobility Data Processing
===========================================================

This module provides comprehensive analysis capabilities for urban mobility networks
across multiple days using trajectory data from mobility datasets. It processes FCM 
(Fuzzy C-Means) clustering results and performs temporal comparative analysis.

MAIN CLASS: NetworkAllDays
==========================

PURPOSE:
--------
Analyzes urban mobility patterns across multiple days by aggregating and comparing
daily network statistics, computing temporal trends, and identifying persistent
mobility patterns including:
- Cross-day class mapping and trajectory clustering consistency
- Temporal evolution of traffic indicators and congestion patterns
- Multi-day statistical distribution fitting and model comparison
- Network topology changes and subnet persistence analysis
- Traffic penetration estimation and validation against open data

INPUT DATA REQUIREMENTS:
-----------------------
- List of DailyNetworkStats objects: pre-computed daily analysis results
- Configuration dictionary: analysis parameters, date ranges, and spatial bounds
- Holiday/weekday classification: temporal aggregation specifications
- Open traffic data: validation datasets for penetration analysis
- Network topology: persistent road network structure across days

KEY FEATURES:
------------

1. TEMPORAL CLASS MAPPING & CONSISTENCY:
   - Maps mobility classes across days based on average speed similarity
   - Generates consistent string class labels for cross-day comparison
   - Handles varying numbers of classes per day through reference alignment
   - Creates day-to-class mapping dictionaries for temporal analysis

2. CROSS-DAY AGGREGATION & STATISTICAL ANALYSIS:
   - Concatenates trajectory data across multiple temporal aggregation levels
   - Performs multi-day statistical distribution fitting (exponential/power-law)
   - Computes aggregated fit parameters and model selection metrics
   - Handles holiday vs. weekday vs. overall aggregation strategies

3. TRAFFIC EVOLUTION & TEMPORAL PATTERNS:
   - Computes daily traffic indicators and congestion evolution
   - Analyzes population-time relationships across multiple days
   - Performs CFAR-based traffic anomaly detection over time
   - Calculates temporal variance in mobility fundamental diagrams

4. NETWORK TOPOLOGY & SUBNET ANALYSIS:
   - Computes union and intersection of mobility subnets across days
   - Analyzes subnet persistence and temporal stability
   - Creates incremental inclusion maps for class-specific road networks
   - Generates interactive HTML visualizations of temporal network changes

5. COMPARATIVE VISUALIZATION & ANALYSIS:
   - Produces grid plots comparing distributions across days
   - Creates time-series analysis of mobility parameters
   - Generates comparative MFD plots for different user classes
   - Plots parameter evolution (fitting coefficients) over time

6. HETEROGENEITY & SCALING ANALYSIS:
   - Implements heterogeneous mobility analysis across classes
   - Computes scaling relationships between class size and mobility features
   - Analyzes class contribution to overall mobility distributions
   - Performs reconstruction analysis from class-specific patterns

7. TRAFFIC PENETRATION & VALIDATION:
   - Estimates traffic penetration rates using network flux data
   - Validates against municipal open traffic datasets
   - Computes temporal correlation with official traffic measurements
   - Performs statistical hypothesis testing for data quality assessment

ANALYSIS WORKFLOW:
-----------------
1. Initialize with list of daily network objects and configuration
2. Map classes across days using speed-based similarity metrics
3. Concatenate trajectory data by temporal aggregation levels
4. Perform cross-day statistical fitting and model comparison
5. Compute temporal evolution of traffic and network indicators
6. Analyze subnet persistence and network topology changes
7. Generate comprehensive comparative visualizations
8. Validate results against external traffic datasets
9. Export aggregated results and temporal trend analysis

AGGREGATION LEVELS:
------------------
- "holidays": Analysis restricted to holiday periods
- "not_holidays": Analysis for regular weekdays/weekends
- "aggregated": Complete dataset analysis across all days

SCIENTIFIC APPLICATIONS:
-----------------------
- Temporal stability analysis of urban mobility patterns
- Long-term traffic trend identification and forecasting
- Transportation policy impact assessment over time
- Urban network resilience and adaptation analysis
- Multi-modal transportation system optimization
- Data quality validation and penetration rate estimation

TECHNICAL DEPENDENCIES:
----------------------
- AnalysisNetwork1Day: Single-day analysis foundation
- GeoPandas: Spatial data processing and network analysis
- Polars/Pandas: High-performance temporal data manipulation
- Folium: Interactive temporal mapping and visualization
- NumPy/SciPy: Statistical analysis and temporal modeling
- Matplotlib: Comprehensive temporal plotting capabilities

CONFIGURATION REQUIREMENTS:
---------------------------
- StrDates: List of date strings for analysis
- holidays/not_holidays: Date classification for aggregation
- InputBaseDir: Base directory for input data files
- Spatial bounding box: Geographic extent for analysis
- Statistical fitting parameters: Model preferences and thresholds
- Temporal aggregation settings: Cut-off times, bin sizes, etc.
- Open data validation sources: URLs or file paths for traffic datasets
- Feature2ScaleCount: Scaling parameters for feature distributions
- Feature2ScaleBins: Binning parameters for feature distributions
- Feature2MaxBins: Maximum bin counts for feature distributions
- Feature2ShiftBin/Count: Parameters for shifting feature distributions
- Feature2IntervalBin/Count: Parameters for interval-based analysis
- Feature2Label: Human-readable labels for features
- Feature2SaveName: Filenames for saving feature distributions
- Feature2Legend: Legends for feature plots
- Feature2IntervalBin: Interval binning parameters for features
- Feature2ShiftBin: Shift binning parameters for features
- Feature2ShiftCount: Shift count parameters for features
- Feature2MaxBins: Maximum bin counts for features
- Feature2Function2Fit2InitialGuess: Initial guess for fitting functions
- Feature2Class2AllFitTry: Dictionary of all fitting attempts per feature
- Feature2Class2Function2Fit2InitialGuess: Initial guess for class-specific fitting functions
- Feature2Class2AllFitTry: Dictionary of all fitting attempts per class
- Feature2Class2Function2Fit2InitialGuess: Initial guess for class-specific fitting functions
- Feature2Class2AllFitTry: Dictionary of all fitting attempts per class
OUTPUT FILES:
------------
- Cross-day statistical fit parameters (CSV/JSON)
- Temporal evolution plots and time-series analysis (PNG)
- Interactive subnet evolution maps (HTML)
- Traffic indicator aggregations and trends (CSV)
- Penetration analysis and validation reports
- Comparative distribution and MFD visualizations

AUTHORS: Alberto Amaduzzi
LAST UPDATED: [12/06/2025]
"""
from AnalysisNetwork1Day import *
from analysisPlot import *
from collections import defaultdict
import numpy as np
from LatexFunctions import *
from UsefulStructures import *
import contextily as ctx
from Heterogeneity import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NetworkAllDays:
    
    def __init__(self,ListDailyNetwork,PlotDir,config,verbose = False): 
        # Analysis flags
        self.AssociateAvSpeed2StrClassBool = False
        self.ConcatenatePerClassBool = False
        self.CreateClass2SubNetAllDaysBool = False
        self.ComputedMFDAggregatedVariablesBool = False
        self.AddStrClassColumn2FcmBool = False
        # Settings
        self.verbose = verbose
        self.PlotDir = PlotDir
        # Initialization all days
        self.ListDailyNetwork = ListDailyNetwork       
        #
        self.CountFunctions = 0
        self.LogFile = os.path.join(self.PlotDir,"LogFile.txt")

        self.Day2Feature2MaxBins = {MobDate.StrDate:defaultdict() for MobDate in self.ListDailyNetwork}
        LocalCount = 0
        # Initialize List of Reference For StrClasses
        self.InitListStrClassReference()        
        self.Day2IntClass2StrClass = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}
        self.Day2StrClass2IntClass = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}
        self.DictClass2AvSpeed = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}
        #
        self.AggregationLevel = ["aggregated"]#["holidays","not_holidays","aggregated"]
        self.AggregationLevel2ListDays = {"holidays": config["holidays"],
                                          "not_holidays": config["not_holidays"],
                                          "aggregated": config["StrDates"]}
        self.StrDates = [] 
        for MobDate in self.ListDailyNetwork:
            self.StrDates.append(MobDate.StrDate)         
            self.Day2Feature2MaxBins[MobDate.StrDate] = MobDate.Feature2MaxBins
            # CONSTANT PARAMETERS
            if LocalCount == 0:
                self.IntClasses = list(MobDate.IntClass2StrClass.keys())
                self.Feature2ScaleCount = MobDate.Feature2ScaleCount
                self.Feature2ScaleBins = MobDate.Feature2ScaleBins
                self.centroid = MobDate.centroid
                self.Class2Color = MobDate.Class2Color
                self.ListColors = MobDate.ListColors
                # Fitting Parameters
                self.Features2Fit = MobDate.Features2Fit
                # PLot Parameters
                self.Feature2Label = MobDate.Feature2Label
                self.Feature2SaveName = MobDate.Feature2SaveName
                self.Feature2Legend = MobDate.Feature2Legend
                self.Feature2IntervalBin = MobDate.Feature2IntervalBin
                self.Feature2IntervalCount = MobDate.Feature2IntervalCount
                self.Feature2ShiftBin = MobDate.Feature2ShiftBin
                self.Feature2ShiftCount = MobDate.Feature2ShiftCount
                self.Feature2MaxBins = MobDate.Feature2MaxBins   
                #
                self.StrClass2MFDAggregated = defaultdict() 
                self.StrClass2MFDNewAggregated = defaultdict()                 
                self.StrClass2MFDAggregated2Plot = defaultdict()
                self.StrClass2MFDNewAggregated2Plot = defaultdict()
                #
                self.bounding_box = MobDate.BoxNumeric
                #
                for StrClass in self.ListStrClassReference:
                    self.StrClass2MFDAggregated[StrClass] = {Key: [] for Key in MobDate.MFD.columns}
                    self.StrClass2MFDNewAggregated[StrClass] = {Key: [] for Key in MobDate.MFD.columns}
                    self.StrClass2MFDAggregated2Plot[StrClass] = {Key: [] for Key in MobDate.MFD2Plot.columns}
                    self.StrClass2MFDNewAggregated2Plot[StrClass] = {Key: [] for Key in MobDate.MFD2Plot.columns}

                self.config = MobDate.config
        # MFD
        self.Aggregation2MFD = {Aggregation:{MobDate.StrDate:MobDate.MFD for MobDate in self.ListDailyNetwork if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]} for Aggregation in self.AggregationLevel}
        # FIT
        self.StrDay2Color = {StrDay: self.ListColors[i] for i,StrDay in enumerate(self.StrDates)}
        # Set The Plot Lim as The Maximum Among all Days
        for Day in self.Day2Feature2MaxBins.keys():
            for Feature in self.Day2Feature2MaxBins[Day].keys():
                for Bins in self.Day2Feature2MaxBins[Day][Feature].keys():
                    self.Feature2MaxBins[Feature][Bins] = max(self.Feature2MaxBins[Feature][Bins],self.Day2Feature2MaxBins[Day][Feature][Bins])
        # Map The Classes among different days according to the closest average speed    
        self.Day2IntClass2StrClass,self.Day2StrClass2IntClass,self.DictClass2AvSpeed = GenerateDay2DictClassAvSpeed(self.ListDailyNetwork,self.ReferenceFcmCenters,self.DictClass2AvSpeed,self.RefIntClass2StrClass,self.Day2IntClass2StrClass,self.Day2StrClass2IntClass)    
        SaveMapsDayInt2Str(self.Day2IntClass2StrClass,self.Day2StrClass2IntClass,self.DictClass2AvSpeed,self.PlotDir)
        self.AddStrClassColumn2Fcm()
        self.AssociateAvSpeed2StrClass()
        # Dictionaries Depending on the Map day 2 common classes
        self.Aggregation2Fcm = {Aggregation: pl.DataFrame() for Aggregation in self.AggregationLevel}
        self.Aggregation2Class2Fcm = {Aggregation: {StrClass: pl.DataFrame() for StrClass in self.Day2StrClass2IntClass[self.DayReferenceClasses].keys()} for Aggregation in self.AggregationLevel}
        self.Aggregation2DictFittedData = {Aggregation:defaultdict(dict) for Aggregation in self.Aggregation2Class2Fcm.keys()}
        self.Aggregation2Class2InfoFittedParameters = {Aggregation:{StrClass: defaultdict(dict) for StrClass in self.Aggregation2Class2Fcm[Aggregation]} for Aggregation in self.Aggregation2Class2Fcm.keys()}
        # 
        self.GpdClasses = None
        # Cut Time To avoid Midnight Where Data are Noisy
        self.CutIndexTime = 8  
        #
        self.Colors = ["red","blue","green","black","yellow","orange","purple","pink","brown","cyan"]
        self.Markers = ["o","s","^","v","<",">","1","2","3","4"]
        self.Day2Marker = {Day: self.Markers[i] for i,Day in enumerate(self.StrDates)}
    def InitListStrClassReference(self):
        # Take The Day with More Classes.
        NumberClasses = 0
        self.DayReferenceClasses = ""
        for MobDate in self.ListDailyNetwork:
            if len(MobDate.StrClass2IntClass) > NumberClasses:
                NumberClasses = len(MobDate.StrClass2IntClass)
                self.StrClass2IntClass = MobDate.StrClass2IntClass
                self.DayReferenceClasses = MobDate.StrDate
                self.ReferenceFcmCenters = MobDate.FcmCenters
                self.RefIntClass2StrClass = MobDate.IntClass2StrClass
            else:
                pass
        self.ListStrClassReference = list(self.StrClass2IntClass.keys())
        self.CountFunctions += 1
        Message = "Function: InitListStrClassReference\n"
        Message += "\tListStrClassReference: {}".format(self.ListStrClassReference)
        AddMessageToLog(Message,self.LogFile)

# USEFUL DICTIONARIES FOR COMPAriSON AMOnG DIFFErent DAYS    
    def AddStrClassColumn2Fcm(self):
        """
            NOTE: Fondmental for the MFD, Fitting and Aggregation Descriptions
            Description:
                Adds to each Fcm the right Class

        """
        for MobDate in self.ListDailyNetwork:
            StrDate = MobDate.StrDate
            MobDate.Fcm = MobDate.Fcm.with_columns(pl.col("class").apply(lambda x: self.Day2IntClass2StrClass[StrDate][x], return_dtype=pl.Utf8).alias("str_class"))
            MobDate.Fcm = MobDate.Fcm.with_columns(pl.col("class_new").apply(lambda x: self.Day2IntClass2StrClass[StrDate][x], return_dtype=pl.Utf8).alias("str_class_new"))
            MobDate.FcmCenters = MobDate.FcmCenters.with_columns(pl.col("class").apply(lambda x: self.Day2IntClass2StrClass[StrDate][x], return_dtype=pl.Utf8).alias("str_class"))
        self.CountFunctions += 1
        Message = "AddStrClassColumn2Fcm: True\n"
        self.AddStrClassColumn2FcmBool = True
        AddMessageToLog(Message,self.LogFile)

    def AssociateAvSpeed2StrClass(self):
        """
            Description:
                1. Create a dictionary that associates the class to the closest speed.
                2. Choose a reference day, that is one with most classes in it.
                3. Compare each FcmCenter (For each day) to the reference day.
                4. Create Day2IntClass2StrClass that associate the string of the velocity class that is  closer
                to the reference day in speed.
                5. Add column str_class to FcmCenter and Fcm in each day.
            Output:
                1. self.Day2IntClass2StrClass = {day: {
                                                    Intclass: StrClass
                                                    }
                                              }
                2. self.Day2StrClass2IntClass = {day: {
                                                    Strclass: IntClass
                                                    }
                                              }
                3. self.DictClass2AvSpeed = {day: {
                                                    Strclass: AvSpeed
                                                    }
                                              }
                4. self.Fcm [str_class]
                5. self.FcmCenters [str_class]
                6. self.ListStrClassReference = [str_class]
            NOTE: To use for comparative analysis among days.

        """
        # Each day.GenerateDay2DictClass2AvSpeed
        self.AssociateAvSpeed2StrClassBool = True
        self.CountFunctions += 1
        MessageAveSpeedStrClass(self.Day2IntClass2StrClass,self.Day2StrClass2IntClass,self.DictClass2AvSpeed,self.LogFile)

    
    def ConcatenateFcm(self):
        """
            Description:
                Contains information about Speed,Time,... for the different aggregation levels.
                Aggregation in [holidays, not_holidays, aggregated]
                They are useful for statistics on the distribution of the features.
            Output:
                Aggregation2Fcm =
                {Aggregation: FcmConcatenated By Aggregated Days}
                Aggregation2Class2Fcm = {Aggregation: {StrClass: FcmConcatenated By Aggregated Days}}

                Day2Class2Fcm = {day: {Strclass: {time: [timetraj0,...,timetrajN], lenght: [lenghttraj0,...,lenghttrajN]}}}
            NOTE: The class are grouped by the speed. From AssociateAvSpeed2StrClass()
        """
        if self.AddStrClassColumn2FcmBool:
            # If Classes are Categorized
            self.CountFunctions += 1
            if self.AssociateAvSpeed2StrClassBool:
                for Aggregation in self.Aggregation2Class2Fcm.keys():
                    for MobDate in self.ListDailyNetwork:
                        if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]:
                            self.Aggregation2Fcm[Aggregation] = pl.concat([self.Aggregation2Fcm[Aggregation],MobDate.Fcm])
                            for StrClass in self.Aggregation2Class2Fcm[Aggregation]:
                            # Append time and lenght of the Iterated Day
                                self.Aggregation2Class2Fcm[Aggregation][StrClass] = pl.concat([self.Aggregation2Class2Fcm[Aggregation][StrClass],MobDate.Fcm.filter(pl.col("str_class") == StrClass)])
                self.ConcatenatePerClassBool = True
                MessageConcatenateFcm(self.Aggregation2Fcm,self.Aggregation2Class2Fcm,self.LogFile)
            else:
                Message = "ConcatenatePerClass: False"
## Put together space and time for all days for each class.
    


    def GetAggregatedFeature2Function2Fit2InitialGuess(self):
        """
            Fill the dictionary with the initial guess for the fit with data from output fit.
            NOTE: Aggregate those days by Aggregation2Fcm
            NOTE: Done
        """
        self.Aggregation2Feature2Function2Fit2InitialGuess = {Aggregation: defaultdict() for Aggregation in self.Aggregation2Class2Fcm.keys()}
        self.Aggregation2Feature2Class2Function2Fit2InitialGuess = {Aggregation: {StrClass: defaultdict() for StrClass in self.Aggregation2Class2Fcm[Aggregation]} for Aggregation in self.Aggregation2Class2Fcm.keys()}
        for Aggregation in self.AggregationLevel:
            for MobDate in self.ListDailyNetwork:
                if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]:
                    for Feature in MobDate.Feature2Class2AllFitTry.keys():
                        self.Aggregation2Feature2Function2Fit2InitialGuess[Aggregation][Feature] = defaultdict()
                        self.Aggregation2Feature2Class2Function2Fit2InitialGuess[Aggregation][Feature] = defaultdict()

                        for Function2Fit in MobDate.Feature2Function2Fit2InitialGuess[Feature].keys():
                            self.Aggregation2Feature2Function2Fit2InitialGuess[Aggregation][Feature][Function2Fit] = MobDate.Feature2Function2Fit2InitialGuess[Feature][Function2Fit]
        for Aggregation in self.AggregationLevel:
            for MobDate in self.ListDailyNetwork:
                if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]:
                    for Feature in MobDate.Feature2Class2AllFitTry.keys():
                        for StrClass in self.Day2StrClass2IntClass[MobDate.StrDate].keys():
                            IntClass = self.Day2StrClass2IntClass[MobDate.StrDate][StrClass]
                            self.Aggregation2Feature2Class2Function2Fit2InitialGuess[Aggregation][Feature][StrClass] = defaultdict()    
                            for Function2Fit in self.Aggregation2Feature2Function2Fit2InitialGuess[Aggregation][Feature].keys():
                                self.Aggregation2Feature2Class2Function2Fit2InitialGuess[Aggregation][Feature][StrClass][Function2Fit] = MobDate.Feature2Class2Function2Fit2InitialGuess[Feature][IntClass][Function2Fit]


    def ComputeAggregatedFit(self):
        """
            Create the dictionary for the Fit (both input and output).
            Put the best_fit, fitted_data, parameters, start_window, end_window, std_error out of the days for the class.
        """
        self.GetAggregatedFeature2Function2Fit2InitialGuess()
        self.Aggregation2Feature2AllFitTry = {Aggregation: InitFeature2AllFitTry(self.Aggregation2Feature2Function2Fit2InitialGuess[Aggregation]) for Aggregation in self.Aggregation2Class2Fcm.keys()}
        for Aggregation in self.Aggregation2Feature2AllFitTry.keys():
            ####################################à
            for Feature in self.Aggregation2Feature2AllFitTry[Aggregation].keys():
                # NOTE: Concatednated Fcm
                ObservedData = self.Aggregation2Fcm[Aggregation][Feature].to_list()
                # Compute the Fit for functions you are Undecided from
                if Feature == "av_speed" or Feature == "speed_kmh":
                    self.Aggregation2Feature2AllFitTry[Aggregation][Feature] = FillIterationFitDicts(ObservedData,
                                                                                                    self.Aggregation2Feature2Function2Fit2InitialGuess[Aggregation][Feature],
                                                                                                    self.Aggregation2Feature2AllFitTry[Aggregation][Feature])
                else:
                    self.Aggregation2Feature2AllFitTry[Aggregation][Feature] = FillIterationFitDictsTimeLength(ObservedData,
                                                                                                    self.Aggregation2Feature2Function2Fit2InitialGuess[Aggregation][Feature],
                                                                                                    self.Aggregation2Feature2AllFitTry[Aggregation][Feature])

    def ConcatenateplExpFits(self):
        """
            @Describe:
                - Aggregate:
                parameters_df:
                    - class - fuzzy - exp
                    - class - pl - exp
                    - class - new - exp
                    - class - pl - exp
                    - aggregated - fuzzy - exp
                    - aggregated - pl - exp
                    - aggregated - new - exp
                    - aggregated - pl - exp
                data_and_fit_df:
                    - class - fuzzy - exp
                    - class - pl - exp
                    - class - new - exp
                    - class - pl - exp
                    - aggregated - fuzzy - exp
                    - aggregated - pl - exp
                    - aggregated - new - exp
                    - aggregated - pl - exp
        """
        for Feature in ["time_hours","lenght_km"]:
            #----------- Parameters -------------#
            # class - fuzzy
            _df_parameters_pl_conditional_on_class = pl.DataFrame()
            _df_parameters_exp_conditional_on_class = pl.DataFrame()
            # class - new
#            _df_parameters_exp_conditional_on_class_new = pl.DataFrame()
#            _df_parameters_pl_conditional_on_class_new = pl.DataFrame()
            
            # aggregated - fuzzy
            _df_parameters_pl = pl.DataFrame()
            _df_parameters_exp = pl.DataFrame()
            # aggregated - new
#            _df_parameters_pl_new = pl.DataFrame()
#            _df_parameters_exp_new = pl.DataFrame()
            # ------------ Data & fit ---------------#
            # Data - class - fuzzy
            _df_fit_and_data_pl = pl.DataFrame()
            _df_fit_and_data_exp = pl.DataFrame()
            # Data - class - new
#            _df_fit_and_data_pl_new = pl.DataFrame()
#            _df_fit_and_data_exp_new = pl.DataFrame()
            # Data - aggregated - fuzzy
            _df_fit_and_data_pl = pl.DataFrame()
            _df_fit_and_data_exp = pl.DataFrame()
            # Data - aggregated - new
#            _df_fit_and_data_pl_new = pl.DataFrame()
#            _df_fit_and_data_exp_new = pl.DataFrame()
            for MobDate in self.ListDailyNetwork:
                StrDate = MobDate.StrDate
                for Aggregation in self.Aggregation2Class2Fcm.keys():
                    if StrDate in self.AggregationLevel2ListDays[Aggregation]:
                        date_dir = os.path.join(self.PlotDir,StrDate)
                        # read params - class - fuzzy - exp
                        if os.path.isfile(os.path.join(date_dir,f"df_parameters_expo_{Feature}_{StrDate}_conditional_class.csv")):
                            df = pl.read_csv(os.path.join(date_dir,f"df_parameters_expo_{Feature}_{StrDate}_conditional_class.csv"))
                            _df_parameters_exp_conditional_on_class = pl.concat([_df_parameters_exp_conditional_on_class,df])
                        # read params - class - fuzzy - pl
                        if os.path.isfile(os.path.join(date_dir,f"df_parameters_pl_{Feature}_{StrDate}_conditional_class.csv")):
                            df = pl.read_csv(os.path.join(date_dir,f"df_parameters_pl_{Feature}_{StrDate}_conditional_class.csv"))
                            _df_parameters_pl_conditional_on_class = pl.concat([_df_parameters_pl_conditional_on_class,df])
                        # read params - class - new - exp
#                        if os.path.isfile(os.path.join(date_dir,f"df_parameters_expo_{Feature}_{StrDate}_conditional_class_new.csv")):
#                            df = pl.read_csv(os.path.join(date_dir,f"df_parameters_expo_{Feature}_{StrDate}_conditional_class_new.csv"))
#                            _df_parameters_exp_conditional_on_class_new = pl.concat([_df_parameters_exp_conditional_on_class_new,df])
                        # read params - class - new - pl
#                        if os.path.isfile(os.path.join(date_dir,f"df_parameters_pl_{Feature}_{StrDate}_conditional_class_new.csv")):
#                            df = pl.read_csv(os.path.join(date_dir,f"df_parameters_pl_{Feature}_{StrDate}_conditional_class_new.csv"))
#                            _df_parameters_pl_conditional_on_class_new = pl.concat([_df_parameters_pl_conditional_on_class_new,df])
                        # read params - aggregated - fuzzy - exp
                        if os.path.isfile(os.path.join(date_dir,f"df_parameters_expo_{Feature}_{StrDate}.csv")):
                            df = pl.read_csv(os.path.join(date_dir,f"df_parameters_expo_{Feature}_{StrDate}.csv"))
                            _df_parameters_exp = pl.concat([_df_parameters_exp,df])
                        # read params - aggregated - fuzzy - pl
                        if os.path.isfile(os.path.join(date_dir,f"df_parameters_pl_{Feature}_{StrDate}.csv")):
                            df = pl.read_csv(os.path.join(date_dir,f"df_parameters_pl_{Feature}_{StrDate}.csv"))
                            _df_parameters_pl = pl.concat([_df_parameters_pl,df])
                        # read params - aggregated - new - exp
#                        if os.path.isfile(os.path.join(date_dir,f"df_parameters_expo_{Feature}_{StrDate}_new.csv")):
#                            df = pl.read_csv(os.path.join(date_dir,f"df_parameters_expo_{Feature}_{StrDate}_new.csv"))
#                            _df_parameters_exp_new = pl.concat([_df_parameters_exp_new,df])
                        # read params - aggregated - new - pl
#                        if os.path.isfile(os.path.join(date_dir,f"df_parameters_pl_{Feature}_{StrDate}_new.csv")):
#                            df = pl.read_csv(os.path.join(date_dir,f"df_parameters_pl_{Feature}_{StrDate}_new.csv"))
#                            _df_parameters_pl_new = pl.concat([_df_parameters_pl_new,df])
                        # read data and fit - class - fuzzy - exp
                        if os.path.isfile(os.path.join(date_dir,f"df_fit_and_data_expo_{Feature}_{StrDate}_conditional_class.csv")):
                            df = pl.read_csv(os.path.join(date_dir,f"df_fit_and_data_expo_{Feature}_{StrDate}_conditional_class.csv"))
                            _df_fit_and_data_exp = pl.concat([_df_fit_and_data_exp,df])
                        # read data and fit - class - fuzzy - pl
                        if os.path.isfile(os.path.join(date_dir,f"df_fit_and_data_pl_{Feature}_{StrDate}_conditional_class.csv")):
                            df = pl.read_csv(os.path.join(date_dir,f"df_fit_and_data_pl_{Feature}_{StrDate}_conditional_class.csv"))
                            _df_fit_and_data_pl = pl.concat([_df_fit_and_data_pl,df])
                        # read data and fit - class - new - exp
 #                       if os.path.isfile(os.path.join(date_dir,f"df_fit_and_data_expo_{Feature}_{StrDate}_conditional_class_new.csv")):
 #                           df = pl.read_csv(os.path.join(date_dir,f"df_fit_and_data_expo_{Feature}_{StrDate}_conditional_class_new.csv"))
 #                           _df_fit_and_data_exp_new = pl.concat([_df_fit_and_data_exp_new,df])
                        # read data and fit - class - new - pl
  #                      if os.path.isfile(os.path.join(date_dir,f"df_fit_and_data_pl_{Feature}_{StrDate}_conditional_class_new.csv")):
  #                          df = pl.read_csv(os.path.join(date_dir,f"df_fit_and_data_pl_{Feature}_{StrDate}_conditional_class_new.csv"))
  #                          _df_fit_and_data_pl_new = pl.concat([_df_fit_and_data_pl_new,df])
                        # read data and fit - aggregated - fuzzy - exp
                        if os.path.isfile(os.path.join(date_dir,f"df_fit_and_data_expo_{Feature}_{StrDate}.csv")):
                            df = pl.read_csv(os.path.join(date_dir,f"df_fit_and_data_expo_{Feature}_{StrDate}.csv"))
                            _df_fit_and_data_exp = pl.concat([_df_fit_and_data_exp,df])
                        # read data and fit - aggregated - fuzzy - pl
                        if os.path.isfile(os.path.join(date_dir,f"df_fit_and_data_pl_{Feature}_{StrDate}.csv")):
                            df = pl.read_csv(os.path.join(date_dir,f"df_fit_and_data_pl_{Feature}_{StrDate}.csv"))
                            _df_fit_and_data_pl = pl.concat([_df_fit_and_data_pl,df])
                        # read data and fit - aggregated - new - exp
#                        if os.path.isfile(os.path.join(date_dir,f"df_fit_and_data_expo_{Feature}_{StrDate}_new.csv")):
#                            df = pl.read_csv(os.path.join(date_dir,f"df_fit_and_data_expo_{Feature}_{StrDate}_new.csv"))
#                            _df_fit_and_data_exp_new = pl.concat([_df_fit_and_data_exp_new,df])
                        # read data and fit - aggregated - new - pl
#                        if os.path.isfile(os.path.join(date_dir,f"df_fit_and_data_pl_{Feature}_{StrDate}_new.csv")):
#                            df = pl.read_csv(os.path.join(date_dir,f"df_fit_and_data_pl_{Feature}_{StrDate}_new.csv"))
#                            _df_fit_and_data_pl_new = pl.concat([_df_fit_and_data_pl_new,df])
            # parameters - pl - class - fuzzy
            if len(_df_parameters_pl_conditional_on_class) > 0:
                _df_parameters_pl_conditional_on_class.write_csv(os.path.join(self.PlotDir,f"df_parameters_pl_{Feature}_conditional_class.csv"))
            # parameters - exp - class - fuzzy
            if len(_df_parameters_exp_conditional_on_class) > 0:
                _df_parameters_exp_conditional_on_class.write_csv(os.path.join(self.PlotDir,f"df_parameters_expo_{Feature}_conditional_class.csv"))
            # parameters - pl - class - new
#            if len(_df_parameters_pl_conditional_on_class_new) > 0:
#                _df_parameters_pl_conditional_on_class_new.write_csv(os.path.join(self.PlotDir,f"df_parameters_pl_{Feature}_conditional_class_new.csv"))
            # parameters - exp - class - new
#            if len(_df_parameters_exp_conditional_on_class_new) > 0:
#                _df_parameters_exp_conditional_on_class_new.write_csv(os.path.join(self.PlotDir,f"df_parameters_expo_{Feature}_conditional_class_new.csv"))
            # parameters - pl - aggregated - fuzzy
            if len(_df_parameters_pl) > 0:
                _df_parameters_pl.write_csv(os.path.join(self.PlotDir,f"df_parameters_pl_{Feature}.csv"))
            # parameters - exp - aggregated - fuzzy
            if len(_df_parameters_exp) > 0:
                _df_parameters_exp.write_csv(os.path.join(self.PlotDir,f"df_parameters_expo_{Feature}.csv"))
            # parameters - pl - aggregated - new
#            if len(_df_parameters_pl_new) > 0:
#                _df_parameters_pl_new.write_csv(os.path.join(self.PlotDir,f"df_parameters_pl_{Feature}_new.csv"))
            # parameters - exp - aggregated - new
#            if len(_df_parameters_exp_new) > 0:
#                _df_parameters_exp_new.write_csv(os.path.join(self.PlotDir,f"df_parameters_expo_{Feature}_new.csv"))
            # data and fit - pl - class - fuzzy
#            if len(_df_fit_and_data_pl) > 0:
#                _df_fit_and_data_pl.write_csv(os.path.join(self.PlotDir,f"df_fit_and_data_pl_{Feature}_conditional_class.csv"))
            # data and fit - exp - class - fuzzy
            if len(_df_fit_and_data_exp) > 0:
                _df_fit_and_data_exp.write_csv(os.path.join(self.PlotDir,f"df_fit_and_data_expo_{Feature}_conditional_class.csv"))
            # data and fit - pl - class - new
#            if len(_df_fit_and_data_pl_new) > 0:
#                _df_fit_and_data_pl_new.write_csv(os.path.join(self.PlotDir,f"df_fit_and_data_pl_{Feature}_conditional_class_new.csv"))
            # data and fit - exp - class - new
#            if len(_df_fit_and_data_exp_new) > 0:
#                _df_fit_and_data_exp_new.write_csv(os.path.join(self.PlotDir,f"df_fit_and_data_expo_{Feature}_conditional_class_new.csv"))
            # data and fit - pl - aggregated - fuzzy
            if len(_df_fit_and_data_pl) > 0:
                _df_fit_and_data_pl.write_csv(os.path.join(self.PlotDir,f"df_fit_and_data_pl_{Feature}.csv"))
            # data and fit - exp - aggregated - fuzzy
            if len(_df_fit_and_data_exp) > 0:
                _df_fit_and_data_exp.write_csv(os.path.join(self.PlotDir,f"df_fit_and_data_expo_{Feature}.csv"))
            # data and fit - pl - aggregated - new
#            if len(_df_fit_and_data_pl_new) > 0:
#                _df_fit_and_data_pl_new.write_csv(os.path.join(self.PlotDir,f"df_fit_and_data_pl_{Feature}_new.csv"))
            # data and fit - exp - aggregated - new
 #           if len(_df_fit_and_data_exp_new) > 0:
  #              _df_fit_and_data_exp_new.write_csv(os.path.join(self.PlotDir,f"df_fit_and_data_expo_{Feature}_new.csv"))
            

    def Concatenate_gauss_max_speed_fit(self):
        """
            @Describe:
                - Aggregate:
                parameters_df:

        """
        # class - fuzzy
        _df_parameters_gs_conditional_on_class = pl.DataFrame()
        _df_parameters_mx_conditional_on_class = pl.DataFrame()
        _df_fit_and_data_gs = pl.DataFrame()
        _df_fit_and_data_mx = pl.DataFrame()
        Feature = "speed_kmh"
        for MobDate in self.ListDailyNetwork:
            StrDate = MobDate.StrDate
            for Aggregation in self.Aggregation2Class2Fcm.keys():
                if StrDate in self.AggregationLevel2ListDays[Aggregation]:
                    date_dir = os.path.join(self.PlotDir,StrDate)
                    # read params - class - fuzzy - gaussian
                    if os.path.isfile(os.path.join(date_dir,f"df_parameters_gs_{Feature}_{StrDate}_conditional_class.csv")):
                        df = pl.read_csv(os.path.join(date_dir,f"df_parameters_gs_{Feature}_{StrDate}_conditional_class.csv"))
                        _df_parameters_gs_conditional_on_class = pl.concat([_df_parameters_gs_conditional_on_class,df])
                    # read params - class - fuzzy - maxwellian
                    if os.path.isfile(os.path.join(date_dir,f"df_parameters_mx_{Feature}_{StrDate}_conditional_class.csv")):
                        df = pl.read_csv(os.path.join(date_dir,f"df_parameters_mx_{Feature}_{StrDate}_conditional_class.csv"))
                        _df_parameters_mx_conditional_on_class = pl.concat([_df_parameters_mx_conditional_on_class,df])
                    # read data and fit - class - fuzzy - gaussian
                    if os.path.isfile(os.path.join(date_dir,f"df_fit_and_data_gs_{Feature}_{StrDate}_conditional_class.csv")):
                        df = pl.read_csv(os.path.join(date_dir,f"df_fit_and_data_gs_{Feature}_{StrDate}_conditional_class.csv"))
                        _df_fit_and_data_gs = pl.concat([_df_fit_and_data_gs,df])
                    # read data and fit - class - fuzzy - maxwellian
                    if os.path.isfile(os.path.join(date_dir,f"df_fit_and_data_mx_{Feature}_{StrDate}_conditional_class.csv")):
                        df = pl.read_csv(os.path.join(date_dir,f"df_fit_and_data_mx_{Feature}_{StrDate}_conditional_class.csv"))
                        _df_fit_and_data_mx = pl.concat([_df_fit_and_data_mx,df])
        # parameters - gs - class - fuzzy
        if len(_df_parameters_gs_conditional_on_class) > 0:
            _df_parameters_gs_conditional_on_class.write_csv(os.path.join(self.PlotDir,f"df_parameters_gs_{Feature}_conditional_class.csv"))
        # parameters - mx - class - fuzzy
        if len(_df_parameters_mx_conditional_on_class) > 0:
            _df_parameters_mx_conditional_on_class.write_csv(os.path.join(self.PlotDir,f"df_parameters_mx_{Feature}_conditional_class.csv"))
        # data and fit - gs - class - fuzzy
        if len(_df_fit_and_data_gs) > 0:
            _df_fit_and_data_gs.write_csv(os.path.join(self.PlotDir,f"df_fit_and_data_gs_{Feature}_conditional_class.csv"))
        # data and fit - mx - class - fuzzy
        if len(_df_fit_and_data_mx) > 0:
            _df_fit_and_data_mx.write_csv(os.path.join(self.PlotDir,f"df_fit_and_data_mx_{Feature}_conditional_class.csv"))
        
                                      
            #########################################
    def ComputeAggregatedFitPerClass(self):
        # Save All the Tried Fit
        self.Aggregation2Feature2Class2AllFitTry = {Aggregation: InitFeature2Class2AllFitTry(self.Aggregation2Feature2Class2Function2Fit2InitialGuess[Aggregation]) for Aggregation in self.Aggregation2Class2Fcm.keys()}
        # Returns for each function to try the best fit.
        for Aggregation in self.Aggregation2Feature2Class2AllFitTry.keys():
            for Feature in self.Aggregation2Feature2Class2AllFitTry[Aggregation].keys():
                for StrClass in self.Aggregation2Feature2Class2AllFitTry[Aggregation][Feature].keys():
                    # NOTE: Change Observed Data Accordingly 
                    ObservedData = self.Aggregation2Class2Fcm[Aggregation][StrClass][Feature].to_list()
                    if Feature == "av_speed" or Feature == "speed_kmh":
                        self.Aggregation2Feature2Class2AllFitTry[Aggregation][Feature][StrClass] = FillIterationFitDicts(ObservedData,
                                                                                                                    self.Aggregation2Feature2Class2Function2Fit2InitialGuess[Aggregation][Feature][StrClass],
                                                                                                                    self.Aggregation2Feature2Class2AllFitTry[Aggregation][Feature][StrClass])
                    else:
                        self.Aggregation2Feature2Class2AllFitTry[Aggregation][Feature][StrClass] = FillIterationFitDictsTimeLength(ObservedData,
                                                                                                                    self.Aggregation2Feature2Class2Function2Fit2InitialGuess[Aggregation][Feature][StrClass],
                                                                                                                    self.Aggregation2Feature2Class2AllFitTry[Aggregation][Feature][StrClass])
                    if self.verbose:
                        print("Feature: ",Feature)
                        print("Class: ",StrClass)

##############################################à
    def PlotExponentsFit(self):
        """
            Description:
                Plot the exponents of the fit for the different days.
        """
        MobDate = self.ListDailyNetwork[0]
        Features = ["time_hours","lenght_km"]
        
        for Feature in Features:        
            Type_Fit = "expo"
            if os.path.isfile(os.path.join(self.PlotDir,f"df_parameters_{Type_Fit}_{Feature}_conditional_class.csv")):
                Class2Par = {"Day":[],"alpha":[],"-log_L_max":[]}
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                df_expo_params = pl.read_csv(os.path.join(self.PlotDir,f"df_parameters_{Type_Fit}_{Feature}_conditional_class.csv"))
                for MobDate in self.ListDailyNetwork:
                    StrDate = MobDate.StrDate
                    av_x = np.array(df_expo_params.filter(pl.col("Day") == StrDate)["<x>"].to_numpy())
                    Classes = np.array(df_expo_params.filter(pl.col("Day") == StrDate)["Class"].to_numpy()).astype(int) + 1
                    fit = np.polyfit(np.log(Classes), np.log(av_x), 1)
                    alpha = fit[0]
                    minus_log_L_max = fit[1]
                    Class2Par["Day"].append(StrDate)
                    Class2Par["alpha"].append(alpha)
                    Class2Par["-log_L_max"].append(minus_log_L_max)
                    ax.scatter(np.log(Classes), np.log(av_x), marker= self.Day2Marker[StrDate],label=f'{StrDate}')    
                    ax.plot(np.log(Classes), alpha * np.log(Classes) + fit[1],label = f'{StrDate}')

            ax.set_xlabel('Class')
            if Feature == "time_hours":
                ax.set_ylabel(r'$\overline{t_k} (h)$',fontsize = 18)
            elif Feature == "lenght_km":
                ax.set_ylabel(r'$\overline{L_k} (km)$',fontsize = 18)
            ax.set_xticks(np.log(Classes))
            ax.set_xticklabels(Classes)
            pl.DataFrame(Class2Par).write_csv(os.path.join(self.PlotDir,f"df_1_Lk_Lmax_{Feature}.csv"))
            plt.savefig(os.path.join(self.PlotDir,'ParameterDistributionDays_{0}.png'.format(Feature)),dpi = 200)
            # Show the plot
            plt.close(fig)
    def PlotExponentsGaussianFit(self):
        """
            Description:
                Plot the exponents of the fit for the different days.
        """
        MobDate = self.ListDailyNetwork[0]
        Features = ["speed_kmh"]
        for Feature in Features:    
            Class2Mu = defaultdict()
            Class2Sigma = defaultdict()
            Days = []
            for MobDate in self.ListDailyNetwork:
                for IntClass in MobDate.Feature2Class2AllFitTry[Feature].keys():
                    if MobDate.Feature2Class2AllFitTry[Feature][IntClass]["best_fit"] == "gaussian":
                        Parameters = MobDate.Feature2Class2AllFitTry[Feature][IntClass]["gaussian"]["parameters"]
                        if IntClass not in Class2Mu.keys():
                            Class2Mu[IntClass] = [Parameters[1]]
                            Class2Sigma[IntClass] = [Parameters[2]]
                        else:
                            Class2Mu[IntClass].append(Parameters[1])
                            Class2Sigma[IntClass].append(Parameters[2])
                        if MobDate.StrDate not in Days:
                            Days.append(MobDate.StrDate)
                    else:
                        if IntClass not in Class2Mu.keys():
                            Class2Mu[IntClass] = [np.nan]
                            Class2Sigma[IntClass] = [np.nan]
                        else:
                            Class2Mu[IntClass].append(np.nan)
                            Class2Sigma[IntClass].append(np.nan)
                        if MobDate.StrDate not in Days:
                            Days.append(MobDate.StrDate)

            # Plot each class with different colors and add a legend
            MuPerday = np.empty((len(Class2Mu.keys()), len(Days)))
            SigmaPerday = np.empty((len(Class2Sigma.keys()), len(Days)))
            for Class in Class2Mu.keys():
                MuPerday[Class] = Class2Mu[Class]
                SigmaPerday[Class] = Class2Sigma[Class]
            AvMu = []
            AvSigma = []
            for ClassIdx in range(len(MuPerday)):
                AvMu.append(np.nanmean(MuPerday[ClassIdx]))
                AvSigma.append(np.nanmean(SigmaPerday[ClassIdx]))
            Mus = np.array(MuPerday).T
            Sigmas = np.array(SigmaPerday).T
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            for i in range(len(Mus)):
                ax.scatter(list(Class2Mu.keys()), Mus[i], label=f'Class {IntClass}')
                ax.scatter(list(Class2Sigma.keys()), AvMu, marker="*", s = 200, label = None)
            # Add legend
            ax.legend(Days)

            # Set labels and title
            ax.set_xlabel('Class')
            ax.set_xticks(list(Class2Mu.keys()))
            ax.set_xticklabels(list(Class2Mu.keys()))
            ax.set_xlabel("Class")
            ax.set_ylabel(r'$\mu (km/h)$')
            plt.savefig(os.path.join(self.PlotDir,'MuDistributionDays_{0}.png'.format(Feature)),dpi = 200)
            plt.close()
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            for i in range(len(Sigmas)):
                ax.scatter(list(Class2Sigma.keys()), Sigmas[i], label=f'Class {IntClass}')
                ax.scatter(list(Class2Sigma.keys()), AvSigma, marker="*",s = 200,label = None)
            # Add legend
            ax.legend(Days)
            ax.set_xlabel('Class')
            ax.set_xticks(list(Class2Mu.keys()))
            ax.set_xticklabels(list(Class2Mu.keys()))
            ax.set_xlabel("Class")
            ax.set_ylabel(r'$\sigma$')
            plt.savefig(os.path.join(self.PlotDir,'SigmaDistributionDays_{0}.png'.format(Feature)),dpi = 200)
            plt.close()

            fig,ax = plt.subplots(1,1,figsize = (10,10))
            ax.scatter(AvMu,AvSigma)
            aq = np.polyfit(AvMu[:-1],AvSigma[:-1],1)
            ax.plot(AvMu,aq[0]*np.array(AvMu) + aq[1],color = "red")
            ax.set_xlabel(r'$\mu (km/h)$')
            ax.set_ylabel(r'$\sigma (km/h)$')
#            ax.legend(["0","1","2","3"])
            pl.DataFrame({"a_<v>":aq[0],"b_<v>":aq[1]}).write_csv(os.path.join(self.PlotDir,f"df_linear_coeffs_mu_sigma_speed_{Feature}.csv"))
            pl.DataFrame({"mu_v":AvMu,"sigma_v":AvSigma}).write_csv(os.path.join(self.PlotDir,f"df_mu_sigma_speed_{Feature}.csv"))
            plt.savefig(os.path.join(self.PlotDir,'mu_sigma_speed_averaged.png'),dpi = 200)
            plt.close()
            # Show the plot
    # Plot of Distribution of Time, Lenght, Av_Speed, Speed_kmh, Lenght_km, Time_hours for all days
    def PlotDistrAggregatedAllDays(self,bins = 100):
        """
            Input:
                label: str -> time, lenght, av_speed, p, a_max, class
            Returns:
                n, bins of velocity distribution
        """
        if self.ConcatenatePerClassBool:
            # Initialize the Dict of DataFrames Containing the Distribution for each Class to Plot
            self.Aggregation2Feature2StrClass2FcmDistr = {Aggregation: 
                                                            {Feature: defaultdict()
                                                              for Feature in self.Features2Fit
                                                            }
                                                           for Aggregation in self.Aggregation2Class2Fcm.keys()
                                                           }
            for Aggregation in self.Aggregation2Class2Fcm.keys():
                for Feature in ["speed_kmh"]:#self.Features2Fit:
                    self.Aggregation2Feature2StrClass2FcmDistr = AggregatedFcmDistr(self.Aggregation2Class2Fcm,
                                                                             Aggregation,
                                                                             Feature,
                                                                             self.Aggregation2Feature2StrClass2FcmDistr,
                                                                            True)
            InfoPlotDistrFeat = {"figsize":(4,4),"minx":0,"miny":0,"maxx":0,"maxy":0}
            # Compute the MinMax for the Plot
            if self.verbose:
                print("Aggregation2Feature2StrClass2FcmDistr:\n",self.Aggregation2Feature2StrClass2FcmDistr)
            PlotFeatureAggregatedAllDays(self.Aggregation2Feature2StrClass2FcmDistr,
                                        self.Aggregation2Feature2Class2AllFitTry,                   
                                        self.Feature2Legend,
                                        self.Feature2IntervalBin,
                                        self.Feature2IntervalCount,
                                        self.Feature2Label,
                                        self.Feature2ShiftBin,
                                        self.Feature2ShiftCount,
                                        self.Feature2ScaleBins,
                                        self.Feature2ScaleCount,
                                        self.PlotDir,
                                        self.Feature2SaveName,
                                        True)

#            PlotFeatureAggregatedWithoutFitRescaledByMean(self.Aggregation2Feature2StrClass2FcmDistr,
#                                        self.Aggregation2Feature2Class2AllFitTry,                   
#                                        self.Feature2Legend,
#                                        self.Feature2IntervalBin,
#                                        self.Feature2IntervalCount,
#                                        self.Feature2Label,
#                                        self.Feature2ShiftBin,
#                                        self.Feature2ShiftCount,
#                                        self.Feature2ScaleBins,
#                                        self.Feature2ScaleCount,
#                                        self.PlotDir,
#                                        self.Feature2SaveName,
#                                        True)            


    def PlotAverageTij(self):
        """
        @ Plots the average Tij 
        """
        self.Tij = None
        CountDays = 0
        for MobDate in self.ListDailyNetwork:
            if self.Tij is None:
                self.Tij = np.array(MobDate.Tij)
            else:
                self.Tij += np.array(MobDate.Tij)
            CountDays += 1
        self.Tij = self.Tij/CountDays
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        cax = ax.matshow(self.Tij, cmap='viridis')
        cbar = fig.colorbar(cax)
        Classes = np.arange(len(self.Tij))
        # Annotate each cell with the numeric value
        for (i, j), val in np.ndenumerate(self.Tij):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')
        ax.set_xticks(Classes)
        ax.set_yticks(Classes)
        ax.set_xlabel('Class Hierarchical')
        ax.set_ylabel('Class')
        ax.set_title('Transition Matrix i -> j')
        plt.savefig(os.path.join(self.PlotDir, 'AveragedTransitionMatrix.png'), dpi=200)
        plt.close()        

    def PlotGridDistrFeat(self):
        for Aggregation in self.AggregationLevel:
            NumberDays = len(self.AggregationLevel2ListDays[Aggregation])
            if NumberDays == 4:
                NumRows = NumberDays//2
                NumCol = 2
            elif NumberDays == 5:
                NumRows = 2
                NumCol = 3
            elif NumberDays == 9:
                NumRows = 3
                NumCol = 3
            for Feature in self.Aggregation2DictFittedData[Aggregation].keys():
                fig,axs = plt.subplots(NumRows,NumCol,figsize = (15,12))
                axs = np.array(axs).reshape(NumRows, NumCol)
                ax_idx = 0
                for MobDate in self.ListDailyNetwork:
                    if MobDate.StrDay in self.AggregationLevel2ListDays[Aggregation]:
                        row_idx, col_idx = divmod(ax_idx, NumCol)
                        axs[row_idx, col_idx] = MobDate.Feature2DistributionPlot[Feature]["ax"]
                        ax_idx += 1
                plt.savefig(os.path.join(self.PlotDir,'Grid_{0}_{1}.png'.format(Aggregation,self.Feature2SaveName[Feature])),dpi = 200)
                plt.close()
    # MFD Aggregated All Days

    def ComputeDay2PopulationTime(self):
        """
            Description:
                Compute the population and time for each day.
        """
        self.Day2PopTime = ComputeDay2PopulationTime(self.ListDailyNetwork,self.IntClasses)


    def PlotComparisonPopulationTime(self):
        """
            Description:
                Plot the population and time for each day.
        """
        PlotDay2PopulationTime(self.Day2PopTime,self.CutIndexTime,self.PlotDir)



    def PlotMFDPerClassCompared(self):
        """
            Plots the MFD for all the different days together in the same plot for each class .
        """
        # linear coefficient fuzzy
        ComputeLinearCoeeficientMFD(self.ListDailyNetwork,
                                    self.IntClasses,
                                    False,
                                    self.PlotDir)
        # linear coefficient hierarchical
        ComputeLinearCoeeficientMFD(self.ListDailyNetwork,
                                    self.IntClasses,
                                    True,
                                    self.PlotDir)

        for Class in self.IntClasses:
            PlotMFDComparison(self.ListDailyNetwork,Class,self.ListColors,True,self.PlotDir)
            PlotMFDComparison(self.ListDailyNetwork,Class,self.ListColors,False,self.PlotDir)
        PlotCoeffClassification(self.PlotDir,self.IntClasses)

            
                    
# HTML SESSION FOR SUBNETS AGGREGATED

    def CreateClass2SubNetAllDays(self):
        """
            Description:
                For each day
                For each Class
                For each road in self.GeojSon, add in column: 
            Output:
                self.Road2StrClass2CountDays = {Road: {StrClass: {Day: Count}}}
                    Stores Informations about the number of days that a road is included in a class.
                self.Road2StrClass2Days = {Road: {StrClass: {Day: [List of Days]}}}
                    Stores Informations about the days that a road is included in a class.
        """
        self.Road2StrClass2CountDays = {int(Road): {StrClass:0 for StrClass in self.ListStrClassReference} for Road in self.ListDailyNetwork[0].GeoJson["poly_lid"]} 
        self.Road2StrClass2Days = {int(Road):{StrClass: [] for StrClass in self.ListStrClassReference} for Road in self.ListDailyNetwork[0].GeoJson["poly_lid"]} 

        for MobDate in self.ListDailyNetwork:
            for StrClass in self.ListStrClassReference:
                # Consider the class of the day associated to the Reference Class (quickest, slowest, etc.)
                IntClassOfDay = MobDate.StrClass2IntClass[StrClass]
                # List of Roads that are included in the class
                for Road in MobDate.IntClass2RoadsIncreasinglyIncludedIntersection[IntClassOfDay]:
                    if int(Road) in self.Road2StrClass2CountDays.keys():
                        self.Road2StrClass2CountDays[int(Road)][StrClass] += 1
                        self.Road2StrClass2Days[int(Road)][StrClass].append(MobDate.StrDate)
        self.CreateClass2SubNetAllDaysBool = True

    def ComputeCmapForSubNet(self):
        """
            Compute Max,Min of Number of Days In Which The Road Appears in the Class 
            Output:
                self.StrClassMinMaxCountPerDay = {StrClass: {"max":0,"min":10000} for StrClass in self.ListStrClassReference}
        """
        if self.CreateClass2SubNetAllDaysBool:
            self.StrClassMinMaxCountPerDay = {StrClass: {"max":0,"min":10000} for StrClass in self.ListStrClassReference}
            for Road in self.Road2StrClass2CountDays.keys():
                for StrClass in self.Road2StrClass2CountDays[Road].keys():
                    if self.Road2StrClass2CountDays[Road][StrClass]>self.StrClassMinMaxCountPerDay[StrClass]["max"]:
                        self.StrClassMinMaxCountPerDay[StrClass]["max"] = self.Road2StrClass2CountDays[Road][StrClass]
                    if self.Road2StrClass2CountDays[Road][StrClass]<self.StrClassMinMaxCountPerDay[StrClass]["min"]:
                        self.StrClassMinMaxCountPerDay[StrClass]["min"] = self.Road2StrClass2CountDays[Road][StrClass] 

    def PlotClass2SubNetAllDays(self):
        if self.CreateClass2SubNetAllDaysBool:
            print("Plotting Incremental Subnetworks in HTML and their Frequency")
            # Init cmap
            cmap = plt.get_cmap('inferno')
            self.ComputeCmapForSubNet()
            # Create a base map
            m = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
            m1 = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
            # Iterate through the Dictionary of list of poly_lid
            for StrClass in self.ListStrClassReference:
                layer_group = folium.FeatureGroup(name="Layer {}".format(StrClass)).add_to(m)
                layer_group1 = folium.FeatureGroup(name="Layer {}".format(StrClass)).add_to(m1)
                for Road in self.Road2StrClass2CountDays.keys():
                    # Create a feature group for the current layer
                    filtered_gdf = self.GeoJson[self.GeoJson['poly_lid'] == Road]
                    for _, road in filtered_gdf.iterrows():
                        RoadNormedCount = (self.Road2StrClass2CountDays[Road][StrClass] - self.StrClassMinMaxCountPerDay[StrClass]["min"])/(self.StrClassMinMaxCountPerDay[StrClass]["max"] - self.StrClassMinMaxCountPerDay[StrClass]["min"])
                        FrequencyInDaysColor = mcolors.to_hex(cmap(norm_count))
                        folium.GeoJson(road.geometry, style_function=lambda x: {'color': FrequencyInDaysColor}).add_to(layer_group)
                        StrPopup = ""
                        for StrDay in self.Road2StrClass2Days[Road][StrClass]:
                            StrPopup = StrPopup + StrDay + "\n"
                        folium.GeoJson(
                            road.geometry,
                            style_function=lambda x: {'color': "green"},
                            popup= StrPopup 
                        ).add_to(layer_group1)                            
                    # Add the feature group to the map
                layer_group.add_to(m)
                layer_group1.add_to(m1)
                # Add layer control to the map
                folium.LayerControl().add_to(m)
                folium.LayerControl().add_to(m1)

            # Save or display the map
            m.save(os.path.join(self.PlotDir,"AggregatedSubnetsIncrementalInclusionCountAllDays.html"))
            m1.save(os.path.join(self.PlotDir,"SubnetsIncrementalInclusionAllDays.html"))
        else:
            print("No Subnetworks to Plot")
            return False            

    def PlotClass2SubnetsComparisonAllDays(self):
        if self.CreateClass2SubNetAllDaysBool:
            print("Plotting Incremental Subnetworks in HTML and their Frequency")
            print("Save in: ",os.path.join(self.PlotDir,"SubnetsIncrementalInclusion_{}.html".format(self.StrDate)))
            # Init cmap
            cmap = plt.get_cmap('inferno')
            self.ComputeCmapForSubNet()
            # Create a base map
            m = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
            m1 = folium.Map(location=[self.centroid.x, self.centroid.y], zoom_start=12)
            # Iterate through the Dictionary of list of poly_lid
            for StrClass in self.ListStrClassReference:
                layer_group1 = folium.FeatureGroup(name="Layer {}".format(StrClass)).add_to(m1)
                layer_group1.add_to(m1)
                folium.LayerControl().add_to(m1)
                for Road in self.Road2StrClass2CountDays.keys():
                    # Create a feature group for the current layer
                    filtered_gdf = self.GeoJson[self.GeoJson['poly_lid'] == Road]
                    for _, road in filtered_gdf.iterrows():
                        for StrDay in self.Road2StrClass2Days[Road][StrClass]:
                            layer_group_day = folium.FeatureGroup(name="Layer {} {}".format(StrClass, StrDay)).add_to(m)
                            folium.GeoJson(road.geometry, style_function=lambda x: {'color': self.StrDay2Color[StrDay]}).add_to(layer_group_day)

                    # Add the feature group to the map
                layer_group_day.add_to(m1)
                # Add layer control to the map
                folium.LayerControl().add_to(m1)

            m1.save(os.path.join(self.PlotDir,"SubnetsIncrementalInclusionAllDays.html"))
        else:
            print("No Subnetworks to Plot")
            return False            

        

                
            
# PLOT FEATURES FOR ALL DAYS ----> 


    def PlotComparisonDistributionEachFeatureAllDaysRescaledByMean(self):
        """
            Description:
                Plot the distribution of each feature given the class for all days
        """
        Features = ["lenght_km","time_hours"]
        Feature2Label = {"lenght_km":"L","speed_kmh":"v","time_hours":"t"}
        Days = [MobDate.StrDate for MobDate in self.ListDailyNetwork]
        Class2ClassStr = {"0":"1 slowest","1":"2 slowest","2":"2 fastest","3":"1 fastest"}
        Plot_distribution_length_time_daily_and_condtioned_to_classes(Days,
                                                                    Features,
                                                                    ["0","1","2","3"],
                                                                    ["New"],
                                                                    self.PlotDir,
                                                                    Class2ClassStr,
                                                                    Feature2Label,
                                                                    self.PlotDir)

        Plot_distribution_length_time_not_conditional_class(["New"],
                                                            Days,
                                                            Features,
                                                            Feature2Label,
                                                            self.PlotDir)


    

    def PlotInsetFitAggregated(self,DfParametersFit,Feature,ax_inset):
        """
            @param DfParametersFit: DataFrame with the parameters of the fit columns: ["Day","A","alpha"]
            @param Feature: Feature to plot
            @param ax_inset: Axes to plot the inset
            @return ax_inset: Axes with the inset -> the plot of the fitting parameters:
            - <x> for exponential
            - alpha for power law
        """
        for Day,DfParDay in DfParametersFit.groupby("Day"):
            # Exp
            if "1/<x>" in DfParDay.columns:
                T = - DfParDay["1/<x>"].to_numpy()[0]
                if "time" in Feature:
                    ax_inset.set_ylabel(r"$\langle t \rangle (hours)$")
                if "lenght" in Feature:
                    ax_inset.set_ylabel(r"$\langle L \rangle (km)$")
            # Pl 
            elif "alpha" in DfParDay.columns:
                T = DfParDay["alpha"].to_numpy()[0]
                ax_inset.set_ylabel(r"$\alpha$")
            ax_inset.set_xlabel("Day")
            ax_inset.set_xticklabels([])
            ax_inset.set_yticklabels([])
            ax_inset.scatter([Day],[T],label = Day,marker=self.Day2Marker[Day],color = self.StrDay2Color[Day])
        return ax_inset

    def PlotParametersAggregatedFit(self):
        """
            Reading from:
              - ExponentialFit_{Aggregation}_{Feature}.csv
              - PowerLawFit_{Aggregation}_{Feature}.csv


        """
        Features = ["lenght_km","time_hours"]
        Feature2Label = {"lenght_km":"L (km)","time_hours":"t (h)"}
        for Feature in Features:
            for Aggregation in self.Aggregation2Class2Fcm.keys():
                if os.path.exists(os.path.join(self.PlotDir,f"ExponentialFit_{Aggregation}_{Feature}.csv")):
                    Day2ExpFit = pl.read_csv(os.path.join(self.PlotDir,f"ExponentialFit_{Aggregation}_{Feature}.csv"))
                    fig,ax = plt.subplots(1,1,figsize = (10,8))
                    for Day in Day2ExpFit["Day"]:
                        T = -Day2ExpFit.filter(pl.col("Day") == Day)["1/<x>"].to_numpy()[0]
                        ax.scatter([Day],[T],label = Day,marker=self.Day2Marker[Day],color = self.StrDay2Color[Day])
                    ax.set_xlabel("Day")
                    if "time" in Feature:
                        ax.set_ylabel(r"$\langle t \rangle (hours)$")
                    if "lenght" in Feature:
                        ax.set_ylabel(r"$\langle L \rangle (km)$")
                    ax.set_title("Exponential Fit Parameters")
                    ax.set_xticklabels(Day2ExpFit["Day"],rotation = 90)
                    ax.legend()
                    fig.savefig(os.path.join(self.PlotDir,f"ParameterExponentialFit_{Aggregation}_{Feature}.png"),dpi = 200)
                    plt.close()
                if os.path.exists(os.path.join(self.PlotDir,f"PowerLawFit_{Aggregation}_{Feature}.csv")):
                    Day2PlFit = pl.read_csv(os.path.join(self.PlotDir,f"PowerLawFit_{Aggregation}_{Feature}.csv"))
                    
                    fig,ax = plt.subplots(1,1,figsize = (10,8))
                    for Day in Day2PlFit["Day"]:
                        alpha = Day2PlFit.filter(pl.col("Day") == Day)["alpha"].to_numpy()[0]
                        ax.scatter([Day],[alpha],label = Day,marker=self.Day2Marker[Day],color = self.StrDay2Color[Day])
                    ax.set_xlabel("A")
                    ax.set_ylabel(r"$\alpha$")
                    ax.set_xticklabels(Day2PlFit["Day"],rotation = 90)
                    ax.set_title("Power Law Fit Parameters")
                    ax.legend()
                    fig.savefig(os.path.join(self.PlotDir,f"ParameterPowerLawFit_{Aggregation}_{Feature}.png"),dpi = 200)
                    plt.close()
    def PlotDistributionTotalNumberPeople(self):
        """
            x: Days
            y: Total Number of People
        """
        fig,ax = plt.subplots(1,1,figsize = (10,8))
        x = [MobDate.StrDate for MobDate in self.ListDailyNetwork]
        y = [len(MobDate.Fcm) for MobDate in self.ListDailyNetwork]
        ax.scatter(x,y)
        ax.set_xticklabels(x,rotation = 90)
        ax.set_xlabel("Days")
        ax.set_ylabel("Total Number of People")
        ax.set_title("Total Number of People")
        plt.savefig(os.path.join(self.PlotDir,"TotalNumberPeople.png"),dpi = 200)
        plt.close()


# NUMBER OF TRAJECTORIES GIVEN THE CLASS
    def PlotNPeopleOverNRoads(self):
        """
            @description:
                For each road in the network, plot the number of people over the number of roads according to the classification
        """
        MobDate = self.ListDailyNetwork[0]
        ks = sorted(list(MobDate.OrderedClass2TimeDeparture2UserId.keys()))
        fig,ax = plt.subplot_mosaic([[ks[0],ks[1]],[ks[2],ks[3]]],figsize = (20,20),sharex=True,sharey=True)        
        for MobDate in self.ListDailyNetwork:
            for Class in ks:
                TimeIntervals = list(MobDate.OrderedClass2TimeDeparture2UserId[Class].keys())
                if type(TimeIntervals[0]) == int:
                    TimeIntervalsDt = [datetime.datetime.fromtimestamp(int(t)).strftime("%Y-%m-%d %H:%M:%S").split(" ")[1] for t in TimeIntervals]
                else:
                    TimeIntervalsDt = TimeIntervals
                    pass
                Npeop = []
                for TimeDeparture in MobDate.OrderedClass2TimeDeparture2UserId[Class].keys():
                    Npeop.append(len(MobDate.OrderedClass2TimeDeparture2UserId[Class][TimeDeparture])/len(MobDate.IntClass2RoadsIncreasinglyIncludedIntersection[Class]))
                ax[Class].scatter(TimeIntervalsDt[self.CutIndexTime:],Npeop[self.CutIndexTime:],label = f"{MobDate.StrDate}")
#                ax[Class].hlines(1,TimeIntervalsDt[self.CutIndexTime],TimeIntervalsDt[-1])
#                ax[Class].text(TimeIntervalsDt[0],0.5,f"Number of Roads {len(MobDate.IntClass2RoadsIncreasinglyIncludedIntersection[Class])}")
#                ax[Class].set_xlabel("Time")
                ax[Class].set_ylabel(r"$\frac{N_p}{N_r}$")
                ax[Class].set_xticks(range(len(TimeIntervalsDt[self.CutIndexTime:]))[::8])  # Set the ticks to correspond to the labels
                ax[Class].set_xticklabels(TimeIntervalsDt[self.CutIndexTime::8], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
                ax[Class].legend()
                ax[Class].set_yscale("log")
        plt.savefig(os.path.join(self.PlotDir,f"EvolutionNumberPeople.png"),dpi = 200)
        plt.close()

    def PlotDensity(self):
        """
            @Description:
                Plots the density for each subnetwork computed as:
                    - Number of people over total length of the subnetwork
        """
        MobDate = self.ListDailyNetwork[0]
        ks = sorted(list(MobDate.OrderedClass2TimeDeparture2UserId.keys()))
        fig,ax = plt.subplot_mosaic([[ks[0],ks[1]],[ks[2],ks[3]]],figsize = (20,20),sharex=True,sharey=True)        
        for MobDate in self.ListDailyNetwork:
            for Class in ks:
                TimeIntervals = list(MobDate.OrderedClass2TimeDeparture2UserId[Class].keys())
                if type(TimeIntervals[0]) == int:
                    TimeIntervalsDt = [datetime.datetime.fromtimestamp(int(t)).strftime("%Y-%m-%d %H:%M:%S").split(" ")[1] for t in TimeIntervals]
                else:
                    TimeIntervalsDt = TimeIntervals
                    pass
                Npeop = []
                print("Plot Density Class: ",Class)
                for TimeDeparture in MobDate.OrderedClass2TimeDeparture2UserId[Class].keys():
                    print("Time Departure: ",TimeDeparture)
                    print("Number of People: ",len(MobDate.OrderedClass2TimeDeparture2UserId[Class][TimeDeparture]),"Total Length: ",(len(MobDate.IntClass2RoadsIncreasinglyIncludedIntersection[Class])*MobDate.Class2TotalLengthOrderedSubnet[Class]))
                    Npeop.append(len(MobDate.OrderedClass2TimeDeparture2UserId[Class][TimeDeparture])/(len(MobDate.IntClass2RoadsIncreasinglyIncludedIntersection[Class])*MobDate.Class2TotalLengthOrderedSubnet[Class]))
                if Class == 3 or Class == "3":
                    ax[Class].scatter(TimeIntervalsDt[self.CutIndexTime:],Npeop[self.CutIndexTime:],label = f"{MobDate.StrDate}")
                else:
                    ax[Class].scatter(TimeIntervalsDt[self.CutIndexTime:],Npeop[self.CutIndexTime:],label = "")
#                ax[Class].hlines(1,TimeIntervalsDt[0],TimeIntervalsDt[-1])
#                ax[Class].text(TimeIntervalsDt[0],0.5,f"Number of Roads {len(MobDate.IntClass2RoadsIncreasinglyIncludedIntersection[Class])}")
#                ax[Class].set_xlabel("Time")
                ax[Class].set_ylabel(r"$\frac{N_p}{L_{r}^{tot}}$")
                ax[Class].set_xticks(range(len(TimeIntervalsDt[self.CutIndexTime:]))[::8])  # Set the ticks to correspond to the labels
                ax[Class].set_xticklabels(TimeIntervalsDt[self.CutIndexTime::8], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
                ax[Class].set_yscale("log")
                if Class == 3:
                    ax[Class].legend()
        plt.savefig(os.path.join(self.PlotDir,f"EvolutionDensitySubnetwork.png"),dpi = 200)
        plt.close()
# TRAFFIC
    def PlotTrafficIndicator(self):
        """
            @brief: Compute and Plot the Daily Traffic Indicator
            It is 1 super trafficked, 0 not trafficked
        """
        MobDate = self.ListDailyNetwork[0]
        ks = sorted(list(MobDate.OrderedClass2TimeDeparture2UserId.keys()))
        fig,ax = plt.subplot_mosaic([[ks[0],ks[1]],[ks[2],ks[3]]],figsize = (20,20))      
        Class2TrafficIndex = {Class: None for Class in ks}  
        Class2CriticalTraffic = {Class: 0 for Class in ks}
        CountDays = 0
        fig,ax = plt.subplots(2,2,figsize = (20,20),sharey = True)
        Class2Idx = {0:(0,0),1:(0,1),2:(1,0),3:(1,1)}
        TrafficIdx = {str(int(Class)): [] for Class in ks}
        TrafficIdx["Day"] = [] 
        TrafficIdx["Time"] = []
        Day2TrafficIndex = {MobDate.StrDate + "_" + str(Class): [] for Class in ks  for MobDate in self.ListDailyNetwork}
        
        for MobDate in self.ListDailyNetwork:
            CountDays += 1
            for Class in ks:
                TimeIntervals = list(MobDate.OrderedClass2TimeDeparture2UserId[Class].keys())
                if type(TimeIntervals[0]) == int:
                    TimeIntervalsDt = [datetime.datetime.fromtimestamp(int(t)).strftime("%Y-%m-%d %H:%M:%S").split(" ")[1] for t in TimeIntervals]
                else:
                    TimeIntervalsDt = TimeIntervals
                    pass

                # Critical Value Traffic Index
                if Class2CriticalTraffic[Class] is None: 
                    Class2CriticalTraffic[Class] = np.array(MobDate.Class2traffickIndex[Class])
                else:
                    Class2CriticalTraffic[Class] += np.array(MobDate.Class2traffickIndex[Class])
                ax0 = Class2Idx[Class][0]
                ax1 = Class2Idx[Class][1]
                if Class == 3:
                    ax[ax0,ax1].plot(TimeIntervalsDt[self.CutIndexTime:],MobDate.Class2traffickIndex[Class][self.CutIndexTime:],label = "{}".format(MobDate.StrDate))
                else:
                    ax[ax0,ax1].plot(TimeIntervalsDt[self.CutIndexTime:],MobDate.Class2traffickIndex[Class][self.CutIndexTime:],label = "")
                Day2TrafficIndex["Time"] = TimeIntervalsDt
                Day2TrafficIndex[MobDate.StrDate + "_" + str(Class)] = list(MobDate.Class2traffickIndex[Class])
                if Class == 0 or Class == "0":
                    TrafficIdx["Day"].extend(list(np.full(len(TimeIntervalsDt[self.CutIndexTime:]),MobDate.StrDate)))
                    TrafficIdx["Time"].extend(TimeIntervalsDt[self.CutIndexTime:])
                
                TrafficIdx[str(int(Class))].extend(list(MobDate.Class2traffickIndex[Class][self.CutIndexTime:]))
                ax[ax0,ax1].set_xlabel("Time")
#                ax[ax0,ax1].set_ylabel(r"$\langle \frac{(v_o(t) - v_h(t))}{\langle v_h \rangle_t}\frac{N_{class}}{N_{max}} \rangle_{days}$")
                ax[ax0,ax1].set_ylabel(r"$\langle \Gamma_{k} (t) \rangle_{days}$",fontsize = 18)

                tick_locations = np.arange(0, len(TimeIntervalsDt[self.CutIndexTime:]), 8)
                tick_labels = MobDate.BinStringHour[self.CutIndexTime::8]
                ax[ax0,ax1].set_xticks(tick_locations)
                ax[ax0,ax1].set_xticklabels(tick_labels, rotation=90)
                if Class == 3:
                    ax[ax0,ax1].legend()

        plt.savefig(os.path.join(self.PlotDir,"TrafficIndicatorAllDays.png"))
        plt.close()
        with open(os.path.join(self.PlotDir,"TrafficIndexAllDays.json"),"w") as f:
            json.dump(Day2TrafficIndex,f,indent=2)
        # Ad Hoc, somehow I inserted numpy.float
        for key in TrafficIdx:
            TrafficIdx[key] = [float(x) if isinstance(x, np.float64) else x for x in TrafficIdx[key]]
        pl.DataFrame(TrafficIdx).write_csv(os.path.join(self.PlotDir,"TrafficIndexAllDays.csv"))
        for Class in Class2CriticalTraffic.keys():     
            Class2CriticalTraffic[Class] = Class2CriticalTraffic[Class]/CountDays
        # NOTE: Compute the average speed and the difference speed
        fig,ax = plt.subplots(2,2,figsize = (20,20),sharey = True)
        Class2Idx = {0:(0,0),1:(0,1),2:(1,0),3:(1,1)}
        for Class in Class2CriticalTraffic.keys():
            ax0 = Class2Idx[Class][0]
            ax1 = Class2Idx[Class][1]
            ax[ax0,ax1].plot(TimeIntervalsDt[self.CutIndexTime:],Class2CriticalTraffic[Class][self.CutIndexTime:],label = "Class {}".format(Class))
#            ax[ax0,ax1].hlines(Class2CriticalTraffic[Class],TimeIntervalsDt[0],TimeIntervalsDt[-1],linestyles = "--",label = "Critical Traffic")
            ax[ax0,ax1].set_xlabel("Time")
            ax[ax0,ax1].set_ylabel(r"$\langle \Gamma_{k} (t) \rangle_{days}$")
#            ax[ax0,ax1].set_ylabel(r"$\langle \frac{(v_o(t) - v_h(t))}{\langle v_h \rangle_t}\frac{N_{class}}{N_{max}} \rangle_{days}$")
            tick_locations = np.arange(0, len(TimeIntervalsDt[self.CutIndexTime:]), 8)
            tick_labels = MobDate.BinStringHour[self.CutIndexTime::8]
            ax[ax0,ax1].set_xticks(tick_locations)
            ax[ax0,ax1].set_xticklabels(tick_labels, rotation=90)

        plt.savefig(os.path.join(self.PlotDir,"TrafficIndicator.png"))
#        with open(os.path.join(self.PlotDir,"Class2traffickIndex.json"),'w') as f:
#            json.dump(Class2TraffickIndex,f,indent=2)
        plt.close()
        Classes = ["0","1","2","3"]
        Days = [MobDate.StrDate for MobDate in self.ListDailyNetwork]
        PlotTrafficIForEachDay(Classes,Days,self.PlotDir)
    

    def PlotNumberTrajectoriesGivenClass(self):
        """
            @brief: Plot the number of trajectories given the class
            This function is the application of heterogenous analysis.
            @requirements:
                df_fit_and_data_expo_{Feature}_conditional_class_new.csv -> contain
                df_1_Lk_Lmax_{Feature}.csv
        """
        from scipy.ndimage import gaussian_filter1d
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        colors = ["blue","red","green","yellow","black","orange","purple","pink","brown","grey"]
        
        Feature2Label = {"lenght_km":"L","time_hours":"t"}
        for Aggregation in ["aggregated"]:            
            DayHeterogeneity = {"Class":[],"Day":[],"y_contribution":[],"x":[]}
            for Feature in ["lenght_km","time_hours"]:
                fig,ax = plt.subplots(1,1,figsize = (10,10))
                sub_ax = inset_axes(
                    parent_axes=ax,
                    width="40%",
                    height="30%",
                    loc='upper left',  # location of the inset axes
                    bbox_to_anchor=(0.05, 0.95, 0.4, 0.3),  # position of the inset axes
                    bbox_transform=ax.figure.transFigure  # transform for the bounding box                
                    )                
                df_fit_feature = pl.read_csv(os.path.join(self.PlotDir,f"df_fit_and_data_expo_{Feature}_conditional_class_new.csv"))
                df_1_lk = pl.read_csv(os.path.join(self.PlotDir,f"df_1_Lk_Lmax_{Feature}.csv"))
                Classes = np.unique(df_fit_feature["Class"].to_numpy()) + 1
                bin_size = 50
                CountDay = 0
                for MobDate in self.ListDailyNetwork:
                    df_fit_feature_day = df_fit_feature.filter(pl.col("Day") == MobDate.StrDate)
                    x_classes = []
                    y_classes = [] 
                    P_reconstructed_from_classes = np.zeros(bin_size)              
                    matrix_class_bins = []
                    matrix_class_distribution = []
                    # parameters heterogeneity
                    alpha = df_1_lk.filter(pl.col("Day") == MobDate.StrDate)["alpha"].to_numpy()[0]
                    x_max = np.exp(- df_1_lk.filter(pl.col("Day") == MobDate.StrDate)["-log_L_max"].to_numpy()[0])
                    # daily distribution
                    y_day = df_fit_feature.filter(pl.col("Day") == MobDate.StrDate,
                                                  pl.col("Class") == 10)["y"].to_numpy()
                    x_day = df_fit_feature.filter(pl.col("Day") == MobDate.StrDate,
                                                  pl.col("Class") == 10)["x"].to_numpy()
                    Fraction_people_classes_vector = Classes**(alpha)/x_max
                    DayHeterogeneity["y_contribution"].extend(y_day)
                    DayHeterogeneity["x"].extend(x_day)
                    DayHeterogeneity["Class"].extend([10]*len(x_day))
                    DayHeterogeneity["Day"].extend([MobDate.StrDate]*len(x_day))                    
                    # Prepare the matrix [class_idx,<distribution_class>], [class_idx,<x_bin_class>]     
                    for Class, df_fit_feature_day_class in df_fit_feature_day.group_by("Class"):
                        if Class != 10:
                            x_class = df_fit_feature_day_class["x"].to_numpy()
                            y_class = df_fit_feature_day_class["y"].to_numpy()
                            x_tmp, y_tmp = map_distributions_of_different_binnings_to_same_bin(x_class,
                                                                                                y_class,
                                                                                                bin_size = bin_size)
                            matrix_class_bins.append(x_tmp)
                            matrix_class_distribution.append(y_tmp)                    
                        # weight parameters
    #                    for class_idx in range(len(matrix_class_bins)):
                            contribution_class_idx2P = Fraction_people_classes_vector[Class]*y_tmp
                            DayHeterogeneity["y_contribution"].extend(contribution_class_idx2P)
                            DayHeterogeneity["x"].extend(x_tmp)
                            DayHeterogeneity["Class"].extend([Class]*len(x_tmp))
                            DayHeterogeneity["Day"].extend([MobDate.StrDate]*len(x_tmp))
                            P_reconstructed_from_classes += contribution_class_idx2P
                    P_reconstructed_from_classes/=len(P_reconstructed_from_classes)
                    P_reconstructed_from_classes = P_reconstructed_from_classes/np.sum(P_reconstructed_from_classes)
                    y_day = enrich_vector_to_length(y_day, bin_size)
                    P_reconstructed_from_classes = gaussian_filter1d(P_reconstructed_from_classes,sigma = 3)
                    ax.plot(x_tmp,P_reconstructed_from_classes,color=colors[CountDay],linestyle = "--",label=None)
                    ax.scatter(x_tmp,y_day,color=colors[CountDay],label=MobDate.StrDate)
                    CountDay += 1
                if Feature == "lenght_km":
                    pl.DataFrame(DayHeterogeneity).write_csv(os.path.join(self.PlotDir,"HeterogeneityL.csv"))
#                    sub_ax.set_title(r"$L_k = \frac{k^{\alpha}}{L_{max}}$")
                elif Feature == "time_hours":
                    pl.DataFrame(DayHeterogeneity).write_csv(os.path.join(self.PlotDir,"HeterogeneityT.csv"))
                ax.legend(fontsize = 'small')
                ax.set_yscale("log")
#                ax.set_xscale("log")
                if Feature == "lenght_km":
                    ax.set_xlabel(Feature2Label[Feature] + " (km)")
                    ax.set_xlim(0.2)
                    ax.set_ylim(1e-3,1)
                elif Feature == "time_hours":
                    ax.set_xlabel(Feature2Label[Feature] + " (h)")
                    ax.set_xlim(0.2,2.5)
                    ax.set_ylim(1e-4,1)
                ax.set_ylabel("P({})".format(Feature2Label[Feature]))
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                ax_inset = inset_axes(ax, width="20%", height="20%", loc="upper left")
                ax_inset.scatter(df_1_lk["Day"],df_1_lk["alpha"],label = r"$\alpha$",s = 25)
                ax_inset.set_xlabel("Day",fontsize = 12)
                ax_inset.set_ylabel(r"$\alpha$",fontsize = 12)
                ax_inset.set_xticklabels([])
                plt.savefig(os.path.join(self.PlotDir,"ComparisonHeterogeneityHpRealDistr_{0}_{1}.png".format(Aggregation,Feature)),dpi = 200)
                plt.close(fig)




    def PlotAveragePerturbationDistributionSpeed(self):
        """
            Description:
                Plot the average perturbation distribution for speed.
        """
        Colors = ["blue","orange","green","red"]
        Class2NewAverageSpeedDistribution = {Type: defaultdict(list) for Type in ["hierarchical","not_hierarchical"]}
        Classes = [IntClass for IntClass in self.ListDailyNetwork[0].Class2DfSpeedAndTimePercorrenceRoads]
        Classes = np.sort(Classes)
        for Class in Classes:
            Class2NAverageNotHierarchical = []
            Class2NAverageHierarchical = []
            for MobDate in self.ListDailyNetwork:
                ClassFcm = MobDate.Fcm.filter(pl.col("class") == Class)
                ClassNewFcm = MobDate.Fcm.filter(pl.col("class_new") == Class)["speed_kmh"]
                n,bins = np.histogram(ClassFcm["speed_kmh"],bins = 100)
                n_h,bins = np.histogram(ClassNewFcm,bins = 100)
                Class2NAverageNotHierarchical.append(n)
                Class2NAverageHierarchical.append(n_h)
            Class2NewAverageSpeedDistribution["not_hierarchical"][Class] = np.mean(np.array(Class2NAverageNotHierarchical),axis = 0)
            Class2NewAverageSpeedDistribution["hierarchical"][Class] = np.mean(np.array(Class2NAverageHierarchical),axis = 0)
        for Class in Classes:    
            fig,ax = plt.subplots(1,1,figsize = (12,10))
            ax.scatter(bins[1:],Class2NewAverageSpeedDistribution["not_hierarchical"][Class],label = "Fuzzy Classification")
            ax.scatter(bins[1:],Class2NewAverageSpeedDistribution["hierarchical"][Class],label = "Hierarchical Classification")
            ax.set_xlabel("Speed (km/h)")
            ax.set_ylabel("Count")
            if Class == len(Classes)-1:
                ax.legend()
            plt.savefig(os.path.join(self.PlotDir,"Average_Speed_Modification_Distribution_Class{}.png".format(Class)),dpi = 200)
            plt.close()


# Comparison Time Percorrence
    def CompareTimePercorrenceAllDays(self):
        """
          Compute the distribution of roads length for each class.
          NOTE: The idea is to try to understand if there is some bias in the length of the roads for each class.
          NOTE: By bias I mean that the distributions are peaked around some length, and the centroid for different
          classes are different, maybe in the sense of sigmas in a gaussian distribution. 
        """
        Slicing = 8
        MobDate = self.ListDailyNetwork[0]
        Classes = [IntClass for IntClass in MobDate.Class2DfSpeedAndTimePercorrenceRoads]
        print("Time Percorrence for Classes: ")
        print("Classes: ",Classes)
        print("Classes For Time Distribution: ",self.ListDailyNetwork[0].Class2Time2Distr.keys())
        print("Classes For Time Percorrence: ",self.ListDailyNetwork[0].Class2AvgTimePercorrence.keys())
        colors = ["blue","red","green","yellow","black","orange","purple","pink","brown","grey"]
        for IntClass in Classes:
            fig,ax = plt.subplots(1,1,figsize = (10,10))
            IDay = 0
            for MobDate in self.ListDailyNetwork:
                # Compute Time (x-axis plot): NOTE: Inefficient, but it is just for the plot 
                StrTimesLabel = []
                if IntClass in MobDate.Class2DfSpeedAndTimePercorrenceRoads.keys():
                    RoadsTimeVel = MobDate.Class2DfSpeedAndTimePercorrenceRoads[IntClass]        
                    RoadsTimeVel = RoadsTimeVel.sort("start_bin")
                    t = 0
                    VarianceVec = []
                    for time,RTV in RoadsTimeVel.groupby("start_bin"):
                        StartInterval = datetime.datetime.fromtimestamp(time)
                        StrTimesLabel.append(StartInterval.strftime("%Y-%m-%d %H:%M:%S").split(" ")[1])
                        VarianceVec.append(np.std(np.array(MobDate.Class2Time2Distr[IntClass][t])))
                    if IntClass in MobDate.Class2AvgTimePercorrence.keys():
                        AvgTimePercorrence = MobDate.Class2AvgTimePercorrence[IntClass]
                        ax.plot(StrTimesLabel[self.CutIndexTime:], AvgTimePercorrence[self.CutIndexTime:], color = colors[IDay], label = MobDate.StrDate)
                        ax.errorbar(StrTimesLabel[self.CutIndexTime:], AvgTimePercorrence[self.CutIndexTime:], yerr=VarianceVec[self.CutIndexTime:], color = colors[IDay],fmt='o',label=None)
                IDay += 1
            ax.set_title("Time Percorrence Distribution for Class {}".format(MobDate.IntClass2StrClass[IntClass]))
            ax.set_xticks(range(len(StrTimesLabel[self.CutIndexTime:]))[::Slicing])  # Set the ticks to correspond to the labels
            Labels = StrTimesLabel[self.CutIndexTime:]
            ax.set_xticklabels(Labels[::Slicing], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
            ax.set_xlabel("time")
            ax.set_ylabel("time Percorrence")
            ax.legend()
            plt.savefig(os.path.join(self.PlotDir,"ComparisonTimePercorrence_{}".format(IntClass)),dpi = 200)
            plt.close()


    def PlotComparisonSubnets(self):
        """
            Description:
                Plot the subnetworks for all days
            Taakes a lot of time to compute
        """
        NewClasses = [0,1,2,3]
        GeoJson = gpd.read_file("/home/aamad/codice/city-pro/output/bologna_mdt_detailed/BolognaMDTClassInfo.geojson")
        Colors = ["red","green","purple","orange","yellow","pink","brown","grey"]
        Index2IJ = {0:(0,0),1:(0,1),2:(0,2),3:(1,0),4:(1,1),5:(1,2),6:(2,0),7:(2,1),8:(2,2)}
        for CountClass,NewClass in enumerate(NewClasses):
            fig, ax = plt.subplots(3, 3, figsize=(15, 10))
            for Index,MobDate in enumerate(self.ListDailyNetwork):
                print("Class {} for Day {}".format(NewClass,MobDate.StrDate))
                try:
                    GdfClass = MobDate.GeoJson.groupby("StrClassOrdered_{}".format(MobDate.StrDate)).get_group(MobDate.IntClass2StrClass[NewClass])
                except:
                    pass
                i = Index2IJ[Index][0]
                j = Index2IJ[Index][1]
                GeoJson.plot(ax=ax[i][j], color="black",alpha = 0.01, linewidth=0.1)
                try:
                    GdfClass.plot(ax=ax[i][j], color=Colors[CountClass], linewidth=2)
                except:
                    pass
                try:
                    ctx.add_basemap(ax[i][j], crs="EPSG:32632")
                except:
                    print("No Basemap")
                ax[i][j].set_title(MobDate.StrDate)
                ax[i][j].set_axis_off()
            # Add legend
#            plt.title('SubNetworks for Class {}'.format(MobDate.IntClass2StrClass[NewClass]))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            fig.suptitle('SubNetworks for Class {}'.format(MobDate.IntClass2StrClass[NewClass]), fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(self.PlotDir,"SubNetworks_{}.png".format(NewClass)),dpi = 200) 
            plt.close()       

    def PlotUnionSubnets(self):
        """
            This snippet works just if I consider 4 classes.

        """
        UniqueClasses = [0,1,2,3]
        Colors = ["red","green","purple","orange","yellow","pink","brown","grey"]
        StrUnion = "OrderedUnion_"
        GeoJson = gpd.read_file(os.path.join(os.environ["WORKSPACE"],"city-pro","output","bologna_mdt_center","BolognaMDTClassInfo.geojson"))
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        Class2Ax = {0:(0,0),1:(0,1),2:(1,0),3:(1,1)}
        IntClass2StrClass = {0:"1 slowest",1:"2 slowest",2:"middle velocity class",3:"1 quickest"}
        for Class in UniqueClasses:
            i = Class2Ax[Class][0]
            j = Class2Ax[Class][1]
            filtered_gdf = GeoJson.loc[GeoJson[StrUnion + IntClass2StrClass[Class]] == True].dropna(subset=['geometry'])
            filtered_gdf.plot(ax=ax[i][j], color=Colors[Class], linewidth=2)
            try:
                ctx.add_basemap(ax[i][j], crs="EPSG:32632")
            except:
                print("No Basemap")
            ax[i][j].set_title("Union All Days Class {}".format(Class))
            ax[i][j].set_axis_off()
        plt.savefig(os.path.join(self.PlotDir,"UnionSubNetworks.png"),dpi = 200) 
        plt.close()


    def PlotIntersectionSubnets(self):
        """
            This snippet works just if I consider 4 classes.

        """
        UniqueClasses = [0,1,2,3]
        Colors = ["red","green","purple","orange","yellow","pink","brown","grey"]
        StrUnion = "OrderedIntersection_"
        IntClass2StrClass = {0:"1 slowest",1:"2 slowest",2:"middle velocity class",3:"1 quickest"}
        GeoJson = gpd.read_file(os.environ["WORKSPACE"],"city-pro","output","bologna_mdt_detailed","BolognaMDTClassInfo.geojson")
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        Class2Ax = {0:(0,0),1:(0,1),2:(1,0),3:(1,1)}
        for Class in UniqueClasses:
            i = Class2Ax[Class][0]
            j = Class2Ax[Class][1]
            filtered_gdf = GeoJson.loc[GeoJson[StrUnion + IntClass2StrClass[Class]] == True].dropna(subset=['geometry'])
            filtered_gdf.plot(ax=ax[i][j], color=Colors[Class], linewidth=2)
            try:
                ctx.add_basemap(ax[i][j], crs="EPSG:32632")
            except:
                print("No Basemap")
            ax[i][j].set_title("Union All Days Class {}".format(Class))
            ax[i][j].set_axis_off()
        plt.savefig(os.path.join(self.PlotDir,"UnionSubNetworks.png"),dpi = 200) 
        plt.close()

    def UnifyGeoJsonAllDays(self):
        """
            Description:
                Unify the GeoJson for all days.
                In particular, obtain from single GeoJson of each day the union and intersection of the subnetworks.
            @brief:
                Obtain the union and intersection of the subnetworks for all days.
        """
        logger.info("Unify GeoJson for all days...")
        FirstDay = True
        for MobDate in self.ListDailyNetwork:
            if FirstDay:
                self.GpdClasses = MobDate.GeoJson
                FirstDay = False
            else:
                invariant_columns = ["poly_lid", "poly_cid", "poly_length", "poly_nF", "poly_nT", "geometry"]
                self.GpdClasses = self.GpdClasses.merge(MobDate.GeoJson, on=invariant_columns, how='outer',suffixes=('', ''))
        if not os.path.exists(os.path.join(self.ListDailyNetwork[0].InputBaseDir,"BolognaMDTClassInfo.geojson")):
            self.GpdClasses.to_file(os.path.join(MobDate.InputBaseDir,"BolognaMDTClassInfo.geojson"))
        
    def ComputeOrderedIntersectionsAllDays(self):
        """
            @brief: Compute the ordered intersections for all days.
            Adds a column for the ordered intersection.
            If all the days ordered classification contain a road, then the class is associated
            to the column OrderedIntersection_{Class}. 
        """
        logger.info("Plot Aggregated Intersection SubNetworks...")
        MobDate = self.ListDailyNetwork[0]
        if self.GpdClasses is None:
            logger.info("GpdClasses is None, reading {}".format(os.path.join(MobDate.InputBaseDir,"BolognaMDTClassInfo.geojson")))                
            self.GpdClasses =  gpd.read_file(os.path.join(MobDate.InputBaseDir,"BolognaMDTClassInfo.geojson"))
        StrClassesOrderedColsDate = {Col: Col.split("_")[1] for Col in self.GpdClasses.columns if Col.startswith("StrClassOrdered_")}
        self.UniqueClassesOrdered = np.unique(self.GpdClasses[list(StrClassesOrderedColsDate.keys())[1]].astype(str).values)
        self.OrderedClass2Road2Intersection,self.OrderedClass2Road2Union = GetIncrementalIntersectionAllDaysClasses(self.GpdClasses,StrClassesOrderedColsDate,self.UniqueClassesOrdered)
        #Class2Road2Intersection,Class2Road2Union = GetIncrementalIntersectionAllDaysClasses(GpdClasses,StrClassesOrderedColsDate,UniqueClasses)
        self.GpdClasses = UpdateGeoJsonWithUnionAndIntersectionColumns(self.GpdClasses,self.OrderedClass2Road2Intersection,self.OrderedClass2Road2Union,StrUnion = "OrderedUnion_",StrIntersection = "OrderedIntersection_")

    def PlotAggregatedSubNetworks(self):
        """
            @brief: Compute the ordered intersections for all days.
            Adds a column for the ordered intersection.
            If all the days ordered classification contain a road, then the class is associated
            to the column OrderedUnion_{Class}.
        """
        logger.info("Plot Aggregated Union SubNetworks...")
        MobDate = self.ListDailyNetwork[0]
        if self.GpdClasses is None:
            logger.info("GpdClasses is None, reading {}".format(os.path.join(MobDate.InputBaseDir,"BolognaMDTClassInfo.geojson")))
            self.GpdClasses =  gpd.read_file(os.path.join(MobDate.InputBaseDir,"BolognaMDTClassInfo.geojson"))
        Colors = ['red','blue','green','orange','purple','yellow','cyan','magenta','lime','pink','teal','lavender','brown','beige','maroon','mint','coral','navy','olive','grey']
        # Ordered Case
        StrClassesOrderedColsDate = {Col: Col.split("_")[1] for Col in self.GpdClasses.columns if Col.startswith("StrClassOrdered_")}
        UniqueClassesOrdered = np.unique(self.GpdClasses[list(StrClassesOrderedColsDate.keys())[1]].values)
        self.StrClassesOrdered2Color = {StrClass: Colors[i] for i, StrClass in enumerate(UniqueClassesOrdered)}
                # Unordered Case
        OrderedClass2Road2Intersection,OrderedClass2Road2Union = GetIncrementalIntersectionAllDaysClasses(self.GpdClasses,StrClassesOrderedColsDate,UniqueClassesOrdered)
        #Class2Road2Intersection,Class2Road2Union = GetIncrementalIntersectionAllDaysClasses(GpdClasses,StrClassesOrderedColsDate,UniqueClasses)
        self.GpdClasses = UpdateGeoJsonWithUnionAndIntersectionColumns(self.GpdClasses,OrderedClass2Road2Intersection,OrderedClass2Road2Union,StrUnion = "OrderedUnion_",StrIntersection = "OrderedIntersection_")
        m1 = PlotUnion(self.GpdClasses,OrderedClass2Road2Union,self.StrClassesOrdered2Color,"OrderedUnion_")
        m1.save(os.path.join(self.PlotDir,"OrderedUnionAllDays.html"))

    def PlotPenetration(self):
        """
            Description:
        """
        logger.info("Plot Penetration")
        from EstimatePenetration import EstimatePenetrationAndPlot
        self.DirTimedFluxes = []
        self.DaysPenetration = []
        for MobDate in self.ListDailyNetwork:
            if "2023" not in MobDate.DictDirInput["timed_fluxes"]:
                self.DirTimedFluxes.append(os.path.join(MobDate.DictDirInput["timed_fluxes"]))
                self.DaysPenetration.append(MobDate.StrDate)
        DfTrafficOpenData = pl.read_csv(os.path.join(os.environ["WORKSPACE"],"city-pro","vars","data","rilevazione-flusso-veicoli-tramite-spire-anno-2022.csv"),separator = ";",truncate_ragged_lines=True)
        EstimatePenetrationAndPlot(self.GpdClasses,DfTrafficOpenData,self.bounding_box,self.DaysPenetration,self.DirTimedFluxes,self.CutIndexTime,self.PlotDir)