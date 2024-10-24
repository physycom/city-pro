from AnalysisNetwork1Day import *
from analysisPlot import *
from collections import defaultdict
import numpy as np
from LatexFunctions import *
from UsefulStructures import *
import contextily as ctx
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
                self.MFDAggregated = {Key: [] for Key in MobDate.MFD.columns}
                self.StrClass2MFDAggregated = defaultdict() 
                self.StrClass2MFDNewAggregated = defaultdict()                 
                self.StrClass2MFDAggregated2Plot = defaultdict()
                self.StrClass2MFDNewAggregated2Plot = defaultdict()
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
        self.MinMaxPlotPerClass = {StrClass: defaultdict() for StrClass in self.ListStrClassReference}
        self.MinMaxPlotPerClassNew = {StrClass: defaultdict() for StrClass in self.ListStrClassReference}

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
            print("StrDate:\n",MobDate.StrDate)
            print("FcmCenters:\n",MobDate.FcmCenters)
            print("Fcm:\n",MobDate.Fcm)

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
                    for Feature in MobDate.Feature2AllFitTry.keys():
                        self.Aggregation2Feature2Function2Fit2InitialGuess[Aggregation][Feature] = defaultdict()
                        self.Aggregation2Feature2Class2Function2Fit2InitialGuess[Aggregation][Feature] = defaultdict()

                        for Function2Fit in MobDate.Feature2Function2Fit2InitialGuess[Feature].keys():
                            self.Aggregation2Feature2Function2Fit2InitialGuess[Aggregation][Feature][Function2Fit] = MobDate.Feature2Function2Fit2InitialGuess[Feature][Function2Fit]
        for Aggregation in self.AggregationLevel:
            for MobDate in self.ListDailyNetwork:
                if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]:
                    for Feature in MobDate.Feature2AllFitTry.keys():
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
                for Feature in self.Features2Fit:
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
            PlotFeatureAggregatedWithoutFitRescaledByMean(self.Aggregation2Feature2StrClass2FcmDistr,
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

    def ComputeAggregatedMFDVariablesObj(self):
        """
            Description:
                Every Day I count for each hour, how many people and the speed of the 
                1. Network -> MFDAggregated = {"population":[],"time":[],"speed_kmh":[]}
                2. SubNetwork -> Class2MFDAggregated = {StrClass: {"population":[sum_i pop_{t0,dayi},...,sum_i pop_{iteration,dayi}],"time":[t0,...,iteration],"speed_kmh":[sum_i speed_{t0,dayi},...,sum_i speed_{iteration,dayi}]}}
                NOTE: time is pl.DateTime
            NOTE: Each Time interval has its own average speed and population. For 15 minutes,
                since iteration in 1 Day Analysis is set in that way. 
        """
        MobDate = self.ListDailyNetwork[0]
        for Class,_ in MobDate.Fcm.groupby("class"):
            # Compute Average MFD per Class
            self.MFDAggregated = ComputeAggregatedMFDVariables(self.ListDailyNetwork,self.MFDAggregated,Class,False)
#            self.Aggregation2MFD = AggregateMFDByHolidays(self.ListDailyNetwork,self.AggregationLevel2ListDays)
# NOTE: The line above in case you want to show the difference between holidays and not holidays.
        self.ComputedMFDAggregatedVariablesBool = True

    def PlotComparisonPopulationTime(self):
        """
            Description:
                Plot the population and time for each day.
        """
        PlotDay2PopulationTime(self.Day2PopTime,self.PlotDir)

    def ComputeMFD2PlotAggregation(self):
        """
            Compute the average MFD for all days.
        """
        self.MFDAggregated2Plot, self.MinMaxPlot,RelativeChange = GetMFDForPlot(MFD = self.MFDAggregated,
                                                                    MFD2Plot = self.MFDAggregated2Plot,
                                                                    MinMaxPlot = self.MinMaxPlot,
                                                                    Class = None,
                                                                    case = "no-classes",
                                                                    NewClass = False,
                                                                    bins_ = 20)
        self.MFDAggregated2Plot, self.MinMaxPlot,RelativeChange = GetMFDForPlot(MFD = self.MFDAggregated,
                                                                    MFD2Plot = self.MFDAggregated2Plot,
                                                                    MinMaxPlot = self.MinMaxPlot,
                                                                    Class = None,
                                                                    case = "no-classes",
                                                                    NewClass = False,
                                                                    bins_ = 20)


    def PlotMFDPerClassCompared(self):
        """
            Plots the MFD for all the different days together in the same plot for each class .
        """
        for Class in self.IntClasses:
            PlotMFDComparison(self.ListDailyNetwork,Class,self.ListColors,True,self.PlotDir)
            PlotMFDComparison(self.ListDailyNetwork,Class,self.ListColors,False,self.PlotDir)
            
                    
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

    def GenerateAndSaveTabAvSpeed(self):
        # Initialize the code string with the start of the tab definition
        Feature2Label = {"lenght_km":"length (km)","speed_kmh":"speed (km/h)","time_hours":"time (h)"}
        self.AvFeat2Class2Day = InitAvFeat2Class2Day(self.StrDates,self.Day2StrClass2IntClass,Feature2Label)
        for Feature in Feature2Label.keys():
            for MobDate in self.ListDailyNetwork:
                for StrClass in self.Day2StrClass2IntClass[MobDate.StrDate].keys():
                    IntClass = self.Day2StrClass2IntClass[MobDate.StrDate][StrClass]
                    if "var" in MobDate.Feature2IntClass2Feat2AvgVar[Feature][IntClass].keys() and "avg" in MobDate.Feature2IntClass2Feat2AvgVar[Feature][IntClass].keys():
                        RoundAvg = round(MobDate.Feature2IntClass2Feat2AvgVar[Feature][IntClass]["avg"],3)
                        RoundVar = round(MobDate.Feature2IntClass2Feat2AvgVar[Feature][IntClass]["var"],3)
                        self.AvFeat2Class2Day[Feature][StrClass][MobDate.StrDate] =  str(RoundAvg) + " $\pm$ " + str(RoundVar)
                    else:
                        self.AvFeat2Class2Day[Feature][StrClass][MobDate.StrDate] =  " "
            LatexTableAvFeat = TableFromDict(self.AvFeat2Class2Day[Feature])
            with open(os.path.join(self.PlotDir,f"LatexTableAvFeat_{Feature}.txt"), "w") as file:
                file.write(LatexTableAvFeat)
            self.GenerateAndSaveTabFit()    

    def GenerateAndSaveTabFit(self):
        """
            Description:
                Generates the table for the fit parameters:
                    Feature2Parameters2Class2Day
                Generates the dictionary with parameters and best fit:
                    Aggregation2Feature2Class2Day2Feature
                NOTE: Used in the scatter plot whein the plane of the parameters.
        """
        Feature2Label = {"lenght_km":"length (km)","speed_kmh":"speed (km/h)","time_hours":"time (h)"}
        self.Feature2Parameters2Class2Day = {Aggregation:{Feature: {StrClass: {StrDay: "" for StrDay in self.StrDates} for StrClass in self.ListStrClassReference} for Feature in Feature2Label.keys()} for Aggregation in self.AggregationLevel}
        self.Aggregation2Feature2Class2Day2Feature = {Aggregation: {Feature: {StrClass: {StrDay: {"parameters":[],"best_fit":""} for StrDay in self.StrDates} for StrClass in self.ListStrClassReference} for Feature in Feature2Label.keys()} for Aggregation in self.AggregationLevel}
        for Aggregation in self.AggregationLevel:
            for Feature in Feature2Label.keys():
                print("Feature: ",Feature)
                for MobDate in self.ListDailyNetwork:
                    print("Day: ",MobDate.StrDate)
                    for StrClass in self.ListStrClassReference:
                        IntClass = self.Day2StrClass2IntClass[MobDate.StrDate][StrClass]
                        StrBestFit = MobDate.Feature2Class2AllFitTry[Feature][IntClass]["best_fit"]
                        RoundedParam0 = round(MobDate.Feature2Class2AllFitTry[Feature][IntClass][StrBestFit]["parameters"][0],3)
                        RoundedParam1 = round(MobDate.Feature2Class2AllFitTry[Feature][IntClass][StrBestFit]["parameters"][1],3)
                        self.Aggregation2Feature2Class2Day2Feature[Aggregation][Feature][StrClass][MobDate.StrDate]["parameters"] =[RoundedParam0,RoundedParam1]
                        self.Aggregation2Feature2Class2Day2Feature[Aggregation][Feature][StrClass][MobDate.StrDate]["best_fit"] =StrBestFit
                        print("Class: ",StrClass," Best Fit: ",StrBestFit)
                        if StrBestFit == "exponential":
                            self.Feature2Parameters2Class2Day[Aggregation][Feature][StrClass][MobDate.StrDate] = "A = " + str(RoundedParam0) + " $\\beta$ = " + str(RoundedParam1)
                        elif StrBestFit == "linear":
                            self.Feature2Parameters2Class2Day[Aggregation][Feature][StrClass][MobDate.StrDate] = "A = " + str(RoundedParam0) + " $\\alpha$ = " + str(RoundedParam1)
                        elif StrBestFit == "gaussian" or StrBestFit == "maxwellian":
                            self.Feature2Parameters2Class2Day[Aggregation][Feature][StrClass][MobDate.StrDate] = "$\mu$ = " + str(RoundedParam0) + " $\sigma$ = " + str(RoundedParam1)                   
                        else:
                            self.Feature2Parameters2Class2Day[Aggregation][Feature][StrClass][MobDate.StrDate] = " "
                LatexTableAvParameters = TableFromDict(self.Feature2Parameters2Class2Day[Aggregation][Feature])
                with open(os.path.join(self.PlotDir,f"LatexTableParameters_{Feature}_{Aggregation}.txt"), "w") as file:
                    file.write(LatexTableAvParameters)
        

                
    def PlotAveragesFeaturesDiscriminatingHolidays(self):
        LocalFeature2AvgStdClass = self.PrepareAvg2StdClassFromAvgInputLatexTable()
        LocalFeature2ParFitClass = self.PrepareParFitClassFromTabFitInput()
        Types = ["holidays","not_holidays"]
        Class2Types2Shape,Class2Types2Colors = GetClass2Type2ShapesAndColors(self.ListStrClassReference,Types)
        for Feature in LocalFeature2AvgStdClass.keys():
            Avg = np.array(LocalFeature2AvgStdClass[Feature]["avg"],dtype = np.float32)
            Std = np.array(LocalFeature2AvgStdClass[Feature]["std"],dtype = np.float32)
            Class = LocalFeature2AvgStdClass[Feature]["class"]
            Types = LocalFeature2AvgStdClass[Feature]["type"]
            Title = "Average and Standard Deviation  {} ".format(Feature)
            print("Plot Comparison Avg and Std for Feature: ",Feature)
            PlotIntervals(Avg,
                          Std,
                          Class,
                          Types,
                          Class2Types2Colors,
                          Class2Types2Shape,
                          Title,
                          Feature,
                          self.PlotDir,
                          "Average_Std_{}".format(Feature))
            if Feature == "speed_kmh" or Feature == "av_speed":
                Xlabel = "mu"
                Ylabel = "sigma"
            elif Feature == "time_hours" or Feature == "lenght_km" or Feature == "time" or Feature == "lenght":
                Xlabel = "A"
                Ylabel = "beta"
            A = LocalFeature2ParFitClass[Feature]["A"]
            b = LocalFeature2ParFitClass[Feature]["b"]
            Class = LocalFeature2ParFitClass[Feature]["class"]
            Types = LocalFeature2ParFitClass[Feature]["type"]
            print("Plot Comparison Fit Parameters for Feature: ",Feature)
            ScatterFitParams(A,
                             b, 
                             Class,
                             Types,
                             Class2Types2Colors,
                             Class2Types2Shape,
                             "Parameters Fit",
                             Xlabel,
                             Ylabel,
                             self.PlotDir,
                             "ParametersFit_{}".format(Feature))
            
    def PrepareAvg2StdClassFromAvgInputLatexTable(self):
        Types = ["holidays","not_holidays"]
        LocalFeature2AvgStdClass = {Feature: {"avg": [], "std": [],"class":[],"type":[]} for Feature in self.Features2Fit}
        print("Preparing Plot Averages")
        for Feature in self.AvFeat2Class2Day.keys():
            for StrClass in self.AvFeat2Class2Day[Feature].keys():
                for Type in Types:
                    for StrDay in self.AvFeat2Class2Day[Feature][StrClass].keys():
                        StrAvgStd = self.AvFeat2Class2Day[Feature][StrClass][StrDay]
                        if StrDay in self.AggregationLevel2ListDays[Type]:
                            if isinstance(StrAvgStd,str):
                                LocalFeature2AvgStdClass[Feature]["avg"].append(StrAvgStd.split(" $\pm$ ")[0])
                                LocalFeature2AvgStdClass[Feature]["std"].append(StrAvgStd.split(" $\pm$ ")[1])
                                LocalFeature2AvgStdClass[Feature]["class"].append(StrClass)
                                LocalFeature2AvgStdClass[Feature]["type"].append(Type)
                            else:
                                print("List in:")
                                print("Feature: ",Feature," StrClass: ",StrClass," StrDay: ",StrDay,"StrAvgStd: ",StrAvgStd," Type: ",Type)
        return LocalFeature2AvgStdClass

    def PrepareParFitClassFromTabFitInput(self):
        """
            Description:
                Prepare the dictionary for the scatter plot of the parameters.
                LocalFeature2ParFitClass = {Feature: {"A": [Aday0,...,Adayn],
                                                      "b": [bDay0,...,bDayn],
                                                      "class":[ClassDay0,...,ClassDayn],
                                                      "type":[TypeDay0,...,TypeDayn]} for Feature in self.Features2Fit}
        """
        Types = ["holidays","not_holidays"]
        LocalFeature2ParFitClass = {Feature: {"A": [], "b": [],"class":[],"type":[]} for Feature in self.Features2Fit}
        for Feature in self.Aggregation2Feature2Class2AllFitTry.keys():
            for StrClass in self.Aggregation2Feature2Class2AllFitTry[Feature].keys():
                for Type in Types:
                    for StrDay in self.Aggregation2Feature2Class2AllFitTry[Feature][StrClass].keys():
                        if StrDay in self.AggregationLevel2ListDays[Type]:
                            A = self.Aggregation2Feature2Class2Day2Feature[Type][Feature][StrClass][StrDay]["parameters"][0]
                            b = self.Aggregation2Feature2Class2Day2Feature[Type][Feature][StrClass][StrDay]["parameters"][1]
                            LocalFeature2ParFitClass[Feature]["A"].append(A)
                            LocalFeature2ParFitClass[Feature]["b"].append(b)
                            LocalFeature2ParFitClass[Feature]["class"].append(StrClass)
                            LocalFeature2ParFitClass[Feature]["type"].append(Type)
        return LocalFeature2ParFitClass
    

# PLOT FEATURES FOR ALL DAYS

    def PlotComparisonDistributionEachFeatureAllDays(self):
        """
            Description:
                Plot the distribution of each feature given the class for all days
        """
        Colors = ["blue","red","green","yellow","black","orange","purple","pink","brown","grey"]
        Features = ["speed_kmh","lenght_km","time_hours"]
        Feature2Label = {"lenght_km":"L (km)","speed_kmh":"v (kmh)","time_hours":"t (h)"}
        if self.AddStrClassColumn2FcmBool:
            if self.AssociateAvSpeed2StrClassBool:
                for Aggregation in self.Aggregation2Class2Fcm.keys():
                    for StrClass in self.Aggregation2Class2Fcm[Aggregation]:
                        for Feature in Features:
                            fig,ax = plt.subplots(1,1,figsize = (10,8))                    
                            legend = []
                            CountDay = 0
                            for MobDate in self.ListDailyNetwork:
                                CountDay += 1
                                if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]:                                
                                    y,x = np.histogram(MobDate.Fcm.filter(pl.col("str_class") == StrClass)[Feature],bins = 50)
                                    P = y/np.sum(y)
                                    x_mean = np.mean(x)
                                    variance = np.var(x)/np.sqrt(len(x))
#                                    legend.append(MobDate.StrDate)
                                    ax.scatter(x[:-1],P,color = Colors[CountDay],label=MobDate.StrDate)
#                                    ax.vlines(x_mean,0,max(P),color = Colors[CountDay],label=None)
#                                    ax.vlines(x_mean - variance,0,1,color = Colors[CountDay])
#                                    ax.vlines(x_mean + variance,0,1,color = Colors[CountDay])
                            ax.set_xlabel(Feature2Label[Feature])
                            ax.set_ylabel("P({})".format(Feature2Label[Feature]))
                            ax.set_title("Distribution of {} for {}".format(Feature2Label[Feature],StrClass))
    #                       ax.legend(legend)
                            if "speed" not in Feature:
                                ax.set_yscale("log")
                                ax.set_xscale("log")
                            plt.savefig(os.path.join(self.PlotDir,"Distribution_{0}_{1}_{2}.png".format(Feature,Aggregation,StrClass)),dpi = 200)
                            plt.close()
            else:
                Message = "Plot Classes"

    def PlotComparisonDistributionEachFeatureAllDaysRescaledByMean(self):
        """
            Description:
                Plot the distribution of each feature given the class for all days
        """
        Colors = ["blue","red","green","yellow","black","orange","purple","pink","brown","grey"]
        Features = ["speed_kmh","lenght_km","time_hours"]
        Feature2Label = {"lenght_km":"L","speed_kmh":"v","time_hours":"t"}
        Feature2AvgLabel = {"lenght_km":"<L>","speed_kmh":"<v>","time_hours":"<t>"}
        if self.AddStrClassColumn2FcmBool:
            if self.AssociateAvSpeed2StrClassBool:
                for Aggregation in self.Aggregation2Class2Fcm.keys():
                    for StrClass in self.Aggregation2Class2Fcm[Aggregation]:
                        for Feature in Features:
                            fig,ax = plt.subplots(1,1,figsize = (10,8))                    
                            legend = []
                            CountDay = 0
                            for MobDate in self.ListDailyNetwork:
                                CountDay += 1
                                if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]:                                
                                    y,x = np.histogram(MobDate.Fcm.filter(pl.col("str_class") == StrClass)[Feature],bins = 50)
                                    P = y/np.sum(y)
                                    x_mean = np.mean(x)
                                    x = x/x_mean
                                    x_mean = np.mean(x)
                                    variance = np.var(x)/np.sqrt(len(x))
#                                    legend.append(MobDate.StrDate)
                                    ax.scatter(x[:-1],P,color = Colors[CountDay],label=MobDate.StrDate)
#                                    ax.vlines(x_mean,0,max(P),color = Colors[CountDay],label=None)
#                                    ax.vlines(x_mean - variance,0,1,color = Colors[CountDay])
#                                    ax.vlines(x_mean + variance,0,1,color = Colors[CountDay])
                            ax.set_xlabel("{0}/{1}".format(Feature2Label[Feature],Feature2AvgLabel[Feature]))
                            ax.set_ylabel("P({0}/{1})".format(Feature2Label[Feature],Feature2AvgLabel[Feature]))
                            ax.set_title("Distribution of {0} for {1} ".format(Feature,StrClass))
                            ax.legend()
                            if "speed" not in Feature:
                                ax.set_yscale("log")
                                ax.set_xscale("log")
                            plt.savefig(os.path.join(self.PlotDir,"RescaledX_Distribution_{0}_{1}_{2}.png".format(Feature,Aggregation,StrClass)),dpi = 200)
                            plt.close()

            else:
                Message = "Plot Classes"


    def PlotComparisonDistributionInDays(self):
        """
            Description:
                Plot the distribution of each feature not conditioned on the class for all days
        """
        Colors = ["blue","red","green","yellow","black","orange","purple","pink","brown","grey"]
        Features = ["speed_kmh","lenght_km","time_hours"]
        Feature2Label = {"lenght_km":"L","speed_kmh":"v","time_hours":"t"}
        Feature2AvgLabel = {"lenght_km":"<L>","speed_kmh":"<v>","time_hours":"<t>"}        
        if self.AddStrClassColumn2FcmBool:
            if self.AssociateAvSpeed2StrClassBool:
                for Aggregation in self.Aggregation2Class2Fcm.keys():
                    for Feature in Features:
                        fig,ax = plt.subplots(1,1,figsize = (10,8))                    
                        legend = []
                        CountDay = 0
                        for MobDate in self.ListDailyNetwork:
                            CountDay += 1
                            if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]:                                
                                y,x = np.histogram(MobDate.Fcm[Feature],bins = 50)
                                P = y/np.sum(y)
                                x_mean = np.mean(x)
                                variance = np.var(x)/np.sqrt(len(x))
#                                legend.append(MobDate.StrDate)
                                ax.scatter(x[:-1],P,color = Colors[CountDay],label=MobDate.StrDate)
                                ax.vlines(x_mean,0,max(P),color = Colors[CountDay],linestyle = "--",label=None)
                        ax.set_xlabel(Feature2Label[Feature])
                        ax.set_ylabel("P({})".format(Feature2Label[Feature]))
                        ax.set_title("Distribution of {0} for {1}".format(Feature,Aggregation))
                        ax.legend()
                        if "speed" not in Feature:
                            ax.set_yscale("log")
                        plt.savefig(os.path.join(self.PlotDir,"Distribution_{0}_{1}.png".format(Feature,Aggregation)),dpi = 200)
                        plt.close()

            else:
                Message = "Plot Classes"


    def PlotDistrFeaturepowerLawComparisonAllDays(self):
        """
            Plots the compared distribution of lenght, time for all days,
            With power law fit.
        """
        colors = ["blue","red","green","yellow","black","orange","purple","pink","brown","grey"]
        Features = ["speed_kmh","lenght_km","time_hours"]
        Feature2Label = {"lenght_km":"L","speed_kmh":"v","time_hours":"t"}
        Feature2AvgLabel = {"lenght_km":"<L>","speed_kmh":"<v>","time_hours":"<t>"}        
        for Feature in ["lenght_km","time_hours"]:
            for Aggregation in self.Aggregation2Class2Fcm.keys():
                fig,ax = plt.subplots(1,1,figsize = (10,10))
                CountDay = 0
                legend = []
                for MobDate in self.ListDailyNetwork:            
                    CountDay += 1
                    if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]:    
                        y,x = np.histogram(MobDate.Fcm[Feature],bins = 50)  
                        y = y/np.sum(y)
                        x_mean = np.mean(x)
                        if "time" in Feature:
                            fit = pwl.Fit(np.array(MobDate.Fcm[Feature]),
                                        xmin = min(np.array(MobDate.Fcm[Feature])[2:]),
                                        xmax = max(np.array(MobDate.Fcm[Feature])),
                                        initial_guess = (-1,1))
                        else:
                            fit = pwl.Fit(np.array(MobDate.Fcm[Feature]),
                                        xmin = min(np.array(MobDate.Fcm[Feature])),
                                        xmax = max(np.array(MobDate.Fcm[Feature])),
                                        initial_guess = (-1,1))
                        ax.scatter(x[:-1],y,color=colors[CountDay],label=MobDate.StrDate)
                        if "time" in Feature:
                            ax.plot(x[2:-1],x[2:-1]**(-fit.alpha),color=colors[CountDay],linestyle = "--",label=None)
                        else:
                            ax.plot(x[1:-1],x[1:-1]**(-fit.alpha),color=colors[CountDay],linestyle = "--",label=None)
                        ax.vlines(x_mean,0,max(y),color=colors[CountDay],linestyles = "dashed",label=None)
                        print(f"Power Law {MobDate.StrDate}: ",fit.alpha)
#                        print(f"Power Lae Truncated {MobDate.StrDate}: ",fit_[0][0],fit_[0][1],fit_[0][2])
#                        ax.plot(x[:-1],fit_[0][0]*x[:-1]**(fit_[0][1])*np.exp(-x[:-1]*fit_[0][2]),color=colors[CountDay],linestyle = "--",label=None)

#                        legend.append(MobDate.StrDate)
                ax.legend()
                ax.set_yscale("log")
                ax.set_xlabel(Feature2Label[Feature])
                ax.set_ylabel("P({})".format(Feature2Label[Feature]))
                ax.set_title(Feature)
                plt.savefig(os.path.join(self.PlotDir,"PowerLawFit_{0}_{1}.png".format(Aggregation,Feature)),dpi = 200)
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

    def PlotNumberTrajectoriesGivenClass(self):
        """
            Description:
                Plots the number of trajectories given the class for all days.
        """
        colors = ["blue","red","green","yellow","black","orange","purple","pink","brown","grey"]
        Feature2Label = {"lenght_km":"L","time_hours":"t"}
        for Aggregation in ["aggregated"]:
            for Feature in ["lenght_km","time_hours"]:
                # Compute the alphas that define for a given day fraction of trajectories that are to a class (RhoK)
                RhoKGivenDay = []
                RhoxGivenKGivenDay = []
                RhoxGivenDay = []
                fig,ax = plt.subplots(1,1,figsize = (10,8))
                for MobDate in self.ListDailyNetwork:
                    ParametersExpoFitDay = []
                    for StrClass in self.ListStrClassReference:
                        IntClass = self.Day2StrClass2IntClass[MobDate.StrDate][StrClass]
                        StrBestFit = MobDate.Feature2Class2AllFitTry[Feature][IntClass]["best_fit"]
                        yClass,xClass = np.histogram(MobDate.Fcm.filter(pl.col("str_class")==StrClass)[Feature],bins = 50)  
                        RhoxGivenKGivenDay.append(yClass/np.sum(yClass)) 
                        if StrBestFit == "exponential":
#                            RoundedParam0 = round(MobDate.Feature2Class2AllFitTry[Feature][IntClass][StrBestFit]["parameters"][0],3)
                            Inversexk = round(MobDate.Feature2Class2AllFitTry[Feature][IntClass][StrBestFit]["parameters"][1],3)
                            ParametersExpoFitDay.append(Inversexk)
                        if IntClass == 0:
                            xmax = Inversexk
                    xk = [1/Inversexk_ for Inversexk_ in ParametersExpoFitDay]
                    k = np.arange(len(xk)) + 1
                    logk = np.log(k) 
                    fit,StdError,ConvergenceSuccess,FittedData,x_windowed,y_measured = FitAndStdError(x = logk,
                                                                        y_measured = np.log(xk),
                                                                        label = "linear",
                                                                        initial_guess = (1,-xmax)
                                                                        )
                    # Extract Distribution Feature For Day
                    y,x = np.histogram(MobDate.Fcm[Feature],bins = 50) 
                    if len(k)!=4:
                        k = np.insert(k,0,2) 
                    RhoxGivenDay.append(y/np.sum(y))    # Shape (Day,50)
                    RhoKGivenDay.append(k[:4]**fit[0][0]/xmax)  # Shape (Day,4)
                    ax.scatter(logk,np.log(xk),label=MobDate.StrDate)
                    ax.plot(np.log(k)*fit[0][0] + fit[0][1],label=MobDate.StrDate + r" $\alpha$: {}".format(round(fit[0][0],2)))
                ax.set_xlabel("log(k)")
                ax.set_ylabel(r"$log(x_k)$")
                ax.set_title(r"$x_k = \frac{k^{\alpha}}{x_{max}}$")
                ax.legend()
                plt.savefig(os.path.join(self.PlotDir,"ScalingClass.png"),dpi = 200)
                plt.close()
                # Show that Rhox = \int_k Rho_k RhoXGivenK dK
                fig,ax = plt.subplots(1,1,figsize = (10,10))
                legend = []
                CountDay = 0
                CountDayPlusClass = 0
                for MobDate in self.ListDailyNetwork:
                    ReconstructedClass = np.zeros(50)
                    for Class in range(len(RhoKGivenDay[CountDay])-1):
                        ReconstructedClass += RhoKGivenDay[CountDay][Class]*RhoxGivenKGivenDay[CountDayPlusClass]
                        CountDayPlusClass += 1
                    ReconstructedClass = ReconstructedClass/len(RhoKGivenDay[CountDay])
                    ax.plot(x[:-1],ReconstructedClass,color=colors[CountDay],linestyle = "--",label=None)
                    ax.scatter(x[:-1],RhoxGivenDay[CountDay],color=colors[CountDay],label=MobDate.StrDate)
                    CountDay += 1
                ax.legend()
                ax.set_yscale("log")
                ax.set_xlabel(Feature2Label[Feature])
                ax.set_ylabel("P({})".format(Feature2Label[Feature]))
                ax.set_title(r"$\rho(x) = \int_k \rho(k) \rho(x|k) dk$")
                plt.savefig(os.path.join(self.PlotDir,"ComparisonHeterogeneityHpRealDistr_{0}_{1}.png".format(Aggregation,Feature)),dpi = 200)
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
                        ax.plot(StrTimesLabel, AvgTimePercorrence, color = colors[IDay], label = MobDate.StrDate)
                        ax.errorbar(StrTimesLabel, AvgTimePercorrence, yerr=VarianceVec, color = colors[IDay],fmt='o',label=None)
                IDay += 1
            ax.set_title("Time Percorrence Distribution for Class {}".format(MobDate.IntClass2StrClass[IntClass]))
            ax.set_xticks(range(len(StrTimesLabel))[::Slicing])  # Set the ticks to correspond to the labels
            ax.set_xticklabels(StrTimesLabel[::Slicing], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
            ax.set_xlabel("Time")
            ax.set_ylabel("Time Percorrence")
            ax.legend()
            plt.savefig(os.path.join(self.PlotDir,"ComparisonTimePercorrence_{}".format(IntClass)),dpi = 200)
            plt.close()


    def PlotComparisonSubnets(self):
        """
            Description:
                Plot the subnetworks for all days
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
        GeoJson = gpd.read_file("/home/aamad/codice/city-pro/output/bologna_mdt_detailed/BolognaMDTClassInfo.geojson")
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
        GeoJson = gpd.read_file("/home/aamad/codice/city-pro/output/bologna_mdt_detailed/BolognaMDTClassInfo.geojson")
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
                GeoJson = MobDate.GeoJson
                FirstDay = False
            else:
                invariant_columns = ["poly_lid", "poly_cid", "poly_length", "poly_nF", "poly_nT", "geometry"]
                GeoJson = GeoJson.merge(MobDate.GeoJson, on=invariant_columns, how='outer',suffixes=('', ''))
        GeoJson.to_file(os.path.join(MobDate.InputBaseDir,"BolognaMDTClassInfo.geojson"))

    def ComputeOrderedIntersectionsAllDays(self):
        MobDate = self.ListDailyNetwork[0]
        GpdClasses =  gpd.read_file(os.path.join(MobDate.InputBaseDir,"BolognaMDTClassInfo.geojson"))
        self.OrderedClass2Road2Intersection,self.OrderedClass2Road2Union = GetIncrementalIntersectionAllDaysClasses(GpdClasses,StrClassesOrderedColsDate,UniqueClassesOrdered)
        #Class2Road2Intersection,Class2Road2Union = GetIncrementalIntersectionAllDaysClasses(GpdClasses,StrClassesOrderedColsDate,UniqueClasses)
        self.GpdClasses = UpdateGeoJsonWithUnionAndIntersectionColumns(GpdClasses,self.OrderedClass2Road2Intersection,self.OrderedClass2Road2Union,StrUnion = "OrderedUnion_",StrIntersection = "OrderedIntersection_")

    def PlotAggregatedSubNetworks(self):
        """
            Plot Union Sub-Networks all days.
        """
        m1 = PlotUnion(self.GpdClasses,self.UniqueClassesOrdered,self.StrClassesOrdered2Color,"OrderedUnion_")
        m1.save("/home/aamad/codice/city-pro/output/bologna_mdt/plots/OrderedUnionAllDays.html")