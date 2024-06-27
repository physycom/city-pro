from AnalysisNetwork1Day import *
from analysisPlot import *
from collections import defaultdict
import numpy as np


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


class NetworkAllDays:
    
    def __init__(self,ListDailyNetwork,PlotDir,verbose = False): 
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
        self.Day2Feature2MaxBins = {MobDate.StrDate:defaultdict() for MobDate in self.ListDailyNetwork}
        LocalCount = 0
        # Initialize List of Reference For StrClasses
        self.InitListStrClassReference()        
        self.Day2IntClass2StrClass = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}
        self.Day2StrClass2IntClass = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}
        self.DictClass2AvSpeed = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}
        #
        self.AggregationLevel = ["holidays","not_holidays","aggregated"]
        self.StrDates = [] 
        for MobDate in self.ListDailyNetwork:
            self.StrDates.append(MobDate.StrDate)         
            self.Day2Feature2MaxBins[MobDate.StrDate] = MobDate.Feature2MaxBins
            # CONSTANT PARAMETERS
            if LocalCount == 0:
                self.Feature2ScaleCount = MobDate.Feature2ScaleCount
                self.Feature2ScaleBins = MobDate.Feature2ScaleBins
                self.centroid = MobDate.centroid
                self.Class2Color = MobDate.Class2Color
                self.ListColors = MobDate.ListColors
                self.DictInitialGuess = MobDate.DictInitialGuess
                self.Column2Label = MobDate.Column2Label
                self.Column2SaveName = MobDate.Column2SaveName
                self.Column2Legend = MobDate.Column2Legend
                self.Feature2MaxBins = MobDate.Feature2MaxBins   
                self.MFDAggregated = {Key: [] for Key in MobDate.MFD.keys()}
                self.StrClass2MFDAggregated = defaultdict() 
                self.StrClass2MFDNewAggregated = defaultdict()                 
                self.StrClass2MFDAggregated2Plot = defaultdict()
                self.StrClass2MFDNewAggregated2Plot = defaultdict()
                for StrClass in self.ListStrClassReference:
                    self.StrClass2MFDAggregated[StrClass] = {Key: [] for Key in MobDate.MFD.keys()}
                    self.StrClass2MFDNewAggregated[StrClass] = {Key: [] for Key in MobDate.MFD.keys()}
                    self.StrClass2MFDAggregated2Plot[StrClass] = {Key: [] for Key in MobDate.MFD2Plot.keys()}
                    self.StrClass2MFDNewAggregated2Plot[StrClass] = {Key: [] for Key in MobDate.MFD2Plot.keys()}
                self.config = MobDate.config

        self.StrDay2Color = {StrDay: self.ListColors[i] for i,StrDay in enumerate(self.StrDates)}
        self.MinMaxPlotPerClass = {StrClass: defaultdict() for StrClass in self.ListStrClassReference}
        self.MinMaxPlotPerClassNew = {StrClass: defaultdict() for StrClass in self.ListStrClassReference}

        # Set The Plot Lim as The Maximum Among all Days
        for Day in self.Day2Feature2MaxBins.keys():
            for Feature in self.Day2Feature2MaxBins[Day].keys():
                for Bins in self.Day2Feature2MaxBins[Day][Feature].keys():
                    self.Feature2MaxBins[Feature][Bins] = max(self.Feature2MaxBins[Feature][Bins],self.Day2Feature2MaxBins[Day][Feature][Bins])
        self.StrDateHolidays = self.config["holiday_dates"]
        # Map The Classes among different days according to the closest average speed        
        self.LogFile = os.path.join(self.PlotDir,"LogFile.txt")
        self.CountFunctions = 0
        self.AddStrClassColumn2Fcm()
        self.AssociateAvSpeed2StrClass()
        self.InitFitInfo()

    def InitListStrClassReference(self):
        RefIntClass2StrClass = {}
        # Take The Day with More Classes.
        NumberClasses = 0
        self.DayReferenceClasses = ""
        for MobDate in self.ListDailyNetwork:
            if len(MobDate.StrClass2IntClass) > NumberClasses:
                NumberClasses = len(MobDate.StrClass2IntClass)
                self.DayReferenceClasses = MobDate.StrDate
                self.ReferenceFcmCenters = MobDate.FcmCenters
                self.RefIntClass2StrClass = MobDate.IntClass2StrClass
            else:
                pass
        self.ListStrClassReference = list(self.RefIntClass2StrClass.keys())
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
        if self.verbose:
            print("Associate AvSpeed to StrClass")
        DayIntClass2IntRef = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}
        # Each day.
        for MobDate in self.ListDailyNetwork:
            # Compare the class speed 
            for class_,Df in MobDate.FcmCenters.group_by("class"):
                # to all the reference class speeds.
                IntClassBestMatch = 0
                MinDifferenceInSpeed = 100000
                # Choose the speed that is closest to the reference
                for class_ref,Df_ref in self.ReferenceFcmCenters.group_by("class"):
                    if np.abs(Df["av_speed"].to_list()[0] - Df_ref["av_speed"].to_list()[0]) < MinDifferenceInSpeed:
                        MinDifferenceInSpeed = np.abs(Df["av_speed"].to_list()[0] - Df_ref["av_speed"].to_list()[0])
                        IntClassBestMatch = class_ref
                    else:
                        pass
                # Fill preparatory dictionary
                self.DictClass2AvSpeed[MobDate.StrDate][class_] = IntClassBestMatch
            # {StrDay: {IntClassStrDay: IntClassRefDay}}
            for Class in self.DictClass2AvSpeed[MobDate.StrDate].keys():
                # {StrDay: {IntClassStrDay: StrClassRefDay}}
                self.Day2IntClass2StrClass[MobDate.StrDate][Class] = self.RefIntClass2StrClass[self.DictClass2AvSpeed[MobDate.StrDate][Class]]
                self.Day2StrClass2IntClass[MobDate.StrDate][self.RefIntClass2StrClass[self.DictClass2AvSpeed[MobDate.StrDate][Class]]] = Class
        if self.verbose:
            print("Day to StrClass to intClass:\n",self.Day2StrClass2IntClass)
            print("Day to IntClass to StrClass:\n",self.Day2IntClass2StrClass)
        # For Each Day Associate a column to FcmCenters and Fcm to the str_class computed above
        for MobDate in self.ListDailyNetwork:
            for Class in MobDate.FcmCenters["class"].unique():
                MobDate.FcmCenters = MobDate.FcmCenters.with_columns(pl.when(pl.col("class") == Class).then(self.Day2IntClass2StrClass[MobDate.StrDate][Class]).otherwise("11").alias("str_class"))
                MobDate.Fcm = MobDate.Fcm.with_columns(pl.when(pl.col("class") == Class).then(self.Day2StrClass2IntClass[MobDate.StrDate][self.Day2IntClass2StrClass[MobDate.StrDate][Class]]).otherwise("11").alias("str_class"))
        self.AssociateAvSpeed2StrClassBool = True
        self.CountFunctions += 1
        Message = "Associate Days Classes to common StrClass:\n"
        Message += "\tself.Day2IntClass2StrClass\n" 
        for Day in self.Day2IntClass2StrClass.keys():
            Message += "\t\t <{}".format(Day)
            for Class in self.Day2IntClass2StrClass[Day].keys():
                Message += ":< {}".format(Class)
                Message += ": {}".format(self.Day2IntClass2StrClass[Day][Class])
            Message += ">"
        Message += ">"
        AddMessageToLog(Message,self.LogFile)
        Message = "\tself.Day2StrClass2IntClass\n"
        for Day in self.Day2StrClass2IntClass.keys():
            Message += "\t\t <{}".format(Day)
            for Class in self.Day2StrClass2IntClass[Day].keys():
                Message += ":< {}".format(Class)
                Message += ": {}".format(self.Day2StrClass2IntClass[Day][Class])
            Message += ">"
        Message = "\tself.DictClass2AvSpeed\n"
        AddMessageToLog(Message,self.LogFile)
        for StrClass in self.DictClass2AvSpeed.keys():
            Message += "\t\t <{}".format(StrClass)
            for AvSpeed in self.DictClass2AvSpeed[StrClass].keys():
                Message += ":< {}".format(AvSpeed)
            Message += ">"
        AddMessageToLog(Message,self.LogFile)

    def InitFitInfo(self):
        """
            This function initializes the dictionaries that will contain the information about the fit.
        """
        if self.AssociateAvSpeed2StrClassBool:
            # FIT
            self.Day2Dict2InitialGuess = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}
            self.Day2DictFittedData = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}
            self.Day2InfoFittedParameters = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}     
            self.Day2Class2DictFittedData = {MobDate.StrDate:{StrClass: defaultdict(dict) for StrClass in self.Day2StrClass2IntClass[MobDate.StrDate]} for MobDate in self.ListDailyNetwork}       
            self.Day2Class2InfoFittedParameters = {MobDate.StrDate:{StrClass: defaultdict(dict) for StrClass in self.Day2StrClass2IntClass[MobDate.StrDate]} for MobDate in self.ListDailyNetwork}
            for MobDate in self.ListDailyNetwork:
                # FIT Features
                self.Day2Dict2InitialGuess[MobDate.StrDate] = MobDate.DictInitialGuess
                self.Day2Dict2InitialGuess[MobDate.StrDate][MobDate.StrDate] = MobDate.Class2DictInitialGuess
                self.Day2DictFittedData[MobDate.StrDate] = MobDate.DictFittedData
                self.Day2InfoFittedParameters[MobDate.StrDate] = MobDate.InfoFittedParameters
                # Per Class
                for StrClass in self.Day2StrClass2IntClass[MobDate.StrDate]:
                    IntClass = self.Day2StrClass2IntClass[MobDate.StrDate][StrClass]
                    self.Day2Class2DictFittedData[MobDate.StrDate][StrClass] = MobDate.Class2DictFittedData[IntClass]
                    self.Day2Class2InfoFittedParameters[MobDate.StrDate][StrClass] = MobDate.Class2InfoFittedParameters[IntClass]

        else:
            raise ValueError("AssociateAvSpeed2StrClassBool is False. Please run AssociateAvSpeed2StrClass()")
    
    def ComparedDaysFit(self):
        """
            Create the Dataframe that contains information about:
                1. Parameters
                2. Best Fit Function
        """
        if self.AssociateAvSpeed2StrClassBool:
            DictFitKeys = ["parameters","best_fit"]
            Columns = ["time","lenght","av_speed","speed_kmh","lenght_km","time_hours"]
            ColumnsDictFittedData = [Feature + "_{}".format() for FI in DictFitKeys for Feature in Columns]
            Index = self.StrDates
            self.DF_FitAggregated = pl.DataFrame(ColumnsDictFittedData,Index)
            self.StrClass2DF_Fit = {StrClass:self.DF_FitAggregated for StrClass in self.ListStrClassReference}
            for MobDate in self.ListDailyNetwork:
                for Feature in Columns:
                    for FitKey in DictFitKeys:
                        self.DF_FitAggregated[Feature+"_{}".format(FitKey)].loc[MobDate.StrDate] = MobDate.Day2DictFittedData[MobDate.StrDate][Feature][FitKey]
            self.CountFunctions += 1
            Message = "ComparedDaysFit\n"
            Message += "\tself.DF_FitAggregated"
            AddMessageToLog(Message,self.LogFile)
            self.DF_FitAggregated.write_csv(os.path.join(self.PlotDir,"FitAllDays.csv"))
            # Per Class
            for StrClass in self.ListStrClassReference:
                for MobDate in self.ListDailyNetwork:
                    DayIntClass = self.Day2StrClass2IntClass[MobDate.StrDate][StrClass]  
                    for Feature in Columns:
                        for FitKey in DictFitKeys:
                            self.StrClass2DF_Fit[StrClass][Feature+"_{}".format(FitKey)].loc[MobDate.StrDate] = MobDate.Day2Class2DictFittedData[MobDate.StrDate][DayIntClass][Feature][FitKey]
                self.StrClass2DF_Fit[StrClass].write_csv(os.path.join(self.PlotDir,"FitAllDays_{}.csv".format(StrClass)))
            self.CountFunctions += 1
            Message += "\tself.StrClass2DF_Fit"
            AddMessageToLog(Message,self.LogFile)


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
            self.Aggregation2Fcm = {Aggregation: pl.DataFrame() for Aggregation in self.AggregationLevel}
            self.Aggregation2Class2Fcm = {Aggregation: {StrClass: pl.DataFrame() for StrClass in self.Day2StrClass2IntClass[self.DayReferenceClasses].keys()} for Aggregation in self.AggregationLevel}
            if self.AssociateAvSpeed2StrClassBool:
                for Aggregation in self.AggregationLevel:
                    for MobDate in self.ListDailyNetwork:
                        self.Aggregation2Fcm[Aggregation] = pl.concat([self.Aggregation2Fcm[Aggregation],MobDate.Fcm])
                        for StrClass in self.ListStrClassReference:
                        # Append time and lenght of the Iterated Day
                            self.Aggregation2Class2Fcm[Aggregation] = pl.concat([self.Aggregation2Class2Fcm[Aggregation],MobDate.Fcm.filter(pl.col("str_class") == StrClass)])
                self.ConcatenatePerClassBool = True
                Message = "ConcatenatePerClass: True\n self.Aggregation2Fcm\n"
                for Aggregation in self.Aggregation2Fcm.keys():
                    Message += "\tAggregation: {}".format(Aggregation)
                    Message += "-> Number of Trajectories: {}".format(len(self.Aggregation2Fcm[Aggregation]))
                    AddMessageToLog(Message,self.LogFile)
                Message = "ConcatenatePerClass: self.Aggregation2Class2Fcm\n"
                for Aggregation in self.Aggregation2Class2Fcm.keys():
                    for StrClass in self.Aggregation2Class2Fcm[Aggregation].keys():
                        Message += "\tAggregation: {}".format(Aggregation)
                        Message += "\tStrClass: {}".format(StrClass)
                        Message += "-> Number of Trajectories: {}".format(len(self.Aggregation2Class2Fcm[Aggregation][StrClass]))
                    AddMessageToLog(Message,self.LogFile)
            else:
                Message = "ConcatenatePerClass: False"
## Put together space and time for all days for each class.

    def ComputeMFDAllDays(self):
        """
            Description:
                Compute the MFD averaged with all days data.
        """
        self.Aggregation2MFD = {Aggregation:{MobDate.StrDate:MobDate.MFD for MobDate in self.ListDailyNetwork} for Aggregation in self.AggregationLevel}
        self.Aggregation2MFDNew = {Aggregation:{MobDate.StrDate:MobDate.MFDNew for MobDate in self.ListDailyNetwork} for Aggregation in self.AggregationLevel}
        self.Aggregation2Class2MFD = {Aggregation:{StrClass:{MobDate.StrDate:MobDate.MFD for MobDate in self.ListDailyNetwork} for StrClass in self.ListStrClassReference} for Aggregation in self.AggregationLevel}
        self.Aggregation2Class2MFDNew = {Aggregation:{StrClass:{MobDate.StrDate:MobDate.MFDNew for MobDate in self.ListDailyNetwork} for StrClass in self.ListStrClassReference} for Aggregation in self.AggregationLevel}
        self.Aggregation2MFD2Plot = {Aggregation:{"binned_av_speed":[],"binned_sqrt_err_speed":[],"bins_population":[]} for Aggregation in self.AggregationLevel}
        for Aggregation in self.AggregationLevel:
            self.Aggregation2MFD[Aggregation] = ComputeAggregatedMFDVariables(self.ListDailyNetwork,self.Aggregation2MFD[Aggregation])
        for Aggregation in self.AggregationLevel:
            for StrClass in self.ListStrClassReference:
                self.Aggregation2Class2MFD[Aggregation][StrClass] = ComputeAggregatedMFDVariables(self.ListDailyNetwork,self.Aggregation2Class2MFD[Aggregation][StrClass])
        self.CountFunctions += 1
        Message = "ComputeMFDAllDays: True\n"
        Message += "\tself.Aggregation2MFD\n"
        for Aggregation in self.Aggregation2MFD.keys():
            Message += "\tAggregation: {}".format(Aggregation)
            Message += "-> Number of Trajectories: {}".format(len(self.Aggregation2MFD[Aggregation]))
        AddMessageToLog(Message,self.LogFile)
        Message = "ComputeMFDAllDays: self.Aggregation2Class2MFD\n"
        for Aggregation in self.Aggregation2Class2MFD.keys():
            for StrClass in self.Aggregation2Class2MFD[Aggregation].keys():
                Message += "\tAggregation: {}".format(Aggregation)
                Message += "\tStrClass: {}".format(StrClass)
                Message += "-> Number of Trajectories: {}".format(len(self.Aggregation2Class2MFD[Aggregation][StrClass]))

    # Plot of Distribution of Time, Lenght, Av_Speed, Speed_kmh, Lenght_km, Time_hours for all days
    def PlotDistributionAggregatedAllDaysPerClass(self,bins = 100):
        """
            Input:
                label: str -> time, lenght, av_speed, p, a_max, class
            Returns:
                n, bins of velocity distribution
        """
        if self.ConcatenatedFcm is None:
            print("No plot Average, lack of concatenated fcm")
        else:
            pass
        if self.ConcatenatePerClassBool:
            legend_ = []
            for StrClass in self.ListStrClassReference:
                fig, ax = plt.subplots(1,1,figsize = (10,8))
                for MobDate in self.ListDailyNetwork:
                    for Feature in self.Column2Label.keys():
                        fig, ax = plt.subplots(1,1,figsize = (10,8))
                        count,bins_ = np.histogram(self.ConcatenatedFcm[Feature],bins = bins)
                        ax.plot(bins_[1:],count, bins = bins)  
                        ax.set_xlim(right = self.MinMaxPlot[Feature]["bins"])
                        ax.set_ylim(left = self.MinMaxPlot[Feature]["count"])
                        ax.set_xscale(self.Feature2ScaleBins[Feature])
                        ax.set_yscale(self.Feature2ScaleCount[Feature])
                        ax.set_xlabel(self.Column2Label[Feature])
                        ax.set_ylabel("Count")
                        ax.set_title(StrClass)               
                        ax.set_xlabel(self.Column2Label[Feature])
                plt.savefig(os.path.join(self.PlotDir,"Average_{0}_Class_{1}.png".format(Feature,StrClass)),dpi = 200)
                plt.close()

    def PlotDistributionComparisonAllDaysPerClass(self,bins = 100):
        """
            Each plot contains the curves of the distrubution of time and lenght and av_speed for each class.
        """
        if self.ConcatenatePerClassBool:
            legend_ = []
            for StrClass in self.ListStrClassReference:
                fig, ax = plt.subplots(1,1,figsize = (10,8))
                for MobDate in self.ListDailyNetwork:
                    for Feature in self.Column2Label.keys():
                        ax.hist(MobDate.Fcm.filter(pl.col("str_class") == StrClass)[Feature], bins = 100)   
                        ax.set_xlim(right = self.MinMaxPlot[Feature]["bins"])
                        ax.set_ylim(left = self.MinMaxPlot[Feature]["count"])
                        ax.set_xscale(self.Feature2ScaleBins[Feature])
                        ax.set_yscale(self.Feature2ScaleCount[Feature])
                        ax.set_xlabel(self.Column2Label[Feature])
                        ax.set_ylabel("Count")
                        ax.set_title(StrClass)
                        legend_.append(MobDate.StrDate)
                ax.set_legend(legend_)
                plt.savefig(os.path.join(self.PlotDir,"ComparisonDaysDistributionsPerClass_{0}_{1}.png".format(StrClass,Feature)))
                plt.close()
 
    def PlotDistributionAggregatedAllDays(self,bins = 100):
        """
            Input:
                label: str -> time, lenght, av_speed, p, a_max, class
            Returns:
                n, bins of velocity distribution
        """
        if self.ConcatenatedFcm is None:
            print("No plot Average, lack of concatenated fcm")
        else:
            pass
        if self.ConcatenatePerClassBool:
            for Feature in self.Column2Label.keys():
                fig, ax = plt.subplots(1,1,figsize = (10,8))
                count,bins_ = np.histogram(self.ConcatenatedFcm[Feature],bins = bins)
                ax.plot(bins_[1:],count, bins = bins)  
                ax.set_xlim(right = self.MinMaxPlot[Feature]["bins"])
                ax.set_ylim(left = self.MinMaxPlot[Feature]["count"])
                ax.set_xscale(self.Feature2ScaleBins[Feature])
                ax.set_yscale(self.Feature2ScaleCount[Feature])
                ax.set_xlabel(self.Column2Label[Feature])
                ax.set_ylabel("Count")
                ax.set_xlabel(self.Column2Label[Feature])
                plt.savefig(os.path.join(self.PlotDir,"Average_{}.png".format(Feature)),dpi = 200)
                plt.close()

    def PlotDistributionComparisonAllDays(self,bins = 100):
        """
            Input:
                label: str -> time, lenght, av_speed, p, a_max, class
            Returns:
                n, bins of velocity distribution
        """
        if self.ConcatenatedFcm is None:
            print("No plot Average, lack of concatenated fcm")
        else:
            if self.ConcatenatePerClassBool:
                legend_ = []
                for NetDay in self.ListDailyNetwork:
                    for Feature in self.Column2Label.keys():
                        fig, ax = plt.subplots(1,1,figsize = (10,8))
                        ax.hist(NetDay.Fcm[Feature], bins = bins)   
                        ax.set_xlim(right = self.MinMaxPlot[Feature]["bins"])
                        ax.set_ylim(left = self.MinMaxPlot[Feature]["count"])
                        ax.set_xscale(self.Feature2ScaleBins[Feature])
                        ax.set_yscale(self.Feature2ScaleCount[Feature])
                        ax.set_xlabel(self.Column2Label[Feature])
                        ax.set_ylabel("Count")
                        legend_.append(NetDay.StrDate)
                        ax.set_xlabel(self.Column2Label[Feature])
                ax.set_legend(legend_)
                plt.savefig(os.path.join(self.PlotDir,"ComparisonAverage_{}.png".format(Feature)),dpi = 200)
                plt.close()
            
    # MFD Aggregated All Days

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
        self.MFDAggregated = ComputeAggregatedMFDVariables(self.ListDailyNetwork,self.MFDAggregated)
        # AGGREGATE MFD FOR ALL DAYS PER CLASS
        LocalDayCount = 0
        for MobDate in self.ListDailyNetwork:
            for StrClass in self.ListStrClassReference:
                if LocalDayCount == 0:
                    # NOTE: This Line Is Essential Not To Confuse The Reference Day Index Map With The Single Day Analysis
                    LocalIntClass = self.Day2StrClass2IntClass[MobDate.StrDate][StrClass]
                    self.StrClass2MFDAggregated[StrClass] = MobDate.Class2MFD[LocalIntClass] # NOTE: The Way I fill this dictionary is not intuitive since I use a map StrClass (For All days) -> IntClass (For the Reference Day)
                    self.StrClass2MFDNewAggregated[StrClass] = MobDate.Class2MFD[LocalIntClass]
                    self.StrClass2MFDAggregated[StrClass]["count_days"] = list(np.zeros(len(self.StrClass2MFDAggregated[StrClass]["time"])))
                    self.StrClass2MFDNewAggregated[StrClass]["count_days"] = list(np.zeros(len(self.StrClass2MFDNewAggregated[StrClass]["time"])))
                    LocalDayCount += 1
                else:
                    # NOTE: This Line Is Essential Not To Confuse The Reference Day Index Map With The Single Day Analysis
                    LocalIntClass = self.Day2StrClass2IntClass[MobDate.StrDate][StrClass]
                    for t in range(len(self.StrClass2MFDAggregated[StrClass]["time"])):
                        SpeedAtTime = MobDate.Class2MFD[LocalIntClass]["speed_kmh"][t]
                        PopulationAtTime = MobDate.Class2MFD[LocalIntClass]["population"][t]
                        if PopulationAtTime != 0 and SpeedAtTime !=0:
                            self.StrClass2MFDAggregated[StrClass]["speed_kmh"][t] += MobDate.Class2MFD[StrClass]["speed_kmh"][t]
                            self.StrClass2MFDAggregated[StrClass]["population"][t] += MobDate.Class2MFD[StrClass]["population"][t]
                            self.StrClass2MFDAggregated[StrClass]["count_days"][t] += 1
                        else:
                            pass
                        SpeedAtTimeNew = MobDate.Class2MFDNew[LocalIntClass]["speed_kmh"][t]
                        PopulationAtTimeNew = MobDate.Class2MFDNew[LocalIntClass]["population"][t]
                        if PopulationAtTimeNew != 0 and SpeedAtTimeNew !=0:
                            self.StrClass2MFDNewAggregated[StrClass]["speed_kmh"][t] += MobDate.Class2MFDNew[StrClass]["speed_kmh"][t]
                            self.StrClass2MFDNewAggregated[StrClass]["population"][t] += MobDate.Class2MFDNew[StrClass]["population"][t]
                            self.StrClass2MFDNewAggregated[StrClass]["count_days"][t] += 1
                        else:
                            pass
        self.ComputedMFDAggregatedVariablesBool = True

    def PlotMFDAggreagated(self):
        if self.ComputedMFDAggregatedVariablesBool: 
            # AGGREGATED 
            self.MFDAggregated2Plot, self.MinMaxPlot,RelativeChange = GetMFDForPlot(MFD = self.MFDAggregated,
                                                                        MFD2Plot = self.MFDAggregated2Plot,
                                                                        MinMaxPlot = self.MinMaxPlot,
                                                                        Class = None,
                                                                        case = "no-classes",
                                                                        verbose = self.verbose,
                                                                        bins_ = 20)
            
            PlotHysteresis(MFD = self.MFDAggregated,
                           Title = "Hysteresis Cycle Aggregated All Days",
                           SaveDir = self.PlotDir,
                           NameFile = "HysteresysAllDaysAllClasses.png")
            if self.verbose:
                print("After GetMFDForPlot:\n")
#                print("\nMFD2Plot:\n",self.MFD2Plot)
#                print("\nMinMaxPlot:\n",self.MinMaxPlot)
            # Plotting and Save Aggregated
            SaveMFDPlot(self.MFDAggregated2Plot["bins_population"],
                        self.MFDAggregated2Plot["binned_av_speed"],
                        self.MFDAggregated2Plot["binned_sqrt_err_speed"],
                        RelativeChange,
                        self.PlotDir,
                        NameFile = "MFD.png")            
            # PER CLASS 
            self.MinMaxPlotPerClass = {StrClass: defaultdict() for Class in self.Class2MFDAggregated.keys()}
            self.MinMaxPlotPerClassNew = {StrClass: defaultdict() for Class in self.Class2MFDNewAggregated.keys()}
            for StrClass in self.Class2MFDAggregated.keys():
                self.Class2MFDAggregated2Plot[StrClass] = {"binned_av_speed": [], "binned_sqrt_err_speed": [], "bins_population": []}
                self.Class2MFDNewAggregated2Plot[StrClass] = {"binned_av_speed": [], "binned_sqrt_err_speed": [], "bins_population": []}
            for StrClass in self.Class2MFDAggregated.keys():
                # Fill Average/Std Speed (to plot)
                # OLD CLASSIFICATION
                self.Class2MFDAggregated2Plot[StrClass], self.MinMaxPlotPerClass,RelativeChange = GetMFDForPlot(MFD = self.Class2MFDAggregated[StrClass],
                                                                                                     MFD2Plot = self.Class2MFDAggregated2Plot[StrClass],
                                                                                                    MinMaxPlot = self.MinMaxPlotPerClass,
                                                                                                    Class = StrClass,
                                                                                                    case = None,
                                                                                                    verbose = self.verbose,
                                                                                                    bins_ = 20)
                # NEW CLASSIFICATION
                self.Class2MFDNewAggregated2Plot[StrClass], self.MinMaxPlotPerClassNew,RelativeChangeNew = GetMFDForPlot(MFD = self.Class2MFDNewAggregated[StrClass],
                                                                                                     MFD2Plot = self.Class2MFDNewAggregated2Plot[StrClass],
                                                                                                    MinMaxPlot = self.MinMaxPlotPerClassNew,
                                                                                                    Class = StrClass,
                                                                                                    case = None,
                                                                                                    verbose = self.verbose,
                                                                                                    bins_ = 20)
                
                PlotHysteresis(MFD = self.Class2MFDAggregated[StrClass],
                            Title = "Hysteresis Cycle Class {}".format(StrClass),
                            SaveDir = self.PlotDir,
                            NameFile = "HysteresysClass_{}.png".format(StrClass))
                
                PlotHysteresis(MFD = self.Class2MFDNewAggregated[StrClass],
                            Title = "Hysteresis Cycle Class New {}".format(StrClass),
                            SaveDir = self.PlotDir,
                            NameFile = "HysteresysClassNew_{}.png".format(StrClass))
                
                if self.verbose:
                    print("After GetMFDForPlot Class {}:\n".format(Class))
#                    print("\nClass2MFD2Plot:\n",self.Class2MFD2Plot)
#                    print("\nMinMaxPlotPerClass:\n",self.MinMaxPlotPerClass)

                # Plotting and Save Per Class
                # OLD CLASSIFICATION
                SaveMFDPlot(self.Class2MFDAggregated2Plot[StrClass]["bins_population"],
                            self.Class2MFDAggregated2Plot[StrClass]["binned_av_speed"],
                            self.Class2MFDAggregated2Plot[StrClass]["binned_sqrt_err_speed"],
                            RelativeChange = RelativeChange,
                            SaveDir = self.PlotDir,
                            Title = "Fondamental Diagram {}".format(StrClass),
                            NameFile = "MFD_{}_AllDays.png".format(StrClass))
                # NEW CLASSIFICATION
                SaveMFDPlot(self.Class2MFDNewAggregated2Plot[StrClass]["bins_population"],
                            self.Class2MFDNewAggregated2Plot[StrClass]["binned_av_speed"],
                            self.Class2MFDNewAggregated2Plot[StrClass]["binned_sqrt_err_speed"],
                            RelativeChange = RelativeChangeNew,
                            SaveDir = self.PlotDir,
                            Title = "Fondamental Diagram New {}".format(StrClass),
                            NameFile = "MFDNew_{}_AllDays.png".format(StrClass))

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
        self.Road2StrClass2CountDays = {Road: {StrClass:0 for StrClass in self.ListStrClassReference} for Road in self.ListDailyNetwork[0].GeoJson["poly_lid"]} 
        self.Road2StrClass2Days = {Road:{StrClass: [] for StrClass in self.ListStrClassReference} for Road in self.ListDailyNetwork[0].GeoJson["poly_lid"]} 

        for MobDate in self.ListDailyNetwork:
            for StrClass in self.ListStrClassReference:
                # Consider the class of the day associated to the Reference Class (quickest, slowest, etc.)
                IntClassOfDay = MobDate.StrClass2IntClass[StrClass]
                # List of Roads that are included in the class
                for Road in MobDate.IntClass2RoadsIncreasinglyIncludedIntersection[IntClassOfDay]:
                    self.Road2StrClass2CountDays[Road][StrClass] += 1
                    self.Road2StrClass2Days[Road][StrClass].append(MobDate.StrDate)
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
            print("Save in: ",os.path.join(self.PlotDir,"SubnetsIncrementalInclusion_{}.html".format(self.StrDate)))
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
