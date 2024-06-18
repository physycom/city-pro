from AnalysisNetwork1Day import *
from collections import defaultdict
import numpy as np


def ComputeAggregatedMFDVariables(ListDailyNetwork,MFDAggregated):
    """
        Description:
            Every Day I count for each hour, how many people and the speed of the 
            1. Network -> MFDAggregated = {"population":[],"time":[],"speed":[]}
            2. SubNetwork -> Class2MFDAggregated = {StrClass: {"population":[sum_i pop_{t0,dayi},...,sum_i pop_{iteration,dayi}],"time":[t0,...,iteration],"speed":[sum_i speed_{t0,dayi},...,sum_i speed_{iteration,dayi}]}}
            NOTE: time is pl.DateTime
        NOTE: Each Time interval has its own average speed and population. For 15 minutes,
            since iteration in 1 Day Analysis is set in that way. 
        NOTE: If at time t there is no population, the speed is set to 0.
    """
    LocalDayCount = 0
    # AGGREGATE MFD FOR ALL DAYS
    for MobDate in self.ListDailyNetwork:
        if LocalDayCount == 0:
            MFDAggregated = MobDate.MFD
            MFDAggregated["count_days"] = list(np.zeros(len(MFDAggregated["time"])))
            LocalDayCount += 1
        else:
            for t in range(len(MobDate.MFD["time"])):
                SpeedAtTime = MobDate.MFD["speed"][t]
                PopulationAtTime = MobDate.MFD["population"][t]
                if PopulationAtTime != 0 and SpeedAtTime !=0:
                    MFDAggregated["speed"][t] += MobDate.MFD["speed"][t]
                    MFDAggregated["population"][t] += MobDate.MFD["population"][t]
                    MFDAggregated["count_days"][t] += 1
                else:
                    pass
    for t in range(len(MFDAggregated["time"])):
        if MFDAggregated["count_days"][t] != 0:
            MFDAggregated["speed"][t] = MFDAggregated["speed"][t]/MFDAggregated["count_days"][t]
            MFDAggregated["population"][t] = MFDAggregated["population"][t]/MFDAggregated["count_days"][t]
        else:
            pass
    MFDAggregated = Dict2PolarsDF(MFDAggregated,schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed":pl.Float64})
    return MFDAggregated


class NetworkAllDays:
    
    def __init__(self,ListDailyNetwork,PlotDir,verbose = False): 
        # Analysis flags
        self.AssociateAvSpeed2StrClassBool = False
        self.ConcatenatePerClassBool = False
        self.CreateClass2SubNetAllDaysBool = False
        self.ComputedMFDAggregatedVariablesBool = False
        # Settings
        self.verbose = verbose
        self.PlotDir = PlotDir
        # Initialization all days
        self.ListDailyNetwork = ListDailyNetwork       
        self.Day2Feature2MaxBins = {MobDate.StrDate:defaultdict() for MobDate in self.ListDailyNetwork}
        LocalCount = 0
        self.StrDates = [] 
        for MobDate in self.ListDailyNetwork:
            self.StrDates.append(MobDate.StrDate)         
            self.Day2Feature2MaxBins[MobDate.StrDate] = MobDate.Feature2MaxBins
            if LocalCount == 0:
                self.Feature2ScaleCount = MobDate.Feature2ScaleCount
                self.Feature2ScaleBins = MobDate.Feature2ScaleBins
                self.centroid = MobDate.centroid
                self.Class2Color = MobDate.Class2Color
                self.ListColors = MobDate.ListColors
        self.StrDay2Color = {StrDay: self.ListColors[i] for i,StrDay in enumerate(self.StrDates)}
        self.Column2Label = {"av_speed":'average speed (km/h)',"av_accel":"average acceleration (m/s^2)","lenght":'lenght (km)',"time_hours":'time (h)',"time":'time (s)'}
        self.Column2SaveName = {"av_speed":"average_speed","av_accel":"average_acceleration","lenght":"lenght","time_hours":"time_hours","time":"time"}
        self.Column2Legend = {"av_speed":"speed (km/h)","av_accel":"acceleration (m/s^2)","lenght":"lenght (km)","time_hours":"time (h)","time":"time (s)"} 
        self.Feature2MaxBins = {"av_speed":{"bins":0,"count":0},"av_accel":{"bins":0,"count":0},"lenght":{"bins":0,"count":0},"time_hours":{"bins":0,"count":0},"time":{"bins":0,"count":0}}
        # MFD
        self.MFDAggregated = {"population":[],"time":[],"speed":[]}
        self.StrClass2MFDAggregated = defaultdict() 
        self.StrClass2MFDNewAggregated = defaultdict() 
        self.MFDAggregated2Plot = {"bins_population":[],"binned_av_speed":[],"binned_sqrt_err_speed":[]}
        self.StrClass2MFDAggregated2Plot = {StrClass: {"bins_population":[],"binned_av_speed":[],"binned_sqrt_err_speed":[]} for StrClass in self.ListStrClassReference}
        self.StrClass2MFDNewAggregated2Plot = {StrClass: {"bins_population":[],"binned_av_speed":[],"binned_sqrt_err_speed":[]} for StrClass in self.ListStrClassReference}
        # Set The Plot Lim as The Maximum Among all Days
        for Day in self.Day2Feature2MaxBins.keys():
            for Feature in self.Day2Feature2MaxBins[Day].keys():
                for Bins in self.Day2Feature2MaxBins[Day][Feature].keys():
                    self.Feature2MaxBins[Feature][Bins] = max(self.Feature2MaxBins[Feature][Bins],self.Day2Feature2MaxBins[Day][Feature][Bins])
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
        self.Day2IntClass2StrClass = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}
        self.Day2StrClass2IntClass = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}
        DayIntClass2IntRef = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}
        self.DictClass2AvSpeed = {MobDate.StrDate:defaultdict(dict) for MobDate in self.ListDailyNetwork}
        # Take The Day with More Classes.
        NumberClasses = 0
        self.DayReferenceClasses = ""
        RefIntClass2StrClass = {}

        for MobDate in self.ListDailyNetwork:
            if len(MobDate.StrClass2IntClass) > NumberClasses:
                NumberClasses = len(MobDate.StrClass2IntClass)
                self.DayReferenceClasses = MobDate.StrDate
                ReferenceFcmCenters = MobDate.FcmCenters
                RefIntClass2StrClass = MobDate.IntClass2StrClass
            else:
                pass
        self.ListStrClassReference = list(RefIntClass2StrClass.keys())
        # Each day.
        for MobDate in self.ListDailyNetwork:
            # Compare the class speed 
            for class_,Df in MobDate.FcmCenters.group_by("class"):
                # to all the reference class speeds.
                IntClassBestMatch = 0
                MinDifferenceInSpeed = 100000

                # Choose the speed that is closest to the reference
                for class_ref,Df_ref in ReferenceFcmCenters.group_by("class"):
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
                self.Day2IntClass2StrClass[MobDate.StrDate][Class] = RefIntClass2StrClass[self.DictClass2AvSpeed[MobDate.StrDate][Class]]
                self.Day2StrClass2IntClass[MobDate.StrDate][RefIntClass2StrClass[self.DictClass2AvSpeed[MobDate.StrDate][Class]]] = Class
        if self.verbose:
            print("Day to StrClass to intClass:\n",self.Day2StrClass2IntClass)
            print("Day to IntClass to StrClass:\n",self.Day2IntClass2StrClass)
        # For Each Day Associate a column to FcmCenters and Fcm to the str_class computed above
        for MobDate in self.ListDailyNetwork:
            for Class in MobDate.FcmCenters["class"].unique():
                MobDate.FcmCenters = MobDate.FcmCenters.with_columns(pl.when(pl.col("class") == Class).then(self.Day2IntClass2StrClass[MobDate.StrDate][Class]).otherwise("11").alias("str_class"))
                MobDate.Fcm = MobDate.Fcm.with_columns(pl.when(pl.col("class") == Class).then(self.Day2StrClass2IntClass[MobDate.StrDate][self.Day2IntClass2StrClass[MobDate.StrDate][Class]]).otherwise("11").alias("str_class"))
        # MFD
        self.StrClass2MFDAggregated = {StrClass: {"population":[],"time":[],"speed":[]} for StrClass in self.ListStrClassReference}
        self.StrClass2MFDNewAggregated = {StrClass: {"population":[],"time":[],"speed":[]} for StrClass in self.ListStrClassReference}
        self.MinMaxPlotPerClass = {StrClass: defaultdict() for StrClass in self.ListStrClassReference}
        self.MinMaxPlotPerClassNew = {StrClass: defaultdict() for StrClass in self.ListStrClassReference}
        
        self.AssociateAvSpeed2StrClassBool = True

    def ConcatenateAllFcms(self):
        """
            Description:
                Concatenate all fcms in self.ListDailyNetwork by label: [lenght, time]
        """
        self.ConcatenatedFcm = None
        for DailyMob in self.ListDailyNetwork:
            if self.ConcatenatedFcm is None:
                if DailyMob.ReadFcmNewBool and DailyMob.ReadFcmBool:
                    self.ConcatenatedFcm = DailyMob.Fcm
                if self.verbose:
                    print("Concatenated Fcm:\n{}".format(self.ConcatenatedFcm.columns))
                    print("Day:\n",DailyMob.StrDate)
            else:
                if self.verbose:
                    print("Concatenated Fcm:\n{}".format(self.ConcatenatedFcm.columns))
                    print("Fcm:\n",DailyMob.Fcm.columns)
                    print("Day:\n",DailyMob.StrDate)
                if DailyMob.ReadFcmNewBool and DailyMob.ReadFcmBool:
                    self.ConcatenatedFcm = pl.concat([self.ConcatenatedFcm,DailyMob.Fcm])



    def ConcatenatePerClass(self):
        """
            Output:
                Class2Fcm =
                {Strclass: {time: [timetraj0,...,timetrajN]
                         lenght: [lenghttraj0,...,lenghttrajN]
                         }
                }
                Day2Class2Fcm = {day: {Strclass: {time: [timetraj0,...,timetrajN], lenght: [lenghttraj0,...,lenghttrajN]}}}
            NOTE: The class are grouped by the speed. From AssociateAvSpeed2StrClass()
        """
        # If Classes are Categorized
        if self.AssociateAvSpeed2StrClassBool:
            self.Class2Fcm = {StrClass: {"time":[],"lenght":[],"av_speed":[]} for StrClass in self.Day2StrClass2IntClass[self.DayReferenceClasses].keys()}
            self.Day2Class2Fcm = {MobDate.StrDate: {StrClass: {"time":[],"lenght":[],"av_speed":[]} for StrClass in self.Day2StrClass2IntClass[MobDate.StrDate].keys()} for MobDate in self.ListDailyNetwork}
            for MobDate in self.ListDailyNetwork:
                for StrClass in self.ListStrClassReference:
                    # Append time and lenght of the Iterated Day
                    self.Class2Fcm[StrClass]["time"].extend(MobDate.Fcm.filter(pl.col("str_class") == StrClass)["time"].to_list())
                    self.Class2Fcm[StrClass]["lenght"].extend(MobDate.Fcm.filter(pl.col("str_class") == StrClass)["lenght"].to_list())
                    self.Class2Fcm[StrClass]["av_speed"].extend(MobDate.Fcm.filter(pl.col("str_class") == StrClass)["av_speed"].to_list())
                    self.Day2Class2Fcm[MobDate.StrDate][StrClass]["time"].extend(MobDate.Fcm.filter(pl.col("str_class") == StrClass)["time"].to_list())
                    self.Day2Class2Fcm[MobDate.StrDate][StrClass]["lenght"].extend(MobDate.Fcm.filter(pl.col("str_class") == StrClass)["lenght"].to_list())
                    self.Day2Class2Fcm[MobDate.StrDate][StrClass]["av_speed"].extend(MobDate.Fcm.filter(pl.col("str_class") == StrClass)["av_speed"].to_list())
#        if self.verbose:
#            print("Class2Fcm:\n",self.Class2Fcm)
        self.ConcatenatePerClassBool = True
## Put together space and time for all days for each class.
## Compare them by the average velocity in self.Day2StrClass2IntClass[MobDate.StrDate]
##
    def SetlenghtTimeCommonBins(self,bins = 100):
        """
            Set the lenght and time common bins:
                1. lenghtTimeBins -> {Class: {"time": {"xmax": 0, "xmin": 0, "ymax": 0, "ymin": 0}, "lenght": {"xmax": 0, "xmin": 0, "ymax": 0, "ymin": 0}}
                2. MaxlenghtTimeBins -> {"time": {"xmax": 0, "xmin": 0, "ymax": 0, "ymin": 0}, "lenght": {"xmax": 0, "xmin": 0, "ymax": 0, "ymin": 0}}
            NOTE:
                This Aligns the distribution of time and lenghts among all classes
                It is essential to have comparable plots among all days.
        
        """
        if self.ConcatenatePerClassBool:
            Labels = ["time","lenght","av_speed"]
            self.lenghtTimeBins = {Class: {"time": {"xmax": 0, "xmin": 0, "ymax": 0, "ymin": 0}, "lenght": {"xmax": 0, "xmin": 0, "ymax": 0, "ymin": 0},"av_speed": {"xmax": 0, "xmin": 0, "ymax": 0, "ymin": 0}} for Class in self.Class2Fcm.keys()}
            self.MaxlenghtTimeBins = {"time": {"xmax": 0, "xmin": 0, "ymax": 0, "ymin": 0}, "lenght": {"xmax": 0, "xmin": 0, "ymax": 0, "ymin": 0},"av_speed": {"xmax": 0, "xmin": 0, "ymax": 0, "ymin": 0}}
            for Label in Labels:
                self.MaxlenghtTimeBins[Label]["xmax"] = max(np.histogram(self.ConcatenatedFcm[Label].to_list(),bins = bins)[1])
                self.MaxlenghtTimeBins[Label]["xmin"] = min(np.histogram(self.ConcatenatedFcm[Label].to_list(),bins = bins)[1])
                self.MaxlenghtTimeBins[Label]["ymax"] = max(np.histogram(self.ConcatenatedFcm[Label].to_list(),bins = bins)[0])
                self.MaxlenghtTimeBins[Label]["ymin"] = min(np.histogram(self.ConcatenatedFcm[Label].to_list(),bins = bins)[0])
                for Class in self.Class2Fcm.keys():
                    print("Class:\n",Class)
                    print("Label:\n",Label)
                    print("len(self.Class2Fcm[Class][Label]):\n",len(self.Class2Fcm[Class][Label]))
                    print("len(self.Class2Fcm[Class]):\n",len(self.Class2Fcm[Class]))
                    print("type(self.Class2Fcm[Class][Label]):\n",type(self.Class2Fcm[Class][Label]))
                    print("self.Class2Fcm[Class][Label]:\n",np.shape(self.Class2Fcm[Class][Label]))
                    self.lenghtTimeBins[Class][Label]["xmax"] = max(np.histogram(self.Class2Fcm[Class][Label],bins = bins)[1])
                    self.lenghtTimeBins[Class][Label]["xmin"] = min(np.histogram(self.Class2Fcm[Class][Label],bins = bins)[1])
                    self.lenghtTimeBins[Class][Label]["ymax"] = max(np.histogram(self.Class2Fcm[Class][Label],bins = bins)[0])
                    self.lenghtTimeBins[Class][Label]["ymin"] = min(np.histogram(self.Class2Fcm[Class][Label],bins = bins)[0])
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
            self.DictLabels = {"time_hours":"time (h)","lenght":"lenght (km)"}
        if self.ConcatenatePerClassBool:
            self.SetlenghtTimeCommonBins()
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
            self.SetlenghtTimeCommonBins()
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
            self.DictLabels = {"time_hours":"time (h)","lenght":"lenght (km)"}
        if self.ConcatenatePerClassBool:
            self.SetlenghtTimeCommonBins()
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
                self.SetlenghtTimeCommonBins()
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
            
    


    def ComputeAggregatedMFDVariablesObj(self):
        """
            Description:
                Every Day I count for each hour, how many people and the speed of the 
                1. Network -> MFDAggregated = {"population":[],"time":[],"speed":[]}
                2. SubNetwork -> Class2MFDAggregated = {StrClass: {"population":[sum_i pop_{t0,dayi},...,sum_i pop_{iteration,dayi}],"time":[t0,...,iteration],"speed":[sum_i speed_{t0,dayi},...,sum_i speed_{iteration,dayi}]}}
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
                        SpeedAtTime = MobDate.Class2MFD[LocalIntClass]["speed"][t]
                        PopulationAtTime = MobDate.Class2MFD[LocalIntClass]["population"][t]
                        if PopulationAtTime != 0 and SpeedAtTime !=0:
                            self.StrClass2MFDAggregated[StrClass]["speed"][t] += MobDate.Class2MFD[StrClass]["speed"][t]
                            self.StrClass2MFDAggregated[StrClass]["population"][t] += MobDate.Class2MFD[StrClass]["population"][t]
                            self.StrClass2MFDAggregated[StrClass]["count_days"][t] += 1
                        else:
                            pass
                        SpeedAtTimeNew = MobDate.Class2MFDNew[LocalIntClass]["speed"][t]
                        PopulationAtTimeNew = MobDate.Class2MFDNew[LocalIntClass]["population"][t]
                        if PopulationAtTimeNew != 0 and SpeedAtTimeNew !=0:
                            self.StrClass2MFDNewAggregated[StrClass]["speed"][t] += MobDate.Class2MFDNew[StrClass]["speed"][t]
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
