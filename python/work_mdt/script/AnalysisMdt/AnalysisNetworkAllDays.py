from AnalysisNetwork1Day import *
from collections import defaultdict
import numpy as np
class NetworkAllDays:
    
    def __init__(self,ListDailyNetwork,PlotDir,verbose = False): 
        self.StrDates = [] 
        # Analysis flags
        self.AssociateAvSpeed2StrClassBool = False
        self.ConcatenatePerClassBool = False
        # Settings
        self.verbose = verbose
        self.PlotDir = PlotDir
        # Initialization all days
        self.ListDailyNetwork = ListDailyNetwork       
        self.Day2Feature2MaxBins = {MobDate.StrDate:defaultdict() for MobDate in self.ListDailyNetwork}
        LocalCount = 0
        for MobDate in self.ListDailyNetwork:
            self.StrDates.append(MobDate.StrDate)         
            self.Day2Feature2MaxBins[MobDate.StrDate] = MobDate.Feature2MaxBins
            if LocalCount == 0:
                self.Feature2ScaleCount = MobDate.Feature2ScaleCount
                self.Feature2ScaleBins = MobDate.Feature2ScaleBins
        self.Column2Label = {"av_speed":'average speed (km/h)',"av_accel":"average acceleration (m/s^2)","lenght":'lenght (km)',"time_hours":'time (h)',"time":'time (s)'}
        self.Column2SaveName = {"av_speed":"average_speed","av_accel":"average_acceleration","lenght":"lenght","time_hours":"time_hours","time":"time"}
        self.Column2Legend = {"av_speed":"speed (km/h)","av_accel":"acceleration (m/s^2)","lenght":"lenght (km)","time_hours":"time (h)","time":"time (s)"} 
        self.Feature2MaxBins = {"av_speed":{"bins":0,"count":0},"av_accel":{"bins":0,"count":0},"lenght":{"bins":0,"count":0},"time_hours":{"bins":0,"count":0},"time":{"bins":0,"count":0}}
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
            
    


    def plot_aggregated_velocity(fcm_data,list_dict_name,i):
        print('all different groups same color')
        fig,ax = plt.subplots(1,1,figsize= (15,12))
        legend = []
        aggregated = []
        aggregation = True
        for cl,df in fcm_data.groupby('class'):
        #    fig,ax = plt.subplots(1,1,figsize= (15,12))
            if cl!=10 and len(list_dict_name[i][cl])!=0:
                aggregated.extend(df['av_speed'].to_numpy())
        plt.hist(aggregated,bins = 50,range = [0,50])
        x,y = get_best_binning_distribution_velocity_plot(fcm_data,list_dict_name,i,aggregation)
        ax.set_xticks(x)
        ax.set_yticks(y)
        ax.set_xlabel('average speed (km/h)')
        ax.set_ylabel('Count')
        plt.savefig(os.path.join(PlotDir,'av_speed_aggregated.png'),dpi = 200)
        plt.show()

