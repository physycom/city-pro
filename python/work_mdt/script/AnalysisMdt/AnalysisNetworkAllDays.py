from AnalysisNetwork1Day import *
from analysisPlot import *
from collections import defaultdict
import numpy as np
from LatexFunctions import *
from UsefulStructures import *



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
        self.AggregationLevel = ["holidays","not_holidays","aggregated"]
        self.AggregationLevel2ListDays = {"holidays": config["holidays"],
                                          "not_holidays": config["not_holidays"],
                                          "aggregated": config["StrDates"]}
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
                    self.StrClass2MFDAggregated2Plot[StrClass] = {Key: [] for Key in MobDate.MFD2Plot.keys()}
                    self.StrClass2MFDNewAggregated2Plot[StrClass] = {Key: [] for Key in MobDate.MFD2Plot.keys()}
                self.config = MobDate.config
        # MFD
        self.Aggregation2MFD = {Aggregation:{MobDate.StrDate:MobDate.MFD for MobDate in self.ListDailyNetwork if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]} for Aggregation in self.AggregationLevel}
        self.Aggregation2MFDNew = {Aggregation:{MobDate.StrDate:MobDate.MFDNew for MobDate in self.ListDailyNetwork if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]} for Aggregation in self.AggregationLevel}
        self.Aggregation2Class2MFD = {Aggregation:{StrClass:{MobDate.StrDate:MobDate.MFD for MobDate in self.ListDailyNetwork if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]} for StrClass in self.ListStrClassReference} for Aggregation in self.AggregationLevel}
        self.Aggregation2Class2MFDNew = {Aggregation:{StrClass:{MobDate.StrDate:MobDate.MFDNew for MobDate in self.ListDailyNetwork if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]} for StrClass in self.ListStrClassReference} for Aggregation in self.AggregationLevel}
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
        if self.verbose:
            print("Associate AvSpeed to StrClass")
            print("Reference Int -> Str Class:\n",self.RefIntClass2StrClass)
            print("Day to StrClass to intClass:\n",self.Day2StrClass2IntClass)
            print("Day to IntClass to StrClass:\n",self.Day2IntClass2StrClass)
        # Each day.GenerateDay2DictClass2AvSpeed
        self.AssociateAvSpeed2StrClassBool = True
        self.CountFunctions += 1
        MessageAveSpeedStrClass(self.Day2IntClass2StrClass,self.Day2StrClass2IntClass,self.DictClass2AvSpeed,self.LogFile)

    
    def ComparedDaysFit(self):
        """
            Create the Dataframe that contains information about:
                1. Parameters
                2. Best Fit Function
        """
        if self.AssociateAvSpeed2StrClassBool:
            DictFitKeys = ["parameters","best_fit"]
            Columns = ["time","lenght","av_speed","speed_kmh","lenght_km","time_hours"]
            ColumnsDictFittedData = [Feature + "_{}".format(FI) for FI in DictFitKeys for Feature in Columns]
            Index = self.StrDates
            self.DF_FitAggregated = pl.DataFrame(ColumnsDictFittedData,Index)
            self.StrClass2DF_Fit = {StrClass:self.DF_FitAggregated for StrClass in self.ListStrClassReference}
            for MobDate in self.ListDailyNetwork:
                for Feature in Columns:
                    for FitKey in DictFitKeys:
                        self.DF_FitAggregated[Feature+"_{}".format(FitKey)].loc[MobDate.StrDate] = MobDate.Aggregation2DictFittedData[MobDate.StrDate][Feature][FitKey]
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
                            self.StrClass2DF_Fit[StrClass][Feature+"_{}".format(FitKey)].loc[MobDate.StrDate] = MobDate.Aggregation2Class2DictFittedData[MobDate.StrDate][DayIntClass][Feature][FitKey]
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
            if self.AssociateAvSpeed2StrClassBool:
                for Aggregation in self.AggregationLevel:
                    for MobDate in self.ListDailyNetwork:
                        if MobDate.StrDate in self.AggregationLevel2ListDays[Aggregation]:
                            self.Aggregation2Fcm[Aggregation] = pl.concat([self.Aggregation2Fcm[Aggregation],MobDate.Fcm])
                            for StrClass in self.ListStrClassReference:
                            # Append time and lenght of the Iterated Day
                                self.Aggregation2Class2Fcm[Aggregation][StrClass] = pl.concat([self.Aggregation2Class2Fcm[Aggregation][StrClass],MobDate.Fcm.filter(pl.col("str_class") == StrClass)])
                self.ConcatenatePerClassBool = True
                MessageConcatenateFcm(self.Aggregation2Fcm,self.Aggregation2Class2Fcm,self.LogFile)
            else:
                Message = "ConcatenatePerClass: False"
## Put together space and time for all days for each class.

    def ComputeMFDAllDays(self):
        """
            Description:
                Compute the MFD averaged with all days data.
        """
        self.Aggregation2MFD2Plot = {Aggregation:{"binned_av_speed":[],"binned_sqrt_err_speed":[],"bins_population":[]} for Aggregation in self.AggregationLevel}
        for Aggregation in self.AggregationLevel:
            self.Aggregation2MFD[Aggregation] = ComputeAggregatedMFDVariables(self.ListDailyNetwork,self.Aggregation2MFD[Aggregation])
        for Aggregation in self.AggregationLevel:
            for StrClass in self.ListStrClassReference:
                self.Aggregation2Class2MFD[Aggregation][StrClass] = ComputeAggregatedMFDVariables(self.ListDailyNetwork,self.Aggregation2Class2MFD[Aggregation][StrClass])
        self.CountFunctions += 1
        MessageComputeMFDAllDays(self.Aggregation2MFD,self.Aggregation2Class2MFD,self.LogFile)

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
                        for Function2Fit in MobDate.Feature2Function2Fit2InitialGuess[Feature][IntClass].keys():
                            self.Aggregation2Feature2Function2Fit2InitialGuess[Aggregation][Feature]["initial_guess"] = tuple(MobDate.Feature2Function2Fit2InitialGuess[Feature][Function2Fit]["parameters"])
                            self.Aggregation2Feature2Function2Fit2InitialGuess[Aggregation][Feature]["interval"] = [MobDate.Feature2Function2Fit2InitialGuess[Feature][Function2Fit]["start_window"],MobDate.Feature2Function2Fit2InitialGuess[Feature][Function2Fit]["end_window"]]
                            for StrClass in self.Day2StrClass2IntClass[MobDate.StrDate].keys():
                                IntClass = self.Day2StrClass2IntClass[MobDate.StrDate][StrClass]
                                self.Aggregation2Feature2Class2Function2Fit2InitialGuess[Aggregation][Feature][StrClass][Function2Fit]["initital_guess"] = tuple(MobDate.Feature2Class2Function2Fit2InitialGuess[Feature][IntClass][Function2Fit]["parameters"])
                                self.Aggregation2Feature2Class2Function2Fit2InitialGuess[Aggregation][Feature][StrClass][Function2Fit]["interval"] = [MobDate.Feature2Class2Function2Fit2InitialGuess[Feature][IntClass][Function2Fit]["start_window"],MobDate.Feature2Class2Function2Fit2InitialGuess[Feature][IntClass][Function2Fit]["end_window"]]
    def ComputeAggregatedFit(self):
        """
            Create the dictionary for the Fit (both input and output).
            Put the best_fit, fitted_data, parameters, start_window, end_window, std_error out of the days for the class.
        """
        self.Aggregation2Feature2AllFitTry = {Aggregation: InitFeature2AllFitTry(self.Aggregation2Feature2Class2Function2Fit2InitialGuess[Aggregation]) for Aggregation in self.Aggregation2Class2Fcm.keys()}
        for Aggregation in self.Aggregation2Class2Fcm.keys():
            ####################################à
            for Feature in self.Aggregation2Feature2AllFitTry[Aggregation].keys():
                # NOTE: Concatednated Fcm
                ObservedData = self.Aggregation2Fcm[Aggregation][Feature].to_list()
                # Compute the Fit for functions you are Undecided from
                self.Aggregation2Feature2AllFitTry[Aggregation][Feature] = FillIterationFitDicts(ObservedData,
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
                    if self.verbose:
                        print("++++++ Class {} Fit ++++++".format(StrClass))
                    ObservedData = self.Aggregation2Class2Fcm[Aggregation][StrClass][Feature].to_list()
                    self.Aggregation2Feature2Class2AllFitTry[Aggregation][Feature][StrClass] = FillIterationFitDicts(ObservedData,
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
            # Prepare the InitialGuess, FitAllTry (Contains info for each function to try), BestFit (Dictionary that Contains info about each fit)
            self.ComputeAggregatedFit()
            self.ComputeAggregatedFitPerClass()
            # Initialize the Dict of DataFrames Containing the Distribution for each Class to Plot
            for Feature in self.Features2Fit:
                self.Aggregation2Feature2StrClass2FcmDistr = AggregatedFcmDistr(self.Aggregation2Class2Fcm,
                                                                             Feature,
                                                                            True)
            InfoPlotDistrFeat = {"figsize":(4,4),"minx":0,"miny":0,"maxx":0,"maxy":0}
            # Compute the MinMax for the Plot
            self.Aggregation2InfoPlotDistrFeat = {Aggregation:
                                                    {Feature: 
                                                     {ComputeMinMaxPlotGivenFeature(self.Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature])}
                                                    for Feature in self.Aggregation2Feature2StrClass2FcmDistr[Aggregation].keys()}
                                                        for Aggregation in self.Aggregation2Feature2StrClass2FcmDistr.keys()}
            
            fig,ax = PlotFeatureAggregatedAllDays(self.Aggregation2Feature2StrClass2FcmDistr,
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
                    print("After GetMFDForPlot Class {}:\n".format(StrClass))
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
            LatexTableAvFeat = FancyTableFromDict(self.AvFeat2Class2Day[Feature])
            with open(os.path.join(self.PlotDir,f"LatexTableAvFeat_{Feature}.txt"), "w") as file:
                file.write(LatexTableAvFeat)
            
    def GenerateAndSaveTabFit(self):
        Feature2Label = {"lenght_km":"length (km)","speed_kmh":"speed (km/h)","time_hours":"time (h)"}
        self.Feature2Parameters2Class2Day = {Feature: {StrClass: {StrDay: [] for StrDay in self.StrDates} for StrClass in self.Day2StrClass2IntClass.keys()} for Feature in Feature2Label.keys()}
        for Aggregation in self.AggregationLevel:
            for Feature in Feature2Label.keys():
                print("Feature: ",Feature)
                for MobDate in self.ListDailyNetwork:
                    print("Day: ",MobDate.StrDate)
                    for StrClass in self.ListStrClassReference:
                        IntClass = self.Day2StrClass2IntClass[MobDate.StrDate][StrClass]
                        RoundedParam0 = round(MobDate.Feature2Class2Feature2AllFitTry[Feature][IntClass][MobDate.Feature2Class2Feature2AllFitTry["best_fit"]]["parameters"][0],3)
                        RoundedParam1 = round(MobDate.Feature2Class2Feature2AllFitTry[Feature][IntClass][MobDate.Feature2Class2Feature2AllFitTry["best_fit"]]["parameters"][1],3)
                        StrBestFit = MobDate.Feature2Class2Feature2AllFitTry[Feature][IntClass]["best_fit"]
                        if StrBestFit == "exponential":
                            self.Aggregation2Feature2Class2AllFitTry[Aggregation][Feature][StrClass][MobDate.StrDate] = "A = " + str(RoundedParam0) + " $\beta$ = " + str(RoundedParam1)
                        elif StrBestFit == "linear":
                            self.Aggregation2Feature2Class2AllFitTry[Aggregation][Feature][StrClass][MobDate.StrDate] = "A = " + str(RoundedParam0) + " $\alpha$ = " + str(RoundedParam1)
                        elif StrBestFit == "gaussian" or StrBestFit == "maxwellian":
                            self.Aggregation2Feature2Class2AllFitTry[Aggregation][Feature][StrClass][MobDate.StrDate] = "$\mu$ = " + str(RoundedParam0) + " $\sigma$ = " + str(RoundedParam1)                   
                        else:
                            self.Aggregation2Feature2Class2AllFitTry[Aggregation][Feature][StrClass][MobDate.StrDate] = " "
                LatexTableAvParameters = TableFromDict(self.Feature2Parameters2Class2Day[Aggregation][Feature])
                with open(os.path.join(self.PlotDir,f"LatexTableParameters_{Feature}.txt"), "w") as file:
                    file.write(LatexTableAvParameters)
