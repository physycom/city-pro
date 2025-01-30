import datetime
import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def Dict2PolarsDF(Dict,schema):
    return pl.DataFrame(Dict,schema=schema)



# FILL TO THE ZEROS 
def fill_zeros_with_average(vector):
    # Convert the vector to a numpy array for easier manipulation
    vector = np.array(vector)
    
    # Iterate through the vector
    for i in range(len(vector)):
        if vector[i] == 0:
            # Find the previous non-zero element
            prev_index = i - 1
            while (prev_index >= 0 and vector[prev_index] == 0):
                prev_index -= 1
            # Find the next non-zero element
            next_index = i + 1
            while(next_index < len(vector) and vector[next_index] == 0):
                next_index += 1
            # Calculate the average of the previous and next non-zero elements
            if (prev_index >= 0 and next_index < len(vector)):
                vector[i] = (vector[prev_index] + vector[next_index]) / 2
            elif (prev_index >= 0):
                vector[i] = vector[prev_index]
            elif (next_index < len(vector)):
                vector[i] = vector[next_index]
    return vector.tolist()


# MFD RELATED FUNCTIONS
def AddColumns2MFD(MFD,FcmClass,Class,BinTimeStamp,NewClass):
    """
        NOTE: The bins in time that have 0 trajectories have 0 average speed
        NOTE: Speed in MFD in km/h
    """
    if not NewClass:
        TmpDict = {"time":[],f"population_{Class}":[],f"speed_kmh_{Class}":[]}
    else:
        TmpDict = {"time":[],f"new_population_{Class}":[],f"new_speed_kmh_{Class}":[]}
    if MFD is None:
        MFD = {"time":[]}
    for t in range(len(BinTimeStamp)-1):
        StartInterval = datetime.datetime.fromtimestamp(int(BinTimeStamp[t]))
        EndInterval = datetime.datetime.fromtimestamp(int(BinTimeStamp[t+1]))                    
        TmpFcm = FcmClass.with_columns(pl.col('start_time').apply(lambda x: datetime.datetime.fromtimestamp(x), return_dtype=pl.Datetime).alias("start_time_datetime"),
                                    pl.col('end_time').apply(lambda x: datetime.datetime.fromtimestamp(x), return_dtype=pl.Datetime).alias("end_time_datetime"))

        TmpFcm = TmpFcm.filter(pl.col('start_time_datetime').is_between(StartInterval,EndInterval))
        Hstr = StartInterval.strftime("%Y-%m-%d %H:%M:%S").split(" ")[1]
        TmpDict["time"].append(Hstr)
        if not NewClass:
            TmpDict[f"population_{Class}"].append(len(TmpFcm))
            if len(TmpFcm) > 0:
                AvSpeed = TmpFcm.select(pl.col("speed_kmh").mean()).to_pandas().iloc[0]["speed_kmh"]
                TmpDict[f"speed_kmh_{Class}"].append(AvSpeed)
            else:
                TmpDict[f"speed_kmh_{Class}"].append(0)
        else:
            TmpDict[f"new_population_{Class}"].append(len(TmpFcm))
            if len(TmpFcm) > 0:
                AvSpeed = TmpFcm.select(pl.col("speed_kmh").mean()).to_pandas().iloc[0]["speed_kmh"]
                TmpDict[f"new_speed_kmh_{Class}"].append(AvSpeed)
            else:
                TmpDict[f"new_speed_kmh_{Class}"].append(0)
    if not NewClass:
        TmpDict = Dict2PolarsDF(TmpDict,schema = {"time":pl.datatypes.Utf8,f"population_{Class}":pl.Int64,f"speed_kmh_{Class}":pl.Float64})
    else:
        TmpDict = Dict2PolarsDF(TmpDict,schema = {"time":pl.datatypes.Utf8,f"new_population_{Class}":pl.Int64,f"new_speed_kmh_{Class}":pl.Float64})
    if not isinstance(MFD, pl.DataFrame):
        MFD = pl.DataFrame(MFD)    
    if len(MFD) == 0:
        MFD = TmpDict
    else:
        # Merge MFD and TmpDict on the column 'time'
        MFD = MFD.join(TmpDict, on="time", how="left")
    return MFD
def ComputeAggregatedMFDVariables(ListDailyNetwork,MFDAggregated,Class,NewClass):
    """
        Description:
            Every Day I count for each hour, how many people and the speed of the 
            1. Network -> MFDAggregated = {"population":[],"time":[],"speed_kmh":[]}
            2. SubNetwork -> Class2MFDAggregated = {StrClass: {"population":[sum_i pop_{t0,dayi},...,sum_i pop_{iteration,dayi}],"time":[t0,...,iteration],"speed_kmh":[sum_i speed_{t0,dayi},...,sum_i speed_{iteration,dayi}]}}
            NOTE: time is pl.DateTime
        NOTE: Each Time interval has its own average speed and population. For 15 minutes,
            since iteration in 1 Day Analysis is set in that way. 
        NOTE: If at time t there is no population, the speed is set to 0.

        NOTE:
            Speed(Road,Time)
            NumberCars(Road,Time)
            Speed(NumberCars) <=> if NumberCars(Road,Time) = NumberCars(Road',Time') => Speed(Road,Time) = Speed(Road',Time') 
            Compute:
                sum_{Day} P(Speed|NumberCars,Day)P(NumberCars|Day)P(Day)
    """
    if NewClass:
        ColPopulation = f"new_population_{Class}"
        ColSpeed = f"new_speed_kmh_{Class}"
    else:
        ColPopulation = f"population_{Class}"
        ColSpeed = f"speed_kmh_{Class}"
    LocalDayCount = 0
    # AGGREGATE MFD FOR ALL DAYS
    for MobDate in ListDailyNetwork:
        if LocalDayCount == 0:
            MFDAggregated = MobDate.MFD
            if isinstance(MFDAggregated,pl.DataFrame):
                MFDAggregated = MFDAggregated.to_pandas()
            else:
                pass
            MFDAggregated["count_days"] = list(np.zeros(len(MFDAggregated["time"])))
            MFDAggregated["total_number_people"] = list(np.zeros(len(MFDAggregated["time"])))
            LocalDayCount += 1
        else:            
            for t in range(len(MobDate.MFD["time"])):
                WeightedSpeedAtTime = MobDate.MFD[ColSpeed][t]*MobDate.MFD[ColPopulation][t]
                PopulationAtTime = MobDate.MFD[ColPopulation][t]
                if PopulationAtTime != 0 and WeightedSpeedAtTime !=0:
                    MFDAggregated[ColSpeed][t] += WeightedSpeedAtTime
                    MFDAggregated[ColPopulation][t] += PopulationAtTime
                    MFDAggregated["count_days"][t] += 1
                    MFDAggregated["total_number_people"][t] += PopulationAtTime
                else:
                    pass
    for t in range(len(MFDAggregated["time"])):
        if MFDAggregated["count_days"][t] != 0:
            MFDAggregated[ColSpeed][t] = MFDAggregated[ColSpeed][t]/(MFDAggregated["count_days"][t]*MFDAggregated["total_number_people"][t])
            MFDAggregated[ColPopulation][t] = int(MFDAggregated[ColPopulation][t]/(MFDAggregated["count_days"][t]*MFDAggregated["total_number_people"][t]))
        else:
            pass
    from pandas import DataFrame
    MFDAggregated = pl.DataFrame(DataFrame(MFDAggregated))
#    MFDAggregated = Dict2PolarsDF(MFDAggregated,schema = {"time":pl.datatypes.Utf8,ColPopulation:pl.Int64,ColSpeed:pl.Float64,"count_days":pl.Int64,"total_number_people":pl.Int64})
    return MFDAggregated

def AggregateMFDByHolidays(ListDailyNetwork,AggregationLevel2ListDays):
    """
        Returns:
            {"Aggregation":[MFD_{DayInAggregation1},...,MFD_{DayInAggregationN}]}
    """
    Aggregation2MFD = defaultdict()
    for MobDate in ListDailyNetwork:
        for Aggregation in AggregationLevel2ListDays:
            if MobDate.StrDate in AggregationLevel2ListDays[Aggregation]:
                Aggregation2MFD[Aggregation] = MobDate.MFD
    return Aggregation2MFD

def GetAverageConditional(Df,ConditioningLabel,ConditionedLabel,binsi,binsi1):
    """
        Df: pl.DataFrame 
        ConditioningLabel: str -> Conditioning Variable name in the DataFrame.
        ConditionedLabel: str -> The Column from which extracting the average.
        bini: int
        binsi1: int
        Return: float -> Average of the ConditionedLabel in the interval [bini,binsi1]

    """
    assert ConditioningLabel in Df.columns,  "Error: ConditioningLabel -> {} Column in the DataFrame".format(ConditioningLabel)
    assert ConditionedLabel in Df.columns,  "Error: ConditionedLabel -> {} Column in the DataFrame".format(ConditionedLabel)
    DfTmp = Df.filter((pl.col(ConditioningLabel) >= binsi) & 
                (pl.col(ConditioningLabel)<=binsi1))
    if len(DfTmp)>0:
        return DfTmp.with_columns(pl.col(ConditionedLabel).mean()).to_pandas().iloc[0][ConditionedLabel]
    else:
        return 0

def GetStdErrorConditional(Df,ConditioningLabel,ConditionedLabel,binsi,binsi1):
    """
        Df: pl.DataFrame 
        ConditioningLabel: str -> Conditioning Variable name in the DataFrame.
        ConditionedLabel: str -> The Column from which extracting the average.
        bini: int
        binsi1: int
        Return: float -> Standard Error of the ConditionedLabel in the interval [bini,binsi1]

    """
    assert ConditioningLabel in Df.columns,  "Error: ConditioningLabel -> {} Column in the DataFrame".format(ConditioningLabel)
    assert ConditionedLabel in Df.columns,  "Error: ConditionedLabel -> {} Column in the DataFrame".format(ConditionedLabel)
    DfTmp = Df.filter((pl.col(ConditioningLabel) >= binsi) & 
                (pl.col(ConditioningLabel)<=binsi1))
    if len(DfTmp)>1:
        return DfTmp.with_columns(pl.col(ConditionedLabel).std()).to_pandas().iloc[0][ConditionedLabel]
    else:
        return 0

def GetLowerBoundsFromBins(bins,label,MinMaxPlot,Class,case):
    if case == "no-classes":
        print("Get Lower Bounds From Bins: {}".format(label))
        MinMaxPlot[label] = {"min":bins[0],"max":bins[-1]}  
        return MinMaxPlot
    else:
        print("Get Lower Bounds From Bins: {0} Class {1}".format(label,Class))
        MinMaxPlot[Class][label] = {"min":bins[0],"max":bins[-1]}
        return MinMaxPlot
def GetRelativeChange(MFD2Plot,Classes,NewClass):
    Class2RelativeChange = defaultdict()
    for Class in Classes:
        if NewClass:
            PopulationColumn = "new_population_{}".format(Class)
            SpeedColumn = "new_speed_kmh_{}".format(Class)
        else:
            PopulationColumn = "population_{}".format(Class)
            SpeedColumn = "speed_kmh_{}".format(Class) 
        Y_Interval = max(MFD2Plot[f'bin_{SpeedColumn}']) - min(MFD2Plot[f'bin_{SpeedColumn}'])
        if max(MFD2Plot[f'bin_{SpeedColumn}'])/100!=0:
            RelativeChange = Y_Interval/max(MFD2Plot[f'bin_{SpeedColumn}'])/100
        else:
            RelativeChange = 0
        Class2RelativeChange[Class] = RelativeChange
    return Class2RelativeChange

def GetMFDForPlot(MFD,MFD2Plot,MinMaxPlot,Class,case,NewClass,bins_ = 12):
    """
        Input:
            MFD: {"population":[],"time":[],"speed_kmh":[]} or {Class:pl.DataFrame{"population":[],"time":[],"speed_kmh":[]}}
        NOTE: Used in self.PlotMFD()
        NOTE: Modifies MDF2Plot = {"bins_population":[p0,..,p19],"binned_av_speed":[v0,..,v19],"binned_sqrt_err_speed":[e0,..,e19]}
        NOTE: Modifies MinMaxPlot = {"speed_kmh":{"min":v0,"max":v19},"population":{"min":p0,"max":p19}}    
    """
    logger.info("Get MFD For Plot: {}".format(Class))
    if NewClass:
        PopulationColumn = "new_population_{}".format(Class)
        SpeedColumn = "new_speed_kmh_{}".format(Class)
    else:
        PopulationColumn = "population_{}".format(Class)
        SpeedColumn = "speed_kmh_{}".format(Class) 
    if MFD2Plot is None:
        MFD2Plot = {f'bin_{SpeedColumn}':[],f'binned_sqrt_err_{SpeedColumn}':[],f"bins_{PopulationColumn}":[]}
    else:
        MFD2Plot[f'bin_{SpeedColumn}'] = []
        MFD2Plot[f'binned_sqrt_err_{SpeedColumn}'] = []
        MFD2Plot[f"bins_{PopulationColumn}"] = []
    n, bins = np.histogram(MFD[PopulationColumn],bins = bins_)
    labels = range(len(bins) - 1)
    for i in range(len(labels)):
        # Fill Average/Std Speed (to plot)
        BinnedAvSpeed = GetAverageConditional(MFD,PopulationColumn,SpeedColumn,bins[i],bins[i+1])
        MFD2Plot[f'bin_{SpeedColumn}'].append(float(BinnedAvSpeed))
        BinnedSqrtSpeed = GetStdErrorConditional(MFD,PopulationColumn,SpeedColumn,bins[i],bins[i+1])
        MFD2Plot[f'binned_sqrt_err_{SpeedColumn}'].append(float(BinnedSqrtSpeed))
    MFD2Plot[f"bins_{PopulationColumn}"] = bins[1:]
    fill_zeros_with_average(MFD2Plot[f'bin_{SpeedColumn}'])
    MinMaxPlot = GetLowerBoundsFromBins(bins = bins,label = "population",MinMaxPlot = MinMaxPlot,Class = Class,case = case)
    MinMaxPlot = GetLowerBoundsFromBins(bins = MFD2Plot[f'bin_{SpeedColumn}'],label = "speed_kmh",MinMaxPlot = MinMaxPlot, Class = Class,case = case)
    Y_Interval = max(MFD2Plot[f'bin_{SpeedColumn}']) - min(MFD2Plot[f'bin_{SpeedColumn}'])
    if max(MFD2Plot[f'bin_{SpeedColumn}'])/100!=0:
        RelativeChange = Y_Interval/max(MFD2Plot[f'bin_{SpeedColumn}'])/100
    else:
        RelativeChange = 0
    return MFD2Plot,MinMaxPlot,RelativeChange

def PlotMFD(binsPop,binsAvSpeed,binsSqrt,RelativeChange,SaveDir,Title = "Fondamental Diagram Aggregated",NameFile = "MFD.png"):
    """
        
    """
    logger.info("Plotting MFD:\n")
    fig, ax = plt.subplots(1,1,figsize = (10,8))
    text = "Relative change : {}%".format(round(RelativeChange,2))
    if len(binsPop) != len(binsAvSpeed): 
        ax.plot(binsPop[1:],binsAvSpeed)
        ax.fill_between(np.array(binsPop[1:]),
                            np.array(binsAvSpeed) - np.array(binsSqrt), 
                            np.array(binsAvSpeed) + np.array(binsSqrt), color='gray', alpha=0.2, label='Std')
    else:
        ax.plot(binsPop,binsAvSpeed)
        ax.fill_between(np.array(binsPop),
                            np.array(binsAvSpeed) - np.array(binsSqrt), 
                            np.array(binsAvSpeed) + np.array(binsSqrt), color='gray', alpha=0.2, label='Std')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10,
    verticalalignment='top', bbox=props)
    ax.set_title(Title)
    ax.set_xlabel("number people")
    ax.set_ylabel("speed (km/h)")
    plt.savefig(os.path.join(SaveDir,NameFile),dpi = 200)
    plt.close()

def PlotHysteresis(MFD,ColSpeed,ColPop,Title,SaveDir,NameFile):
    logger.info("Plot Hysteresis: {}".format(NameFile))
    if isinstance(MFD,pl.DataFrame):
        x = MFD[ColPop].to_list()
        y = MFD[ColSpeed].to_list()
    else:
        x = MFD[ColPop]
        y = MFD[ColSpeed]
    if len(x)!= 0 and len(y)!=0:
        u = [x[i+1]-x[i] for i in range(len(x)-1)]
        v = [y[i+1]-y[i] for i in range(len(y)-1)]
        u.append(x[len(x)-1] -x[0])
        v.append(y[len(y)-1] -y[0])
        plt.quiver(x,y,u,v,angles='xy', scale_units='xy', scale=1,width = 0.0025)
        plt.xlabel('Number People')
        plt.ylabel('Speed (km/h)')
        plt.title(Title)
        plt.savefig(os.path.join(SaveDir,NameFile),dpi = 200)
        plt.close()
    else:
        print("No Data for: ",os.path.join(SaveDir,NameFile))
# END MFD RELATED FUNCTIONS
def PlotMFDComparison(ListDailyNetwork,Class,Colors,NewClass,PlotDir):
    if NewClass:
        PopulationColumn = "new_population_{}".format(Class)
        SpeedColumn = "new_speed_kmh_{}".format(Class)
    else:
        PopulationColumn = "population_{}".format(Class)
        SpeedColumn = "speed_kmh_{}".format(Class) 
    fig, ax = plt.subplots(1,1,figsize = (10,8))
    CountDate = 0
    # Will Plot also the distribution of the linear coefficient to see if it is Decreasing
    LinearCoeffPerDay = []
    Days = []
    for MobDate in ListDailyNetwork:
        MFD2Plot = MobDate.MFD2Plot
        MFD2PlotBinSpeed = np.array(fill_zeros_with_average(MFD2Plot[f'bin_{SpeedColumn}']))
        MFD2PlotBinSpeed = np.array(fill_zeros_with_average(MFD2PlotBinSpeed))
        MFD2PlotBinSquareErr = np.array(fill_zeros_with_average(MFD2Plot[f'binned_sqrt_err_{SpeedColumn}']))
        MFD2PlotBinSquareErr = fill_zeros_with_average(MFD2PlotBinSquareErr)
        aq = np.polyfit(MFD2Plot[f'bins_{PopulationColumn}'],MFD2PlotBinSpeed,1)
        LinearCoeffPerDay.append(aq[0])
        ax.plot(MFD2Plot[f'bins_{PopulationColumn}'],MFD2PlotBinSpeed,color = Colors[CountDate],label=f"{MobDate.StrDate}")
        ax.fill_between(MFD2Plot[f'bins_{PopulationColumn}'],
                        np.array(MFD2PlotBinSpeed) - np.array(MFD2PlotBinSquareErr), 
                        np.array(MFD2PlotBinSpeed) + np.array(MFD2PlotBinSquareErr), color='gray', alpha=0.2, label=None)
        ax.plot(MFD2Plot[f'bins_{PopulationColumn}'],aq[0]*np.array(MFD2Plot[f'bins_{PopulationColumn}'])+aq[1],color = Colors[CountDate],linestyle = "--")
        Days.append(MobDate.StrDate)
        CountDate += 1
    ax.set_title("Fondamental Diagram All Days {}".format(Class))
    ax.set_xlabel("number people")
    ax.set_ylabel("v (km/h)")
    ax.legend(fontsize = "small")
    if NewClass:
        plt.savefig(os.path.join(PlotDir,f"ComparisonMFD_{Class}_NewClass"),dpi = 200)
    else:
        plt.savefig(os.path.join(PlotDir,f"ComparisonMFD_{Class}"),dpi = 200)
    plt.close()
    fig, ax = plt.subplots(1,1,figsize = (10,8))
    ax.scatter(Days,LinearCoeffPerDay)
    ax.set_xticks(np.arange(0,len(Days),1))
    ax.set_xticklabels(Days,rotation = 90)
    if NewClass:
        plt.savefig(os.path.join(PlotDir,f"LinearCoeff_{Class}_NewClass"),dpi = 200)
        pl.DataFrame({"Days":Days,"LinearCoeff":LinearCoeffPerDay}).write_csv(os.path.join(PlotDir,f"LinearCoeff_{Class}_NewClass.csv"))            
    else:
        plt.savefig(os.path.join(PlotDir,f"LinearCoeff_{Class}"),dpi = 200)
        pl.DataFrame({"Days":Days,"LinearCoeff":LinearCoeffPerDay}).write_csv(os.path.join(PlotDir,f"LinearCoeff_{Class}.csv"))            
    plt.close()


def PlotCoeffClassification(PlotDir,Classes):
    """
        @Classes: list -> List of Classes
        Plots the linear coefficient of the fundamental diagram for each class
    """
    Colors = ["red","blue","green","black","yellow","orange","purple","pink","brown","cyan"]
    Markers = ["o","s","^","v","<",">","1","2","3","4"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('viridis')  # You can choose any colormap you like
    z_values = Classes  # Example z values for hyperplanes
    num_layers = len(z_values)
    # Generate colors for each layer
    Layer2Color = {z_values[i]: cmap(i / num_layers) for i in range(num_layers)}
    for Class in Classes:
        ClassCoord = []
        OldCoeff = []
        NewCoeff = []
        DfOld = pl.read_csv(os.path.join(PlotDir,f"LinearCoeff_{Class}.csv"))
        LinearCoeffPerDay = DfOld.to_pandas()["LinearCoeff"].to_numpy()
        DfNew = pl.read_csv(os.path.join(PlotDir,f"LinearCoeff_{Class}_NewClass.csv"))
        Days = DfNew["Days"].to_numpy()
        Day2Marker = {Days[i]:Markers[i] for i in range(len(Days))}
        Day2Color = {Days[i]:Colors[i] for i in range(len(Days))}
        LinearCoeffPerDayNew = DfNew.to_pandas()["LinearCoeff"].to_numpy()
        for Day in range(len(LinearCoeffPerDay)):
            OldCoeff.append(LinearCoeffPerDay[Day])
            NewCoeff.append(LinearCoeffPerDayNew[Day])
            ClassCoord.append(Class)
            ax.scatter(NewCoeff[Day], OldCoeff[Day], ClassCoord[Day], c=Day2Color[Days[Day]], marker=Day2Marker[Days[Day]])
    ax.set_xlabel(r'$\alpha_k^{f}$')
    ax.set_ylabel(r'$\alpha_k^{h}$')
    x_range = np.linspace(min(LinearCoeffPerDayNew), max(LinearCoeffPerDayNew), 10)
    y_range = np.linspace(min(LinearCoeffPerDay), max(LinearCoeffPerDay), 10)
    X, Y = np.meshgrid(x_range, y_range)
    for z in z_values:
        Z = np.full_like(X, z)
        ax.plot_surface(X, Y, Z, alpha=0.1, color=Layer2Color[z], rstride=100, cstride=100)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker=Markers[i], color='w', label=f'Day {Days[i]}', markerfacecolor=Day2Color[Days[i]], markersize=10) for i in range(len(Days))]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    fig.savefig(os.path.join(PlotDir,"CoeffClassification3D.png"),dpi = 200)
    plt.close()    