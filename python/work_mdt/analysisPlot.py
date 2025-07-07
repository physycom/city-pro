import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import folium
import polars as pl
from collections import defaultdict
import seaborn as sns
from matplotlib.lines import Line2D
from MFDAnalysis import *
from CastVariables import *
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
VERBOSE = True      
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
    if not os.path.exists(os.path.join(save_dir,"Hysteresys")):
        os.makedirs(os.path.join(save_dir,"Hysteresys"))
    if average_all_days:
        plt.savefig(os.path.join(save_dir,"Hysteresys",'Hysteresis_Average_{0}_Class_{1}.png'.format(day,dict_name[idx])),dpi = 200)
    else:
        plt.savefig(os.path.join(save_dir,"Hysteresys",'Hysteresis_{0}_Class_{1}.png'.format(day,dict_name[idx])),dpi = 200)
    plt.close()

def MFDByClass(population,velocity,dict_name,idx,save_dir,day,verbose = False): 

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
    plt.close()



##----------------------------------- PLOT TIMES -----------------------------------##

def PlotTimePercorrenceDistribution(RoadsTimeVel,Time2Distr,AvgTimePercorrence,StrTimesLabel,File2Save):    
    """
        Input:
            RoadsTimeVel: pl.DataFrame -> DataFrame with the Roads Time Velocities 
        Description:
            Time2Distr: array -> Array with the Time Percorrence Distribution (should be list of 96 values, Number of roads available)
            AvgTimePercorrence: array -> Array with the Average Time Percorrence (should be list of 96 values)
        NOTE: In the case I do not have information about a street of the subnet I will not have any information about the time_percorrence
    """
    Slicing = 8
    VarianceVec = []
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    RoadsTimeVel = RoadsTimeVel.sort("start_bin")
    CountNonNull = 0
    for time,RTV in RoadsTimeVel.groupby("start_bin"):
        ValidTime = RTV.filter(pl.col("time_percorrence")>0)
        if len(ValidTime) == 0:
            pass
        else:
            CountNonNull += 1
        Time2Distr.append(ValidTime["time_percorrence"].to_list()/np.sqrt(len(ValidTime["time_percorrence"].to_list())))
        AvgTimePercorrence.append(np.mean(ValidTime["time_percorrence"].to_list()))
        VarianceVec.append(np.std(ValidTime["time_percorrence"].to_list())/np.sqrt(len(ValidTime["time_percorrence"].to_list())))
        StartInterval = datetime.datetime.fromtimestamp(time)
        StrTimesLabel.append(StartInterval.strftime("%Y-%m-%d %H:%M:%S").split(" ")[1])
    # Count how many values are not null
    
    if VERBOSE:
        print("Number of Null Time Slots: ",len(AvgTimePercorrence) - CountNonNull)
        print("Save in: ",File2Save)
        print("Number of Time Slot with valid values: ",CountNonNull)
    # Plot just those partition that contain half of valid numbers
    if CountNonNull > len(AvgTimePercorrence)/2:
        ax.plot(StrTimesLabel, AvgTimePercorrence)
#        ax.boxplot(Time2Distr,sym='')
        ax.errorbar(StrTimesLabel, AvgTimePercorrence, yerr=VarianceVec, fmt='o')
        ax.set_xticks(range(len(StrTimesLabel))[::Slicing])  # Set the ticks to correspond to the labels
        ax.set_xticklabels(StrTimesLabel[::Slicing], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
        ax.set_xlabel("Time")
        ax.set_ylabel("Time Percorrence")
        plt.savefig(File2Save,dpi = 200)
        plt.close()
    return Time2Distr,AvgTimePercorrence


def ComputeTimePercorrence(GeoJson,RoadsTimeVel,Class,StrDay):
    RoadsTimeVel = RoadsTimeVel.sort("start_bin")
    for time,RTV in RoadsTimeVel.groupby("start_bin"):
        StartInterval = datetime.datetime.fromtimestamp(time).strftime("%Y-%m-%d %H:%M:%S").split(" ")[1]
        GeoJson["TimePercorrence_" + StartInterval + "_" + str(Class) + "_" + StrDay] = np.ones(len(GeoJson))*(-1)
        GeoJson["AvSpeed_" + StartInterval + "_" + str(Class) + "_" + StrDay] = np.ones(len(GeoJson))*(-1)
        ValidTime = RTV.filter(pl.col("time_percorrence")>0)
        Roads = ValidTime["poly_id"].to_list()
        for Road in Roads:
            GeoJson.loc[GeoJson["poly_lid"] == Road,"TimePercorrence_" + StartInterval + "_" + str(Class) + "_" + StrDay] = ValidTime.filter(pl.col("poly_id") == Road)["time_percorrence"].to_list()[0]
            GeoJson.loc[GeoJson["poly_lid"] == Road,"AvSpeed_" + StartInterval + "_" + str(Class) + "_" + StrDay] = ValidTime.filter(pl.col("poly_id") == Road)["av_speed"].to_list()[0]
    return GeoJson
def AggregateTimePercorrence(GeoJson,StrDay):
    Hour2Road2TimePercorrence = {datetime.datetime.fromtimestamp(time).strftime("%Y-%m-%d %H:%M:%S").split(" ")[1]: {Road:[] for Road in GeoJson["poly_lid"].to_numpy()} for time,_ in RoadsTimeVel.groupby("start_bin")}
    Hour2Road2AvSpeed = Hour2Road2TimePercorrence


##----------------------------------- PLOT DISTRIBUTIONS -----------------------------------##

def ComputeMinMaxPlotGivenFeature(Class2FcmDistr,InfoPlotDistrFeat):
    maxx = 0
    maxy = 0
    minx = 10000000
    miny = 10000000
    for IntClass in Class2FcmDistr.keys():
        if maxx < Class2FcmDistr[IntClass]["maxx"]:
            maxx = Class2FcmDistr[IntClass]["maxx"]
        if maxy < Class2FcmDistr[IntClass]["maxy"]:
            maxy = Class2FcmDistr[IntClass]["maxy"]
        if minx > Class2FcmDistr[IntClass]["minx"]:
            minx = Class2FcmDistr[IntClass]["minx"]
        if miny > Class2FcmDistr[IntClass]["miny"]:
            miny = Class2FcmDistr[IntClass]["miny"]
    InfoPlotDistrFeat["maxx"] = maxx
    InfoPlotDistrFeat["maxy"] = maxy
    InfoPlotDistrFeat["minx"] = minx
    InfoPlotDistrFeat["miny"] = miny
    return InfoPlotDistrFeat

def ScatterAndPlotLegend(ax,Class2FcmDistr,Feature,IntClass2StrClass,DictFittedData,Column2Legend,IntClass,legend):
    ax.scatter(Class2FcmDistr[IntClass]["x"][1:],Class2FcmDistr[IntClass]["y"])
    legend.append(str(IntClass2StrClass[IntClass]) + " " + Column2Legend[Feature] + " " + str(round(Class2FcmDistr[IntClass]["mean"],3)))
    # Fit
    if len(Class2FcmDistr[IntClass]["x"][1:]) == len(DictFittedData[Feature]["fitted_data"]):
        ax.plot(Class2FcmDistr[IntClass]["x"][1:],np.array(DictFittedData[Feature]["fitted_data"]),label = DictFittedData[Feature]["best_fit"])
        legend.append(str(IntClass2StrClass[IntClass]) + " " + Column2Legend[Feature] + " " + str(round(Class2FcmDistr[IntClass]["mean"],3)))
    return ax,legend

def ScatterAndPlotSingleClass(ax,Class2FcmDistr,Feature,DictFittedData,IntClass):
    ax.scatter(Class2FcmDistr[IntClass]["x"][1:],Class2FcmDistr[IntClass]["y"])
    # Fit
    if len(Class2FcmDistr[IntClass]["x"][1:]) == len(DictFittedData[Feature]["fitted_data"]):
        ax.plot(Class2FcmDistr[IntClass]["x"][1:],np.array(DictFittedData[Feature]["fitted_data"]),label = DictFittedData[Feature]["best_fit"])
    return ax

def PlotTransitionClass2ClassNew(DfComparison,PlotDir):
    """
        @brief:
            Plot the Transition Class to Class New
            x-axis: Class
            y-axis: Number of People

    """
    ListColors = ["green", "yellow", "black", "orange", "purple", "pink", "brown", "grey"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Get unique classes
    Classes = np.sort(DfComparison["ClassBefore"].unique().to_list())
    TotalPeopleHierarchicalNet = []
    Tij = []
    Nij = []
    for OldClass in Classes:
        OutGoingOld = DfComparison.filter(pl.col("ClassBefore") == OldClass).sort("ClassAfter")
        # Fraction People Transit From i -> j
        Ni = np.array(OutGoingOld["NumberBefore"])[0]
        Tij.append(OutGoingOld["Tij"])
        Nij.append(Ni*np.array(OutGoingOld["Tij"]))
        if len(TotalPeopleHierarchicalNet) == 0:
            TotalPeopleHierarchicalNet = DfComparison.filter(pl.col("ClassBefore") == OldClass).sort("ClassAfter")["NumberAfter"]
            TotalPeopleNet = DfComparison.filter(pl.col("ClassAfter") == OldClass).sort("ClassBefore")["NumberBefore"]
    width = 0.45
    for i, Class in enumerate(Classes):
        ax.bar(np.array(Classes), Ni*np.array(Tij[i]), width = width,color=ListColors[i % len(ListColors)], label= Class)
    ax.scatter(np.array(Classes),TotalPeopleHierarchicalNet,label = "N Traj Hierarchical Net")
    ax.scatter(np.array(Classes),TotalPeopleNet,label = "N Traj Net")
    ax.set_xticks(Classes)
    # Set labels and title
    ax.set_xlabel('Class')
    ax.set_ylabel('Composition Trajectories after hierarchical re-organization')
    ax.legend()
    ax.set_title('Redistribution of Trajectories to Hierarchical Classes')
    # Save the plot
    plt.savefig(os.path.join(PlotDir, 'TransitionClass2ClassNew.png'), dpi=200)
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    cax = ax.matshow(Tij, cmap='viridis')
    cbar = fig.colorbar(cax)
    # Annotate each cell with the numeric value
    for (i, j), val in np.ndenumerate(Tij):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')
    ax.set_xticks(Classes)
    ax.set_yticks(Classes)
    ax.set_xlabel('Class Hierarchical')
    ax.set_ylabel('Class')
    ax.set_title('Transition Matrix i -> j')
    plt.savefig(os.path.join(PlotDir, 'TransitionMatrix.png'), dpi=200)
    plt.close()
    return Tij
# Plot Fit Single Day
def PlotFeatureDistrSeparatedByClass(Feature2IntClass2FcmDistr,
                                    Feature2InfoPlotDistrFeat,
                                    IntClass2StrClass,
                                    Feature2Class2AllFitTry,
                                    Feature2Legend,
                                    Feature2IntervalBin,
                                    Feature2IntervalCount,
                                    Feature2Label,
                                    Feature2ShiftBin,
                                    Feature2ShiftCount,
                                    Feature2ScaleBins,
                                    Feature2ScaleCount,
                                    Feature2DistributionPlot,
                                    PlotDir,
                                    Feature2SaveName,
                                    NormBool = True,
                                    Case = ""):
    """
        Class2FcmDistr: dict -> {IntClass: {"x":x,"y":y,"maxx":max(x),"maxy":max(y),"minx":min(x),"miny":min(y),"mean":np.mean(Fcm.filter(pl.col("class") == IntClass)[Feature].to_list())}}
        InfoPlotDistrFeat: dict -> {"maxx":0,"maxy":0,"minx":10000000,"miny":10000000}
        Feature: str -> Feature to Plot
        IntClass2StrClass: dict -> {IntClass: StrClass}
        DictFittedData: dict -> {Feature: {"fitted_data": [],"best_fit": str}}
        Column2Legend: dict -> {Feature: Legend}
        Feature2IntervalBin: dict -> {Feature: IntervalBin}
        Feature2IntervalCount: dict -> {Feature: IntervalCount}
        Feature2Label: dict -> {Feature: Label}
        Feature2ShiftBin: dict -> {Feature: ShiftBin}
        Feature2ShiftCount: dict -> {Feature: ShiftCount}
        Feature2ScaleBins: dict -> {Feature: ScaleBins}
        Feature2ScaleCount: dict -> {Feature: ScaleCount}
        PlotDir: str -> Path to Save the Plot

        NOTE: 
         Plots: P(Feature|Class,Day)P(Class|Day)P(Day) as if P(Day) = 1, P(Class|Day) = 1
    """
    for Feature in Feature2IntClass2FcmDistr.keys():
        fig,ax = plt.subplots(1,1,figsize = Feature2InfoPlotDistrFeat[Feature]["figsize"])
        legend = []
        if "time" in Feature:
            minx = 0.1
            maxx = 22
            miny = 0.000001
            maxy = 1
        if "lenght" in Feature:
            minx = 0.1
            maxx = 11
            miny = 0.000001
            maxy = 1
        else:
            minx,maxx,miny,maxy = ComputeCommonBins(Feature2IntClass2FcmDistr[Feature])
        for IntClass in Feature2IntClass2FcmDistr[Feature].keys():
            mean = np.nanmean(np.array(Feature2IntClass2FcmDistr[Feature][IntClass]["y"])*np.array(Feature2IntClass2FcmDistr[Feature][IntClass]["x"][1:]))

            if NormBool:
                Feature2IntervalCount[Feature] = 0.05
                Feature2ShiftCount[Feature] = 0.1
            LabelBestFit = Feature2Class2AllFitTry[Feature][IntClass]["best_fit"]
            if LabelBestFit != "":
                mean = Feature2IntClass2FcmDistr[Feature][IntClass]["mean"]
                if LabelBestFit == "exponential":
                    x_windowed = np.array(Feature2Class2AllFitTry[Feature][IntClass][LabelBestFit]["x_windowed"])/mean
                else:
                    x_windowed = Feature2Class2AllFitTry[Feature][IntClass][LabelBestFit]["x_windowed"]
                fitted_data_windowed = Feature2Class2AllFitTry[Feature][IntClass][LabelBestFit]["fitted_data_windowed"]
            if LabelBestFit == "exponential":
                x = np.array(Feature2IntClass2FcmDistr[Feature][IntClass]["x"][1:])/mean
            elif LabelBestFit != "gaussian":
                x = Feature2IntClass2FcmDistr[Feature][IntClass]["x"][1:]/mean
            else:
                x = Feature2IntClass2FcmDistr[Feature][IntClass]["x"][1:]
            y = Feature2IntClass2FcmDistr[Feature][IntClass]["y"]
            
            # Scatter Points
            ax.scatter(x_windowed,y)
#            legend.append(str(IntClass2StrClass[IntClass]) + " " + Feature2Legend[Feature] + " " + str(round(mean,3)))
            legend.append(str(IntClass2StrClass[IntClass]))
            # Fit
 #           if LabelBestFit != "":
 #               if len(x_windowed) == len(fitted_data_windowed):
#                    ax.plot(x_windowed,fitted_data_windowed,label = "",linestyle = "--")
#                    legend.append(str(IntClass2StrClass[IntClass]))
#                    legend.append(str(IntClass2StrClass[IntClass]) + " " + Feature2Legend[Feature] + " " + str(round(mean,3)))
        ax.set_xticks(np.arange(minx,maxx,Feature2IntervalBin[Feature]))
        ax.set_yticks(np.arange(miny,maxy,Feature2IntervalCount[Feature]))
        if LabelBestFit == "exponential":
            if "lenght" in Feature:
                Feat = "L"
            elif "time" in Feature:
                Feat = "t"
            else:
                Feat = "v"
            ax.set_xlabel(Feat + f"/ <{Feat}>")
            ax.set_ylabel("P(" + Feat + f"/ <{Feat}>" + ")")
        else:
            ax.set_xlabel(Feature2Label[Feature])
            ax.set_ylabel('P({0})'.format(Feature2Label[Feature]))
        ax.set_xlim(left = 0.1, right = maxx + Feature2ShiftBin[Feature])
        ax.set_ylim(bottom = 0.000001,top = maxy + Feature2ShiftCount[Feature])
#        ax.set_xscale(Feature2ScaleBins[Feature])
        ax.set_yscale(Feature2ScaleCount[Feature])
#        ax.set_title(f"Best Fit: {Feature2Class2AllFitTry[Feature][IntClass]["best_fit"]}")
        legend_ = plt.legend(legend)
        frame = legend_.get_frame()
        frame.set_facecolor('white')
        Feature2DistributionPlot[Feature]["fig"] = fig
        Feature2DistributionPlot[Feature]["ax"] = ax
        Date = os.path.basename(PlotDir)
        plt.savefig(os.path.join(PlotDir,'{0}_{1}_{2}{3}.png'.format("Aggregated",Feature2SaveName[Feature],Date,Case)),dpi = 200)
        plt.close()
#        if VERBOSE:
#            print("Plot Distributions With All Classes")
#            print("Feature: ",Feature)
#            print("min x: ",minx," max x: ",maxx," min y: ",miny," max y: ",maxy)
#            print("Plotted x:\n",x)
#            print("Plotted y:\n",y)
#            print("Plotted x_windowed:\n",x_windowed)
#            print("Plotted fitted_data_windowed:\n",fitted_data_windowed)
#            print("min x windowed",min(x_windowed),"max x windowed: ",max(x_windowed),"min y windowed", min(fitted_data_windowed)," max y windowed: ",max(fitted_data_windowed))
    return fig,ax
def ComputeCommonBins(IntClass2FcmDistr):
    minx = 10000000
    maxx = 0
    miny = 10000000
    maxy = 0
    for IntClass in IntClass2FcmDistr:
        if minx > min(IntClass2FcmDistr[IntClass]["x"]):
            minx = min(IntClass2FcmDistr[IntClass]["x"])
        if maxx < max(IntClass2FcmDistr[IntClass]["x"]):
            maxx = max(IntClass2FcmDistr[IntClass]["x"])
        if miny > min(IntClass2FcmDistr[IntClass]["y"]):
            miny = min(IntClass2FcmDistr[IntClass]["y"])
        if maxy < max(IntClass2FcmDistr[IntClass]["y"]):
            maxy = max(IntClass2FcmDistr[IntClass]["y"])
    return minx,maxx,miny,maxy
def PlotFeatureSingleClass(FcmDistr,
                           AllFitTry,
                           IntervalBin,
                           IntervalCount,
                           Label,
                           ShiftBin,
                           ShiftCount,
                           ScaleBins,
                           ScaleCount,):  
    """
        P(Feature|Class,Day)P(Class|Day)P(Day) 
        NOTE: Consider as if P(Day) = 1, P(Class|Day) = 1
    """   
    fig,ax = plt.subplots(1,1,figsize = (12,10))
    # Scatter Points
    x = FcmDistr["x"][1:]
    y = FcmDistr["y"]
    LabelBestFit = AllFitTry["best_fit"]
    if AllFitTry["best_fit"] != "":
        if AllFitTry["best_fit"] == "exponential":
            mean = np.sum(np.array(x)*np.array(y))
            x_windowed = np.array(AllFitTry[LabelBestFit]["x_windowed"])/mean
            x = np.array(x)/mean
        else:
            x_windowed = AllFitTry[LabelBestFit]["x_windowed"]
        fitted_data_windowed = AllFitTry[LabelBestFit]["fitted_data_windowed"]
    ax.scatter(x,y)
    # Fit
    if AllFitTry["best_fit"] != "":
        if len(x_windowed) == len(fitted_data_windowed):
            ax.plot(x_windowed,fitted_data_windowed)
    ax.set_xticks(np.arange(min(x),max(x),IntervalBin))
    ax.set_yticks(np.arange(min(y),max(y),IntervalCount))
    if AllFitTry["best_fit"] == "exponential":
        ax.set_xlabel(Label + f"/ <{Label}>")
        ax.set_ylabel("P(" + Label + f"/ <{Label}>" + ")")
    else:
        ax.set_xlabel(Label)
        ax.set_ylabel(f'P({Label})')
    ax.set_xlim(left = 0.01,right = max(x) + ShiftBin)
    ax.set_ylim(bottom = 0.000001,top = max(y) + ShiftCount)
    ax.set_xscale(ScaleBins)
    ax.set_yscale(ScaleCount)
    ax.set_title(f"Best Fit: {AllFitTry["best_fit"]}")

    return fig,ax



def PlotFeatureAggregatedAllDays(Aggregation2Feature2StrClass2FcmDistr,                   
                                    Aggregation2Feature2Class2AllFitTry,
                                    Feature2Legend,
                                    Feature2IntervalBin,
                                    Feature2IntervalCount,
                                    Feature2Label,
                                    Feature2ShiftBin,
                                    Feature2ShiftCount,
                                    Feature2ScaleBins,
                                    Feature2ScaleCount,
                                    PlotDir,
                                    Feature2SaveName,
                                    NormBool = True):
    """
        Plot aggregated over all days distribution of speed, time and length
        \sum_{Day}^{Aggregation} P(Feature|Class,Day)P(Class|Day)P(Day)
        NOTE: 
        This represent the dependence of the Feature to the class integrating out the day
    """
#    Features = ["time_hours","lenght_km","speed_kmh"]
    Features = ["speed_kmh"]
    Feature2Label = {"time_hours":"t","lenght_km":"L","speed_kmh":"v (km/h)"}
    for Aggregation in Aggregation2Feature2StrClass2FcmDistr.keys():
        for Feature in Features:
            fig,ax = plt.subplots(1,1,figsize = (12,10))
            legend = []
            StrClass2Fit = {StrClass: {"x":[],"y":[],"x_fit":[],"y_fit":[]} for StrClass in Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature].keys()}
            for StrClass in Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature].keys():
                if NormBool:
                    print(f"Distribution {Feature}: ",Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature][StrClass]["y"][:3])
                    print("maxy: ",max(Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature][StrClass]["y"]))
                    Feature2IntervalCount[Feature] = 0.05
                    Feature2ShiftCount[Feature] = 0.1
                # Scatter Points

                x = Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature][StrClass]["x"][1:]
                y = Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature][StrClass]["y"]
                LabelBestFit = Aggregation2Feature2Class2AllFitTry[Aggregation][Feature][StrClass]["best_fit"]
                if LabelBestFit != "":
                    mean = Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature][StrClass]["mean"]
                    if LabelBestFit == "exponential":
                        x_windowed = Aggregation2Feature2Class2AllFitTry[Aggregation][Feature][StrClass][LabelBestFit]["x_windowed"]/mean
                        x = x/mean
                    else:
                        x_windowed = Aggregation2Feature2Class2AllFitTry[Aggregation][Feature][StrClass][LabelBestFit]["x_windowed"]
                    y_data = Aggregation2Feature2Class2AllFitTry[Aggregation][Feature][StrClass][LabelBestFit]["fitted_data_windowed"]
                ax.scatter(x,y)
                legend.append(StrClass + " " + Feature2Legend[Feature] + " " + str(round(mean,3)))
                # Fit
                if LabelBestFit != "":                
                    if len(x_windowed) == len(y_data):
                        ax.plot(x_windowed,y_data)
                        legend.append(StrClass + " " + Feature2Legend[Feature] + " " + str(round(mean,3)))
                ax.set_xticks(np.arange(x[0],x[-1],Feature2IntervalBin[Feature]))
                ax.set_yticks(np.arange(min(y),max(y),Feature2IntervalCount[Feature]))
                if LabelBestFit == "exponential":
                    ax.set_xlabel(Feature2Label[Feature] + f"/ <{Feature2Label[Feature]}>")
                    ax.set_ylabel("P(" + Feature2Label[Feature] + f"/ <{Feature2Label[Feature]}>" + ")")
                else:
                    ax.set_xlabel(Feature2Label[Feature])
                    ax.set_ylabel("P(" + Feature2Label[Feature] + ")")
                if Feature == "speed_kmh":
                    ax.set_xlim(left = 0,right = 150)
                else:
                    ax.set_xlim(left = 0.1,right = x[-1] + Feature2ShiftBin[Feature])
#                ax.set_ylim(bottom = 0.000001,top =max(y) + Feature2ShiftCount[Feature])
                ax.set_xscale(Feature2ScaleBins[Feature])
                ax.set_yscale(Feature2ScaleCount[Feature])
                StrClass2Fit[StrClass]["x"] = list(x)
                StrClass2Fit[StrClass]["y"] = list(y)
                StrClass2Fit[StrClass]["x_fit"] = list(x_windowed)
                StrClass2Fit[StrClass]["y_fit"] = list(y_data)
#            ax.set_title(f"Best Fit: {LabelBestFit}")
#            legend_ = plt.legend(legend,fontsize = "small")
#            frame = legend_.get_frame()
#            frame.set_facecolor('white')
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            ax_inset = inset_axes(ax, width="20%", height="20%")
            PlotInsetSpeedAverage(Feature,ax_inset,PlotDir)
            Date = os.path.basename(PlotDir)
            fig.savefig(os.path.join(PlotDir,'{0}_{1}_{2}.png'.format(Aggregation,Feature2SaveName[Feature],Date)),dpi = 200)
            plt.close()
            import json
            with open(os.path.join(PlotDir,'{0}_{1}_{2}.json'.format(Aggregation,Feature2SaveName[Feature],Date)),"w") as f:
                json.dump(StrClass2Fit,f)

def PlotInsetSpeedAverage(Feature,ax_inset,PlotDir):
    """
        @describe:
            Plot the average speed of the inset
    """
    df_v_coeff = pl.read_csv(os.path.join(PlotDir,f"df_linear_coeffs_mu_sigma_speed_{Feature}.csv"))
    df_v = pl.read_csv(os.path.join(PlotDir,f"df_mu_sigma_speed_{Feature}.csv"))
    a_v = df_v_coeff["a_<v>"].to_numpy()[0]
    b_v = df_v_coeff["b_<v>"].to_numpy()[0]
    mu_v = df_v["mu_v"].to_numpy()
    sigma_v = df_v["sigma_v"].to_numpy()
    ax_inset.scatter(mu_v,sigma_v,s = 30)
    ax_inset.set_xlabel(r'$\mu (km/h)$',fontsize = 12)
    ax_inset.set_ylabel(r'$\sigma (km/h)$',fontsize = 12)
    ax_inset.plot(mu_v,a_v*np.array(mu_v) + b_v,color = "black",linestyle = "--")

def PlotFeatureAggregatedWithoutFitRescaledByMean(Aggregation2Feature2StrClass2FcmDistr,                   
                                    Aggregation2Feature2Class2AllFitTry,
                                    Feature2Legend,
                                    Feature2IntervalBin,
                                    Feature2IntervalCount,
                                    Feature2Label,
                                    Feature2ShiftBin,
                                    Feature2ShiftCount,
                                    Feature2ScaleBins,
                                    Feature2ScaleCount,
                                    PlotDir,
                                    Feature2SaveName,
                                    NormBool = True):
    """
        Plots aggregated over all days distribution of speed, time and length rescaled by mean.
        NOTE: Do this plot to show that all the curve fall in the same when divided by mean as length and speed are exponentially distributed. 
    """
    
    Features = ["time_hours","lenght_km"]
    Feature2Label = {"time_hours":"t (h)","lenght_km":"L (km)"}
    for Aggregation in Aggregation2Feature2StrClass2FcmDistr.keys():
        for Feature in Features:
            fig,ax = plt.subplots(1,1,figsize = (12,10))
            legend = []

            df_fit_and_data = pl.read_csv(os.path.join(PlotDir,f"df_fit_and_data_expo_{Feature}.csv"))
            df_parameters = pl.read_csv(os.path.join(PlotDir,f"df_parameters_expo_{Feature}.csv"))
            
            for StrClass in Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature].keys():
                if NormBool:
                    print(f"Distribution {Feature}: ",Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature][StrClass]["y"][:3])
                    print("maxy: ",max(Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature][StrClass]["y"]))
                    Feature2IntervalCount[Feature] = 0.05
                    Feature2ShiftCount[Feature] = 0.1
                # Scatter Points
                x = Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature][StrClass]["x"][1:]
                y = Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature][StrClass]["y"]
                x_mean = np.mean(x)
                x = np.array(x)/x_mean
                ax.scatter(x,y)
#                legend.append(StrClass + " " + Feature2Legend[Feature])
                ax.vlines(x_mean,0,max(y),label = None)
#                legend.append(str(round(x_mean,3)))
                ax.set_xticks(np.arange(x[0],x[-1],Feature2IntervalBin[Feature]))
                ax.set_yticks(np.arange(min(y),max(y),Feature2IntervalCount[Feature]))
                ax.set_xlabel(Feature2Label[Feature] + f"/ <{Feature2Label[Feature]}>")
                ax.set_ylabel("P(" + Feature2Label[Feature] + f"/ <{Feature2Label[Feature]}>" + ")")
                ax.set_xlim(left = 0.01,right = x[-1] + Feature2ShiftBin[Feature])
                ax.set_ylim(bottom = 0.000001,top =max(y) + Feature2ShiftCount[Feature])
                ax.set_xscale(Feature2ScaleBins[Feature])
                ax.set_yscale(Feature2ScaleCount[Feature])
#            ax.set_title("Distribution " + Feature2Label[Feature] + " Rescaled by Mean")
            ax.legend()
#            legend_ = plt.legend(legend)
#            frame = legend_.get_frame()
#            frame.set_facecolor('white')
            Date = os.path.basename(PlotDir)
            fig.savefig(os.path.join(PlotDir,'WithoutFit_{0}_{1}_{2}.png'.format(Aggregation,Feature2SaveName[Feature],Date)),dpi = 200)
            plt.close()



"""
def PlotAggregatedAllDaysPerClass(Aggregation2Feature2StrClass2FcmDistr,                   
                                    Aggregation2Feature2Class2InfoOutputFit,
                                    Feature2IntervalBin,
                                    Feature2IntervalCount,
                                    Feature2Label,
                                    Feature2ShiftBin,
                                    Feature2ShiftCount,
                                    Feature2ScaleBins,
                                    Feature2ScaleCount,
                                    PlotDir,
                                    Feature2SaveName,
                                    NormBool = True):
    for Aggregation in Aggregation2Feature2StrClass2FcmDistr.keys():
        for Feature in Aggregation2Feature2StrClass2FcmDistr[Aggregation].keys():
            fig,ax = plt.subplots(1,1,figsize = (12,10))
            legend = []
            for StrClass in Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature].keys():
                if NormBool:
                    print(f"Distribution {Feature}: ",Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature][StrClass]["y"][:3])
                    print("maxy: ",max(Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature][StrClass]["y"]))
                    Feature2IntervalCount[Feature] = 0.05
                    Feature2ShiftCount[Feature] = 0.1
                # Scatter Points
                x = Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature][StrClass]["x"][1:]
                y = Aggregation2Feature2StrClass2FcmDistr[Aggregation][Feature][StrClass]["y"]
                y_data = Aggregation2Feature2Class2InfoOutputFit[Aggregation][Feature][StrClass]["fitted_data"]
                LabelBestFit = Aggregation2Feature2Class2InfoOutputFit[Aggregation][Feature][StrClass]["best_fit"]
                ax.scatter(x,y)
                # Fit
                if len(x) == len(y_data):
                    ax.plot(x,y_data,label = LabelBestFit)
                ax.set_xticks(np.arange(x[0],x[-1],Feature2IntervalBin[Feature]))
                ax.set_yticks(np.arange(min(y),max(y),Feature2IntervalCount[Feature]))
                ax.set_xlabel(Feature2Label[Feature])
                ax.set_ylabel('Count')
                ax.set_xlim(right = x[-1] + Feature2ShiftBin[Feature])
                ax.set_ylim(bottom = 1,top =max(y) + Feature2ShiftCount[Feature])
                ax.set_xscale(Feature2ScaleBins[Feature])
                ax.set_yscale(Feature2ScaleCount[Feature])
                Date = os.path.basename(PlotDir)
                fig.savefig(os.path.join(PlotDir,'{0}_{1}_{2}_{3}.png'.format(Aggregation,Feature2SaveName[Feature],StrClass,Date)),dpi = 200)
                plt.close()
        return fig,ax
"""

##----------------------------------- PLOT SPEED TIME -----------------------------------##

def PlotSpeedEvolutionFromGeoJson(Class2TimeInterval2Speed,Class2TimeInterval2Road2Speed,BinStringHour,PlotDir):
    Class2Speed = defaultdict()
    NewClass2Speed = defaultdict()
    BinHour = [datetime.datetime.strptime(StrTime,"%H:%M:%S") for StrTime in BinStringHour]
    for Class in Class2TimeInterval2Speed.keys():
        fig,ax = plt.subplots(1,1,figsize = (12,10))
        Class2Speed[Class] = []
        NewClass2Speed[Class] = []
        for TimeInterval in Class2TimeInterval2Speed[Class].keys():
            if len(Class2TimeInterval2Speed[Class][TimeInterval])!=0:
                FirstRoad = list(Class2TimeInterval2Speed[Class][TimeInterval].keys())[0]
                Class2Speed[Class].append(Class2TimeInterval2Speed[Class][TimeInterval][FirstRoad])
            else:
                Class2Speed[Class].append(0)
            if len(Class2TimeInterval2Road2Speed[Class][TimeInterval])!=0:                
                FirstRoad = list(Class2TimeInterval2Road2Speed[Class][TimeInterval].keys())[0]
                NewClass2Speed[Class].append(Class2TimeInterval2Road2Speed[Class][TimeInterval][FirstRoad])
            else:
                NewClass2Speed[Class].append(0)
        if len(BinHour) == len(Class2Speed[Class]):
            ax.plot(BinHour,list(Class2Speed[Class]),label = r"$\langle v \rangle_{f}$")
            ax.plot(BinHour,list(NewClass2Speed[Class]),label = r"$\langle v \rangle_{h}$ ")
        else:
            bh = BinHour[1:]
            ax.plot(bh,list(Class2Speed[Class]),label = r"$\langle v \rangle_{f}$")
            ax.plot(bh,list(NewClass2Speed[Class]),label = r"$\langle v \rangle_{h}$ ")
        ax.set_xlabel("Time")
        ax.set_ylabel("Speed")
        ax.set_title("Speed Evolution Sub-Net Class {}".format(Class))
        ax.set_xticks(range(len(BinHour[1:]))[::8])
        ax.set_xticklabels(BinHour[1::8], rotation=90)

        ax.legend()
        plt.savefig(os.path.join(PlotDir,"SpeedEvolutionClass{}.png".format(Class)),dpi = 200)
        plt.close()
    
def SortClass2SpeedBySpeed(Class2Speed):
    """
        @param: Class2Speed: dict -> {Class: [Speed]}
        @brief:
            Sort the Speeds of each Class and reassign classes based on the sorted order
    """
    # Calculate the minimum speed for each class
    ClassMinSpeed = {}
    for Class, Speeds in Class2Speed.items():
        valid_speeds = [Speed for Speed in Speeds if Speed > 0 and not np.isnan(Speed)]
        if valid_speeds:
            ClassMinSpeed[Class] = min(valid_speeds)
        else:
            ClassMinSpeed[Class] = float('inf')  # Assign a high value if no valid speeds

    # Sort classes based on their minimum speed
    SortedClasses = sorted(ClassMinSpeed, key=ClassMinSpeed.get)

    # Reassign classes in Class2SpeedOrdered based on the sorted order
    Class2SpeedOrdered = defaultdict(list)
    for i, Class in enumerate(SortedClasses):
        Class2SpeedOrdered[i] = [Speed for Speed in Class2Speed[Class]]

    return Class2SpeedOrdered

def PrepareSpeedEvolutionNewClassConsideringRoadClassification(ClassNew2TimeInterval2Road2SpeedActualRoads,Class2TimeInterval2Road2SpeedNew,Class2TimeInterval2Road2Speed):
    """
        @param: ClassNew2TimeInterval2Road2SpeedActualRoads: dict -> {Class: {TimeInterval: {Road: Speed}}}
        @param: Class2TimeInterval2Road2SpeedNew: dict -> {Class: {TimeInterval: {Road: Speed}}}
        @param: Class2TimeInterval2Road2Speed: dict -> {Class: {TimeInterval: {Road: Speed}}}
        @brief:
            Prepare the Speed Evolution for the New Class considering
            the Road Classification
        @return: Class2Speed: dict -> {Class: [Speed]}
        @return: Class2SpeedH: dict -> {Class: [Speed]}
        @return: Class2SpeedO: dict -> {Class: [Speed]}
        The values are in length as the timeintervals used are in length
    """
    Classes = list(ClassNew2TimeInterval2Road2SpeedActualRoads.keys())
    TimeIntervals = list(ClassNew2TimeInterval2Road2SpeedActualRoads[Classes[0]].keys())
    if type(TimeIntervals[0]) == int:
        TimeIntervalsDt = [datetime.datetime.fromtimestamp(int(t)).strftime("%Y-%m-%d %H:%M:%S").split(" ")[1] for t in TimeIntervals]
    else:
        TimeIntervalsDt = TimeIntervals
        pass

    Class2Speed = defaultdict()
    Class2SpeedH = defaultdict()
    Class2SpeedO = defaultdict()
    for i,Class in enumerate(list(ClassNew2TimeInterval2Road2SpeedActualRoads.keys())):
        Class2Speed[Class] = []
        Class2SpeedH[Class] = []
        Class2SpeedO[Class] = []
        for t,TimeInterval in enumerate(list(ClassNew2TimeInterval2Road2SpeedActualRoads[Class].keys())):
            if len(list(ClassNew2TimeInterval2Road2SpeedActualRoads[Class][TimeInterval].keys()))!=0:
                FirstRoad = list(ClassNew2TimeInterval2Road2SpeedActualRoads[Class][TimeInterval].keys())[0]
                Class2Speed[Class].append(ClassNew2TimeInterval2Road2SpeedActualRoads[Class][TimeInterval][FirstRoad])
            else:
                Class2Speed[Class].append(0)
            k = CastKeys(list(Class2TimeInterval2Road2SpeedNew.keys())[i])
            if len(list(Class2TimeInterval2Road2SpeedNew[k].keys()))!=0:
                kt = CastKeys(list(Class2TimeInterval2Road2SpeedNew[k].keys())[t])
                if len(list(Class2TimeInterval2Road2SpeedNew[k][kt].keys()))!=0:
                    FirstRoad = list(Class2TimeInterval2Road2SpeedNew[k][kt].keys())[0]
                    FirstRoad = CastKeys(FirstRoad)
                    Class2SpeedH[Class].append(Class2TimeInterval2Road2SpeedNew[k][kt][FirstRoad])
                else:
                    Class2SpeedH[Class].append(0)
            else:
                Class2SpeedH[Class].append(0)
            k1 = CastKeys(list(Class2TimeInterval2Road2Speed.keys())[i])
            if len(list(Class2TimeInterval2Road2Speed[k1].keys()))!=0:
                kt2 = CastKeys(list(Class2TimeInterval2Road2Speed[k1].keys())[t])
                if len(list(Class2TimeInterval2Road2Speed[k1][kt2].keys()))!=0:
                    FirstRoad = list(Class2TimeInterval2Road2Speed[k1][kt].keys())[0]
                    FirstRoad = CastKeys(FirstRoad)
                    Class2SpeedO[Class].append(Class2TimeInterval2Road2Speed[k1][kt2][FirstRoad])
                else:
                    Class2SpeedO[Class].append(0)
            else:
                Class2SpeedO[Class].append(0)    
        #    logger.info(f"Time: {t}, ClassRoads: {Class}, ClassHierarchical: {k}, ClassOriginal: {k1}")
        assert len(Class2Speed[Class]) == len(Class2SpeedH[Class]) == len(Class2SpeedO[Class]), "Lengths Class2Spped: {0}, Class2SpeedH: {1}, Class2SpeedO: {0}".format(len(Class2Speed[Class]),len(Class2SpeedH[Class]),len(Class2SpeedO[Class]))

    Class2Speed = SortClass2SpeedBySpeed(Class2Speed)
    Class2SpeedH = SortClass2SpeedBySpeed(Class2SpeedH)
    Class2SpeedO = SortClass2SpeedBySpeed(Class2SpeedO)
    return Class2Speed,Class2SpeedH,Class2SpeedO,TimeIntervalsDt

def PrepareStdDevSpeedEvolutionNewClassConsideringRoadClassification(ClassNew2TimeInterval2Road2SpeedActualRoads,Class2TimeInterval2Road2SpeedNew,Class2TimeInterval2Road2Speed):
    """
        @param: Class2Speed: dict -> {Class: [Speed]}
        @param: Class2SpeedH: dict -> {Class: [Speed]}
        @param: Class2SpeedO: dict -> {Class: [Speed]}
        @return: Class2StdDevSpeed: dict -> {Class: StdDev Speed}
        @return: Class2StdDevSpeedH: dict -> {Class: StdDev Speed Hierarchical}
        @return: Class2StdDevSpeedO: dict -> {Class: StdDev Speed Original}
    """
    # Number Classes
    Classes = list(ClassNew2TimeInterval2Road2SpeedActualRoads.keys())
    # Number Time Intervals
    TimeIntervals = list(ClassNew2TimeInterval2Road2SpeedActualRoads[Classes[0]].keys())
    if type(TimeIntervals[0]) == int:
        TimeIntervalsDt = [datetime.datetime.fromtimestamp(int(t)).strftime("%Y-%m-%d %H:%M:%S").split(" ")[1] for t in TimeIntervals]
    else:
        TimeIntervalsDt = TimeIntervals
        pass

    Class2StdSpeed = defaultdict()
    Class2StdSpeedH = defaultdict()
    Class2StdSpeedO = defaultdict()
    for i,Class in enumerate(list(ClassNew2TimeInterval2Road2SpeedActualRoads.keys())):
        Class2StdSpeed[Class] = []
        Class2StdSpeedH[Class] = []
        Class2StdSpeedO[Class] = []
        for t,TimeInterval in enumerate(list(ClassNew2TimeInterval2Road2SpeedActualRoads[Class].keys())):
            if len(list(ClassNew2TimeInterval2Road2SpeedActualRoads[Class][TimeInterval].keys()))!=0:
                FirstRoad = list(ClassNew2TimeInterval2Road2SpeedActualRoads[Class][TimeInterval].keys())[0]
                Class2StdSpeed[Class].append(ClassNew2TimeInterval2Road2SpeedActualRoads[Class][TimeInterval][FirstRoad])
            else:
                Class2StdSpeed[Class].append(0)
            k = CastKeys(list(Class2TimeInterval2Road2SpeedNew.keys())[i])
            if len(list(Class2TimeInterval2Road2SpeedNew[k].keys()))!=0:
                kt = CastKeys(list(Class2TimeInterval2Road2SpeedNew[k].keys())[t])
                if len(list(Class2TimeInterval2Road2SpeedNew[k][kt].keys()))!=0:
                    FirstRoad = list(Class2TimeInterval2Road2SpeedNew[k][kt].keys())[0]
                    FirstRoad = CastKeys(FirstRoad)
                    Class2StdSpeedH[Class].append(Class2TimeInterval2Road2SpeedNew[k][kt][FirstRoad])
                else:
                    Class2StdSpeedH[Class].append(0)
            else:
                Class2StdSpeedH[Class].append(0)
            k1 = CastKeys(list(Class2TimeInterval2Road2Speed.keys())[i])
            if len(list(Class2TimeInterval2Road2Speed[k1].keys()))!=0:
                kt2 = CastKeys(list(Class2TimeInterval2Road2Speed[k1].keys())[t])
                if len(list(Class2TimeInterval2Road2Speed[k1][kt2].keys()))!=0:
                    FirstRoad = list(Class2TimeInterval2Road2Speed[k1][kt].keys())[0]
                    FirstRoad = CastKeys(FirstRoad)
                    Class2StdSpeedO[Class].append(Class2TimeInterval2Road2Speed[k1][kt2][FirstRoad])
                else:
                    Class2StdSpeedO[Class].append(0)
            else:
                Class2StdSpeedO[Class].append(0)    
        #    logger.info(f"Time: {t}, ClassRoads: {Class}, ClassHierarchical: {k}, ClassOriginal: {k1}")
        assert len(Class2StdSpeed[Class]) == len(Class2StdSpeedH[Class]) == len(Class2StdSpeedO[Class]), "Lengths Class2Spped: {0}, Class2SpeedH: {1}, Class2SpeedO: {0}".format(len(Class2Speed[Class]),len(Class2SpeedH[Class]),len(Class2SpeedO[Class]))

    Class2StdSpeed = SortClass2SpeedBySpeed(Class2StdSpeed)
    Class2StdSpeedH = SortClass2SpeedBySpeed(Class2StdSpeedH)
    Class2StdSpeedO = SortClass2SpeedBySpeed(Class2StdSpeedO)
    return Class2Speed,Class2SpeedH,Class2SpeedO,TimeIntervalsDt


def PlotSingleSpeedEvolutionNewClassConsideringRoadClassification(Class2Speed,Class2SpeedH,Class2SpeedO,TimeIntervalsDt,Class,ax,StrDate,CutIndexTime,PlotDir):
    """
        @param: Class2Speed: dict -> {Class: [Speed]}
        @param: Class2SpeedH: dict -> {Class: [Speed]}
        @param: Class2SpeedO: dict -> {Class: [Speed]}
        @param: TimeIntervalsDt: list -> Time Intervals
        @param: Class: int -> Class
        @param: PlotDir: str -> Path to Save the Plot
        @brief:
            Plot the speed evolution considering the road classification (If a trajectory at some time t has a point in the road then consider it for the average)
            x-axis: Time
            y-axis: Speed

    """
#    logger.info(f"Class2Speed: {Class2Speed[Class]}, Class2SpeedH: {Class2SpeedH[Class]}, Class2SpeedO: {Class2SpeedO[Class]}")
#    fig,ax = plt.subplots(1,1,figsize = (12,10))        
    ax.scatter(TimeIntervalsDt[CutIndexTime:],list(Class2Speed[Class][CutIndexTime:]),label = r"$\langle v_R \rangle$ ")
    ax.plot(TimeIntervalsDt[CutIndexTime:],list(Class2SpeedH[Class][CutIndexTime:]),label = r"$\langle v_h \rangle$ ")
#    ax.plot(TimeIntervalsDt[CutIndexTime:],list(Class2SpeedO[Class][CutIndexTime:]),label = r"$\langle v_o \rangle$ ")
    ax.set_xlabel("Time")
    ax.set_ylabel("v (km/h)")
#    ax.set_title("Speed Evolution Hierarchical Sub-Net Class {}".format(Class))
    ax.set_xticks(range(len(TimeIntervalsDt[CutIndexTime:]))[::8])  # Set the ticks to correspond to the labels
    ax.set_xticklabels(TimeIntervalsDt[CutIndexTime::8], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
    ax.legend(["Road Classification","Hierarchical","No Hierarchical"])
    plt.savefig(os.path.join(PlotDir,"SpeedEvolutionClass_RoadReClassificaion_{0}_{1}.png".format(Class,StrDate)),dpi = 200)
    plt.close()
    return ax


# Def Plot Number People Per Class and Contribution Per Class 2 Speed

def PrepareSpeedEvolutionTransitionClasses(ClassOld2ClassNewTimeInterval2Road2SpeedNew,Class2TimeInterval2Road2Speed,ClassOld2ClassNewTimeInterval2Transition,OrderedClass2TimeDeparture2UserId):
    """
        @param: ClassOld2ClassNewTimeInterval2Road2SpeedNew: dict -> {ClassOld: {ClassNew: {TimeInterval: {Road: Speed}}}}
        @param: Class2TimeInterval2Road2Speed: dict -> {Class: {TimeInterval: {Road: Speed}}}
        @param: ClassOld2ClassNewTimeInterval2Transition: dict -> {ClassOld: {ClassNew: {TimeInterval: NumberPeople}}}
        @param: OrderedClass2TimeDeparture2UserId: dict -> {Class: {TimeDeparture: UserId}}
        @return: T_new_old_t: np.array -> Transition Matrix
        @return: v_new_old_t: np.array -> Speed of People Transitioning
        @return: v_new_t: np.array -> Speed of People in the New Class
        @return: weighted_v_new_t: np.array -> Weighted Speed of People Transitioning
        @return: N_new_t: np.array -> Number of People Transitioning
        @return: TimeIntervalsDt: list -> Time Intervals
        @return: ClassesOld: list -> Old Classes
        @return: ClassesNew: list -> New Classes
        NOTE: Used as input for PlotSingleSpeedEvolutionTransitionClasses
    """
    NumberClasses = len(list(ClassOld2ClassNewTimeInterval2Road2SpeedNew.keys()))
    ClassesOld = sorted(list(ClassOld2ClassNewTimeInterval2Road2SpeedNew.keys()))
    ClassesNew = sorted(list(ClassOld2ClassNewTimeInterval2Road2SpeedNew[ClassesOld[0]].keys()))
    # Transition Matrix
    TimeIntervals = list(ClassOld2ClassNewTimeInterval2Road2SpeedNew[ClassesOld[0]][ClassesNew[0]].keys())
    if type(TimeIntervals[0]) == int:
        TimeIntervalsDt = [datetime.datetime.fromtimestamp(int(t)).strftime("%Y-%m-%d %H:%M:%S").split(" ")[1] for t in TimeIntervals]
    elif type(TimeIntervals[0]) == datetime.datetime:
        TimeIntervalsDt = [t.replace(minute=0, second=0, microsecond=0) for t in TimeIntervals]
    elif type(TimeIntervals[0]) == str:
        TimeIntervalsDt = [datetime.datetime.fromtimestamp(int(t)).strftime("%Y-%m-%d %H:%M:%S").split(" ")[1] for t in TimeIntervals]
    else:
        pass
    NTimeIntervals = len(TimeIntervals)
    T_new_old_t = np.zeros((NumberClasses,NumberClasses,NTimeIntervals))
    v_new_old_t = np.zeros((NumberClasses,NumberClasses,NTimeIntervals))
    v_new_t = np.zeros((NumberClasses,NTimeIntervals))
    weighted_v_new_t = np.zeros((NumberClasses,NTimeIntervals))
    # Take The Old Partition 
    for i,ClassOld in enumerate(ClassesOld):
        for j,ClassNew in enumerate(ClassesNew):
#            KeysT2R = list(OrderedClass2TimeDeparture2UserId[ClassNew].keys())
            for t in range(len(TimeIntervals)):
                TimeInterval = TimeIntervals[t]
                # Consider The Average Speed On the New Sub Network (Considering Just the speed of Trajectories Newly Classified)
                Key0 = CastKeys(list(Class2TimeInterval2Road2Speed.keys())[i])
                Key1 = list(Class2TimeInterval2Road2Speed[Key0].keys())[t]
                RoadsForSpeed = list(Class2TimeInterval2Road2Speed[Key0][Key1].keys())
                RoadsForNewSpeed = list(ClassOld2ClassNewTimeInterval2Road2SpeedNew[ClassOld][ClassNew][TimeInterval].keys())
                NOld2New = ClassOld2ClassNewTimeInterval2Transition[ClassOld][ClassNew][TimeInterval]
                # Fill the Number of People That at time t were In class i and now are class j
                assert not isinstance(NOld2New,str)
                T_new_old_t[j][i][t] = NOld2New
                if len(RoadsForSpeed)!=0:
                    # Average speed of the people that have transitioned
                    FirstRoad = CastKeys(list(Class2TimeInterval2Road2Speed[Key0][Key1].keys())[0])
                    v_new_t[j][t] = Class2TimeInterval2Road2Speed[Key0][Key1][FirstRoad]
                    assert not isinstance(v_new_t[j][t],str)
                else:
                    v_new_t[j][t] = 0                
                if len(RoadsForNewSpeed) != 0:
                    FirstRoad = CastKeys(RoadsForNewSpeed[0])
                    # Speed On the Sub Network For People Transitioning (<v>_{Old \cap New})
                    v_new_old_t[j][i][t] = ClassOld2ClassNewTimeInterval2Road2SpeedNew[ClassOld][ClassNew][TimeInterval][FirstRoad]
                    assert not isinstance(v_new_old_t[j][i][t],str)
                else:
                    v_new_old_t[j][i][t] = 0
                weighted_v_new_t[j][t] += v_new_old_t[j][i][t]*T_new_old_t[j][i][t]
        # NOTE: If ClassNew and ClassOld where not the same set, this would not be good
        N_new_t = np.nansum(T_new_old_t,axis = 1)
    return T_new_old_t,v_new_old_t,v_new_t,weighted_v_new_t,N_new_t,TimeIntervalsDt,ClassesOld,ClassesNew



def PlotTransitionClassesInTime(ClassOld2ClassNewTimeInterval2Road2Transition,CutIndexTime,PlotDir):
    for ClassOld in ClassOld2ClassNewTimeInterval2Road2Transition.keys():
        for ClassNew in ClassOld2ClassNewTimeInterval2Road2Transition[ClassOld].keys():
            fig,ax = plt.subplots(1,1,figsize = (12,10))
            TimeInvervals = []
            NumberInIntervals = []
            for TimeInterval in ClassOld2ClassNewTimeInterval2Road2Transition[ClassOld][ClassNew].keys():
                TimeInvervals.append(TimeInterval)    
                NumberInIntervals.append(ClassOld2ClassNewTimeInterval2Road2Transition[ClassOld][ClassNew][TimeInterval])
            ax.plot(TimeInterval[CutIndexTime:],NumberInIntervals[CutIndexTime:],label = f"{TimeInterval}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Transition")
            ax.set_title(f"Transition {ClassOld} -> {ClassNew}")
            ax.legend()
            plt.savefig(os.path.join(PlotDir,f"Transition_{ClassOld}_{ClassNew}.png"),dpi = 200)
            plt.close()

def PlotComparisonDistributionSpeedNewOld(Fcm,PlotDir):
    for Class,ClassFcm in Fcm.groupby("class"):
        fig,ax = plt.subplots(1,1,figsize = (12,10))
        n,bins = np.histogram(ClassFcm["speed_kmh"],bins = 100)
        ax.scatter(bins[1:],n,label = "Class {}".format(Class))
        ClassNewFcm = Fcm.filter(pl.col("class_new") == Class)["speed_kmh"]
        n,bins = np.histogram(ClassNewFcm,bins = 100)
        ax.scatter(bins[1:],n,label = "Class Hierarchical {}".format(Class))
        ax.set_xlabel("Speed (km/h)")
        ax.set_ylabel("Count")
        ax.set_title("Speed Distribution Class {}".format(Class))
        ax.legend()
        plt.savefig(os.path.join(PlotDir,"SpeedDistributionClass{}.png".format(Class)),dpi = 200)
        plt.close()


## PLOT FIT
def Plot_distribution_length_time_daily_and_condtioned_to_classes(Days,Features,Classes,Cases,InputDir,Class2ClassStr,Feature2Label,PlotDir):
    """
        @param Days: List of Days in the DataSet
        @param Features: List of Features to be analyzed (length_km,time_hours)
        @param Classes: List of Classes to be analyzed (0,1,2,3)
        @param Cases: List of Cases to be analyzed ("","New")
        @param InputDir: Directory where the data is stored
        @param PlotDir: Directory where the plots will be stored
        @Description: 
            This function compares the exponential and power law fits for the distribution of the features
            It saves the results in csv files:
            1. aggregated_fit_parameters_{Feature}_powerlaw_new.csv
            2. aggregated_fit_parameters_{Feature}_exponential_new.csv
            Each of these files contains the following columns:
                - Day: Day of the dataset
                - A: Amplitude of the fit
                - class: Class of the distribution
            if powerlaw:
                - alpha: Exponent of the power law
            if exponential:
                - <x>: Average value of the feature
                - 1/x0: Decay constant of the exponential
    """
    import polars as pl
    import os
    for Feature in Features:
        for Day in Days:
            fig,ax = plt.subplots(1,1,figsize = (10,10))
            for Case in Cases:
                if Case == "New":
                    post_fix = "_new"
                else:
                    post_fix = ""
                for Class in Classes:
                    if os.path.isfile(os.path.join(InputDir,f"df_fit_and_data_expo_{Feature}_conditional_class{post_fix}.csv")):
                        DfDataFit = pl.read_csv(os.path.join(InputDir,f"df_fit_and_data_expo_{Feature}_conditional_class{post_fix}.csv"))
                        is_exponential = True
                    else:
                        DfDataFit = pl.read_csv(os.path.join(InputDir,f"df_fit_and_data_pl_{Feature}_conditional_class{post_fix}.csv"))
                        is_exponential = False
                    x = DfDataFit.filter(pl.col("Class") == int(Class),
                                         pl.col("Day") == Day)["x"].to_numpy()
                    y = DfDataFit.filter(pl.col("Class") == int(Class),
                                         pl.col("Day") == Day)["y"].to_numpy()
                    y_fit = DfDataFit.filter(pl.col("Class") == int(Class),
                                            pl.col("Day") == Day)["y_fit"].to_numpy()
                    x_over_xmean = x/(np.sum(x)/len(x))
                    y_scaled = y/np.sum(y)
                    y_fit = y_fit/np.sum(y_fit)
                    ax.scatter(x_over_xmean,y_scaled,label = Class2ClassStr[Class])
                    ax.plot(x_over_xmean,y_fit,label ="",linestyle = "--")
            str_ = Feature2Label[Feature]
            fraction_label = r"$\langle $" + str_ + r"$\rangle$"
            ax.set_xlabel(f"{Feature2Label[Feature]}/{fraction_label}")
            ax.set_ylabel(f"P({Feature2Label[Feature]}/{fraction_label})" )
            if Feature == "lenght_km":
                ax.set_ylim(0.0001)
            else:
                ax.set_ylim(0.004)
            ax.set_xlim(0.1,1.8)
            ax.legend(loc = "best")
            ax.set_yscale("log")
            plt.savefig(os.path.join(PlotDir,Day,f"distribution_{Feature}_{Day}{post_fix}.png"))
            plt.close(fig)



def Plot_distribution_length_time_not_conditional_class(Cases,Days,Features,Feature2Label,PlotDir):
    """
        @plot distribution of 
    """
    id_aggregated_col = 10
    for Case in Cases:
        for Feature in Features:
            if Case == "New":
                post_fix = "_new"
            else:
                post_fix = ""
            if os.path.isfile(os.path.join(PlotDir,f"df_fit_and_data_expo_{Feature}_conditional_class{post_fix}.csv")):
                DfDataFit = pl.read_csv(os.path.join(PlotDir,f"df_fit_and_data_expo_{Feature}_conditional_class{post_fix}.csv"))
            else:
                DfDataFit = pl.read_csv(os.path.join(PlotDir,f"df_fit_and_data_pl_{Feature}_conditional_class{post_fix}.csv"))
            fig,ax = plt.subplots(1,1,figsize = (10,10))
            for Day in Days:
                DfDataFitDay = DfDataFit.filter(pl.col("Day") == Day,
                                                pl.col("Class") == id_aggregated_col)              
                x = DfDataFitDay["x"].to_numpy()
                y = DfDataFitDay["y"].to_numpy()
                x_over_xmean = x/(np.sum(x)/len(x))
                y_scaled = y/np.sum(y)
                ax.scatter(x_over_xmean,y_scaled,label = Day)
            ax.plot(x_over_xmean,np.exp(-x_over_xmean),label ="",linestyle = "--")
            ax.set_xlabel(Feature2Label[Feature])
            ax.set_ylabel(f"P({Feature2Label[Feature]})")
            if Feature == "lenght_km":
                ax.set_ylim(0.0001)
            else:
                ax.set_ylim(0.004)
            ax.set_xlim(0.1,1.8)
            ax.legend(loc = "best")
            ax.set_yscale("log")
            plt.savefig(os.path.join(PlotDir,f"distribution_{Feature}_{post_fix}.png"))

### SPARSENESS DATA

def PlotNPeopNRoadsClass(OrderedClass2TimeDeparture2UserId,IntClass2RoadsIncreasinglyIncludedIntersection,case,PlotDir):
    """
        Plot the number of people and number of roads in the class
        @params OrderedClass2TimeDeparture2UserId: dict -> {Class: {TimeDeparture: [UserId]}}
        @params IntClass2RoadsIncreasinglyIncludedIntersection: dict -> {Class: [Roads]}
        @params case: str -> "Hierarchical" or "Not-Hierarchical"
    """
    ks = sorted(list(OrderedClass2TimeDeparture2UserId.keys()))
    fig,ax = plt.subplot_mosaic([[ks[0],ks[1]],[ks[2],ks[3]]],figsize = (20,20))
    for Class in ks:
        TimeIntervals = list(OrderedClass2TimeDeparture2UserId[Class].keys())
        if type(TimeIntervals[0]) == int:
            TimeIntervalsDt = [datetime.datetime.fromtimestamp(int(t)).strftime("%Y-%m-%d %H:%M:%S").split(" ")[1] for t in TimeIntervals]
        else:
            TimeIntervalsDt = TimeIntervals
            pass
        Npeop = []
        for TimeDeparture in OrderedClass2TimeDeparture2UserId[Class].keys():
            Npeop.append(len(OrderedClass2TimeDeparture2UserId[Class][TimeDeparture])/len(IntClass2RoadsIncreasinglyIncludedIntersection[Class]))
        ax[Class].plot(TimeIntervalsDt,Npeop,label = f"Class {Class}")
        ax[Class].hlines(1,TimeIntervalsDt[0],TimeIntervalsDt[-1],label = f"Number of Roads Class {Class}")
        ax[Class].text(TimeIntervalsDt[0],0.9,f"Number of Roads {len(IntClass2RoadsIncreasinglyIncludedIntersection[Class])}")
        ax[Class].set_xlabel("Time")
        ax[Class].set_ylabel("Number people/Number Roads")
        ax[Class].set_xticks(range(len(TimeIntervalsDt))[::8])  # Set the ticks to correspond to the labels
        ax[Class].set_xticklabels(TimeIntervalsDt[::8], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
        ax[Class].legend()
    plt.savefig(os.path.join(PlotDir,f"EvolutionNumberPeople_{case}.png"),dpi = 200)
    plt.close()

def ScatterParametersFit(VectorDay2Class2Parameter,Labels,PlotDir):
    """
        @params VectorDay2Class2Parameter: np.array -> Parameters (Ndays, Nclasses) [Given a day take the class parameter]
        @params Labels: list -> Labels
        @params PlotDir: str -> Plot Directory
    """
    ClassesTicks = ["slow","second slow","second fast","fast"]
    assert len(ClassesTicks) == len(VectorDay2Class2Parameter[0]), "Length ClassesTicks: {0}, Length VectorDay2Class2Parameter: {1}".format(len(ClassesTicks),len(VectorDay2Class2Parameter[0]))
    fig, ax = plt.subplots(figsize=(10, 6))
    for day in range(VectorDay2Class2Parameter.shape[0]):
        ax.scatter(day, VectorDay2Class2Parameter[day], label=Labels[day])
    # Set custom ticks and rotate them 90 degrees
    ax.set_xticks(VectorDay2Class2Parameter.shape[1])
    ax.set_xticklabels(ClassesTicks, rotation=90)

    # Add legend
    ax.legend()

    # Set labels and title
    ax.set_xlabel("Speed Category")
    ax.set_ylabel("Value")
    ax.set_title("Scatter Plot of Vectors")

## --------------------------------- PLOT NETWORKS ---------------------------------- ##
def PlotIncrementSubnetHTML(GeoJson,IntClass2StrClass,centroid,PlotDir,StrDate,ReadFluxesSubIncreasinglyIncludedIntersectionBool,ReadGeojsonBool,Class2Color,BaseNameFile = "SubnetsIncrementalInclusion",verbose = False):
    if ReadFluxesSubIncreasinglyIncludedIntersectionBool and ReadGeojsonBool and ReadGeojsonBool:
        print("Plotting Daily Incremental Subnetworks in HTML")
        if not os.path.isfile(os.path.join(PlotDir,"Subnets_{}.html".format(StrDate))) or True:
            print("Save in: ",os.path.join(PlotDir,"SubnetsIncrementalInclusion_{}.html".format(StrDate)))
            # Create a base map
            m = folium.Map(location=[centroid.x, centroid.y], zoom_start=12)
            # Iterate through the Dictionary of list of poly_lid
            for IntClass in np.unique(GeoJson["IntClassOrdered_{}".format(StrDate)]):
                mclass = folium.Map(location=[centroid.x, centroid.y], zoom_start=12)
                filtered_gdf = GeoJson.groupby("IntClassOrdered_{}".format(StrDate)).get_group(IntClass)
                # Create a feature group for the current layer
                layer_group = folium.FeatureGroup(name="Layer {}".format(IntClass)).add_to(m)
                layer_group_class = folium.FeatureGroup(name="Layer {}".format(IntClass)).add_to(mclass)
                # Add roads to the feature group with a unique color                
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
                folium.TileLayer("CartoDB positron", show=False).add_to(m)

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
            for IntClass in np.unique(GeoJson["IntClassOrdered_{}".format(StrDate)]):
                mclass = folium.Map(location=[centroid.x, centroid.y], zoom_start=12)
#                    for index_list in self.IntClass2Roads[IntClass]:
#                if isinstance(index_list,int):
#                    index_list = [index_list]
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


# -------------------------- SPECIFIC ALL DAYS ----------------------------#

def ComputeDay2PopulationTime(ListDailyNetwork,Classes):
    """
        Description:
            Compute for each day the vector of 96 elements of population over time.
    """
    Day2PopulationTime = {MobDate.StrDate: {"population":[],"time":[]} for MobDate in ListDailyNetwork}
    LocalDayCount = 0
    for MobDate in ListDailyNetwork:
        StrDate = MobDate.StrDate
        MFDAggregated = MobDate.MFD
        if isinstance(MFDAggregated,pl.DataFrame):
            MFDAggregated = MFDAggregated.to_pandas()
        else:
            pass
        Day2PopulationTime[StrDate]["time"] = list(np.zeros(len(MobDate.MFD["time"])))
        Day2PopulationTime[StrDate]["population"] = list(np.zeros(len(MobDate.MFD["time"])))
        for t in range(len(MobDate.MFD["time"])-1):
            Day2PopulationTime[StrDate]["time"][t] = MobDate.MFD["time"][t]
            for Class in Classes:
                Day2PopulationTime[StrDate]["population"][t] += MobDate.MFD[f"population_{Class}"][t]
    return Day2PopulationTime

def PlotDay2PopulationTime(Day2PopulationTime,IndexTime,PlotDir):
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    for StrDate in Day2PopulationTime.keys():
        ax.plot(Day2PopulationTime[StrDate]["time"][IndexTime:],Day2PopulationTime[StrDate]["population"][IndexTime:],label = StrDate)
    time_labels =  [str(t) for t in Day2PopulationTime[StrDate]["time"]]
    ax.set_xticks(range(len(time_labels))[IndexTime::8])
    ax.set_xticklabels(time_labels[IndexTime::8], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of paths")
    ax.legend()
    plt.savefig(os.path.join(PlotDir,"AllDaysPopulationOverTime.png"))
    plt.close()
    return fig,ax    


def PlotIntervals(Avareges, std_devs, classes, types,Class2Type2Colors,Class2Type2Shapes,Title,Xlabel,PlotDir,SaveName):
    """
        Input:
            Averages: List of average values
            std_devs: List of standard deviations
            classes: List of classes
            types: List of types
            NOTE: They could be 3 different columns of a dataframe (they must hold the same length)
    
    """
    assert len(Avareges) == len(std_devs) == len(classes) == len(types), 'The input lists must have the same length'
    legend = []
    legend_entries = set()
    fig, ax = plt.subplots()
    for point, std_dev, cls,type in zip(Avareges, std_devs, classes,types):
        color = Class2Type2Colors[cls][str(type)]
        shape = Class2Type2Shapes[cls][str(type)]
        # Plot point
        ax.plot(point, 0, shape, color=color)
        class_type_identifier = (cls, str(type))
        if class_type_identifier not in legend_entries:
            legend.append(Line2D([0], [0], marker=shape, color='w', markerfacecolor=color, markersize=10, label=f"Class {cls}, Type {type}"))
            legend_entries.add(class_type_identifier)
            TemporaryAddLegend = True

        # Plot interval
        ax.plot([point - std_dev, point + std_dev], [0, 0], color=color, marker='_', markersize=20, alpha = 0.3)
    # Customize the plot
    ax.set_yticks([])  # Hide y-axis
    ax.set_xlabel(Xlabel)
    plt.title(Title)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Resize plot to make space for the legend
    print(legend)
    ax.legend(handles = legend,loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend outside the plot
    plt.savefig(os.path.join(PlotDir,SaveName + ".png"))
    plt.close()

def ScatterFitParams(A, b, classes, types,Class2Type2Colors,Class2Type2Shapes,Title,Xlabel,Ylabel,PlotDir,SaveName):
    assert len(A) == len(b) == len(classes), 'The input lists must have the same length'
    legend = []
    legend_entries = set()
    fig, ax = plt.subplots()
    for point, std_dev, cls,type in zip(A, b, classes,types):
        color = Class2Type2Colors[cls][str(type)]
        shape = Class2Type2Shapes[cls][str(type)]
        # Plot point
        class_type_identifier = (cls, str(type))
        print("A: ",round(point,2)," b: ",round(std_dev,2)," Class:",cls," type: ",type," Color: ",color," Shape: ",shape," Identifier: ",class_type_identifier)
        ax.scatter(point, std_dev, marker=shape, color=color)
        if class_type_identifier not in legend_entries:
            legend.append(Line2D([0], [0], marker=shape, color='w', markerfacecolor=color, markersize=10, label=f"Class {cls}, Type {type}"))
            legend_entries.add(class_type_identifier)
            TemporaryAddLegend = True
    # Customize the plot
    ax.set_yticks([])  # Hide y-axis
    ax.set_xlabel(Xlabel)
    ax.set_ylabel(Ylabel)
#    legend_ = plt.legend(legend)
#    frame = legend_.get_frame()
#    frame.set_facecolor('white')    
    plt.title(Title)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Resize plot to make space for the legend
    print(legend)
    ax.legend(handles = legend,loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend outside the plot
    plt.savefig(os.path.join(PlotDir,SaveName + ".png"))
    plt.close()


# UNION ALL DAYS SUBNETWORK
def PlotIntersection(GpdClasses,UniqueClasses,StrClasses2Color,StrIntersection = "Intersection_"):
    First = True
    for Class in UniqueClasses:
        filtered_gdf = GpdClasses.loc[GpdClasses[StrIntersection + Class] == True].dropna(subset=['geometry'])
        if len(filtered_gdf) == 0:
            continue
        else:        
            if not First:
                print("class: ",Class," Color: ",StrClasses2Color[Class]," Column: ",StrIntersection + Class)
                print("Number of roads to Color: ",len([True for i in GpdClasses[StrIntersection + Class] if i if i == True]))
                
                filtered_gdf.explore(column = StrIntersection + Class,
                                    color = StrClasses2Color[Class],
                                    categories = [True,False],
                                    legend = True,
                                    legend_kwds = {'loc': 'upper right'},
                                    figsize = (8,5),
                                    tooltip = "poly_lid",
                                    tooltip_kwds = dict(labels= False),
                                    name = "Intersection All Days " + Class,
                                    m = m)
            else:
                print("First: Class ",Class," Color: ",StrClasses2Color[Class]," Column: ",StrIntersection + Class)
                print("Number of roads to Color: ",len([True for i in GpdClasses[StrIntersection + Class] if i if i == True]))
                m = filtered_gdf.explore(column = "Intersection_" + Class,
                                    color = StrClasses2Color[Class],
                                    categories = [True,False],
                                    legend = True,
                                    legend_kwds = {'loc': 'upper right'},
                                    figsize = (8,5),
                                    tooltip = "poly_lid",
                                    tooltip_kwds = dict(labels= False),
                                    name = "Intersection All Days " + Class
                                    )
                folium.TileLayer("CartoDB positron", show=False).add_to(m)
                First = False
    folium.LayerControl().add_to(m)        
    return m

        
def PlotUnion(GpdClasses,UniqueClasses,StrClasses2Color,StrUnion = "Union_"):
    """
        @param GpdClasses: GeoDataFrame with the classes column informat OrderedUnion_{Class}
        @param UniqueClasses: List of Unique Classes
        @param StrClasses2Color: Dictionary with the classes and the colors
        @param StrUnion: String to identify the Union Column
        @return m: folium map with the roads colored by class
    """
    First = True
    # For each Class
    for Class in UniqueClasses:
        print(StrUnion + Class)
        # Select Roads That Belong to the Class
        filtered_gdf = GpdClasses.loc[GpdClasses[StrUnion + Class] == True].dropna(subset=['geometry'])
        # If There are no Roads in the Class
        if len(filtered_gdf) == 0:
            continue
        else:
            if not First:
                print("class: ",Class," Color: ",StrClasses2Color[Class])
                print("Number of roads to Color: ",len([True for i in GpdClasses[StrUnion + Class] if i if i == True]))
                # Plot Roads
                filtered_gdf.explore(column = StrUnion + Class,
                                    color = StrClasses2Color[Class],
                                    categories = [True,False],
                                    legend = True,
                                    legend_kwds = {'loc': 'upper right'},
                                    figsize = (8,5),
                                    tooltip = "poly_lid",
                                    tooltip_kwds = dict(labels= False),
                                    name = "Union All Days " + Class,
                                    m = m)
            else:
                print("First: Class ",Class," Color: ",StrClasses2Color[Class])
                print("Number of roads to Color: ",len([True for i in GpdClasses[StrUnion + Class] if i if i == True]))
                m = filtered_gdf.explore(column = StrUnion + Class,
                                    color = StrClasses2Color[Class],
                                    categories = [True,False],
                                    legend = True,
                                    legend_kwds = {'loc': 'upper right'},
                                    figsize = (8,5),
                                    tooltip = "poly_lid",
                                    tooltip_kwds = dict(labels= False),
                                    name = "Union All Days " + Class
                                    )
                folium.TileLayer("CartoDB positron", show=False).add_to(m)
                First = False
    folium.LayerControl().add_to(m)        
    return m

def PlotUnionPlotply(GpdClasses,UniqueClasses,StrUnion = "OrderedUnion_"):
    """
        @param GpdClasses: GeoDataFrame with the classes column informat OrderedUnion_{Class}
        @param UniqueClasses: List of Unique Classes
    """
    from plotly import express as px
    import shapely.geometry as sg
    for Class in UniqueClasses:
        # Select Roads That Belong to the Class
        filtered_gdf = GpdClasses.loc[GpdClasses[StrUnion + Class] == True].dropna(subset=['geometry'])
        # If There are no Roads in the Class
        if len(filtered_gdf) == 0:
            continue
        else:
            lats = []
            lons = []
            names = []    
            for feature, name in zip(filtered_gdf.geometry, filtered_gdf.poly_lid):
                if isinstance(feature, sg.linestring.LineString):
                    linestrings = [feature]
                elif isinstance(feature, sg.multilinestring.MultiLineString):
                    linestrings = feature.geoms
                else:
                    continue
                for linestring in linestrings:
                    x, y = linestring.xy
                    lats = np.append(lats, y)
                    lons = np.append(lons, x)
                    names = np.append(names, [name]*len(y))
                    lats = np.append(lats, None)
                    lons = np.append(lons, None)
                    names = np.append(names, None)

            fig = px.line_geo(lat=lats, lon=lons, hover_name=names)
            fig.show()    


def PlotTruncatedInset(Day2TruncatedFit,ax_inset,Colors):
    """
        @param Day2TruncatedFit: Dictionary with the truncated fit {Day: gamma,delta}
        @param ax_inset: axis to plot the inset

    """
    Gammas = []
    Deltas = []
    Days = list(Day2TruncatedFit.keys())
    for i,Day in enumerate(Days):
        Gammas.append(Day2TruncatedFit[Day]["gamma"])
        Deltas.append(Day2TruncatedFit[Day]["delta"])
        ax_inset.scatter(Day2TruncatedFit[Day]["gamma"],Day2TruncatedFit[Day]["delta"],Colors[i])
    ax_inset.set_xlabel(r"\gamma",fontsize = 12)
    ax_inset.set_ylabel(r"\delta",fontsize = 12)
    return ax_inset
    
def PlotTrafficIForEachDay(Classes,Days,PlotDir):
    """Plot the comparison of the highest two peaks."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    import json
    Class2AvgPeak = {Class:{"LinCoef":[], "AvPeak":[] }for Class in Classes}
    for Class in Classes:
        DfLinCoefClass = pl.read_csv(os.path.join(PlotDir,f"LinearCoeff_NewClass.csv"))
        DfLinCoefClass = DfLinCoefClass.filter(pl.col("Class")==int(Class))
        fig,ax = plt.subplots(1,1,figsize = (12,12))
        DfLinCoefClassDay = DfLinCoefClass["LinearCoeff"].to_numpy()#.filter(pl.col("Days")==Day)
        DaysNew = DfLinCoefClass["Days"].unique().to_numpy()
        DaysNew, DfLinCoefClassDay = sort_according_to_order(Days, DaysNew, DfLinCoefClassDay)
        i = 0
        for Day in Days:
            with open(os.path.join(PlotDir,"TrafficIndexAllDays.json"), "r") as file:
                TrafficIndexAllDays = json.load(file)
            TrafficIndexDayClass = TrafficIndexAllDays[Day + "_" + Class]
            if Class == "0" or Class == 0:
                y = gaussian_filter1d(TrafficIndexDayClass, sigma=3)
            else:
                y = gaussian_filter1d(TrafficIndexDayClass, sigma=3)            
            peaks, _ = find_peaks(y)
            x = np.arange(len(y))
            ax.plot(x, y, label=Day)
            ax.plot(x[peaks], y[peaks], 'ro')
            plt.title(f"Class {Class}")
            plt.legend()
            if len(peaks) >= 2:
                mean = np.sort(y[peaks])
                Class2AvgPeak[Class]["AvPeak"].append(np.mean(mean[-2:]))
            else:
                Class2AvgPeak[Class]["AvPeak"].append(0)
            Class2AvgPeak[Class]["LinCoef"].append(DfLinCoefClassDay[i])
    #        print(Class,"Linear Coefficient: ",Class2AvgPeak[Class]["LinCoef"],"\nAverage Peak: ",Class2AvgPeak[Class]["AvPeak"])
            i += 1
        plt.show()
        plt.close()
    marker_style = {0: "o", 1: "s", 2: "^", 3: "v"}
    Colors = ["blue","orange","green","red"]
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    for Class in Classes:
        ax.scatter(Days,Class2AvgPeak[Class]["AvPeak"],label="Class " +Class,color =Colors[int(Class)],marker = marker_style[int(Class)],s = 100)
        ax.hlines(np.mean(Class2AvgPeak[Class]["AvPeak"]),0,len(Days),colors=Colors[int(Class)],linestyles='dashed')
        ax.set_xlabel("Day")
        ax.set_ylabel(r"$\langle \Gamma_k \rangle_{peaks}$",fontsize = 18)
        ax.legend(fontsize = 10)
        ax.set_xticklabels(Days,rotation = 90)  # Set the ticks to correspond to the labels
    plt.savefig(os.path.join(PlotDir,"AveragePeak.png"))
    plt.show()
    plt.close()

def sort_according_to_order(fixed_order, values_list, associated_list):
    # Pair the elements of the values_list with the associated_list
    paired_list = list(zip(values_list, associated_list))
    
    # Create a dictionary to map values to their positions in the fixed order
    order_dict = {value: index for index, value in enumerate(fixed_order)}
    
    # Sort the paired list according to the fixed order
    sorted_paired_list = sorted(paired_list, key=lambda x: order_dict[x[0]])
    
    # Unzip the sorted pairs back into separate lists
    sorted_values_list, sorted_associated_list = zip(*sorted_paired_list)
    
    return list(sorted_values_list), list(sorted_associated_list)


def PlotVideoTrafficFromGeoJsonWithSpeedColumn(gdf,case,PlotDir):
    """
        @param gdf: GeoDataFrame with the speed column 
            - new_class_Speed
            - class
        
    """
    import contextily as ctx
    if case == "New":
        timestamps = []
        Columns2Plot = []
        for col in gdf.columns:
            if 'new_class_Speed' in col:
                Day = col.split("_")[3]
                TimeInterval = int(col.split("_")[4])
                timestamps.append(TimeInterval)
                Columns2Plot.append(col)
    elif case == "Fuzzy":
        timestamps = []
        Columns2Plot = []
        for col in gdf.columns:
            if 'class_Speed' in col and "new" not in col:
                Day = col.split("_")[3]
                TimeInterval = int(col.split("_")[4])
                timestamps.append(TimeInterval)
                Columns2Plot.append(col)
    else:
        raise ValueError("Case not recognized")
    timestamps = sorted(timestamps)
    Time = [datetime.datetime.fromtimestamp(int(t)).strftime("%Y-%m-%d %H:%M:%S").split(" ")[1] for t in timestamps]
    cmap = "viridis"
    y_label = r" \langle v \rangle km/h"
    ColorMap = plt.Normalize(vmin=0, vmax=100)
    gdf = gdf.to_crs(epsg=3857)
    Class2AvSpeed = {}

    for Class, gdfClass in gdf.groupby(f"IntClassOrdered_{Day}"):
        AvSpeed = []
        for t in range(len(timestamps)):         
            AvSpeed.append(np.mean(gdfClass[Columns2Plot[t]]))
        Class2AvSpeed[Class] = AvSpeed

    for t in range(len(timestamps)): 
        if t >0:   
    #            fig, ax = plt.subplots(1,1,figsize=(10,5))
            fig, (ax_curve,ax_geo) = plt.subplots(1,2,width_ratios=[1,4],figsize = (20,20))#,gridspec_kw={'height_ratios': [1, 2]}
            for Class, gdfClass in gdf.groupby(f"IntClassOrdered_{Day}"):
                AvSpeed = Class2AvSpeed[Class]
                # Global Variables
                ax_curve.plot(Time[:t], AvSpeed[:t], linewidth=1)
            ax_curve.set_xticks(range(len(Time))[::8])
            ax_curve.set_xticklabels(Time[::8], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
            ax_curve.set(xlabel='Time', ylabel=y_label)
            ax_curve.set_aspect(aspect= 0.5)
            ax_curve.set_position([0.05, 0, 0.2, 0.8])  # [left, bottom, width, height]
            ax_curve.set_xlim([0, len(Time)])
            ax_curve.set_ylim([0, 140])
            # Base Geo-Map
            linewidth = 0.1
            linewidth_traffic = 2           
            gdf.plot(Columns2Plot[t], ax=ax_geo,cmap = cmap,norm=ColorMap,linewidth=linewidth_traffic, alpha=1)
            ctx.add_basemap(ax_geo, crs =gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik,alpha = 0.5)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=ColorMap)
            sm._A = []  # Dummy array for the ScalarMappable
            cbar = fig.colorbar(sm, ax=ax_geo,orientation='horizontal', shrink=1, pad=0.05)
            cbar.set_label(y_label)    
            ax_geo.set_title(f"{Day} {Time[t]}")
            ax_geo.set_aspect(aspect= 1.5)
            if t < 10:
                strt = "00"+str(t)
            elif t < 100:
                strt = "0"+str(t)
            else:
                strt = str(t)    
            os.makedirs(os.path.join(PlotDir,"Video","Aggregated"),exist_ok = True)
            plt.savefig(os.path.join(PlotDir,"Video","Aggregated",f"Speed_{Day}_{strt}.png"),dpi = 200)
            plt.close()
    CreateVideoFromImages(os.path.join(PlotDir,"Video","Aggregated"),f"Speed_{Day}")
    DeleteImages(os.path.join(PlotDir,"Video","Aggregated"))

def CreateVideoFromImages(DirImagesAndVideo,PrefixFile):
    """
    Launches the ffmpeg command to create a video from a sequence of images.
    NOTE: The images must be named as Density_000.png, Density_001.png, etc.
    """
    import subprocess
    ImagesString = os.path.join(DirImagesAndVideo,"{}_%03d.png".format(PrefixFile))
    VideoString = os.path.join(DirImagesAndVideo,"{}.mp4".format(PrefixFile))
    command = [
        'ffmpeg',
        '-f', 'image2',
        '-framerate', '1.5',
        '-start_number', '0',
        '-i', ImagesString,
        '-vf', 'pad=width=iw:height=ih+ih/1.5:color=black',
        '-frames:v', '281',
        '-vcodec', 'libx264',
        VideoString
    ]
    
    try:
        subprocess.run(command, check=True)
        print("Video created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while creating the video: {e}")

def DeleteImages(DirImagesAndVideo):
    """
    Deletes the images created for the video.
    """
    for file in os.listdir(DirImagesAndVideo):
        if file.endswith(".png"):
            os.remove(os.path.join(DirImagesAndVideo, file))
    print("Images deleted successfully.")



###############################################################################################################################
#################################################### DEPRECATED ###############################################################
###############################################################################################################################

def PlotCFAR(Class2Signal,Class2Cut,Class2CFARClassification,TimeIntervalsDt,CutIndexTime,PlotDir):
    """
        @param: Class2Signal: dict -> {Class: [Signal]}
        @param: Class2Cut: dict -> {Class: [Cut]}
        @param: Class2CFARClassification: dict -> {Class: [CFARClassification]}
        @param: TimeIntervalsDt: list -> Time Intervals
        @param: PlotDir: str -> Path to Save the Plot
        @brief:
            Plot the CFAR Classification
            x-axis: Time
            y-axis: Signal
    """
    fig,axs = plt.subplots(2,2,figsize = (20,20)) 
    for Class in Class2Signal.keys():
        if int(Class) == 0:
            i = 0
            j = 0
        elif int(Class) == 1:
            i = 0
            j = 1
        elif int(Class) == 2:
            i = 1
            j = 0
        else:
            i = 1
            j = 1
        
        axs[i,j].scatter(TimeIntervalsDt,Class2Signal[Class],label = "Signal")
#        axs[i,j].plot(TimeIntervalsDt,Class2Cut[Class],label = "Cut")
        Class2CFARClassification[Class] = [0 if x < 0 else x for x in Class2CFARClassification[Class]]
        axs[i,j].plot(TimeIntervalsDt,Class2CFARClassification[Class],label = "CFAR Classification")
        axs[i,j].set_xlabel("Time")
        axs[i,j].set_ylabel("Signal")
        axs[i,j].set_title("CFAR Classification Class {}".format(Class))
        axs[i,j].set_xticks(range(len(TimeIntervalsDt[CutIndexTime:]))[::8])
        axs[i,j].set_xticklabels(TimeIntervalsDt[CutIndexTime::8], rotation=90)
        axs[i,j].legend()
#    plt.savefig(os.path.join(PlotDir,"CFARClassificationClass.png"),dpi = 200)
    plt.close()

def PlotPtest(Class2Ptest,TimeIntervalsDt,CutIndexTime,PlotDir):
    """
        @param: Class2Ptest: dict -> {Class: [Ptest]}
        @param: TimeIntervalsDt: list -> Time Intervals
        @param: PlotDir: str -> Path to Save the Plot
        @brief:
            Plot the Ptest
            x-axis: Time
            y-axis: Ptest
    """
    fig,axs = plt.subplots(2,2,figsize = (12,10)) 
    for Class in Class2Ptest.keys():
        if int(Class) == 0: 
            i = 0
            j = 0
        elif int(Class) == 1:
            i = 0
            j = 1
        elif int(Class) == 2:
            i = 1
            j = 0
        else:
            i = 1
            j = 1
        axs[i,j].scatter(TimeIntervalsDt,Class2Ptest[Class],label = "Ptest")
        axs[i,j].set_xlabel("Time")
        axs[i,j].set_ylabel("is traffic")
        axs[i,j].set_title("Ptest Class {}".format(Class))
        axs[i,j].set_xticks(range(len(TimeIntervalsDt[CutIndexTime:]))[::8])
        axs[i,j].set_xticklabels(TimeIntervalsDt[CutIndexTime::8], rotation=90)
    plt.savefig(os.path.join(PlotDir,"PtestClass.png"),dpi = 200)
    plt.close()
    #PlotPtestSingleClass(Class2Ptest,TimeIntervalsDt,CutIndexTime,PlotDir)

def PlotPtestSingleClass(Class2Ptest,TimeIntervalsDt,CutIndexTime,PlotDir):
    """
        @param: Class2Ptest: dict -> {Class: [Ptest]}
        @param: TimeIntervalsDt: list -> Time Intervals
        @param: PlotDir: str -> Path to Save the Plot
        @brief:
            Plot the Ptest
            x-axis: Time
            y-axis: Ptest
        NOTE: Useless -> Does not work
    """
    for Class in Class2Ptest.keys():
        fig,ax = plt.subplots(1,1,figsize = (12,10)) 
        ax.scatter(TimeIntervalsDt,Class2Ptest[Class],label = "Ptest")
        ax.set_xlabel("Time")
        ax.set_ylabel("is traffic")
        ax.set_title("Ptest Class {}".format(Class))
        ax.set_xticks(range(len(TimeIntervalsDt[CutIndexTime:]))[::8])
        ax.set_xticklabels(TimeIntervalsDt[CutIndexTime::8], rotation=90)
        plt.savefig(os.path.join(PlotDir,f"Ptest_{Class}.png"),dpi = 200)
        plt.close()



def PlotSingleSpeedEvolutionTransitionClasses(weighted_v_new_t,N_new_t,T_new_old_t,v_new_old_t,TimeIntervalsDt,i,ClassesOld,ClassNew,ax00,ax01,PlotDir):
    """
        @params weighted_v_new_t: np.array -> Weighted Speed of People Transitioning
        @params N_new_t: np.array -> Number of People Transitioning
        @params T_new_old_t: np.array -> Transition Matrix
        @params v_new_old_t: np.array -> Speed of People Transitioning
        @params TimeIntervalsDt: list -> Time Intervals
        @params i: int -> Index of the Class
        @params ClassesOld: list -> Old Classes
        @params ClassNew: str -> New Class
        @params PlotDir: str -> Plot Directory
        @return: ax00: plt -> Plot of Number of People Transitioning
        @return: ax01: plt -> Plot of Speed of People Transitioning
        NOTE: Used as input for PlotComparisonSpeedClassesAndNPeopleTogether
    """    
    colors = ["red","green","yellow","black","orange","purple","pink","brown","grey"]    
    weighted_v_new_t[i] = weighted_v_new_t[i]/N_new_t[i] 
    # Average Speed of People Transitioning
#    fig00,ax00 = plt.subplots(1,1,figsize = (12,10))
#    fig01,ax01 = plt.subplots(1,1,figsize = (12,10))
    ax00.plot(TimeIntervalsDt,N_new_t[i],color = "blue")
    ax01.plot(TimeIntervalsDt,weighted_v_new_t[i],color = "blue")
    for j,ClassOld in enumerate(ClassesOld):
        ax00.scatter(TimeIntervalsDt,T_new_old_t[i][j],color = colors[j],label = f"{ClassOld} -> {ClassNew}")
        ax01.scatter(TimeIntervalsDt,v_new_old_t[i][j],color = colors[j],label = f"{ClassOld} -> {ClassNew}")
    ax01.set_xlabel("Time")
    ax01.set_ylabel(r"$\langle v_h \rangle$")
    ax01.set_title(f"v(t) in Hierarchical Sub-Net {ClassNew}")
    ax01.set_xticks(range(len(TimeIntervalsDt))[::8])  # Set the ticks to correspond to the labels
    ax01.set_xticklabels(TimeIntervalsDt[::8], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
    ax01.legend(loc='upper left', bbox_to_anchor=(0.5, 1))
    ax00.set_xlabel("Time")
    ax00.set_ylabel("Number People")
    ax00.set_xticks(range(len(TimeIntervalsDt))[::8])  # Set the ticks to correspond to the labels
    ax00.set_xticklabels(TimeIntervalsDt[::8], rotation=90)  # Set the labels with rotation    ax.set_title("Time Percorrence Distribution")
    ax00.set_title(f"N(t) in Hierarchical Sub-Net {ClassNew}")
    ax00.legend(loc='upper left', bbox_to_anchor=(0.5, 1))
#    plt.savefig(os.path.join(PlotDir,f"SpeedEvolution_Class_{ClassNew}.png"),dpi = 200)
#    plt.savefig(os.path.join(PlotDir,f"Nt_Class_{ClassNew}.png"),dpi = 200)            
    return ax00,ax01


def PlotComparisonSpeedClassesAndNPeopleTogether(Class2TimeInterval2Road2Speed,Class2TimeInterval2Road2SpeedNew,ClassNew2TimeInterval2Road2SpeedActualRoads,ClassOld2ClassNewTimeInterval2Road2SpeedNew,ClassOld2ClassNewTimeInterval2Transition,OrderedClass2TimeDeparture2UserId,CutIndexTime,PlotDir):
    """
        @params Class2TimeInterval2Road2Speed: dict -> {Class: {TimeInterval: {Road: Speed}}}
        @params Class2TimeInterval2Road2SpeedNew: dict -> {Class: {TimeInterval: {Road: Speed}}}
        @params ClassNew2TimeInterval2Road2SpeedActualRoads: dict -> {Class: {TimeInterval: {Road: Speed}}}
        @params ClassOld2ClassNewTimeInterval2Road2SpeedNew: dict -> {ClassOld: {ClassNew: {TimeInterval: {Road: Speed}}}}
        @params ClassOld2ClassNewTimeInterval2Transition: dict -> {ClassOld: {ClassNew: {TimeInterval: NumberPeople}}}
        @params OrderedClass2TimeDeparture2UserId: dict -> {Class: {TimeDeparture: UserId}}
        @params CutIndexTime: int -> Cut Index Time
        @params PlotDir: str -> Plot Directory
        @brief:
            Plot the Speed Evolution Transition Classes
            x-axis: Time
            y-axis: Speed

    """
    T_new_old_t,v_new_old_t,_,weighted_v_new_t,N_new_t,TimeIntervalsDt,ClassesOld,ClassesNew = PrepareSpeedEvolutionTransitionClasses(ClassOld2ClassNewTimeInterval2Road2SpeedNew,Class2TimeInterval2Road2Speed,ClassOld2ClassNewTimeInterval2Transition,OrderedClass2TimeDeparture2UserId) 
    Class2Speed,Class2SpeedH,Class2SpeedO,TimeIntervalsDt = PrepareSpeedEvolutionNewClassConsideringRoadClassification(ClassNew2TimeInterval2Road2SpeedActualRoads,Class2TimeInterval2Road2SpeedNew,Class2TimeInterval2Road2Speed)    
    for Class in Class2Speed.keys():
        print(f"Class: {Class}, len Speed: {len(Class2Speed[Class])}, len SpeedH: {len(Class2SpeedH[Class])}, len SpeedO: {len(Class2SpeedO[Class])}")
    logger.info("PlotSpeedEvolutionTransitionClasses")
    StrDate = PlotDir.split("/")[-1]
    ClassesNew = list(Class2Speed.keys())
    for i,ClassNew in enumerate(ClassesNew):
        fig, axd = plt.subplot_mosaic([['people', 'division_class_speed'],
                                    ['three_associations','three_associations']],
                                    figsize=(20,20),constrained_layout=True)   

        axd["people"],axd["division_class_speed"] = PlotSingleSpeedEvolutionTransitionClasses(weighted_v_new_t,N_new_t,T_new_old_t,v_new_old_t,TimeIntervalsDt,i,ClassesOld,ClassNew,axd["people"],axd["division_class_speed"],PlotDir)        
        axd["three_associations"] = PlotSingleSpeedEvolutionNewClassConsideringRoadClassification(Class2Speed,Class2SpeedH,Class2SpeedO,TimeIntervalsDt,ClassNew,axd["three_associations"],StrDate,CutIndexTime,PlotDir)        
        plt.savefig(os.path.join(PlotDir,f"SpeedEvolutionComparison_{ClassNew}.png"),dpi = 200)
        plt.close(fig)
        fig,ax = plt.subplots(1,1,figsize = (12,10))
        PlotSingleSpeedEvolutionNewClassConsideringRoadClassification(Class2Speed,Class2SpeedH,Class2SpeedO,TimeIntervalsDt,ClassNew,ax,StrDate,CutIndexTime,PlotDir)
