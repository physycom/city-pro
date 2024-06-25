import datetime
import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
def Dict2PolarsDF(Dict,schema):
    return pl.DataFrame(Dict,schema=schema)

# MFD RELATED FUNCTIONS
def ComputeMFDVariables(Df,MFD,TimeStampDate,dt,iterations,verbose = False):
    """
        NOTE: The bins in time that have 0 trajectories have 0 average speed
        NOTE: Speed in MFD in km/h
    """
    print("Compute MFD Variables:")
    TmpDict = {"time":[],"population":[],"speed_kmh":[],"av_speed":[]}
    for t in range(int(iterations)):
        StartInterval = datetime.datetime.fromtimestamp(int(TimeStampDate)+t*dt)
        EndInterval = datetime.datetime.fromtimestamp(int(TimeStampDate)+(t+1)*dt)                    
        TmpDf = Df.with_columns(pl.col('start_time').apply(lambda x: datetime.datetime.fromtimestamp(x), return_dtype=pl.Datetime).alias("start_time_datetime"),
                                    pl.col('end_time').apply(lambda x: datetime.datetime.fromtimestamp(x), return_dtype=pl.Datetime).alias("end_time_datetime"))

        TmpFcm = TmpDf.filter(pl.col('start_time_datetime').is_between(StartInterval,EndInterval))
        Hstr = StartInterval.strftime("%Y-%m-%d %H:%M:%S").split(" ")[1]
        TmpDict["time"].append(Hstr)
        TmpDict["population"].append(len(TmpFcm))

        if len(TmpFcm) > 0:
            AvSpeed = TmpFcm.select(pl.col("speed_kmh").mean()).to_pandas().iloc[0]["speed_kmh"]
            TmpDict["speed_kmh"].append(AvSpeed)
            AvSpeed = TmpFcm.select(pl.col("av_speed").mean()).to_pandas().iloc[0]["av_speed"]
            TmpDict["av_speed"].append(AvSpeed)
            MoreThan0Traj = True
        else:
            TmpDict["speed_kmh"].append(0)
            TmpDict["av_speed"].append(0)
            MoreThan0Traj = False
#        if verbose:
#            print("Iteration: ",t)
#            print("Considered Hour: ",Hstr)
#            print("Population: ",len(TmpFcm))
#            print("Size dict: ",len(TmpDict["time"]))
#            if MoreThan0Traj:
#                print("Speed: ",AvSpeed)
#    if verbose:
#        print("Dict: ",TmpDict)
    MFD = Dict2PolarsDF(TmpDict,schema = {"time":pl.datatypes.Utf8,"population":pl.Int64,"speed_kmh":pl.Float64,"av_speed":pl.Float64})
    return MFD,Df

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

def GetMFDForPlot(MFD,MFD2Plot,MinMaxPlot,Class,case,verbose = False,bins_ = 15):
    """
        Input:
            MFD: {"population":[],"time":[],"speed_kmh":[]} or {Class:pl.DataFrame{"population":[],"time":[],"speed_kmh":[]}}
        NOTE: Used in self.PlotMFD()
        NOTE: Modifies MDF2Plot = {"bins_population":[p0,..,p19],"binned_av_speed":[v0,..,v19],"binned_sqrt_err_speed":[e0,..,e19]}
        NOTE: Modifies MinMaxPlot = {"speed_kmh":{"min":v0,"max":v19},"population":{"min":p0,"max":p19}}    
    """
    assert "population" in MFD.columns, "population not in MFD"
    assert "speed_kmh" in MFD.columns, "speed not in MFD"
#    assert "bins_population" in MFD2Plot.columns, "bins_population not in MFD2Plot"
#    assert "binned_av_speed" in MFD2Plot.columns, "binned_av_speed not in MFD2Plot"
#    assert "binned_sqrt_err_speed" in MFD2Plot.columns, "binned_sqrt_err_speed not in MFD2Plot"
    print("Get MFD For Plot: {}".format(Class))
    n, bins = np.histogram(MFD["population"],bins = bins_)
    labels = range(len(bins) - 1)
    for i in range(len(labels)):
        # Fill Average/Std Speed (to plot)
        BinnedAvSpeed = GetAverageConditional(MFD,"population","speed_kmh",bins[i],bins[i+1])
        MFD2Plot['binned_av_speed'].append(BinnedAvSpeed)
        BinnedSqrtSpeed = GetStdErrorConditional(MFD,"population","speed_kmh",bins[i],bins[i+1])
        MFD2Plot['binned_sqrt_err_speed'].append(BinnedSqrtSpeed)
#        if verbose:
#            print("Bin [",bins[i],',',bins[i+1],']')
#            print("Av Speed: ",BinnedAvSpeed)
#            print("SqrtError: ",BinnedSqrtSpeed)
    MFD2Plot["bins_population"] = bins
    if verbose:
        print("MFD Features Aggregated: ")
#        print("Bins Population:\n",MFD2Plot['bins_population'])
#        print("\nBins Average Speed:\n",MFD2Plot['binned_av_speed'])
#        print("\nBins Standard Deviation:\n",MFD2Plot['binned_sqrt_err_speed'])
    MinMaxPlot = GetLowerBoundsFromBins(bins = bins,label = "population",MinMaxPlot = MinMaxPlot,Class = Class,case = case)
    MinMaxPlot = GetLowerBoundsFromBins(bins = MFD2Plot['binned_av_speed'],label = "speed_kmh",MinMaxPlot = MinMaxPlot, Class = Class,case = case)
    Y_Interval = max(MFD2Plot['binned_av_speed']) - min(MFD2Plot['binned_av_speed'])
    RelativeChange = Y_Interval/max(MFD2Plot['binned_av_speed'])/100
    if verbose:
#        print("\nMinMaxPlot:\n",MinMaxPlot)
        print("\nInterval Error: ",Y_Interval)            
    return MFD2Plot,MinMaxPlot,RelativeChange

def SaveMFDPlot(binsPop,binsAvSpeed,binsSqrt,RelativeChange,SaveDir,Title = "Fondamental Diagram Aggregated",NameFile = "MFD.png"):
    """
        
    """
#    assert "bins_population" in MFD2Plot.columns, "bins_population not in MFD2Plot"
#    assert "binned_av_speed" in MFD2Plot.columns, "binned_av_speed not in MFD2Plot"
#    assert "binned_sqrt_err_speed" in MFD2Plot.columns, "binned_sqrt_err_speed not in MFD2Plot"
    print("Plotting MFD:\n")
    fig, ax = plt.subplots(1,1,figsize = (10,8))
    text = "Relative change : {}%".format(round(RelativeChange,2))
    ax.plot(binsPop[1:],binsAvSpeed)
    ax.fill_between(np.array(binsPop[1:]),
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

def PlotHysteresis(MFD,Title,SaveDir,NameFile):
    x = MFD['population'].to_list()
    y = MFD['speed_kmh'].to_list()
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

# END MFD RELATED FUNCTIONS
