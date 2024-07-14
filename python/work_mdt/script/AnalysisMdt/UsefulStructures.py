import numpy as np
import pandas as pd
import os
from collections import defaultdict
import polars as pl


# FIT INITIAL GUESS (No Class/ Class)
def InitFeature2Function2Fit2InitialGuess(Features2Fit):
    Feature2Function2Fit2InitialGuess = {Feature:None for Feature in Features2Fit}
    for Feature in Features2Fit:
        if Feature == "av_speed" or Feature == "speed_kmh":
            Feature2Function2Fit2InitialGuess[Feature] = {"maxwellian":{"initial_guess":[0,0],"interval":[]},
             "gaussian":{"initial_guess":[0,0],"interval":[]}
            }
        else:
            Feature2Function2Fit2InitialGuess[Feature] = {"exponential":{"initial_guess":[0,0],"interval":[]},
             "powerlaw":{"initial_guess":[0,0],"interval":[]}
            }
    return Feature2Function2Fit2InitialGuess

def InitFeature2Class2Function2Fit2InitialGuess(Features2Fit,IntClass2StrClass):
    Feature2Class2Function2Fit2InitialGuess = {Feature:None for Feature in Features2Fit}
    for Feature in Features2Fit:
        Feature2Class2Function2Fit2InitialGuess[Feature] = {IntClass:None for IntClass in IntClass2StrClass.keys()}
        for IntClass in IntClass2StrClass:
            if Feature == "av_speed" or Feature == "speed_kmh":
                Feature2Class2Function2Fit2InitialGuess[Feature][IntClass] = {"maxwellian":{"initial_guess":[0,0],"interval":[]},
                 "gaussian":{"initial_guess":[0,0],"interval":[]}
                }
            else:
                Feature2Class2Function2Fit2InitialGuess[Feature][IntClass] = {"exponential":{"initial_guess":[0,0],"interval":[]},
                 "powerlaw":{"initial_guess":[0,0],"interval":[]}
                }
    return Feature2Class2Function2Fit2InitialGuess

# FIt OUTPUT (Already Selected) (No Class/ Class) ReturnInfoFit -> Gives this back as output
def InitFeature2InfoOutputFit(Features2Fit):
    Feature2InfoOutputFit = {Feature:{"best_fit":[], "fitted_data":[],"parameters":[],"start_window":None,"end_window":None,"std_error":None} for Feature in Features2Fit}
    return Feature2InfoOutputFit
def InitFeature2Class2InfoOutputFit(Features2Fit,IntClass2StrClass):
    Feature2Class2InfoOutputFit = {Feature: {IntClass: {"best_fit":[], "fitted_data":[],"parameters":[],"start_window":None,"end_window":None,"std_error":None} for IntClass in IntClass2StrClass.keys()} for Feature in list(Features2Fit)}
    return Feature2Class2InfoOutputFit

def InitFeature2AllFitTry(Feature2Function2Fit2InitialGuess):
    Feature2AllFitTry = {Feature:defaultdict() for Feature in Feature2Function2Fit2InitialGuess.keys()}
    for Feature in Feature2Function2Fit2InitialGuess.keys():
        for Function2Fit in Feature2Function2Fit2InitialGuess[Feature].keys():            
            Feature2AllFitTry[Feature][Function2Fit] = {"fitted_data":[],"parameters":[],"start_window":None,"end_window":None,"std_error":None,"success": False}
    return Feature2AllFitTry

def InitFeature2Class2AllFitTry(Feature2Class2Function2Fit2InitialGuess):
    Feature2Class2AllFitTry = {Feature:defaultdict() for Feature in Feature2Class2Function2Fit2InitialGuess.keys()}
    for Feature in Feature2Class2Function2Fit2InitialGuess.keys():
        Feature2Class2AllFitTry[Feature] = {IntClass:defaultdict() for IntClass in Feature2Class2Function2Fit2InitialGuess[Feature].keys()}
        for IntClass in Feature2Class2Function2Fit2InitialGuess[Feature].keys():
            for Function2Fit in Feature2Class2Function2Fit2InitialGuess[Feature][IntClass].keys():            
                Feature2Class2AllFitTry[Feature][IntClass][Function2Fit] = {"fitted_data":[],"parameters":[],"start_window":None,"end_window":None,"std_error":None,"success": False}
    return Feature2Class2AllFitTry

def GenerateDay2DictClassAvSpeed(ListDailyNetwork,ReferenceFcmCenters,DictClass2AvSpeed,RefIntClass2StrClass,Day2IntClass2StrClass,Day2StrClass2IntClass):
    for MobDate in ListDailyNetwork:
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
            DictClass2AvSpeed[MobDate.StrDate][class_] = IntClassBestMatch
        # {StrDay: {IntClassStrDay: IntClassRefDay}}            
        for Class in DictClass2AvSpeed[MobDate.StrDate].keys():
            # {StrDay: {IntClassStrDay: StrClassRefDay}}
            Day2IntClass2StrClass[MobDate.StrDate][Class] = RefIntClass2StrClass[DictClass2AvSpeed[MobDate.StrDate][Class]]
            Day2StrClass2IntClass[MobDate.StrDate][RefIntClass2StrClass[DictClass2AvSpeed[MobDate.StrDate][Class]]] = Class
    return Day2IntClass2StrClass,Day2StrClass2IntClass,DictClass2AvSpeed

def SplitFcmByClass(Fcm,Feature,IntClass2StrClass,NormBool = True):
    Class2FcmDistr = defaultdict(dict)
    IntClass2Feat2AvgVar = defaultdict(dict)
#    IntClass2ResultFit = defaultdict(dict)
    for IntClass in IntClass2StrClass:
        y,x = np.histogram(Fcm.filter(pl.col("class") == IntClass)[Feature].to_list(),bins = 50)
        if NormBool:
            y = y/np.sum(y)
        Class2FcmDistr[IntClass] = {"x":x,"y":y,"maxx":max(x),"maxy":max(y),"minx":min(x),"miny":min(y),"mean":np.mean(Fcm.filter(pl.col("class") == IntClass)[Feature].to_list())}
        IntClass2Feat2AvgVar[IntClass] = {"avg":np.mean(Fcm.filter(pl.col("class") == IntClass)[Feature].to_list()),"var":np.var(Fcm.filter(pl.col("class") == IntClass)[Feature].to_list())}
    print("Class2FcmDistr Feature: ",Feature)
    for IntClass in IntClass2StrClass:
        print("Class: ",IntClass," mean: ",Class2FcmDistr[IntClass]["mean"])
    return Class2FcmDistr,IntClass2Feat2AvgVar#,IntClass2ResultFit



def FitDataFrame(Class2Feature2FittedData,Feature2Class2FcmDistr,PlotDir):
    """
        Save Df With x,y,fitted_data, for each class
    """
    Feature2FitDf = {Feature:None for Feature in Feature2Class2FcmDistr.keys()}
    for Feature in Feature2Class2FcmDistr.keys():
        Cols = []
        ColsVal = []
        for IntClass in Class2Feature2FittedData.keys():
            Cols.append("x_{}_{}".format(Feature,IntClass))
            Cols.append("y_{}_{}".format(Feature,IntClass))
            Cols.append("fitted_data_{}_{}".format(Feature,IntClass))
            ColsVal.append(Feature2Class2FcmDistr[Feature][IntClass]["x"][1:])
            ColsVal.append(Feature2Class2FcmDistr[Feature][IntClass]["y"])
            ColsVal.append(Class2Feature2FittedData[IntClass][Feature]["fitted_data"])
        ColsVal = list(map(list, zip(*ColsVal)))
        Df = pd.DataFrame(ColsVal,columns = Cols)
        Feature2FitDf[Feature] = Df
        Df.to_csv(os.path.join(PlotDir,f"DistributionAllClassesSeparated_{Feature}.csv"))
    return Feature2FitDf