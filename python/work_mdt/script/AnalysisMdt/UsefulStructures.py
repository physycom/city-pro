import numpy as np
import pandas as pd
import os
from collections import defaultdict
import polars as pl
import json

VERBOSE = False
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
    if VERBOSE:
        print("Feature2Function2Fit2InitialGuess: ")
        for Feature in Feature2Function2Fit2InitialGuess.keys():
            for Function2Fit in Feature2Function2Fit2InitialGuess[Feature].keys():
                print("Feature: ",Feature," Function2Fit: ",Function2Fit," InitialGuess: ",Feature2Function2Fit2InitialGuess[Feature][Function2Fit]["initial_guess"]," Interval: ",Feature2Function2Fit2InitialGuess[Feature][Function2Fit]["interval"])
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
    if VERBOSE:
        print("Feature2Class2Function2Fit2InitialGuess: ")
        for Feature in Feature2Class2Function2Fit2InitialGuess.keys():
            for IntClass in Feature2Class2Function2Fit2InitialGuess[Feature].keys():
                for Function2Fit in Feature2Class2Function2Fit2InitialGuess[Feature][IntClass].keys():
                    print("Feature: ",Feature," Class: ",IntClass," Function2Fit: ",Function2Fit," InitialGuess: ",Feature2Class2Function2Fit2InitialGuess[Feature][IntClass][Function2Fit]["initial_guess"]," Interval: ",Feature2Class2Function2Fit2InitialGuess[Feature][IntClass][Function2Fit]["interval"])
    return Feature2Class2Function2Fit2InitialGuess

# FIt OUTPUT (Already Selected) (No Class/ Class) ReturnInfoFit -> Gives this back as output
def InitFeature2AllFitTry(Feature2Function2Fit2InitialGuess):
    Feature2AllFitTry = {Feature:defaultdict() for Feature in Feature2Function2Fit2InitialGuess.keys()}
    for Feature in Feature2Function2Fit2InitialGuess.keys():
        for Function2Fit in Feature2Function2Fit2InitialGuess[Feature].keys():            
            Feature2AllFitTry[Feature][Function2Fit] = {"x_windowed":[],"y_windowed":[],"fitted_data_windowed":[],"parameters":[],"start_window":None,"end_window":None,"std_error":None,"success": False}
        Feature2AllFitTry[Feature]["best_fit"] = ""
    if VERBOSE:
        print("Feature2AllFitTry: ")
        for Feature in Feature2AllFitTry.keys():
            for Function2Fit in Feature2AllFitTry[Feature].keys():
                print("Feature: ",Feature," Function2Fit: ",Function2Fit," AllFitTry: ",Feature2AllFitTry[Feature][Function2Fit])
    return Feature2AllFitTry

def InitFeature2Class2AllFitTry(Feature2Class2Function2Fit2InitialGuess):
    Feature2Class2AllFitTry = {Feature:defaultdict() for Feature in Feature2Class2Function2Fit2InitialGuess.keys()}
    for Feature in Feature2Class2Function2Fit2InitialGuess.keys():
        Feature2Class2AllFitTry[Feature] = {IntClass:defaultdict() for IntClass in Feature2Class2Function2Fit2InitialGuess[Feature].keys()}
        for IntClass in Feature2Class2Function2Fit2InitialGuess[Feature].keys():
            for Function2Fit in Feature2Class2Function2Fit2InitialGuess[Feature][IntClass].keys():            
                Feature2Class2AllFitTry[Feature][IntClass][Function2Fit] = {"x_windowed":[],"y_windowed":[],"fitted_data_windowed":[],"parameters":[],"start_window":None,"end_window":None,"std_error":None,"success": False}
            Feature2Class2AllFitTry[Feature][IntClass]["best_fit"] = ""
    if VERBOSE:
        print("Feature2Class2AllFitTry: ")
        for Feature in Feature2Class2AllFitTry.keys():
            for IntClass in Feature2Class2AllFitTry[Feature].keys():
                for Function2Fit in Feature2Class2AllFitTry[Feature][IntClass].keys():
                    print("Feature: ",Feature," Class: ",IntClass," Function2Fit: ",Function2Fit," AllFitTry: ",Feature2Class2AllFitTry[Feature][IntClass][Function2Fit])
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
        IntClass2Feat2AvgVar[IntClass] = {"avg":np.mean(Fcm.filter(pl.col("class") == IntClass)[Feature].to_list()),"var":np.std(Fcm.filter(pl.col("class") == IntClass)[Feature].to_list())/np.sqrt(len(Fcm.filter(pl.col("class") == IntClass)[Feature].to_list()))}
    if VERBOSE:
        print("Class2FcmDistr Feature: ",Feature)
        for IntClass in IntClass2StrClass:
            print("Class: ",IntClass," mean: ",Class2FcmDistr[IntClass]["mean"]," x: ",Class2FcmDistr[IntClass]["x"][:3]," y: ",Class2FcmDistr[IntClass]["y"][:3])

    return Class2FcmDistr,IntClass2Feat2AvgVar#,IntClass2ResultFit

def AggregatedFcmDistr(Aggregation2Class2Fcm,Feature,NormBool = True):
    Aggregated2Class2FcmDistr = defaultdict(dict)
    for Aggregation in Aggregation2Class2Fcm:
        Aggregated2Class2FcmDistr[Aggregation] = defaultdict(dict)
        for StrClass in Aggregation2Class2Fcm[Aggregation]:
            y,x = np.histogram(Aggregated2Class2FcmDistr[Aggregation][StrClass][Feature].to_list(),bins = 50)
            if NormBool:
                y = y/np.sum(y)
            Aggregated2Class2FcmDistr[Aggregation][StrClass] = {"x":x,"y":y,"maxx":max(x),"maxy":max(y),"minx":min(x),"miny":min(y),"mean":np.mean(Aggregated2Class2FcmDistr[Aggregation][StrClass][Feature].to_list())}
    if VERBOSE:
        print("Aggregated2Class2FcmDistr Feature: ",Feature)
        for Aggregation in Aggregated2Class2FcmDistr:
            for StrClass in Aggregated2Class2FcmDistr[Aggregation]:
                print("Aggregation: ",Aggregation," Class: ",StrClass," mean: ",Aggregated2Class2FcmDistr[Aggregation][StrClass]["mean"])
    return Aggregated2Class2FcmDistr

def FitDataFrame(Feature2Class2AllTryFit,PlotDir):
    """
        Save Df With x,y,fitted_data, for each class
    """
    Feature2FitDf = {Feature:None for Feature in Feature2Class2AllTryFit.keys()}
    for Feature in Feature2Class2AllTryFit.keys():
        for IntClass in Feature2Class2AllTryFit[Feature].keys():
            Cols = []
            ColsVal = []
            BestFitLabel = Feature2Class2AllTryFit[Feature][IntClass]["best_fit"]
            if BestFitLabel == "":
                continue
            else:
                Cols.append("x_{}_{}".format(Feature,IntClass))
                Cols.append("y_{}_{}".format(Feature,IntClass))
                Cols.append("fitted_data_{}_{}".format(Feature,IntClass))
                ColsVal.append(Feature2Class2AllTryFit[Feature][IntClass][BestFitLabel]["x_windowed"][1:])
                ColsVal.append(Feature2Class2AllTryFit[Feature][IntClass][BestFitLabel]["y_windowed"])
                ColsVal.append(Feature2Class2AllTryFit[Feature][IntClass][BestFitLabel]["fitted_data_windowed"])
                ColsVal = list(map(list, zip(*ColsVal)))
                Df = pd.DataFrame(ColsVal,columns = Cols)
                Feature2FitDf[Feature] = Df
                if not os.path.exists(os.path.join(PlotDir,"Fit")):
                    os.makedirs(os.path.join(PlotDir,"Fit"))
                Df.to_csv(os.path.join(PlotDir,"Fit",f"Distribution_{IntClass}_{Feature}.csv"))
    return Feature2FitDf


def SaveMapsDayInt2Str(Day2IntClass2StrClass,Day2StrClass2IntClass,DictClass2AvSpeed,PlotDir):
    with open(os.path.join(PlotDir,"Day2IntClass2StrClass.json"),"w") as f:
        json.dump(Day2IntClass2StrClass,f,indent=2)
    with open(os.path.join(PlotDir,"Day2StrClass2IntClass.json"),"w") as f:
        json.dump(Day2StrClass2IntClass,f,indent=2)
    with open(os.path.join(PlotDir,"DictClass2AvSpeed.json"),"w") as f:
        json.dump(DictClass2AvSpeed,f,indent=2)

def InitAvFeat2Class2Day(StrDates,Day2StrClass2IntClass,Feature2Label):
    AvFeat2Class2Day = defaultdict(dict)
    for Feature in Feature2Label.keys():
        AvFeat2Class2Day[Feature] = defaultdict(dict)
        for StrDay in StrDates:
            for StrClass in Day2StrClass2IntClass[StrDay].keys():
                AvFeat2Class2Day[Feature][StrDay] = {StrClass:[]}
    return AvFeat2Class2Day