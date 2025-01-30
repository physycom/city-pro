import numpy as np
import pandas as pd
import os
from collections import defaultdict
import polars as pl
import json
import logging
logger = logging.getLogger(__name__)
VERBOSE = False
def CreateIntClass2StrClass(FcmCenters,IntClass2StrClass,StrClass2IntClass):
    """
        @brief: Create the Dictionary that maps the Integer Class to the String Class
        @param FcmCenters: DataFrame [class,av_speed,v_min,v_max,sinuosity]
        @param IntClass2StrClass: {IntClass: StrClass}
    """
    logger.info("Create IntClass2StrClass")
    number_classes = len(FcmCenters["class"]) 
    for i in range(number_classes):
        if FcmCenters.filter(pl.col("class") == i)["av_speed"].to_list()[0] > 130:
            pass
        else:
            if i<number_classes/2:
                IntClass2StrClass[list(FcmCenters["class"])[i]] = '{} slowest'.format(i+1)
                StrClass2IntClass['{} slowest'.format(i+1)] = list(FcmCenters["class"])[i]
            elif i == number_classes/2:
                IntClass2StrClass[list(FcmCenters["class"])[i]] = 'middle velocity class'
                StrClass2IntClass['middle velocity class'] = list(FcmCenters["class"])[i]
            else:
                IntClass2StrClass[list(FcmCenters["class"])[i]] = '{} quickest'.format(number_classes - i)             
                StrClass2IntClass['{} quickest'.format(number_classes - i)] = list(FcmCenters["class"])[i]
    BoolStrClass2IntClass = True
    return IntClass2StrClass,StrClass2IntClass,BoolStrClass2IntClass

# FIT INITIAL GUESS (No Class/ Class)
def InitFeature2Function2Fit2InitialGuess(Features2Fit):
    Feature2Function2Fit2InitialGuess = {Feature:None for Feature in Features2Fit}
    
    for Feature in Features2Fit:
        if Feature == "av_speed" or Feature == "speed_kmh":
            Feature2Function2Fit2InitialGuess[Feature] = {"maxwellian":{"initial_guess":[1,0,0],"interval":[]},
             "gaussian":{"initial_guess":[1,0,0],"interval":[]}
            }
        else:
            Feature2Function2Fit2InitialGuess[Feature] = {"exponential":{"initial_guess":[1,-1],"interval":[]},
             "powerlaw":{"initial_guess":[1,-1],"interval":[]},"truncated_powerlaw":{"initial_guess":[1,-1,2],"interval":[]}
            }
    if VERBOSE:
        print("Feature2Function2Fit2InitialGuess: ")
        for Feature in Feature2Function2Fit2InitialGuess.keys():
            for Function2Fit in Feature2Function2Fit2InitialGuess[Feature].keys():
                print("Feature: ",Feature," Function2Fit: ",Function2Fit," InitialGuess: ",Feature2Function2Fit2InitialGuess[Feature][Function2Fit]["initial_guess"]," Interval: ",Feature2Function2Fit2InitialGuess[Feature][Function2Fit]["interval"])
    return Feature2Function2Fit2InitialGuess

def InitFeature2Class2Function2Fit2InitialGuess(Features2Fit,IntClass2StrClass,Day):
    Day2ClassExpoTime = {
        "2022-12-30": [1.331, 0.776, 0.903, 0.441],
        "2022-12-31": [1.403, 0.799, 0.918, 0.465],
        "2023-01-01": [1.447, 0.781, 0.944, 0.434],
        "2022-05-12": [0.99, 0.574, 0.676, 0.383],
        "2022-11-11": [1.221, 0.7, 0.881, 0.78],
        "2022-07-01": [1.061, 0.619, 0.731, 0.344],
        "2022-08-05": [1.006, 0.558, 0.643, 0.366],
        "2022-01-31": [1.032, 0.606, 0.714, 0.411]
    }    
    Day2ClassExpoLength = {
    "2022-12-30": [2.488, 5.267, 15.828, 37.529],
    "2022-12-31": [2.131, 4.449, 12.066, 36.24],
    "2023-01-01": [2.096, 4.99, 15.637, 38.368],
    "2022-05-12": [1.915, 3.989, 10.696, 25.939],
    "2022-11-11": [2.707, 6.866, 21.964, 44.086],
    "2022-07-01": [2.257, 4.906, 14.289, 24.714],
    "2022-08-05": [2.142, 4.236, 11.797, 27.937],
    "2022-01-31": [1.838, 10.656, 4.003, 28.045]    
    }
    Feature2Class2Function2Fit2InitialGuess = {Feature:None for Feature in Features2Fit}
    for Feature in Features2Fit:
        Feature2Class2Function2Fit2InitialGuess[Feature] = {IntClass:None for IntClass in IntClass2StrClass.keys()}
        for IntClass in IntClass2StrClass:
            if Feature == "av_speed" or Feature == "speed_kmh":
                Feature2Class2Function2Fit2InitialGuess[Feature][IntClass] = {"maxwellian":{"initial_guess":[1,0,0],"interval":[]},
                 "gaussian":{"initial_guess":[1,0,0],"interval":[]}
                }
            elif Feature == "time_hour":
                Feature2Class2Function2Fit2InitialGuess[Feature][IntClass] = {"exponential":{"initial_guess":[1,-Day2ClassExpoTime[Day][IntClass]],"interval":[]},
                 "powerlaw":{"initial_guess":[1,-1],"interval":[]},
                 "truncated_powerlaw":{"initial_guess":[1,-1,2],"interval":[]}
                }
            elif Feature == "length_km":
                Feature2Class2Function2Fit2InitialGuess[Feature][IntClass] = {"exponential":{"initial_guess":[1,-Day2ClassExpoLength[Day][IntClass]],"interval:":[]},
                 "powerlaw":{"initial_guess":[0,0],"interval":[]},
                 "truncated_powerlaw":{"initial_guess":[1,-1,2],"interval":[]}
                }
            else:
                Feature2Class2Function2Fit2InitialGuess[Feature][IntClass] = {"exponential":{"initial_guess":[1,-1],"interval":[]},
                 "powerlaw":{"initial_guess":[1,-1],"interval":[]},
                 "truncated_powerlaw":{"initial_guess":[1,-1,2],"interval":[]}
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

def AggregatedFcmDistr(Aggregation2Class2Fcm,Aggregation,Feature,Aggregated2Class2FcmDistr,NormBool = True):
    for StrClass in Aggregation2Class2Fcm[Aggregation]:
        y,x = np.histogram(Aggregation2Class2Fcm[Aggregation][StrClass][Feature].to_list(),bins = 50)
        if NormBool:
            y = y/np.sum(y)
        Aggregated2Class2FcmDistr[Aggregation][Feature][StrClass] = {"x":x,
                                                            "y":y,
                                                            "maxx":max(x),
                                                            "maxy":max(y),
                                                            "minx":min(x),
                                                            "miny":min(y),
                                                            "mean":np.mean(Aggregation2Class2Fcm[Aggregation][StrClass][Feature].to_list())}
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
                AvFeat2Class2Day[Feature][StrClass] = {StrDay:[]}
    return AvFeat2Class2Day

def GetClass2Type2ShapesAndColors(Classes,Types):
    """
        Return:
            {Class: {Type: Shape}}
            {Class: {Type: Color}}
    """
    PossibleShapes = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'P', '*', 'X']
    PossibleColors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black']
    Class2Type2Colors = {Class: {Type:"" for Type in Types} for Class in Classes}
    Class2Type2Shapes = {Class: {Type:"" for Type in Types} for Class in Classes}
    count = 0
    for cl in range(len(Classes)):
        Class = Classes[cl]
        for t in range(len(Types)):
            Type = Types[t]
            Class2Type2Shapes[Class][Type] = PossibleShapes[count]
            Class2Type2Colors[Class][Type] = PossibleColors[count]
            count += 1
    return Class2Type2Shapes,Class2Type2Colors




