import json
import numpy as np
import polars as pl
from FittingProcedures import *
from JsonFunctions import *



# PRINT
def PrintMFDDictInfo(MFD,StartingString = "Class 2 MFD: "):
    print(StartingString)
    print(MFD)

# FIT AND DICT

def FillInitGuessIntervalPlExp(DictInitGuessInterval,MaxCount,Avg,StartWindow,EndWindow,Function2Test):
    """
        Input:
            - DictInitGuessInterval: dict -> Dictionary with initial_guess and interval
                NOTE: Usage In Program:
                    1) Class2InitialGuess[IntClass][Function2Test][Feature]
                    2) DictInitialGuess[Function2Test][Feature]
            - MaxCount: int -> Maximum Count of the Class
            - Avg: float -> Average of the Class
            - StartWindow: int -> Start Window in Seconds
            - EndWindow: int -> End Window in Seconds
                NOTE: Usage in Program:
                    For (time,lenght):
                        1) LocalDict2Params[StrClass]["MaxCount"]
                        2) LocalDict2Params[StrClass]["Avg"]
                        3) LocalDict2Params[StrClass]["StartWindowS"]
                        4) LocalDict2Params[StrClass]["EndWindowS"]
                    For (time_hours,lenght_km):
                        1) LocalDict2Params[StrClass]["MaxCount"]
                        2) LocalDict2Params[StrClass]["Avg"]/(SecondsInHour-MetersInKm)
                        3) LocalDict2Params[StrClass]["StartWindowS"]/(SecondsInHour-MetersInKm)
                        4) LocalDict2Params[StrClass]["EndWindowS"]/(SecondsInHour-MetersInKm)
            - Function2Test: str -> Function to Test (exponential,powerlaw)

            LocalDict2Params: dict -> Dictionary with parameters for the class

    """
    if Function2Test == "exponential":
        DictInitGuessInterval["initial_guess"] = [MaxCount,Avg]
        DictInitGuessInterval["interval"] = [StartWindow,EndWindow]        
    elif Function2Test == "powerlaw":
        DictInitGuessInterval["initial_guess"] = [MaxCount*StartWindow,-1]
        DictInitGuessInterval["interval"] = [StartWindow,EndWindow]    
    return DictInitGuessInterval

def FillInitGuessIntervalMxGs(DictInitGuessInterval,Fcm,Feature,IntClass):
    """
        Input:
            - DictInitGuessInterval: dict -> Dictionary with initial_guess and interval
                NOTE: Usage In Program:
                    1) Class2InitialGuess[IntClass][Function2Test][Feature]
                    2) DictInitialGuess[Function2Test][Feature]
            - Fcm: pl.DataFrame -> DataFrame with the FCM
            - Feature: str -> Feature to Test
    """
    if IntClass is None:
        LocalMeanMs = np.mean(Fcm[Feature].to_list())
        LocalStdMs = np.std(Fcm[Feature].to_list())
    else:
        LocalMeanMs = Fcm.filter(pl.col("class") == IntClass)[Feature].mean()
        LocalStdMs = Fcm.filter(pl.col("class") == IntClass)[Feature].std()
    DictInitGuessInterval["initial_guess"] = [LocalStdMs,LocalMeanMs]
    return DictInitGuessInterval

def ReturnFitInfoFromDict(Fcm,InitialGuess,DictFittedData,InfoFittedParameters,Feature2Label,FitFile,FittedDataFile):
    """
        Input:
            Fcm: pl.DataFrame -> Fcm Dataframe
            InfoFittedParameters: dict -> {Feature: {Function: [A,b]}}
            DictFittedData: dict -> {Feature: {"fitted_data": [],"best_fit": str}}
            InitialGuess: dict -> {"exponential":{"time":{"initial_guess":[],"interval":[]},"time_hours":{"initial_guess":[],"interval":[]}},
            Feature2Label: dict -> {Feature: Label}
            FitFile: str -> Path to the Fit File
            FittedDataFile: str -> Path to the Fitted Data File
        Description:
            For each Feature of ColumnLabel: ['time','lenght','av_speed','time_hours','lenght_km','speed_kmh']
            Fit the distribution of the feature and plot the distribution.
        Output:
            InfoFittedParameters: dict -> {Feature: {Function: [A,b]}}
            DictFittedData: dict -> {Feature: {"fitted_data": [],"best_fit": str}}
    """
    print("Return Fit Info From Dict:")
#    if os.path.isfile(FitFile) and os.path.isfile(FittedDataFile):
#        with open(FitFile,'r') as f:
#            InfoFittedParameters = json.load(f)
#        with open(FittedDataFile,'r') as f:
#            DictFittedData = json.load(f)
#       for Feature in DictFittedData.keys():
#           with open(FitFile+"{}.json".format(Feature),'r') as f:
#               InfoFittedParameters = json.load(f)
#           with open(FittedDataFile+"{}.json".format(Feature),'r') as f:
#               DictFittedData = json.load(f)
#        Uploading = True
#        SuccessFit = True
#    else:
    for Feature in DictFittedData.keys():
        y,x = np.histogram(Fcm[Feature].to_list(),bins = 50)
        if Feature == "speed_kmh" or Feature == "av_speed":
            y = y/np.sum(y)
        else:
            y = y/np.sum(y)
        InfoFittedParameters, DictFittedData,SuccessFit,FunctionFitted = FitAndPlot(x[1:],y,InitialGuess,Feature,InfoFittedParameters,DictFittedData)
        if SuccessFit:
            with open(FitFile+"{}.json".format(Feature),'w') as f:
                json.dump(InfoFittedParameters[FunctionFitted][Feature],f,cls=NumpyArrayEncoder,indent = 4)
            with open(FittedDataFile+"{}.json".format(Feature),'w') as f:
                json.dump(DictFittedData[Feature],f,cls=NumpyArrayEncoder,indent = 4)

    with open(FitFile,'w') as f:
        json.dump(InfoFittedParameters,f,cls=NumpyArrayEncoder,indent = 4)
    with open(FittedDataFile,'w') as f:
        json.dump(DictFittedData,f,cls=NumpyArrayEncoder,indent = 4)
    Uploading = False
    return InfoFittedParameters,DictFittedData,Uploading,SuccessFit
