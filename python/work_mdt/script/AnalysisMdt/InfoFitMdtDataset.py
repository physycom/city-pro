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

