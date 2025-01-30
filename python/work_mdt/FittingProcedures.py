try:
    import pymc3 as pm
    FoundPyMC3 = True
except:
    print('PyMC3 not installed')
    FoundPyMC3 = False
from scipy.special import gamma
from scipy.optimize import curve_fit,minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy as kl_div
from scipy.stats import kstest
from tqdm import tqdm
import powerlaw as pwl
from FittingGuesser import *
from FittingFunctions import *
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VERBOSE = False
# FUCNTIONS FOR FITTING



# LOSS FUNCTIONS
def quadratic_loss_function(y_predict, y_measured):
    return np.sum((y_predict-y_measured)**2)

def objective_function_powerlaw(params,x,y_measured):
    if len(x)!=len(y_measured):
        raise ValueError('Power Law Loss: x and y measured do not have the same shape')
    if len(params)!=2:
        raise ValueError('Power Law Loss: the parameters are not 2 but {}'.format(len(params)))
    return quadratic_loss_function(linear_per_powerlaw(x, params[0], params[1]), y_measured)

def objective_function_exponential(params,x,y_measured):
    if len(x)!=len(y_measured):
        raise ValueError('Expo Loss: x and y measured do not have the same shape')
    if len(params)!=2:
        raise ValueError('Expo Loss: the parameters are not 2 but {}'.format(len(params)))
    return quadratic_loss_function(exponential(x, params[0], params[1]), y_measured)

def objective_function_linear(params,x,y_measured):
    if len(x)!=len(y_measured):
        raise ValueError('Linear Loss: x and y measured do not have the same shape')
    if len(params)!=2:
        raise ValueError('Linear Law Loss: the parameters are not 2 but {}'.format(len(params)))
    return quadratic_loss_function(linear(x, params[0], params[1]), y_measured)

def objective_function_multilinear4variables(params,x,y_measured):
    if len(params)!=4:
        raise ValueError('The parameters must be an array of length 4')
    if len(x)!=3:
        raise ValueError('The x must be an array of shape (3,N)')
    if len(x[0])!=len(y_measured):
        raise ValueError('The log of the fluxes must be of the same length as the masses')
    y_guessed = multilinear4variables(x, params[0],params[1],params[2],params[3])
    return quadratic_loss_function(y_guessed,y_measured)

def objective_function_lognormal(params,x,y_measured):
    if len(params)!=3:
        raise ValueError('The parameters must be an array of length 2')
    if len(x)!=len(y_measured):
        raise ValueError('The x and y must have the same shape')
    y_guessed = lognormal(x, params[0], params[1])
    return quadratic_loss_function(y_guessed,y_measured)

def objective_function_gamma(params,x,y_measured):
    if len(params)!=3:
        raise ValueError('The parameters must be an array of length 2')
    if len(x)!=len(y_measured):
        raise ValueError('The x and y must have the same shape')
    y_guessed = gamma_(x, params[0], params[1],params[2])
    return quadratic_loss_function(y_guessed,y_measured)

def objective_function_weibull(params,x,y_measured):
    if len(params)!=3:
        raise ValueError('The parameters must be an array of length 2')
    if len(x)!=len(y_measured):
        raise ValueError('The x and y must have the same shape')
    y_guessed = weibull(x, params[0], params[1],params[2])
    return quadratic_loss_function(y_guessed,y_measured)

def objective_function_maxwellian(params,x,y_measured):
    if len(params)!=3:
        raise ValueError('The parameters must be an array of length 3 {}'.format(params))
    if len(x)!=len(y_measured):
        raise ValueError('The x and y must have the same shape')
    y_guessed = maxwellian(x, params[0], params[1],params[2])
    return quadratic_loss_function(y_guessed,y_measured)

def objective_function_gaussian(params,x,y_measured):
    if len(params)!=3:
        raise ValueError('The parameters must be an array of length 3 {}'.format(params))
    if len(x)!=len(y_measured):
        raise ValueError('The x and y must have the same shape')
    y_guessed = gaussian(x, params[0], params[1],params[2])
    return quadratic_loss_function(y_guessed,y_measured)

def objective_function_truncated_powerlaw(params,x,y_measured):
    if len(params)!=3:
        raise ValueError('The parameters must be an array of length 3')
    if len(x)!=len(y_measured):
        raise ValueError('The x and y must have the same shape')
    y_guessed = truncated_powerlaw(x, params[0], params[1], params[2])
    return quadratic_loss_function(y_guessed,y_measured)
## DICTIONARY FOR LOSS FUNCTIONS
Name2Function = {'powerlaw':powerlaw,
                'exponential':exponential,
                'linear':linear,
                'vespignani':multilinear4variables,
                'lognormal':lognormal,
                'gamma':gamma_,
                'weibull':weibull,
                'maxwellian':maxwellian,
                'gaussian':gaussian,
                'truncated_powerlaw':truncated_powerlaw}
Name2LossFunction = {'powerlaw':objective_function_powerlaw,
                    'exponential':objective_function_exponential,
                    'linear':objective_function_linear,
                    'vespignani':objective_function_multilinear4variables,
                    'lognormal':objective_function_lognormal,
                    'gamma':objective_function_gamma,
                    'weibull':objective_function_weibull,
                    'maxwellian':objective_function_maxwellian,
                    'gaussian':objective_function_gaussian,
                    'truncated_powerlaw':objective_function_truncated_powerlaw}
    

Name2EstimateParameters = {'powerlaw':best_parameter_powerlaw,
                            'exponential':best_parameter_exponential,
                            'linear':best_parameter_linear,
                            'maxwellian':best_parameter_maxwellian,
                            'gaussian':best_parameter_gaussian,
                            'truncated_powerlaw':best_parameter_powerlaw}


### POLISH DATA ####
def NormalizeIntegral2One(y,x):
    """
        @params y: np.array 1D
        @params x: np.array 1D
        @describe: Normalize the integral of the data to 1.
        @return y,x,Z (p(x),x,Normalization)
    """
    dx = np.diff(x)
    if len(dx) == len(y):
        Z = np.sum(dx*y)
    else:
        Z = np.sum(dx*y[:-1])
    y = y/Z
    return y,x,Z

def FromObservation2NormDistr(ObservedData,bins = 50):
    """
        @params: ObservedData: np.array 1D
        @params: bins: int (number of bins)
        @describe: Compute the histogram of the data and normalize it such that:
        \int_{x_min}^{x_max} p(x) dx = 1
        @return y,x,Z (p(x),x,Normalization)
    """
    y,x = np.histogram(ObservedData,bins = bins)
    # Assumes that if some bin has 0 elements is beacouse the data is not enough to fill it (so it is a problem of the choice of bin that we solve in this way)
    y = AdjustZerosDataWithAverage(y)
    y,x,Z = NormalizeIntegral2One(y,x)
    return y,x,Z

def CutFromIntervalX(x,y_measured,interval):
    """
        @params x: np.array 1D
        @params y: np.array 1D (p(x))
        @params interval: np.array [low,high]
        @describe: Cut the data from the interval [low,high] if there are 2 values.
        @return x,y
        """
    if len(interval)!=0:
        x = np.array([x[i] for i in range(len(x)) if x[i] >= interval[0] and x[i] <= interval[1]])
        y_measured = np.array([y_measured[i] for i in range(len(x)) if x[i] >= interval[0] and x[i] <= interval[1]])
    elif len(interval)==0:
        x = np.array(x)
        y_measured = np.array(y_measured)
    else:
        raise ValueError("CutFromIntervalX: Interval =\n\t{}".format(interval))
    assert len(x) != 0, "CutFromIntervalX: The length of x is 0"
    assert len(y_measured) != 0, "CutFromIntervalX: The length of y_measured is 0"
    return x,y_measured

def EstimateInitialGuessIfAbsent(x,y_measured,label):
    """
        @params x: np.array 1D
        @params y_measured: np.array 1D
        @params label: str
        @describe: Estimate the initial guess if it is not present
        @return initial_guess
    """
    if label != "gaussian" and label != "maxwellian" and label != "truncated_powerlaw":
        initial_guess = Name2EstimateParameters[label](x,y_measured)
        return initial_guess
    else:
        mu = sum(x * y_measured) / sum(y_measured)                  
        sigma = np.sqrt(sum(y_measured*(x - mu)**2)/sum(y_measured))
        initial_guess = (max(y_measured),mu,sigma)
        return initial_guess

def EnsureContraintOnFunctionsMet(x,y_measured,label):
    """
        @params x: np.array 1D
        @params y_measured: np.array 1D
        @params label: str
        @describe: Ensure that the function is well defined for the data.
    """
    assert len(x) == len(y_measured), f"EnsureContraintOnFunctionsMet: The length of x {len(x)} and y_measured {len(y_measured)} must be the same"
    if label == "powerlaw":
        mask = np.logical_and(np.array(x) > 0, np.array(y_measured) > 0)
        x = x[mask]
        y_measured = y_measured[mask]
        assert len(x) != 0, "EnsureContraintOnFunctionsMet: The length of x is 0"
        assert len(y_measured) != 0, "EnsureContraintOnFunctionsMet: The length of y_measured is 0"
        return x,y_measured
    elif label == "truncated_powerlaw":
        mask = np.logical_and(np.array(x) > 0, np.array(y_measured) > 0)
        x = x[mask]
        y_measured = y_measured[mask]
        assert len(x) != 0, "EnsureContraintOnFunctionsMet: The length of x is 0"
        assert len(y_measured) != 0, "EnsureContraintOnFunctionsMet: The length of y_measured is 0"
        return x,y_measured
    else:
        assert len(x) != 0, "EnsureContraintOnFunctionsMet: The length of x is 0"
        assert len(y_measured) != 0, "EnsureContraintOnFunctionsMet: The length of y_measured is 0"
        return x,y_measured

def MaskInfinities(x,y_measured,y_fit):
    """
        @params x: np.array 1D
        @params y_measured: np.array 1D
        @params y_fit: np.array 1D
        @describe: Mask the infinities in the fit. x**(-a) is infinite for x = 0
        Mask at the beginning of the procedure, and at the end, after computing the fit.
        """
    if y_fit is None:
        x = x[np.logical_not(np.isinf(y_measured))]
        y_measured = y_measured[np.logical_not(np.isinf(y_measured))]
        return x,y_measured,y_fit
    else:
        y_measured[np.logical_not(np.isinf(y_fit))]
        x = x[np.logical_not(np.isinf(y_fit))]
        y_fit = y_fit[np.logical_not(np.isinf(y_fit))]
    return x,y_measured,y_fit


def RescaleOptimalAmplitudeFromDataAndFit(OptimalParameters,x,y_measured,label):
    """
        @params OptimalParameters: tuple 1D
        @params y_measured: np.array 1D
        @params y_fit: np.array 1D
        @params label: str
        Since the amplitude is missed by the optimizer, I rescale the function with the first value of the y_measured
    """
    if label == "powerlaw":
        logy_fit = linear_per_powerlaw(x,OptimalParameters[0],OptimalParameters[1])            
        A_minus_Afit = y_measured[1] - logy_fit[1]
        OptimalParameters = (np.exp(OptimalParameters[0] + A_minus_Afit),OptimalParameters[1])
    elif label == "gaussian" or label == "maxwellian" or label == "weibull" or label == "truncated_powerlaw":
        pass
    else:
        y_fit = Name2Function[label](x,OptimalParameters[0],OptimalParameters[1])
        A_over_Afit = y_measured[1]/y_fit[1]
        OptimalParameters = (OptimalParameters[0]*A_over_Afit,OptimalParameters[1])
    return OptimalParameters

def ExtractFit(x,y_measured,label,initial_guess,maxfev,method):
    """
        @params x: np.array 1D
        @params y_measured: np.array 1D
        @params label: str (powerlaw,exponential,linear)
        @params initial_guess: tuple 2D
        @describe: According to the case at hand transforms the input and returns the output.
        @return fit, success of the fit
    """
    assert len(x) == len(y_measured), f"ExtractFit: The length of x {len(x)} and y_measured {len(y_measured)} must be the same"
    assert len(x) != 0, "ExtractFit: The length of x is 0"
    if label == 'powerlaw':
        assert initial_guess[0] > 0, "ExtractFit: Amplitude {} must be positive".format(initial_guess[0])
        initial_guess = (initial_guess[0],initial_guess[1])
        y_measured = np.log(y_measured)
        x = np.log(x)
        mask_nan_log = np.logical_and(np.logical_not(np.isnan(x)),np.logical_not(np.isnan(y_measured)))
        x = x[mask_nan_log]
        y_measured = y_measured[mask_nan_log]
    else:
        pass
    MinimumFunctionFit = minimize(Name2LossFunction[label], initial_guess,method = method, args = (x, y_measured))#,maxfev = maxfev
    OptimalParameters = MinimumFunctionFit.x
    # Handle powerlaw as linear with log indices
    OptimalParameters = RescaleOptimalAmplitudeFromDataAndFit(OptimalParameters,x,y_measured,label)
    logger.info(f"Optimal Parameters: {OptimalParameters}")
    if label == 'powerlaw':
        y_measured = np.exp(y_measured)
        x = np.exp(x)    
    else:
        pass
    try:
        fit = curve_fit(Name2Function[label], xdata = x, ydata = y_measured,p0 = list(OptimalParameters),maxfev = maxfev)
        Success = True
    except RuntimeError:
        fit = OptimalParameters
        Success = False
    
    if label == 'powerlaw':
        fit[0][0] = np.exp(fit[0][0])
#    logger.info(f"Fit {label} success: {MinimumFunctionFit.success}, Optimal Parameters: {OptimalParameters}, Fit: {fit}")
    return fit,Success

def ComputeFittedValues(x,fit,label,Success):
    """
        @params x: np.array 1D
        @params fit: tuple 2D
        @describe: Compute the fitted values
        @return y_fit: np.array 1D
    """
    if label == 'powerlaw' or label == 'exponential' or label == 'linear':
        if not Success:
            y_fit = Name2Function[label](x,fit[0],fit[1])
        else:
            y_fit = Name2Function[label](x,fit[0][0],fit[0][1])
    elif label == "gaussian" or label == "maxwellian" or label == "weibull" or label == "truncated_powerlaw":
        if not Success:
            y_fit = Name2Function[label](x,fit[0],fit[1],fit[2])
        else:
            y_fit = Name2Function[label](x,fit[0][0],fit[0][1],fit[0][2])
    else:
        raise ValueError("ComputeFittedValues: label =\n\t{}".format(label))
    return y_fit

### FITs PROCEDURE ###



def FitProbabilityFromData(ObservedData,label = 'powerlaw',initial_guess = (1,-1),maxfev = 50000,interval = [],method = "Nelder-Mead"):
    '''
        @param ObservedData: np.array 1D (The Data Observed)
        @param label: str (powerlaw,exponential,linear)
        @param initial_guess: tuple 2D (Initial Guess)
        @param maxfev: int (Maximum number of function evaluations)
        @param interval: list [low,high] (Interval to cut the data)
        @param method: str (Method to use for the minimization) NOTE: Default is Nelder-Mead
        NOTE: https://docs.scipy.org/doc/scipy/tutorial/optimize.html
        @return 1) fit [param0,...,param_i]        
                2) Success: bool
                3) x: np.array 1D (filtered from the window)
                4) y_measured: np.array 1D (filtered from the window)
        @Description:
            1) Normalize the data to a probability distribution
            2) Cut the data from the interval if it is present
            3) Estimate the initial guess if it is not present
            4) Ensure the constraint on the functions are met
            5) Extract the fit


    '''
    if label != "gaussian" and label != "maxwellian" and label != "truncated_powerlaw":
        y_measured,x,Z = FromObservation2NormDistr(ObservedData,100)
    else:
        y_measured,x = np.histogram(ObservedData,bins = 50)
        y_measured = AdjustZerosDataWithAverage(y_measured)
#        y_measured,x = NormalizeIntegral2One(y_measured,x)
#    x,y_measured,_ = MaskInfinities(x,y_measured,None)
    # Filter the Infinities
    if len(x) == len(y_measured) + 1:
        x = x[:-1]
    else:
        pass
    x,y_measured = CutFromIntervalX(x,y_measured,interval)
    if label == "gaussian" or label == "maxwellian":
        initial_guess = EstimateInitialGuessIfAbsent(x,y_measured,label)
    else:
        pass    
    x,y_measured = EnsureContraintOnFunctionsMet(x,y_measured,label)
#    initial_guess = (initial_guess[0]/Z,initial_guess[1])
    fit,Success = ExtractFit(x,y_measured,label,initial_guess,maxfev,method)
    y_fit = ComputeFittedValues(x,fit,label,Success)
    # Mask the infinities
    x,y_measured,y_fit = MaskInfinities(x,y_measured,y_fit)
    if label == "gaussian" and label == "maxwellian" and label == "truncated_powerlaw":
#        dx = np.diff(x)
#        Z_fit = np.sum(dx*y_fit[:-1])
#        y_fit = Z*y_fit/Z_fit
        y_measured = Z*y_measured
    return fit,Success,y_fit,x,y_measured



def FitGivenXY(x,y_measured,label = 'powerlaw',initial_guess = (6000,0.3),maxfev = 50000,interval = [],method = "Nelder-Mead"):
    """
        @params x: np.array 1D
        @params y_measured: np.array 1D
        @params label: str 

    """    
    x,y_measured,_ = MaskInfinities(x,y_measured,None)
    if label != "gaussian" and label != "maxwellian" and label != "truncated_powerlaw":
        y_measured,x,Z = NormalizeIntegral2One(y_measured,x)
    x,y_measured = CutFromIntervalX(x,y_measured,interval)
    if label == "gaussian" or label == "maxwellian":
        initial_guess = EstimateInitialGuessIfAbsent(x,y_measured,label)
    else:
        pass    
    x,y_measured = EnsureContraintOnFunctionsMet(x,y_measured,label)
#    initial_guess = (initial_guess[0]/Z,initial_guess[1])
    fit,Success = ExtractFit(x,y_measured,label,initial_guess,maxfev,method)
    # Normalize the data to the original one
    y_fit = ComputeFittedValues(x,fit,label,Success)
    x,y_measured,y_fit = MaskInfinities(x,y_measured,y_fit)
    if label != "gaussian" and label != "maxwellian" and label != "truncated_powerlaw":
        Z = 1/np.sum(y_fit)
        y_fit = Z*y_fit
        y_measured = Z*y_measured
    return fit,Success,y_fit,x,y_measured

def ComputeStdError(y_measured,y_fit):
    """
        @params y_measured: np.array 1D
        @params y_fit: np.array 1D
        @describe: Compute the standard error of the fit
        @return std_error: float
    """
    assert len(y_measured) == len(y_fit), "The length of the measured and fitted data must be the same"
    SqrtN = np.sqrt(len(y_measured))
    StdError = np.sqrt(np.sum((np.array(y_measured) - np.array(y_fit))**2))/SqrtN
    return StdError
# City - Pro Usage

def FitAndStdErrorFromXY(x,y_measured,label,initial_guess,maxfev = 50000,interval = [],method = "Nelder-Mead"):
    """
        @params x: np.array 1D
        @params y_measured: np.array 1D 
        @params label: str (powerlaw,exponential,linear,weibull....)
        @params initial_guess: tuple 1D (Initial Guess Parameters) 
        @params maxfev: int (Maximum number of function evaluations)
        @params interval: list [low,high]
    """
    fit,ConvergenceSuccess,y_fit,x,y_measured = FitGivenXY(x,y_measured,label,initial_guess = initial_guess,maxfev = maxfev,interval = interval,method = method)
    StdError = ComputeStdError(y_measured,y_fit)
    return fit,StdError,ConvergenceSuccess,y_fit,x,y_measured

def FitAndStdErrorFromObservedData(ObservedData,label,initial_guess,maxfev = 50000,interval = [],method = "Nelder-Mead"):
    """
        @params ObservedData: np.array 1D
        @params label: str (powerlaw,exponential,linear,weibull....)
        @params initial_guess: tuple 1D (Initial Guess Parameters)
        @params maxfev: int (Maximum number of function evaluations)
        @params interval: list [low,high]
        @params method: str (Method to use for the minimization) NOTE: Default is Nelder-Mead    
    """
    fit,ConvergenceSuccess,y_fit,x,y_measured = FitProbabilityFromData(ObservedData,label,initial_guess,maxfev = maxfev,interval = interval,method = method)
    StdError = ComputeStdError(y_measured,y_fit)
    return fit,StdError,ConvergenceSuccess,y_fit,x,y_measured

def ReturnFitInfoFromDict(ObservedData,Function2InitialGuess,FitAllTry,NormBool = True):
    """
        Input:
            ObservedData: Column from a dataframe or array of observations
            Function2InitialGuess: dict -> {Function0: (A,b),Function1: (A,b),...}
            1) FitAllTry: dict -> {Function2Fit: {"fitted_data": [],"best_fit": str,"parameters": [],"start_window":None,"end_window":None,"std_error":None,"success": False}}
        Description:
            Usage in cycles over features and conditionalities of a big dataframe.
            Computes the fit and store them in entrance of FitAllTry
        NOTE: FitAllTry must be a dictionary whose entrancies respect the same structure  of the conditional search one wants to do.
            i.e. 1) Function2Fit: ["eponential","powerlaw"] if ObservedData represents -> Feature in ["lenght","lenght_km","time","time_hours"]
                    Function2Fit: ["gaussian","maxwellian"] if ObservedData represents -> Feature in ["av_speed","av_speed_kmh"]
        NOTE: In AnalysisNetwork1Day and AnalysisNetworkAllDays this function is called for each feature and conditionalities.
            For example groupby the Fcm dataframe by "Class" and the ObservedData will be the vector of observation of some Feature.
    """
    # Bin Data: number of bins is chosen randomly (no optimal principle of any kind)

    for Function2Fit in Function2InitialGuess.keys():
        print("Function2Fit: ",Function2Fit)
        # Consider the Fitting without the interval. (Case of Average Speed) <- It fits well.
        y,x = np.histogram(ObservedData,bins = 50)
        if len(Function2InitialGuess[Function2Fit]["interval"])==0:
            fit,StdError,ConvergenceSuccess,FittedData,x_windowed,y_measured = FitAndStdErrorFromXY(x = x[1:],
                                                                                y_measured= y,
                                                                                label = Function2Fit,
                                                                                initial_guess = Function2InitialGuess[Function2Fit]["initial_guess"],
                                                                                maxfev = 50000,
                                                                                interval = [],
                                                                                method = "Nelder-Mead")
        else:
            fit,StdError,ConvergenceSuccess,FittedData,x_windowed,y_measured = FitAndStdErrorFromXY(x = x[1:],
                                                                                y_measured= y,
                                                                                label = Function2Fit,
                                                                                initial_guess = Function2InitialGuess[Function2Fit]["initial_guess"],
                                                                                maxfev = 50000,
                                                                                interval = Function2InitialGuess[Function2Fit]["interval"],
                                                                                method = "Nelder-Mead")
        Z = np.sum(y_measured)
        Z_fit = np.sum(FittedData)
        y_measured = y_measured/Z
        FittedData = FittedData/Z_fit
        FitAllTry[Function2Fit]["x_windowed"] = x_windowed
        FitAllTry[Function2Fit]["y_windowed"] = y_measured
        FitAllTry[Function2Fit]["fitted_data_windowed"] = FittedData
        if ConvergenceSuccess:
            FitAllTry[Function2Fit]["parameters"] = list(fit[0])
        else:
            FitAllTry[Function2Fit]["parameters"] = list(fit)
        FitAllTry[Function2Fit]["std_error"] = StdError
        FitAllTry[Function2Fit]["success"] = ConvergenceSuccess
        if len(Function2InitialGuess[Function2Fit]["interval"])!=0:
            FitAllTry[Function2Fit]["start_window"] = Function2InitialGuess[Function2Fit]["interval"][0]
            FitAllTry[Function2Fit]["end_window"] = Function2InitialGuess[Function2Fit]["interval"][1]
        else:
            FitAllTry[Function2Fit]["start_window"] = x[1]
            FitAllTry[Function2Fit]["end_window"] = x[-1]
        
    return FitAllTry



def ComputeAndChooseBestFit(ObservedData,Function2InitialGuess,FitAllTry,NormBool = True):
    """
        This function is used to compute the best fit for Feature in [lenght,time,lenght_km,time_hours].  .
        NOTE: Feature2Class2Function2Fit2InitialGuess 
        {Feature:
          {Class:
            {Function:
              {"initial_guess": (A,b),"interval": (start,end)}}}}
        Initialized from the configuration file in CreateDictClass2FitInit.
            """
    y,x = np.histogram(ObservedData,bins = 50)
    y = AdjustZerosDataWithAverage(y)
    if NormBool:
        y = y/np.sum(y)
    else:
        pass
    if VERBOSE:
        print("interval considered for fit: ",Function2InitialGuess["exponential"]["interval"])
    # Compare exponential and powerlaw
    fit = pwl.Fit(ObservedData,
                  xmin = Function2InitialGuess["exponential"]["interval"][0],
                  xmax = Function2InitialGuess["exponential"]["interval"][1]
                  )
    R,p = fit.distribution_compare('power_law', 'exponential',normalized_ratio=True)
    Aexp = fit.exponential.parameter1
    bexp = fit.exponential.parameter2
    if R>0:
        BestFit = "powerlaw"
    else:
        BestFit = "exponential"
    print("Best Fit: ",BestFit)
    print("R: ",R)
    print("p: ",round(p,3))
    print("Aexp: ",Aexp)
    print("bexp: ",bexp)
    print("xmin: ",fit.xmin)
    print("xmax: ",fit.xmax)
    for Function2Fit in Function2InitialGuess.keys():
        #  Windowing the data between xmin and xmax
        if fit.xmax is not None:            
            x_windowed = [x_ for x_ in x if (x_>=fit.xmin) and (x_<=fit.xmax)]
            y_windowed = [y_ for i,y_ in enumerate(y) if (x[i]>=fit.xmin) and (x[i]<=fit.xmax)]
            FitAllTry[Function2Fit]["x_windowed"] = x_windowed
            FitAllTry[Function2Fit]["y_windowed"] = y_windowed
        else:
            x_windowed = [x_ for x_ in x if (x_>=fit.xmin)]
            y_windowed = [y_ for i,y_ in enumerate(y) if (x[i]>=fit.xmin)]
            FitAllTry[Function2Fit]["x_windowed"] = x_windowed
            FitAllTry[Function2Fit]["y_windowed"] = y_windowed
        # Get the parameters for the fit
        if Function2Fit == "powerlaw":
            A = 1
            b = fit.power_law.parameter1
            FitAllTry[Function2Fit]["fitted_data_windowed"] = list(Name2Function[Function2Fit](np.array(x_windowed),A,b))
            FitAllTry[Function2Fit]["parameters"] = (A,b)
        elif Function2Fit == "exponential":
            A = 1
            b = -Aexp
            FitAllTry[Function2Fit]["fitted_data_windowed"] = list(Name2Function[Function2Fit](np.array(x_windowed),A,b))
            FitAllTry[Function2Fit]["parameters"] = (A,b)
        elif Function2Fit == "truncated_powerlaw":
            fit_,StdError,ConvergenceSuccess,FittedData,x_windowed,y_measured = FitAndStdErrorFromXY(x = x[1:],
                                                                y_measured = y,
                                                                label = "truncated_powerlaw",
                                                                initial_guess = Function2InitialGuess[Function2Fit]["initial_guess"],
                                                                interval=Function2InitialGuess["exponential"]["interval"]
                                                                )

            A = fit_[0][0]
            b = fit_[0][1]
            c = fit_[0][2]
            FitAllTry[Function2Fit]["fitted_data_windowed"] = list(Name2Function[Function2Fit](np.array(x_windowed),A,b,c))
            FitAllTry[Function2Fit]["parameters"] = (A,b,c)

        # Compare with truncated powerlaw the fits and choose the best.
        if BestFit == "powerlaw":
            kstats, pval = kstest(y_windowed, Name2Function["powerlaw"](x_windowed,A,b))
        elif BestFit == "exponential":
            kstats, pval = kstest(y_windowed, Name2Function["exponential"](x_windowed,A,b))
        if Function2Fit == "truncated_powerlaw":
            kstats_trun, pval_trun = kstest(y_windowed, Name2Function["truncated_powerlaw"](x_windowed,A,b,c))
        DimensionSmallestArray = min(len(y_windowed),len(FitAllTry[Function2Fit]["fitted_data_windowed"]))
        # Fill the error
        if Function2Fit == "powerlaw":
            FitAllTry[Function2Fit]["std_error"] = fit.sigma
        elif Function2Fit == "exponential":
            SqrtN = np.sqrt(len(y))
            StdError = np.sqrt(np.sum((np.array(y_windowed)[:DimensionSmallestArray] - np.array(FitAllTry[Function2Fit]["fitted_data_windowed"][:DimensionSmallestArray]))**2))/SqrtN            
            FitAllTry[Function2Fit]["std_error"] = StdError
        elif Function2Fit == "truncated_powerlaw":
            FitAllTry[Function2Fit]["std_error"] = StdError
        FitAllTry[Function2Fit]["success"] = True
        FitAllTry[Function2Fit]["start_window"] = fit.xmin
        FitAllTry[Function2Fit]["end_window"] = fit.xmax
    if kstats_trun > kstats:
        BestFit = "truncated_powerlaw"
    else:
        pass

    FitAllTry["best_fit"] = BestFit
    return FitAllTry


def ChooseBestFit(AllFitTry):
    for Function2Fit in AllFitTry.keys():
        InfError = 10000000000
        # If the Fit for the Feature -> Class -> Function is successful and has smaller error than the previous one
        if Function2Fit != "best_fit":
            if AllFitTry[Function2Fit]["std_error"] < InfError and AllFitTry[Function2Fit]["success"]:
                AllFitTry["best_fit"] = Function2Fit
            else:
                pass
    return AllFitTry

def FillIterationFitDicts(ObservedData,Function2Fit2InitialGuess,AllFitTry):
    y,x = np.histogram(ObservedData,bins = 50)
    AllFitTry = ReturnFitInfoFromDict(ObservedData,
                                        Function2Fit2InitialGuess,
                                        AllFitTry,
                                        True)
    # Choose the Best Fit among all the tried feature
    AllFitTry = ChooseBestFit(AllFitTry)
    if VERBOSE:
        print("Info Fit All Try: ")
        print("best_fit: ",AllFitTry["best_fit"])
        for Function2Fit in AllFitTry.keys():
            if Function2Fit != "best_fit":
                print("Function Considered: ",Function2Fit)
                print("Number element x windowed: ",len(AllFitTry[Function2Fit]["x_windowed"]))
                print("Number element y windowed: ",len(AllFitTry[Function2Fit]["y_windowed"]))
                print("Number element fitted data windowed: ",len(AllFitTry[Function2Fit]["fitted_data_windowed"]))
                print("Parameters: ",AllFitTry[Function2Fit]["parameters"])
                print("Std Error: ",AllFitTry[Function2Fit]["std_error"])
                print("Success: ",AllFitTry[Function2Fit]["success"])
    return AllFitTry 

def FillIterationFitDictsTimeLength(ObservedData,Function2Fit2InitialGuess,AllFitTry):
    AllFitTry = ComputeAndChooseBestFit(ObservedData,Function2Fit2InitialGuess,AllFitTry,True)
    return AllFitTry


def AdjustZerosDataWithAverage(data):
    """
        Fill with average the zeros in the data.
        When there are 3 zeros in a row, just the edges will be filled
    """
    for i in range(len(data)-1):
        # Take the average of the two closest values o
        if data[i] == 0:
            if i > 0:
                if data[i+1] != 0 or data[i-1] != 0:
                    data[i] = (data[i-1] + data[i+1])/2
            # if the index is 0 then allow the 1st value that is not 0.
            else:
                for j in range(1,4):
                    if data[i+j] != 0:
                        data[i] = data[i+j]
                        break
    return data


def Ptest(x,vector_signal,mu,sigma,percentile = 0.95):
    """
        Input:
            x: np.array 1D (linspace of the bin of an histogram that could be done by np.histogram(vector_signal))
            vector_signal: np.array 1D (signal to be filtered)
            mu: float
            sigma: float
            percentile: float
        Output:
            is_signal: np.array 1D (boolean array)
        Description:
            The function is used to filter the signal from the noise.
            That is, chooses the values in vector_signal that are above the percentile given
            a gaussian distribution with mean mu and sigma.
        

    """
    
    Z = np.sum(np.exp(-(x-mu)**2/(2*sigma**2)))
#    idx_mean = np.argmin(np.abs(x-mu))
    idx_mean = int(len(x)/2)
    idces_percentile = []
    erf_from_mean = 0
    for i in range(len(x)):
        erf_from_mean += np.exp(-(x[i]-mu)**2/(2*sigma**2))
        if erf_from_mean > Z*percentile:
            idces_percentile.append(x[i])
    # True if outside 95 % distribution
    is_signal = [True if vector_signal[i] > idces_percentile[0] else False for i in range(len(vector_signal))]
    return is_signal,idces_percentile


