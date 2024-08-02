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

VERBOSE = False
# FUCNTIONS FOR FITTING
def powerlaw(x, amp, index):
    return amp * (np.array(x)**index)


def exponential(x, amp, index):
    return amp * np.exp(-index*np.array(x))


def linear(x, amp,q):
    return amp * np.array(x) + q

def linear_per_powerlaw(x, index, log_amp):
    return index * x + log_amp

def multilinear4variables(x, a,b,c,log_d):
    '''
        N is the couples of Origin and Destination
        Fitting like Vespignani:
            Vectors - Data:
                1) log Ni -> 1 Dimensional vector of length N being the mass of the grid i.
                2) log Nj -> 1 Dimensional vector of length N being the mass of the grid j.
                3) log dij -> 1 Dimensional vector of length N being the distance between centroids of grid i and j.
            Scalar - Parameters:
                1) log_d: -> k in Ramasco Paper
                2) a: exponent mass i
                3) b: exponent mass j
                4) c: exp(1/d0)                        
    '''
    return a * x[0] + b * x[1] + c * x[2] + log_d


def lognormal(x, mean, sigma):
    return (np.exp(-(np.log(x) - mean)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))


def gamma_(x, shape, scale):
    return ((x**(shape - 1)) * np.exp(-x / scale)) / (scale**shape * gamma(shape))


def weibull(x, shape, scale):
    return (shape / scale) * (x / scale)**(shape - 1) * np.exp(-(x / scale)**shape)

def maxwellian(x,sigma,mu):
    return (x/sigma)**2 * np.exp(-(x - mu)**2 / (2 * sigma**2))


def gaussian(x,sigma,mu):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def truncated_powerlaw(x, amp, index, beta):
    return amp * (np.array(x)**index) * np.exp(-beta*np.array(x))

####### DEPRECATED
# Best Parameter from data
def PlotParameters(ListA,Listb):
    plt.hist2d(ListA,Listb,bins = 100)
    plt.xaxis('Amplitude')
    plt.yaxis('Index')
    plt.show()

def best_parameter_powerlaw(x,y):
    """
        Input: 
            Arrays: For each element of the 
    """
    avg_amp = []
    avg_index = []
    index = 1
    logy = np.log(y)
    logx = np.log(x)
    MinErr = 10000000
    BestIndex = 0
    BestAmp = 0
    Threshold = 100
    # Look for the best couple of parameters by trying with data
    for elem0_couple in range(len(x)):
        if x[elem0_couple] > 0 and y[elem0_couple] > 0:
            amp = index*logy[elem0_couple]/logx[elem0_couple]
            for elem1_couple in range(len(x)):
                if elem0_couple != elem1_couple and x[elem1_couple] > 0 and y[elem1_couple] > 0:
                    index = (logy[elem1_couple] - np.log(amp))/logx[elem1_couple]
                    # Control Not to Overflow
                    if not np.isnan(powerlaw(x[elem1_couple],amp,index)):
                        Err = np.sum(np.abs(y - powerlaw(x,amp,index)))
                        if Err < MinErr and Err < Threshold:
                            MinErr = Err
                            BestIndex = index
                            BestAmp = amp
                    avg_amp.append(amp)
                    avg_index.append(index)
#    PlotParameters(avg_amp,avg_index)
    return BestAmp,BestIndex

def best_parameter_exponential(x,y):
    """
        Input: 
            Arrays: For each element of the 
    """
    avg_amp = []
    avg_index = []
    index = 1
    logy = np.log(y)
    MinErr = 10000000
    BestIndex = 0
    BestAmp = 0
    StopThreshold = 0.0001
    for elem0_couple in range(len(x)):
        if x[elem0_couple] > 0 and y[elem0_couple] >0:
            amp = y[elem0_couple]/np.exp(-index*x[elem0_couple])
            for elem1_couple in range(len(x)):
                if elem0_couple != elem1_couple and x[elem1_couple] > 0 and y[elem1_couple] >0:
                    index = - (logy[elem1_couple] - np.log(amp))/x[elem1_couple]
                    Err = np.sum(np.abs(y - exponential(x,amp,index)))
                    if Err < MinErr:
                        MinErr = Err
                        BestIndex = index
                        BestAmp = amp
                    if Err < StopThreshold:
                        break
                    avg_amp.append(amp)
                    avg_index.append(index)
#    PlotParameters(avg_amp,avg_index)
    return BestAmp,BestIndex

def best_parameter_linear(x,y):
    """
        Input: 
            Arrays: For each element of the 
    """
    avg_amp = []
    avg_index = []
    q = 1
    logy = y
    logx = x
    MinErr = 10000000
    BestIndex = 0
    BestAmp = 0
    for elem0_couple in range(len(x)):
        if x[elem0_couple] != 0:
            amp = (y[elem0_couple] - q)/x[elem0_couple]
            for elem1_couple in range(len(x)):
                if elem0_couple != elem1_couple and x[elem1_couple] != 0:
                    q = (y[elem1_couple] - amp)/x[elem1_couple]
                    Err = np.sum(np.abs(y - linear(x,amp,q)))
                    if Err < MinErr:
                        MinErr = Err
                        BestIndex = q
                        BestAmp = amp
                    avg_amp.append(amp)
                    avg_index.append(q)
#    PlotParameters(avg_amp,avg_index)
    return BestAmp,BestIndex

def best_parameter_maxwellian(x,y):
    """
        Input: 
            Arrays: For each element of the 
    """
    avg_sigma = []
    avg_mu = []
    mu = 1
    sigma = 1
    MinErr = 10000000
    logx = np.log(x)
    logy = np.log(y)
    BestSigma = 0
    BestMu = 0
    for elem0_couple in range(len(x)):
        if x[elem0_couple] != 0:
            UnderSqrt = 2*sigma**2*(logy[elem0_couple] - 2*logx[elem0_couple] + 2*np.log(sigma)) + x**2
            mu = x + np.sqrt(x**2 + UnderSqrt)
            for elem1_couple in range(len(x)):
                if elem0_couple != elem1_couple and x[elem1_couple] != 0:
                    UnderSqrt1 = -(x[elem1_couple]-mu)**2/(logy[elem1_couple] - 2*logx[elem1_couple] + 2*np.log(sigma))
                    if UnderSqrt1 > 0:
                        sigma = np.sqrt(-(x[elem1_couple]-mu)**2/(logy[elem1_couple] - 2*logx[elem1_couple] + 2*np.log(sigma)))/2
                    else:
                        sigma = sigma
                    Err = np.sum(np.abs(y - maxwellian(x,sigma,mu)))
                    if Err < MinErr:
                        MinErr = Err
                        BestSigma = sigma
                        BestMu = mu
                    avg_sigma.append(sigma)
                    avg_mu.append(mu)
#    PlotParameters(avg_sigma,avg_mu)
    return BestSigma,BestMu

def best_parameter_gaussian(x,y):
    """
        Input: 
            Arrays: For each element of the 
    """
    avg_sigma = []
    avg_mu = []
    mu = 1
    sigma = 1
    MinErr = 10000000
    BestSigma = 0
    BestMu = 0
    for elem0_couple in range(len(x)):
        if x[elem0_couple] != 0:
            sigma = np.sqrt(-2*np.log(y[elem0_couple]*sigma*np.sqrt(2*np.pi)))
            for elem1_couple in range(len(x)):
                if elem0_couple != elem1_couple and x[elem1_couple] != 0:
                    mu = mu - sigma*np.sqrt(-2*np.log(y[elem1_couple]*sigma*np.sqrt(2*np.pi)))
                    Err = np.sum(np.abs(y - gaussian(x,sigma,mu)))
                    if Err < MinErr:
                        MinErr = Err
                        BestSigma = sigma
                        BestMu = mu
                    avg_sigma.append(sigma)
                    avg_mu.append(mu)
#    PlotParameters(avg_sigma,avg_mu)
    return BestMu,BestSigma
########################################
# Inverse Functions

def inverse_powerlaw(y, amp, index):
    return 1/amp * (np.array(y)**(1/index))

def inverse_exponential(y, amp, index):
    return -1/index * np.log(np.array(y)/amp)

def inverse_linear(y, amp,q):
    return (np.array(y)-q)/amp

def inverse_maxwellian(y,sigma,mu):
    return np.sqrt(-2*sigma**2*np.log(y/(mu/sigma)**2))

def inverse_gaussian(y,sigma,mu):
    return mu - sigma*np.sqrt(-2*np.log(y*sigma*np.sqrt(2*np.pi)))

# LOSS FUNCTIONS
def quadratic_loss_function(y_predict, y_measured):
    return np.sum((y_predict-y_measured)**2)

def objective_function_powerlaw(params,x,y_measured):
    if len(x)!=len(y_measured):
        raise ValueError('Power Law Loss: x and y measured do not have the same shape')
    if len(params)!=2:
        raise ValueError('Power Law Loss: the parameters are not 2 but {}'.format(len(params)))
    return quadratic_loss_function(linear_per_powerlaw(np.log(x), params[0], params[1]), np.log(y_measured))

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
    if len(params)!=2:
        raise ValueError('The parameters must be an array of length 2')
    if len(x)!=len(y_measured):
        raise ValueError('The x and y must have the same shape')
    y_guessed = lognormal(x, params[0], params[1])
    return quadratic_loss_function(y_guessed,y_measured)

def objective_function_gamma(params,x,y_measured):
    if len(params)!=2:
        raise ValueError('The parameters must be an array of length 2')
    if len(x)!=len(y_measured):
        raise ValueError('The x and y must have the same shape')
    y_guessed = gamma_(x, params[0], params[1])
    return quadratic_loss_function(y_guessed,y_measured)

def objective_function_weibull(params,x,y_measured):
    if len(params)!=2:
        raise ValueError('The parameters must be an array of length 2')
    if len(x)!=len(y_measured):
        raise ValueError('The x and y must have the same shape')
    y_guessed = weibull(x, params[0], params[1])
    return quadratic_loss_function(y_guessed,y_measured)

def objective_function_maxwellian(params,x,y_measured):
    if len(params)!=2:
        raise ValueError('The parameters must be an array of length 2')
    if len(x)!=len(y_measured):
        raise ValueError('The x and y must have the same shape')
    y_guessed = maxwellian(x, params[0], params[1])
    return quadratic_loss_function(y_guessed,y_measured)

def objective_function_gaussian(params,x,y_measured):
    if len(params)!=2:
        raise ValueError('The parameters must be an array of length 2')
    if len(x)!=len(y_measured):
        raise ValueError('The x and y must have the same shape')
    y_guessed = gaussian(x, params[0], params[1])
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
    

Name2InverseFunction = {'powerlaw':inverse_powerlaw,
                        'exponential':inverse_exponential,
                        'linear':inverse_linear,
                        'maxwellian':inverse_maxwellian,
                        'gaussian':inverse_gaussian}

Name2EstimateParameters = {'powerlaw':best_parameter_powerlaw,
                            'exponential':best_parameter_exponential,
                            'linear':best_parameter_linear,
                            'maxwellian':best_parameter_maxwellian,
                            'gaussian':best_parameter_gaussian}

def Fitting(x,y_measured,label = 'powerlaw',initial_guess = (6000,0.3),maxfev = 50000,interval = []):
    '''
        Input:
            label: 'powerlaw' or 'exponential' or 'linear'
            x: (np.array 1D) x-axis
            y_measured: (np.array 1D) y-axis
            initial_guess: (tuple 2D) parameters for fitting
            maxfev: (int) maximum number of iterations
            interval: (tuple 2D) interval on x for fitting
        USAGE:

    '''
    if len(interval)!=0:
#        print("x Not Filtered:\n",x)
        x = np.array([x[i] for i in range(len(x)) if x[i] >= interval[0] and x[i] <= interval[1]])
#        print("x Filtered:\n",x)
        y_measured = np.array([y_measured[i] for i in range(len(x)) if x[i] >= interval[0] and x[i] <= interval[1]])
    else:
        x = np.array(x)
        y_measured = np.array(y_measured)
        assert len(x) == len(y_measured), "x and y_measured must have the same length"
    if label != "gaussian" and label != "maxwellian":
        A,b = Name2EstimateParameters[label](x,y_measured)
#        print("Estimate Parameters: ",A,b)
        initial_guess = (A,b)
    result_powerlaw = minimize(Name2LossFunction[label], initial_guess, args = (x, y_measured))#,maxfev = maxfev
    optimal_params_plw = result_powerlaw.x
    # Handle powerlaw as linear with log indices
    if label == 'powerlaw':
        mask = np.logical_and(np.array(x) > 0, np.array(y_measured) > 0)
        x = x[mask]
        y_measured = y_measured[mask]
        fit = curve_fit(linear_per_powerlaw, xdata = np.log(x), ydata = np.log(y_measured),p0 = list(optimal_params_plw),maxfev = maxfev)
    fit = curve_fit(Name2Function[label], xdata = x, ydata = y_measured,p0 = list(optimal_params_plw),maxfev = maxfev)
    if label == 'powerlaw':
        fit[0][0] = np.exp(fit[0][0])
    if VERBOSE:
        print("Fitting:")
        print("windowing: ",interval)
        print("Size of x after windowing: ",len(x))
        print("Function: ",label,' Message: ',result_powerlaw.message)
    return fit,result_powerlaw.success,x,y_measured

# City - Pro Usage

def FitAndStdError(x,y_measured,label,initial_guess,maxfev = 50000,interval = []):
    fit,ConvergenceSuccess,x,y_measured = Fitting(x,y_measured,label,initial_guess = initial_guess,maxfev = maxfev,interval = interval)
    if len(fit[0]) == 2:
        A = fit[0][0]
        b = fit[0][1]
    else:
        raise ValueError("Fit Of more than 2 parameters need to be handled separately")
    FittedData = list(Name2Function[label](x,A,b))
    SqrtN = np.sqrt(len(y_measured))
    StdError = np.sqrt(np.sum((np.array(y_measured) - np.array(FittedData))**2))/SqrtN
#    kullback_leibler = kl_div(FittedData, y_measured)
#    ks_stat, ks_pval = kstest(data, FittedData.cdf)

    return fit,StdError,ConvergenceSuccess,FittedData,x,y_measured


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
    y,x = np.histogram(ObservedData,bins = 50)
    # Assumes that if some bin has 0 elements is beacouse the data is not enough to fill it (so it is a problem of the choice of bin that we solve in this way)
    y = AdjustZerosDataWithAverage(y)
    if NormBool:
        y = y/np.sum(y)
    else:
        pass

    for Function2Fit in Function2InitialGuess.keys():
        if VERBOSE:
            print("Function2Fit: ",Function2Fit)
        # Consider the Fitting without the interval. (Case of Average Speed) <- It fits well.
        if len(Function2InitialGuess[Function2Fit]["interval"])==0:
            fit,StdError,ConvergenceSuccess,FittedData,x_windowed,y_measured = FitAndStdError(x = x[1:],
                                                                    y_measured = y,
                                                                    label = Function2Fit,
                                                                    initial_guess = Function2InitialGuess[Function2Fit]["initial_guess"]
                                                                    )
        else:
            fit,StdError,ConvergenceSuccess,FittedData,x_windowed,y_measured = FitAndStdError(x = x[1:],
                                                                    y_measured = y,
                                                                    label = Function2Fit,
                                                                    initial_guess = Function2InitialGuess[Function2Fit]["initial_guess"],
                                                                    maxfev = 50000,
                                                                    interval=Function2InitialGuess[Function2Fit]["interval"]
                                                                    )
        FitAllTry[Function2Fit]["x_windowed"] = x_windowed
        FitAllTry[Function2Fit]["y_windowed"] = y_measured
        FitAllTry[Function2Fit]["fitted_data_windowed"] = FittedData
        FitAllTry[Function2Fit]["parameters"] = list(fit[0])
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
        NOTE: Feature2Class2Function2Fit2InitialGuess {Feature: {Class: {Function: {"initial_guess": (A,b),"interval": (start,end)}}}}
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
    # Choose the interval Coming from the Configuration File and Put in Feature2Class2Function2Fit2InitialGuess
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
    if VERBOSE:
        print("Best Fit: ",BestFit)
        print("R: ",R)
        print("p: ",round(p,3))
        print("Aexp: ",Aexp)
        print("bexp: ",bexp)
        print("xmin: ",fit.xmin)
        print("xmax: ",fit.xmax)
    for Function2Fit in Function2InitialGuess.keys():
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
        if Function2Fit == "powerlaw":
            A = 1
            b = fit.power_law.parameter1
        elif Function2Fit == "exponential":
            A = 1
            b = Aexp
        FitAllTry[Function2Fit]["fitted_data_windowed"] = list(Name2Function[Function2Fit](np.array(x_windowed),A,b))
        FitAllTry[Function2Fit]["parameters"] = (A,b)
        if Function2Fit == "powerlaw":
            FitAllTry[Function2Fit]["std_error"] = fit.sigma
        else:
            SqrtN = np.sqrt(len(y))
            StdError = np.sqrt(np.sum((np.array(y_windowed) - np.array(FitAllTry[Function2Fit]["fitted_data_windowed"]))**2))/SqrtN            
            FitAllTry[Function2Fit]["std_error"] = StdError
        FitAllTry[Function2Fit]["success"] = True
        FitAllTry[Function2Fit]["start_window"] = fit.xmin
        FitAllTry[Function2Fit]["end_window"] = fit.xmax
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


if FoundPyMC3:
    """
        Want to define a method that is capable to understand what is the best fit for the data and gives me back some 
        refined information for telling why the choice.
    """
    def FitWithPymc(x,y,label):
        return x,y,label
    import pytensor
    from pytensor.graph.op import Op
    class BestFit(Op):
        __props__ = ()

        #itypes and otypes attributes are
        #compulsory if make_node method is not defined.
        #They're the type of input and output respectively
        itypes = None
        otypes = None        
