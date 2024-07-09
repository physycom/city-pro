try:
    import pymc3 as pm
    FoundPyMC3 = True
except:
    print('PyMC3 not installed')
    FoundPyMC3 = False
from scipy.special import gamma
from scipy.optimize import curve_fit,minimize
from scipy import stats
from scipy.stats import powerlaw as plw
import numpy as np
import matplotlib.pyplot as plt

# FUCNTIONS FOR FITTING
def powerlaw(x, amp, index):
    return amp * (np.array(x)**index)


def exponential(x, amp, index):
    return amp * np.exp(-index*np.array(x))


def linear(x, amp,q):
    return amp * np.array(x) + q


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
    return quadratic_loss_function(powerlaw(x, params[0], params[1]), y_measured)

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
## DICTIONARY FOR LOSS FUNCTIONS
Name2Function = {'powerlaw':powerlaw,
                'exponential':exponential,
                'linear':linear,
                'vespignani':multilinear4variables,
                'lognormal':lognormal,
                'gamma':gamma_,
                'weibull':weibull,
                'maxwellian':maxwellian,
                'gaussian':gaussian}
Name2LossFunction = {'powerlaw':objective_function_powerlaw,
                    'exponential':objective_function_exponential,
                    'linear':objective_function_linear,
                    'vespignani':objective_function_multilinear4variables,
                    'lognormal':objective_function_lognormal,
                    'gamma':objective_function_gamma,
                    'weibull':objective_function_weibull,
                    'maxwellian':objective_function_maxwellian,
                    'gaussian':objective_function_gaussian}
    

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
        x = np.array([x[i] for i in range(len(x)) if x[i] >= interval[0] and x[i] <= interval[1]])
        y_measured = np.array([y_measured[i] for i in range(len(x)) if x[i] >= interval[0] and x[i] <= interval[1]])
    else:
        x = np.array(x)
        y_measured = np.array(y_measured)
        assert len(x) == len(y_measured), "x and y_measured must have the same length"
#    print('Fitting {}'.format(label))
#    if label == 'powerlaw':
#        x = np.log(x)
#        y = np.log(y_measured)
#        initial_guess = (initial_guess[1],np.log(initial_guess[0]))
#        result_powerlaw = minimize(objective_function_linear, initial_guess, args = (x, y_measured))
#        optimal_params_plw = result_powerlaw.x
#        fit = curve_fit(linear, xdata = x, ydata = y_measured,p0 = list(optimal_params_plw),maxfev = maxfev)
#    else:
    if label != "gaussian" and label != "maxwellian":
        A,b = Name2EstimateParameters[label](x,y_measured)
        print("Estimate Parameters: ",A,b)
        initial_guess = (A,b)
    result_powerlaw = minimize(Name2LossFunction[label], initial_guess, args = (x, y_measured))#,maxfev = maxfev
    optimal_params_plw = result_powerlaw.x
    fit = curve_fit(Name2Function[label], xdata = x, ydata = y_measured,p0 = list(optimal_params_plw),maxfev = maxfev)
#    print(fit)
#    print('{} fit: '.format(label),fit[0][0],' ',fit[0][1])
#    print('Convergence fit {}: '.format(label),result_powerlaw.success)
#    print('Optimal parameters: ',result_powerlaw.x)
    print('Message: ',result_powerlaw.message)
    return fit,result_powerlaw.success

# City - Pro Usage

def FitAndStdError(x,y_measured,label,initial_guess,maxfev = 50000,interval = []):
    fit,ConvergenceSuccess = Fitting(x,y_measured,label,initial_guess = initial_guess,maxfev = maxfev,interval = interval)
    if len(fit[0]) == 2:
        A = fit[0][0]
        b = fit[0][1]
    else:
        raise ValueError("Fit Of more than 2 parameters need to be handled separately")
    FittedData = list(Name2Function[label](x,A,b))
    SqrtN = np.sqrt(len(y_measured))
    StdError = np.sqrt(np.sum((np.array(y_measured) - np.array(FittedData))**2))/SqrtN
    return fit,StdError,ConvergenceSuccess

def FitAndPlot(x,y_measured,DictInitialGuess,Feature,InfoFit,DictFittedData):
    """
        Input:
            x: np.array -> x-axis
            y_measured: np.array -> y-axis
            DictInitialGuess: dict -> Dictionary of the initial guess for the fit
            Feature: str -> Feature to fit from [lenght_km,lenght,time,time_hours,av_speed,speed_kmh]
            Function2Fit: str -> Function to fit from [powerlaw,exponential,gaussian,maxwellian]
        Return: 
            InfoFit: {label:{"fit":None,"StdError":None} for label in DictInitialGuess.keys()}
            FittedData: np.array -> Fitted Data
            BestFit: str -> Best Fit Label
        Usage:
            Prepare the Dictionary of the Fit you want to try:
                Example:
                    DictInitialGuess = {"powerlaw":
                                            {"initial_guess":[6000,0.3],"interval":[0,100]},
                                        "exponential":
                                            {"initial_guess":[6000,0.3],"interval":[0,100]},
                                        "linear":
                                            {"initial_guess":[6000,0.3],"interval":[0,100]}}
        NOTE: The Function will update InfoFit and DictFittedData each time a new Feature is fitted
        NOTE: This fact is embedded in the Usage
    """
    print("Fit and Plot:")
    for Function2Fit in DictInitialGuess.keys():
        if Feature in DictInitialGuess[Function2Fit].keys():
            InfoFit[Function2Fit][Feature]['fit'],InfoFit[Function2Fit][Feature]['StdError'],InfoFit[Function2Fit][Feature]["success"] = FitAndStdError(x,y_measured,Function2Fit,tuple(DictInitialGuess[Function2Fit][Feature]["initial_guess"]),10000,DictInitialGuess[Function2Fit][Feature]["interval"])
            print('Fitting Function {}'.format(Function2Fit))
            print('Feature {}'.format(Feature))
            print('Fit: ',InfoFit[Function2Fit][Feature]['fit'])
            print('StdError: ',InfoFit[Function2Fit][Feature]['StdError'])
            print('Success: ',InfoFit[Function2Fit][Feature]["success"])
    # Compare the <function for a given Feature> and update the DictFittedData
    for Function2Fit in DictInitialGuess.keys():
        InfError = 10000000000
        BestFitFunction = None
        BestFitParameters = None
                
        if Feature in DictInitialGuess[Function2Fit].keys(): 
            Error = InfoFit[Function2Fit][Feature]['StdError']
            SuccessFeatureFunction = InfoFit[Function2Fit][Feature]["success"]
            if Error is not None and SuccessFeatureFunction:
                if InfoFit[Function2Fit][Feature]['StdError'] < InfError:
                    InfError = Error
                    BestFitFunction = Function2Fit
                    BestFitParameters = list(InfoFit[Function2Fit][Feature]['fit'])
            else:
                print("The Case with Feature {0} and Function {1} has Nan Error".format(Feature,Function2Fit))
    if BestFitFunction is not None:
        if Feature in DictInitialGuess[Function2Fit].keys():
            print("\tFeature: ",Feature)
            print("\tBest Fit Function: ",BestFitFunction)
            print("\tBest Fit Parameters: ",BestFitParameters)
            if len(BestFitParameters[0])==2:
                A = BestFitParameters[0][0]
                b = BestFitParameters[0][1]
                DictFittedData[Feature]["best_fit"] = BestFitFunction
                DictFittedData[Feature]["fitted_data"] = list(Name2Function[BestFitFunction](x,A,b))
                DictFittedData[Feature]["parameters"] = BestFitParameters
                if len(DictInitialGuess[Function2Fit][Feature]["interval"]) == 2:
                    DictFittedData[Feature]["start_window"] = DictInitialGuess[Function2Fit][Feature]["interval"][0]
                    DictFittedData[Feature]["end_window"] = DictInitialGuess[Function2Fit][Feature]["interval"][1]
                else:
                    DictFittedData[Feature]["start_window"] = None
                    DictFittedData[Feature]["end_window"] = None
    else:
        if Feature in DictInitialGuess[Function2Fit].keys():
            print("\tFeature: ",Feature)
            print("\tBest Fit Function: ",BestFitFunction)
            print("\tBest Fit Parameters: ",BestFitParameters)
            DictFittedData[Feature]["best_fit"] = None
            DictFittedData[Feature]["fitted_data"] = []
            DictFittedData[Feature]["parameters"] = None
            DictFittedData[Feature]["start_window"] = None
            DictFittedData[Feature]["end_window"] = None
    if BestFitFunction is None:
        SuccessFit = False
    else:
        SuccessFit = True
    return InfoFit,DictFittedData,SuccessFit,BestFitFunction


if FoundPyMC3:
    def FitWithPymc(x,y,label):
        return x,y,label
