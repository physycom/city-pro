import numpy as np
import matplotlib.pyplot as plt
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
    mu = sum(x * y) / sum(y)                  
    sigma = np.sqrt(sum(y*(x - mu)**2)/sum(y))    
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