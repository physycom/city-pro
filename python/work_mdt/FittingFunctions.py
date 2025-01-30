import numpy as np
def powerlaw(x, A, alpha):
    """
        y = Ax^{\alpha}
    """
    return A*x**alpha

def exponential(x, A, beta):
    """
        y = A*exp(\beta*x)
    """
    return A * np.exp(beta*np.array(x))


def linear(x, A,b):
    """
        y = A*x + b
    """
    return A * np.array(x) + b

def linear_per_powerlaw(x, log_amp,index):
    return index * x + log_amp

def multilinear4variables(x, log_k,alpha,gamma,d0minus1):
    '''
        N is the couples of Origin and Destination
        Fitting like Vespignani:
            Vectors - Data:
                1) log Mi -> 1 Dimensional vector of length N being the mass of the grid i.
                2) log Mj -> 1 Dimensional vector of length N being the mass of the grid j.
                3) log dij -> 1 Dimensional vector of length N being the distance between centroids of grid i and j.
            Scalar - Parameters:
                1) log_k: k in Ramasco Paper
                2) alpha: exponent mass i
                3) gamma: exponent mass j
                4) d0minus1: exp(1/d0)                        
    '''
    return log_k + alpha * x[0] + gamma * x[1] + d0minus1 * x[2] 


def lognormal(x, A,mean, sigma):
    return A*(np.exp(-(np.log(x) - mean)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))


def gamma_(x, A,shape, scale):
    return A*((x**(shape - 1)) * np.exp(-x / scale)) / (scale**shape * gamma(shape))


def weibull(x, A, shape, scale):
    return A*(shape / scale) * (x / scale)**(shape - 1) * np.exp(-(x / scale)**shape)

def maxwellian(x,A,mu,sigma):
    return A*(x/sigma)**2 * np.exp(-(x - mu)**2 / (2 * sigma**2))


def gaussian(x,A,mu,sigma):
    return A/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def truncated_powerlaw(x, amp, index, beta):
    return amp * (np.array(x)**index) * np.exp(-beta*np.array(x))
