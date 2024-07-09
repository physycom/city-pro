import numpy as np
import pymc as pm
import scipy.stats as stats


class PyMCFitter:
    """
        The Class works just with data that has already been binned. The granularity decision
        must be handled by another code. The topology here is given.
        The compute the most probable values of parameters given:
        Example:
            Priors:
                P(alpha): ["exponential","uniform","normal","bernoulli","poisson","discrete_uniform"]
                P(beta): ["exponential","uniform","normal","bernoulli","poisson","discrete_uniform"]
                P(sigma): ["exponential","uniform","normal","bernoulli","poisson","discrete_uniform"]
            Likelihood:
                P(y|x,alpha,beta,sigma) = N(alpha + beta*x, sigma)
            Posterior:
                P(alpha,beta,sigma|y,x) = P(y|x,alpha,beta,sigma)P(alpha)P(beta)P(sigma)
    """
    def __init__(self, x,y, ModelName):
        self.x = x
        self.y = y
        self.maxx = max(x)
        self.minx = min(x)
        self.maxy = max(y)
        self.miny = min(y)
        self.n = len(x)
        self.dimension = np.shape(x)
        self.ModelName2Parmas = {"exponential":pm.Exponential,
                                "uniform":pm.Uniform,
                                "normal":pm.Normal,
                                "maxwellian":pm.Maxwellian,
                                "powerlaw":pm.PowerLaw}
        if ModelName in self.ModelName2Parmas.keys():
            self.ModelName = ModelName
        else:
            raise ValueError("ModelName not in the list of available models (exponential,uniform,normal,maxwellian,powerlaw)")
        
    def ExponentialModel(self):
        @stochastic(observed=False)
        def Tau

    def TensorProduct(self):
        if np.shape(self.x) == np.shape(self.y):
            self.xy = np.dot(self.x[:,None],self.y[None,:])
        

    