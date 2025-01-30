import numpy as np
import pymc as pm
import scipy.stats as stats
from collections import defaultdict
RANDOM_SEED = 111111

class PyMCFitter:
    """
        @param x: The x values of the data
        @param P: The probability values of the data (or any function you want to fit)
        @param ModelName: The name of the model to be used
        @description:
            Given the x,y values of the data, the class will fit the data to a model using PyMC.
        
        @usage:
            x = np.linspace(0,10,100)
            y = 2*x + 1 + np.random.normal(0,1,100)

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
    def __init__(self, x,y, ModelToFit):
        assert np.shape(x) == np.shape(y), "The shape of x and y must be the same"
        assert len(np.shape(x)) == 1, f"The shape of x must be 1 {np.shape(x)}"
        self.x = np.array(x)
        self.y = np.array(y)
        self.Z = np.sum(x*y)
        if np.sum(x*y) == 1:
            self.y_norm = y
        else:
            self.y_norm = y/self.Z
        assert np.sum(self.y_norm*self.x) == 1, "The sum of the normalized y values must be 1"
        self.maxx = max(x)
        self.minx = min(x)
        self.maxy = max(y)
        self.miny = min(y)
        self.n = len(x)
        self.dimension = np.shape(x)
        # Names of Parameters for each model name
        self.ModelName2ParamNames = {"exponential":["A","beta"],
                                     "uniform":["A"],
                                    "normal":["A","mu","sigma"],
                                    "maxwellian":["A","mu","sigma"],
                                    "powerlaw":["A","alpha"]}
        
        # Names Models
        self.ModelName2Parmas = {"exponential":pm.Exponential,
                                "uniform":pm.Uniform,
                                "normal":pm.Normal,
                                "maxwellian":pm.Maxwellian,
                                "powerlaw":pm.PowerLaw}
        # Parameters of the model {ModelName2ParamNames[ModelName]:{param_name:param_value}}
        self.ParamModel = defaultdict(list)
        # Each variable to fit with all the models informations 
        self.VariablesFit2ParamsPrior = None
        if ModelToFit in self.ModelName2Parmas.keys():
            self.ModelToFit = ModelToFit
        else:
            raise ValueError("ModelName not in the list of available models (exponential,uniform,normal,maxwellian,powerlaw)")



## PREPROCESSING

    def CutYPercentOnMax(self,percentile):
        """
            @brief: Cut the x,y values that are above the percentile on the y values
            x = [1,2,3,4,5]
            y = [10,1,5,100,7] -> percentile = 90 -> x = [1,2,3,5], y = [10,1,5,7]

        """
        self.y = self.y[np.where(self.y < np.percentile(self.y,percentile))]
        self.x = self.x[np.where(self.y < np.percentile(self.y,percentile))]
    
    def CutXPercentOnMax(self,percentile):
        """
            @brief: Cut the x,y values that are above the percentile on the x values
            x = [1,2,3,4,5]
            y = [10,1,5,100,7] -> percentile = 90 -> x = [1,2,3,4], y = [10,1,5,100] 
            NOTE: Cut the indices for which x is above the percentile
        """
        self.y = self.y[np.where(self.x < np.percentile(self.x,percentile))]
        self.x = self.x[np.where(self.x < np.percentile(self.x,percentile))]

    def CutOnValueX(self,value):
        """
            @brief: Cut the x,y values that are above the value on the x values
            x = [1,2,3,4,5]
            y = [10,1,5,100,7] -> value = 3 -> x = [1,2,3], y = [10,1,5]
            NOTE: Cut the indices for which x is above the value
        """
        self.y = self.y[np.where(self.x < value)]
        self.x = self.x[np.where(self.x < value)]
    
    def CutOnValueY(self,value):
        """
            @brief: Cut the x,y values that are above the value on the y values
            x = [1,2,3,4,5]
            y = [10,1,5,100,7] -> value = 10 -> x = [1,2,3,4], y = [10,1,5,7]
            NOTE: Cut the indices for which y is above the value
        """
        self.y = self.y[np.where(self.y < value)]
        self.x = self.x[np.where(self.y < value)]
# MODEL
    def InitParam(self,name,prior_name,initial_condition):
        """
            @brief: Initialize the parameters of the model
            @param name: The name of the parameter
            @param prior_name: The name of the prior
            @param initial_condition: The initial condition of the parameter
        """
        if prior_name == "uniform":
            assert len(initial_condition) == 2, f"The initial condition must be a tuple with two elements, given: {len(initial_condition)}"
            assert initial_condition[0] < initial_condition[1], f"The lower bound must be less than the upper bound, given: {initial_condition}"
            self.ParamModel[name] = self.ModelName2Parmas[self.ModelToFit](name,lower=initial_condition[0],upper=initial_condition[1])
        elif prior_name == "exponential":
            self.ParamModel[name] = self.ModelName2Parmas[self.ModelToFit](name,value=initial_condition)
        elif prior_name == "normal":
            assert len(initial_condition) == 2, f"The initial condition must be a tuple with two elements, given: {len(initial_condition)}"
            self.ParamModel[name] = self.ModelName2Parmas[self.ModelToFit](name,mu=initial_condition[0],sigma=initial_condition[1])

    def InitPriorParams(self,name_variable2fit,list_prior_parameters,list_initial_conditions,list_name_parameters,list_names_priors):
        """

            @param args: The parameters of the prior (It is a variable number since each model has different number of parameters)
            NOTE: Each arg of args = []
            @description: For each distribuution I want a set of parameters, and their respective initial conditions
            The input must be  something like:
            ["prior_name",[param1,param2,param3],initial_conditions]
            NOTE: According to prior name we have different initial conditions.
            NOTE: VariablesFit2ParamsPrior is used to store informations about each fitting procedure 
        """
        # betas = pm.Uniform("betas", 0, 1, size=number_parameters)  if they are in the same domain
        # Check how many parameters I have and how they are distributed
        assert len(list_names_priors) == len(list_prior_parameters) == len(list_initial_conditions) == len(list_name_parameters), "The number of priors is not the same as the number of parameters"
        if self.VariablesFit2ParamsPrior is None:
            self.VariablesFit2ParamsPrior = defaultdict()
            self.VariablesFit2ParamsPrior[name_variable2fit] = defaultdict()
        for i in range(len(list_prior_parameters)):
            if list_names_priors[i] == "uniform":
                assert len(list_initial_conditions[i]) == 2, "The number of initial conditions is not the same as the number of parameters"
                par_min = list_initial_conditions[i][0]
                par_max = list_initial_conditions[i][1]
                self.VariablesFit2ParamsPrior[name_variable2fit][list_name_parameters[i]] = pm.Uniform(list_names_priors[i],par_min,par_max)
            if list_names_priors[i] == "exponential":
                self.VariablesFit2ParamsPrior[name_variable2fit][list_name_parameters[i]] = pm.Exponential(list_name_parameters[i],value=list_initial_conditions[i])
            self.VariablesFit2ParamsPrior[name_variable2fit][list_name_parameters[i]] = pm.stochastic(list_names_priors[i],list_prior_parameters[i],value=list_initial_conditions[i])

            pm.stochastic(list_prior_names[i],list_prior_parameters[i],value=list_initial_conditions[i])
        if isinstance(distribution_prior_parameters,list) or isinstance(distribution_prior_parameters,tuple) or isinstance(distribution_prior_parameters,np.ndarray):
            number_parameters = len(distribution_prior_parameters)
        else:
            pass
        assert len(args) == number_parameters, "The number of parameters is not the same as the number of parameters of the model"
        for i in range(number_parameters):

        if self.ModelName == "exponential":
            self.alpha = self.ModelName2Parmas[self.ModelName]("alpha",value=0)
            self.beta = self.ModelName2Parmas[self.ModelName]("beta",value=0)
            self.sigma = self.ModelName2Parmas[self.ModelName]("sigma",value=1)
        self.alpha = self.ModelName2Parmas[self.ModelName]("alpha",value=0)
        self.beta = self.ModelName2Parmas[self.ModelName]("beta",value=0)
        self.sigma = self.ModelName2Parmas[self.ModelName]("sigma",value=1)

    def GenerateModel(self,params):
        """
            @brief: Initialize the model
        """

        self.Model = pm.Model()

    def ExponentialModel(self):
        @stochastic(observed=False)
        def Tau

    def TensorProduct(self):
        if np.shape(self.x) == np.shape(self.y):
            self.xy = np.dot(self.x[:,None],self.y[None,:])
        

    