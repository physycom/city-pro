import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class generator_points:
    def __init__(self,size,x0,x1,function):
        self.noise = np.random.randn(size)
        self.x = np.linspace(x0,x1,size)
        self.y = self.dict_f[function](self.x) +self.noise
        
    def plot_generated_points(self):
        
class fit:
    def __init__(self,data,column1,column2):
        '''
        Initialize dataset:
            Set x and y columns
        '''
        self.data = data.dropna()
        self.size= len(self.data) 
        if len(data.columns())<2:
            print('not possible to make a fitting procedure')
        self.x = data[column1].to_numpy()
        self.y = data[column2].to_numpy()
        self.cost = 0
        self.history_cost = []
    def distance_cost(self,key):
        eucl_dist = lambda x,y: sum((np.array(x)-np.array(y))**2)
        kul_leibl = lambda x,y: sum(np.array(x)*np.log2(np.array(x)/np.array(y)))
        self.dict_fc = {'euclidean':eucl_dist,'KL':kul_leibl}
        self.cost = self.dict_fc[key](self.x,self.y)
    
    def _polynomial(self):
        
        