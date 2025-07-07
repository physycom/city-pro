import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

class Distribution:
    """
        Handles distributions using numpy functions.
        @param _v_data: numpy array
        @param name: str (is the name of the feature)
        @param cut_x: dict ({"bottom":None,"top":None})
        @param bins: int
        @param filter_name: str (gaussian,savitzky_golay)
        @param params: float or list
        @param choose_bins_without_0: bool (if True, it will iteratively diminish the bins until no 0s are found)
        NOTE: This structure is valid just for one dimensional data.
    """
    def __init__(self, data,name,bins=100,filter_name = None,params = None,choose_bins_without_0 = False):
        self._v_data = data
        self.name = name
        self._bins = bins
        self._y,self._x = np.histogram(self._v_data,bins=self._bins)
        self._x_size_y = self._x[:-1]
        self._Z: float = None
        self._norm_y: bool = False
        self._str_norm_y = None
        self._filter_name = filter_name
        self._params = params
        self._choose_bins_without_0 = choose_bins_without_0
        self._polish_v_data()
        self._smooth_v_data(filter_name,params)
        if self._choose_bins_without_0:
            self._iteratively_diminish_bins_until_no_0_from_end()
        self._normalize_y()


    def _cut_x(self,
               cut_x = {"bottom":None,"top":None}):
        """
            @param cut_x: dict ({"bottom":None,"top":None})
        """
        self._cut_x = cut_x
        self._is_cut = True        

    def _polish_v_data(self):
        """
            1- Sorts the data and removes NaN values.
            2- Applies the cut, controlling for NaN values. (if found,invalid cut, and nedglect it)
        """
        if isinstance(self._v_data,np.ndarray):
            self._v_data = np.sort(self._v_data)
            self._v_data = self._v_data[~np.isnan(self._v_data)]
        else:
            raise NotImplementedError("Data must be a numpy array.")
        if isinstance(self._cut_x,np.ndarray) or isinstance(self._cut_x,list) or isinstance(self._cut_x,tuple):
            # Sort the cut_x and remove NaN values.
            self._cut_x = np.sort(self._cut_x)
            self._cut_x = self._cut_x[~np.isnan(self._cut_x)]
            if len(self._cut_x) >= 1:
                self._x = self._x[self._cut_x[0]<self._x]
            
            self._x = self._x[self._x<self._cut_x[1]]
        else:
            pass

    def _iteratively_diminish_bins_until_no_0_from_end(self):
        """
            NOTE: This function must be used to avoid 0s in the bins when we want to 
            have this distribution to be fitted with powerlaw or exponential.
            Indeed, it happens, that these cases are not well defined when we have 0s in the bins or values.
            Indeed, the fit is done usually with linear functions with logarithm in the y-axis and x-axis.
        """
        while 0 in self._y:
            self._bins -= 1
            self._y,self._x = np.histogram(self._v_data,bins=self._bins)
            self._x_size_y = self._x[:-1]
        return self._y,self._x
    

    def _probability_density(self):
        """
            Returns the probability density of the data.
            That is:
                \int_dx y(x) = 1
        """
        dx = self._x[1]-self._x[0]
        self._y_density = self._y/np.sum(self._y*dx)
        return self._y_density
    
    def _normalize_y(self):
        """
            Normalizes the y values.
        """
        self._y_norm = self._y/np.sum(self._y)
        self._norm_y = True
        self._str_norm_y = "Normalized"
        return self._y_norm
    
    def _smooth_v_data(self,filter_name,params):
        """
            Smooths the data.
        """
        if filter_name == "gaussian":
            assert isinstance(params,float) or isinstance(params,int), f"params must be float or int. {params} given."
            self._y = gaussian_filter1d(self._y, sigma=params)
        elif filter_name == "savitzky_golay":
            assert isinstance(params,list) or isinstance(params,np.ndarray) or isinstance(params,tuple), f"params must be list,array or tuple. {params} given."
            assert len(params) == 2, f"params must have length 2. {len(params)} given."
            assert isinstance(params[0],int), f"windows size must be int or float. {params[0]} given."
            assert isinstance(params[1],int), f"polynomial order must be int or float. {params[1]} given."
            window_size = params[0]
            polynomial_order = params[1]
            self._y = savgol_filter(self._y, window_size, polynomial_order)
        else:
            pass

    def _mean(self):
        """
            Returns the mean of the data.
        """
        return np.mean(self._v_data)
    def _std(self):
        """
            Returns the standard deviation of the data.
        """
        return np.std(self._v_data)
if __name__ == "__main__":
    seed = np.random.seed(0)
    data = np.random.exponential(scale=1.0,size=1000)
    dist_gaussian_filtered = Distribution(data,"exponential_generated",bins=100,filter_name="gaussian",params=0.5)
    dist_solay_filtered = Distribution(data,"exponential_generated",bins=100,filter_name="savitzky_golay",params=[5,2])