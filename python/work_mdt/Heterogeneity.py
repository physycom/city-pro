"""
    This script is useful to explain power law as heterogenous sum of exponentials.
    The idea is the following:
    - Look at distribution of a given feature conditional to ORDERED type of populations:
        P(x|k) \sim e^{-x/x_k}
    - The size of the average feature is linearly related to the class:
        <x>_k = k^a*<x>_max
    INPUT:
        - distributions: x,P(x|k)

    PIPELINE:

        
"""
import numpy as np
import polars as pl
from Distributions import *
from FittingProcedures import *
from HeterogeneityPlot import *


def split_equally_P_given_bins(x,y,x_new):
    """
        @params start_bins: vector bins
        @params end_bins: vector bins
        @params y: vector of values
        @return y_split: vector of values 
        NOTE:
            This function is the realization of the intersection function for a couple of 1D intervals 
            weighted by the distribution value. 
    """
    dx = x[1] - x[0]
    dx_new = x_new[1] - x_new[0]
    y_split = np.zeros(len(x_new))
    for j_n in range(len(x_new)-1):
        for i in range(len(x)-1):
            # Intersection conditions
            dx_i_jn = x[i] - x_new[j_n]
            dx_iplus1_jnplus1 = x[i+1] - x_new[j_n + 1]
            # Null intersection conditions   
            dx_i_jnplus1 = x[i] - x_new[j_n+1]
            dx_iplus1_jn = x[i+1] - x_new[j_n]
            null_intersection = (dx_iplus1_jn < 0) or (dx_i_jnplus1 > 0)
            if null_intersection:
                pass                                                            # NOTE: if the intersection of the old bins to the new bins is 0 then I do not need to pass anything in distribution terms
            else:
                # j|---i|----j+1|---i+1|
                if dx_i_jn > 0 and dx_iplus1_jnplus1 > 0:
                    fraction = (x_new[j_n + 1] - x[i])/dx
                    y_split[j_n] = y[i]*fraction
                # j|---i|----i+1|---i|
                elif dx_i_jn > 0 and dx_iplus1_jnplus1 < 0:
                    y_split[j_n] += y[i]
                # i|---j|----j+1|---i+1|
                elif dx_i_jn < 0 and dx_iplus1_jnplus1 > 0:
                    fraction = dx/dx_new
                    y_split[j_n] = y[i]*fraction
                # i|---j|----i+1|---j+1|
                elif dx_i_jn < 0 and dx_iplus1_jnplus1 < 0:
                    fraction = (x_new[j_n] - x[i + 1])/dx
                    y_split[j_n] += y[i]
    return y_split

def map_distributions_of_different_binnings_to_same_bin(x_classes,y_classes,bin_size = 50):
    """
        @params x_classes: np.array: x values of the classes.
        @params y_classes: np.array: y values of the classes.
        @params bin_size: int: number of bins to be used in the histogram.
        @return matrix_x_bins: np.array: matrix of x bins.
                                 (the first axis is the number of classes,
                                 the second axis is the number of bins) 
                                -> the bins are shared by all classes.
        @return matrix_y_bins: np.array: matrix of y bins.
    """
    if isinstance(x_classes,np.ndarray) and isinstance(y_classes,np.ndarray):
        pass
    else:
        x_classes = np.array(x_classes)
        y_classes = np.array(y_classes)
    # number of indices array
    tensor_dim_x = len(x_classes.shape)
    tensor_dim_y = len(y_classes.shape)
    # get maximum range on the x axis
    x_max = x_classes.reshape(-1).max()
    x_min = x_classes.reshape(-1).min()
    # generate new bins
    x_bins = np.linspace(x_min,x_max,bin_size)
    # 1D case
    if tensor_dim_x == 1 and tensor_dim_y == 1:
        matrix_new_bins_y = split_equally_P_given_bins(x_classes,y_classes,x_bins)
        matrix_new_bins_x = x_bins
        assert matrix_new_bins_x.shape == matrix_new_bins_y.shape, "The shapes of the new bins must be equal."
        return matrix_new_bins_x,matrix_new_bins_y
    #2D case
    elif tensor_dim_x == 2 and tensor_dim_y == 2:
        # new vector cotaining new distributions
        matrix_new_bins_y = np.empty((tensor_dim_x,len(x_bins)))
        matrix_new_bins_x = np.empty((tensor_dim_y,len(x_bins)))
        for class_idx in range(x_classes.shape[0]):
            matrix_new_bins_x = np.append(matrix_new_bins_x,x_bins)
            y_tmp = np.zeros(bin_size)
            # generate the new distribution for the class
            y_tmp = split_equally_P_given_bins(x_classes[class_idx],y_classes[class_idx],matrix_new_bins_x[class_idx])
            matrix_new_bins_y = np.append(matrix_new_bins_y,y_tmp)
            assert y_tmp.shape == x_bins.shape, "The shapes of the new bins must be equal."
        assert matrix_new_bins_x.shape == matrix_new_bins_y.shape, "The shapes of the new bins must be equal."
        return matrix_new_bins_x,matrix_new_bins_y 
    elif tensor_dim_x != tensor_dim_y:
        raise ValueError("The tensor dimensions must be equal.")
    else:
        raise ValueError("The tensor dimensions must be at most 2.")        

def enrich_vector_to_length(vector, target_length):
    """
        @params vector: np.array: vector to be enriched.
        @params target_length: int: target length of the vector.
        @return vector: np.array: enriched vector.
        Transform the vector in a vector of length target_length.
        It adds iteratviely poits that are in the middle of the vector
    """
    while len(vector) < target_length:
        new_vector = []
        for i in range(len(vector) - 1):
            new_vector.append(vector[i])
            new_vector.append((vector[i] + vector[i + 1]) / 2)
        new_vector.append(vector[-1])
        vector = new_vector[:target_length]
    return vector
class HeterogeneousPowerLaw:
    """
        @brief:
            - This class is useful to explain power law as heterogenous sum of exponentials.

        @param data_used: pl.DataFrame: 
            - generic dataframe that must contain:
                1 column_feature
                2 column_order
        @param column_feature: str: column of the feature to be analyzed.
        @param column_order: str: column of the order of the feature to be analyzed.
        @param cut_x: dict: {"bottom":float,"top":float} NOTE: default is None. -> No cut.
        @bins: int: number of bins to be used in the histogram. NOTE: default is 100.


        @return:
            - fit a: <x>_k = k^a*<x>_max
    """
    def __init__(self,
                 data,
                 tab_exponential_parameters,
                 column_feature,
                 column_order,
                 PlotDir,
                 NameFig,
                 cut_x = {"bottom":None,
                          "top":None},
                 bins = 100):
        self._data = data
        self._column_feature = column_feature
        self._column_order = column_order
        # Day,Class,<x>,beta,error_exp,R2
        self._xks = tab_exponential_parameters["<x>"].to_numpy()
        self._cut_x = cut_x
        self._bins = bins
        # These are the classes of the data
        self._classes = self._data[self._column_order].unique().to_numpy()
        if 0 in self._classes:
            self._ks = self._classes + 1
        else:
            self._ks = self._classes
        self.PlotDir = PlotDir
        self.NameFig = NameFig
        self._asserts()
        self._cut_data()
        self._get_distribution()



    def _cut_data(self):
        """
            @brief:
                - This method is useful to filter the data.
            @param cut_x: dict: {"bottom":float,"top":float}
            NOTE: The cut must bet  
        """
        # 
        self._is_cut_top = False
        self._is_cut_bottom = False
        self._is_cut = False
        self._data_used = self._data.filter(pl.col(column_feature).is_not_null(),
                              pl.col(column_order).is_not_null())
        # cut lowest values
        if self._cut_x["bottom"] is not None:
            self._data_used = self._data_used.filter(
                                pl.col(column_feature)>self._cut_x["bottom"])
            self._is_cut_bottom = True
        # cut highest values
        if self._cut_x["top"] is not None:
            self._data_used = self._data_used.filter(
                                pl.col(column_feature)<self._cut_x["top"]
                                )
            self._is_cut_top = True
        if self._is_cut_top or self._is_cut_bottom:
            self._is_cut = True
        else:
            self._data_used = self._data  

    def _get_distribution(self):
        """
            @brief:
                - This method is useful to get the distribution of the feature.
        """

        self._distribution = Distribution(data = self.data_used[self._column_feature].to_numpy(),
                                          name = self._column_feature,
                                          bins = self._bins)
        self._x = self._distribution._x
        self._y_norm = self._distribution._normalize_y()
        self._y_density = self._distribution._probability_density()
        del self._distribution

    def _get_xmax(self):
        """
            @brief:
            - extract the highest feature from the parameters.
            NOTE: The parameters fit: e(ax), therefore:
            <x> = - 1/a
            _x_max_bar = 1/a
        """
        max_parameter = -1
        for k in self._exponential_parameter_data.keys():
            if self._column_order_2_exponential_parameter[k] > max_parameter:
                max_parameter = 1/self._column_order_2_exponential_parameter[k]
                self.class_max_characteristic = k
        self._x_max_bar= max_parameter
        return max_parameter

    def _compute_best_fit_x_max_x_k(self):
        """
            @brief:
                Computes:


        """
        from scipy.optimize import curve_fit
        logk = np.log(self._ks) 
        logx = np.log(self._xks)
        self._logx_max = np.log(self._x_max_bar)
        fit = curve_fit(lambda x,a,b: a*x + b,logk,logx)
        self._alpha = fit[0][0]
        self._A = fit[0][1]


    def _plot_linear_extrapolation_xmax_x_k(self,x_label,y_label):
        """
            @brief:
                - This method is useful to plot the linear extrapolation of the x_k and x_max.
        """
        Plot_linear_extrapolation_xmax_x_k(self._xks,
                                    self._ks,
                                    max(self._xks),
                                    self._alpha,
                                    x_label,
                                    y_label,
                                    self.PlotDir,
                                    self.NameFig,
                                    SaveFig = False,
                                    )


    def _asserts(self):
        """
            @brief:
                Makes sure that the input data is correct.
        """
        assert isinstance(self.data_used,pl.DataFrame), "data_used must be a pandas DataFrame."
        assert isinstance(self.column_feature,str), "column_feature must be a string."
        assert isinstance(self.column_order,str), "column_order must be a string."
        assert np.isnan(self._cut_x["bottom"]), "cut_x[bottom] must be a float."
        assert np.isnan(self._cut_x["top"]), "cut_x[top] must be a float."
        #assert np.array([k for k in self._column_order_2_exponential_parameter.keys()]) == self._data[self._column_order].unique().to_numpy(), "column_order_2_exponential_parameter must be a dictionary with keys equal to the unique values of column_order."
if __name__ == "__main__":
    feature_2_ylabel_linear_extr = {"length_km":r"log($\langle lenght (km) \rangle_k$)",
                                    "time_hour":r"log($\langle time (h) \rangle_k$)",}
    column_feature = "length_km"
    
    column_order = "class"
    cut_x = {"bottom":0.1,"top":10}
    x_label = "log(k)"
    y_label = "log(<x>_k)"
    Fcm = pl.read_csv("/path/to/fcm/fcm.csv")

    hpl = HeterogeneousPowerLaw(Fcm,
                          column_feature,
                          column_order,
                          cut_x = cut_x,
                          bins = 100
                        )
    hpl._plot_linear_extrapolation_xmax_x_k(x_label,
                                            feature_2_ylabel_linear_extr[column_feature])
    pass