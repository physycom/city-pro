"""
    This file contains debuggin tools and plots related to heterogeneity analysis.
    Heterogeneity.py
"""
from matplotlib.pyplot import subplots,show 
from numpy import log
from os.path import join
from polars import DataFrame

def Plot_linear_extrapolation_xmax_x_k(x_ks,
                                       ks,
                                       x_max,
                                       alpha,
                                       x_label,
                                       y_label,
                                       PlotDir,
                                       NameFig,
                                       SaveFig = False,
                                       ):
    """
        @brief:
            - This method is useful to plot the linear extrapolation of the x_k and x_max.
        @param x_ks: list
        @param ks: list
        @param x_max: float
        The identity we look for is:
            k^(-alpha) = <x>_k/<x>_max -> - alpha*log(k) = log(<x>_k) - log(<x>_max)
        @param alpha: float
        @param x_label: str
        @param y_label: str
        @param PlotDir: str
        @param NameFig: str
        @param SaveFig: bool
        @return:
            - DataFrame({"log_k":logk,"logx":logx,"y":y}),alpha
            
    """
    assert len(x_ks) == len(ks), "The length of x_ks and ks must be the same."
    assert isinstance(x_max,float) or isinstance(x_max,int), f"x_max must be float or int. {x_max} given."
    assert isinstance(alpha,float) or isinstance(alpha,int), f"alpha must be float or int. {alpha} given."
    fig,ax = subplots()
    logk = log(ks)
    logx = log(x_ks)
    logx_max = log(x_max)
    slope = -alpha
    y = slope*logk + logx_max
    ax.scatter(logk,logx)
    ax.plot(logx,y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if SaveFig:
        fig.savefig(join(PlotDir,NameFig +".png"))
    return DataFrame({"log_k":logk,"logx":logx,"y":y}),alpha
    
