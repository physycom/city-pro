from lmfit import Model
import numpy as np
import pandas as pd

from scipy.stats import lognorm,gengamma
#from scipy.special import gammainc


# NOTE: the shift is done to avoid the problem of the log of zero

#Weibull
def cdf_weib_analytic(x,a,b):
    return  1-np.exp(-(x/a)**b)

def cdf_weib_analytic_shift(x,a,b,minX):
    shift = cdf_weib_analytic(minX,a,b)
    if shift < 1:
        return  (cdf_weib_analytic(x,a,b) - shift) / (1. - shift)
    else:
        return 0
    
    
#can I do this directly on the log?

#Lognormal
def cdf_logn_analytic_shift(x,s,scale,minX):
    shift = lognorm.cdf(minX, s, 0, scale)
    if shift < 1:
        return  (lognorm.cdf(x, s, 0, scale) - shift) / (1. - shift)
    else:
        return 0

#Random Acceleration
def cdf_randacc_analytic_shift(x,scale,minX):
    shift = gengamma.cdf(minX, 0.5, 0.5, 0, scale)
    if shift < 1:
        return  (gengamma.cdf(x, 0.5, 0.5, 0, scale) - shift) / (1. - shift)
    else:
        return 0
    
    
#note: this can be extended to   
def cdf_geng(x,a,scale):
    return gengamma.cdf(x,a,0.5, 0, scale)


def cdf_geng_numerical(x,a,scale,maxX=5000,dx=0.1):
    x_v = np.arange(dx,maxX,dx)
    y = x_v**(a*0.5-1) * np.exp(-(x_v/scale)**0.5)
    yy = np.cumsum(y)/np.sum(y)
    return np.array([yy[int((xx-dx)/dx)] for xx in x])

def cdf_geng_shift(x,a,scale,minX):
    shift = gengamma.cdf(minX, a, 0.5, 0, scale)
    if shift < 1:
        return  (gengamma.cdf(x, a, 0.5, 0, scale) - shift) / (1. - shift)
    else:
        return 0
    
def cdf_geng_numerical_shift(x,a,scale,minX):
    shift = cdf_geng_numerical([minX],a,scale)
    if shift < 1:
        return  (cdf_geng_numerical(x,a,scale) - shift) / (1. - shift)
    else:
        return 0

#TPL (done numerically, if not we don't have all exponents)
def cdf_tpl_numerical(x,a,scale,maxX=5000,dx=0.1):
    x_v = np.arange(dx,maxX,dx)
    y = x_v**(a-1) * np.exp(-x_v/scale)
    yy = np.cumsum(y)/np.sum(y)
    return np.array([yy[int((xx-dx)/dx)] for xx in x])


def cdf_tpl_numerical_shift(x,a,scale,minX):
    shift = cdf_tpl_numerical([minX],a,scale)
    if shift < 1:
        return  (cdf_tpl_numerical(x,a,scale) - shift) / (1. - shift)
    else:
        return 0

    
#TPL_3par (done numerically, if not we don't have all exponents)
def cdf_tpl_numerical_3par(x,a,scale,r0,maxX=5000,dx=0.1):
    x_v = np.arange(dx,maxX,dx)
    y = (x_v+r0)**(a-1) * np.exp(-x_v/scale)
    yy = np.cumsum(y)/np.sum(y)
    return np.array([yy[int((xx-dx)/dx)] for xx in x])

def cdf_tpl_numerical_shift_3par(x,a,scale,r0,minX):
    shift = cdf_tpl_numerical_3par([minX],a,scale,r0)
    if shift < 1:
        return  (cdf_tpl_numerical_3par(x,a,scale,r0) - shift) / (1. - shift)
    else:
        return 0

def test_4curves_CCDF(ccdf_x,ccdf_y,minX=0,max_gga=6):
        

    minL = minX

    #def ccdf_weib_analytic_shift_log(x,a,b):
    #    return  np.nan_to_num(np.log(1-cdf_weib_analytic_shift(x,a,b,minL)))
    
    def ccdf_weib_analytic_shift_log_simple(x,a,b):
        return  -(x/a)**b + (minL/a)**b 

    def ccdf_logn_analytic_shift_log(x,s,scale):
        return  np.nan_to_num(np.log(1-cdf_logn_analytic_shift(x,s,scale,minL)))
    
    def ccdf_randacc_analytic_shift_log(x,scale):
        return  np.nan_to_num(np.log(1-cdf_randacc_analytic_shift(x,scale,minL)))

    def ccdf_geng_analytic_shift_log(x,a,scale):
        return  np.nan_to_num(np.log(1-cdf_geng_shift(x,a,scale,minL)))
    
    def ccdf_geng_numerical_shift_log(x,a,scale):
        return  np.nan_to_num(np.log(1-cdf_geng_numerical_shift(x,a,scale,minL)))
    
    def ccdf_tpl_numerical_shift_log(x,a,scale):
        return  np.nan_to_num(np.log(1-cdf_tpl_numerical_shift(x,a,scale,minL)))
    
    def ccdf_tpl_numerical_shift_3par_log(x,a,scale,r0):
        return  np.nan_to_num(np.log(1-cdf_tpl_numerical_shift_3par(x,a,scale,r0,minL)))
    
    #Random Acceleration
    ra_ccdf_model = Model(ccdf_randacc_analytic_shift_log)
    ra_ccdf_model.set_param_hint('scale', min=0,max=1000)
    result_fit_ra = ra_ccdf_model.fit(np.log(ccdf_y),x=ccdf_x,scale=10)

    #Gengamma
    gg_ccdf_model = Model(ccdf_geng_numerical_shift_log)
    gg_ccdf_model.set_param_hint('scale', min=0,max=100000)
    gg_ccdf_model.set_param_hint('a',     min=-10, max=max_gga)
    result_fit_gg = gg_ccdf_model.fit(np.log(ccdf_y),x=ccdf_x,a=0.5,scale=50)
    
    #Lognormal
    logn_ccdf_model = Model(ccdf_logn_analytic_shift_log)
    logn_ccdf_model.set_param_hint('scale', min=1e-10, max=200)
    logn_ccdf_model.set_param_hint('s',     min=1e-10, max=20)
    result_fit_ln = logn_ccdf_model.fit(np.log(ccdf_y),x=ccdf_x,scale=10,s=1)

    #Weibull √
    weib_ccdf_model = Model(ccdf_weib_analytic_shift_log_simple)
    weib_ccdf_model.set_param_hint('a', min=0, max=100)
    weib_ccdf_model.set_param_hint('b', min=0.05, max=2)
    result_fit_wb = weib_ccdf_model.fit(np.log(ccdf_y),x=ccdf_x,a=1,b=0.5)
    
    #TPL 
    tpl2_ccdf_model = Model(ccdf_tpl_numerical_shift_log)
    tpl2_ccdf_model.set_param_hint('a',     min=-10, max=10)
    tpl2_ccdf_model.set_param_hint('scale', min=0, max= 1e8)
    result_fit_tpl2 = tpl2_ccdf_model.fit(np.log(ccdf_y),x=ccdf_x,a=-0.5,scale=200)

    #TPL3 
    tpl3_ccdf_model = Model(ccdf_tpl_numerical_shift_3par_log)
    tpl3_ccdf_model.set_param_hint('a',     min=-3, max=3)
    tpl3_ccdf_model.set_param_hint('scale', min=0, max= 1e8)
    tpl3_ccdf_model.set_param_hint('r0',    min=0, max= 100)
    result_fit_tpl3 = tpl3_ccdf_model.fit(np.log(ccdf_y),x=ccdf_x,a=-0.5,scale=200,r0=5.8)
    
    #Combine results
    df_res_fit_log= {'ln':result_fit_ln,
                 'rua':result_fit_ra,
                 'wb':result_fit_wb,
                 'tpl2':result_fit_tpl2,
                 'tpl3':result_fit_tpl3,
                 'rak':result_fit_gg}


    df_akaike = pd.DataFrame(pd.Series({'ln':result_fit_ln.aic,
                                        'rua':result_fit_ra.aic,
                                        'wb':result_fit_wb.aic,
                                        'tpl2':result_fit_tpl2.aic,
                                        'tpl3':result_fit_tpl3.aic,
                                        'rak':result_fit_gg.aic}),columns=['aic'])
    df_akaike['delta'] = df_akaike['aic'] - df_akaike['aic'].min()
    df_akaike['relative_likelihood'] = df_akaike.apply(lambda x:np.exp(-x.delta/2),axis=1)
    df_akaike['weights'] = df_akaike['relative_likelihood']/df_akaike['relative_likelihood'].sum()
    df_akaike.sort_values(by='weights',ascending=False,inplace=True)
    

    #R squared
    rsq = {'ln':1-np.sum(result_fit_ln.residual**2)/np.sum((result_fit_ln.data-result_fit_ln.data.mean())**2),
                                'rua':1-np.sum(result_fit_ra.residual**2)/np.sum((result_fit_ra.data-result_fit_ra.data.mean())**2),
                                'wb':1-np.sum(result_fit_wb.residual**2)/np.sum((result_fit_wb.data-result_fit_wb.data.mean())**2),
                                'tpl2':1-np.sum(result_fit_tpl2.residual**2)/np.sum((result_fit_tpl2.data-result_fit_tpl2.data.mean())**2),
                                'tpl3':1-np.sum(result_fit_tpl3.residual**2)/np.sum((result_fit_tpl3.data-result_fit_tpl3.data.mean())**2),
                                'rak':1-np.sum(result_fit_gg.residual**2)/np.sum((result_fit_gg.data-result_fit_gg.data.mean())**2)}
    df_rsq = pd.DataFrame(pd.Series(rsq).sort_values(ascending=False),columns=['Rsquared'])
    
    #A series with all parameters
    
    res_par = {}
    res_par['wb_a'] = df_res_fit_log['wb'].values['a']
    res_par['wb_b'] = df_res_fit_log['wb'].values['b']
    res_par['wb_rsq'] = rsq['wb']

    res_par['ln_scale'] = df_res_fit_log['ln'].values['scale']
    res_par['ln_s'] = df_res_fit_log['ln'].values['s']
    res_par['ln_rsq'] = rsq['ln']
    
    res_par['tpl2_a'] = df_res_fit_log['tpl2'].values['a']
    res_par['tpl2_scale'] = df_res_fit_log['tpl2'].values['scale']
    res_par['tpl2_rsq'] = rsq['tpl2']
    
    res_par['rua_scale'] = df_res_fit_log['rua'].values['scale']
    res_par['rua_rsq'] = rsq['rua']
    
    res_par['rak_scale'] = df_res_fit_log['rak'].values['scale']
    res_par['rak_a'] = df_res_fit_log['rak'].values['a']
    res_par['rak_rsq'] = rsq['rak']
    
    res_par['tpl3_scale'] = df_res_fit_log['tpl3'].values['scale']
    res_par['tpl3_a']     = df_res_fit_log['tpl3'].values['a']
    res_par['tpl3_r0']    = df_res_fit_log['tpl3'].values['r0']

    res_par['tpl3_rsq'] = rsq['tpl3']
    
    res_par['wb_alpha'] = 1/res_par['wb_b']-1
    res_par['rak_gamma'] = 1-res_par['rak_a']/2
    
    df_par = pd.Series(res_par)
    return df_res_fit_log,df_akaike,df_rsq,df_par


def CDF_column(df_,column,minL=20,maxL = 1e6,xcolumn='length_km',minY=1e-6):
    df_ = df_[(df_[xcolumn] >= minL) & (df_[xcolumn] <= maxL)][[xcolumn,column]].sort_values(by=xcolumn)
    df_ = df_.groupby(xcolumn)[column].sum().reset_index()
    df_['cumulative'] = df_[column].cumsum()/df_[column].sum()
    x = df_[xcolumn].values
    y = df_.cumulative.values

    ind = y < 1-minY
    return x[ind],y[ind]

def CDF_column_ax(df_,ax,column,minL=0,xcolumn='length_km',curveformat='-', maxL = 1e6,ifplot=True):
    df_ = df_[(df_[xcolumn] >= minL) & (df_[xcolumn] <= maxL)][[xcolumn,column]].sort_values(by=xcolumn)
    df_ = df_.groupby(xcolumn)[column].sum().reset_index()
    df_['cumulative'] = df_[column].cumsum()/df_[column].sum()
    if ifplot:
        ax.loglog(df_[xcolumn],1-df_.cumulative,curveformat,lw=2)
    return df_[xcolumn].values,df_.cumulative.values