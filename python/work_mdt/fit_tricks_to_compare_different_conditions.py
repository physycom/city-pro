import numpy as np

def _iteratively_diminish_bins_until_no_0_from_end(_v_data,_y,_x,_bins):
    """
        NOTE: This function must be used to avoid 0s in the bins when we want to 
        have this distribution to be fitted with powerlaw or exponential.
        Indeed, it happens, that these cases are not well defined when we have 0s in the bins or values.
        Indeed, the fit is done usually with linear functions with logarithm in the y-axis and x-axis.
    """
    is_diminished = False
    while 0 in _y:
        _bins -= 1
        _y,_x = np.histogram(_v_data,bins=_bins)
        is_diminished = True
        
    if is_diminished:
        return _y,_x[1:]
    else:
        return _y,_x

def enrich_vector_to_length(vector, target_length):
    if len(vector) >= target_length:
        return vector[:target_length]
    
    # Ensure we always keep first and last elements
    if target_length >= 2:
        result = [vector[0]]  # First element
        
        # Calculate how many points to insert between each pair
        remaining_points = target_length - 2  # -2 for first and last elements
        segments = len(vector) - 1
        points_per_segment = [remaining_points // segments] * segments
        
        # Distribute any remainder points
        remainder = remaining_points % segments
        for i in range(remainder):
            points_per_segment[i] += 1
            
        # Generate the interpolated points
        for i in range(len(vector) - 1):
            a, b = vector[i], vector[i + 1]
            for j in range(points_per_segment[i]):
                t = (j + 1) / (points_per_segment[i] + 1)
                result.append(a * (1 - t) + b * t)
                
        result.append(vector[-1])  # Last element
        return result
    else:
        return vector[:target_length]

### ---- HANDLE DISTIRBUTIONS ---- ###
def from_feature_to_nin_bin_range(Feature,range_time_hours,bin_size_time_hours,range_length_km,bin_size_length_km):
    """
        @params Feature: str: name of the feature.
        @params range_time_hours: tuple: range of the time in hours.
        @params bin_size_time_hours: float: size of the bins for the time in hours.
        @params range_length_km: tuple: range of the length in km.
        @params bin_size_length_km: float: size of the bins for the length in km.
        @description:
            Computes the number of bins and the range of the bins for the feature.
    """
    if Feature == "time_hours":
        bin_range = range_time_hours
        bins = int((bin_range[1] - bin_range[0])/bin_size_time_hours)
        print("bin_size: ",bin_size_time_hours)
    elif Feature == "lenght_km":
        bin_range = range_length_km
        bins = int((bin_range[1] - bin_range[0])/bin_size_length_km)
        print("bin_size: ",bin_size_length_km)
    return bins,bin_range

def from_data_to_cut_distribution(FcmNormalUsersClass,bins,bin_range):
    """
        @params FcmNormalUsersClass: list: list of values of the feature.
        @params bins: int: number of bins.
        @params
        @description:
            Computes the distribution of the feature for the class and the fit.
            - the bins are computed in the range of the feature
    """
    n,x = np.histogram(FcmNormalUsersClass,bins = bins,range = bin_range)
    x = x[:]
    n,x = _iteratively_diminish_bins_until_no_0_from_end(FcmNormalUsersClass,n,x,len(x))
    return x,n

def from_data_2_distributions(FcmNormalUsersClass,Feature,bin_size_time_hours,bin_size_length_km,range_time_hours = (0.1,2),range_length_km = (0.1,10),enriched_vector_length = 50):
    """
        @params FcmNormalUsersClass: list: list of values of the feature.
        @params Feature: str: name of the feature.
        @params bin_size_time_hours: float: size of the bins for the time in hours.
        @params bin_size_length_km: float: size of the bins for the length in km.       NOTE: Cutting
        @params range_time_hours: tuple: range of the time in hours.                    NOTE: The cut is done here.
        @params range_length_km: tuple: range of the length in km.                      NOTE: The cut is done here.
        @params enriched_vector_length: int: length of the enriched vector.             NOTE: size of the final vector of the probability distribution
        @description:
            Computes the distribution of the feature for the class and the fit.
            - the bins are computed in the range of the feature
    """
    # Choose the range for the bins
    bins,bin_range = from_feature_to_nin_bin_range(Feature,range_time_hours,bin_size_time_hours,range_length_km,bin_size_length_km)
    print("number bins: ",bins)
    print("bin_range: ",bin_range)
    x,n = from_data_to_cut_distribution(FcmNormalUsersClass,bins,bin_range)
    print("x: ",x)
    print("n: ",n)
    A_exp,beta_exp,exp_,error_exp,R2_exp,A_pl,alpha_pl,pl_,error_pl,R2_pl,bins_plot = compare_exponential_power_law_from_xy(x,n)
#    x = np.array(enrich_vector_to_length(x, enriched_vector_length))
    x_mean = np.nanmean(FcmNormalUsersClass)
#    if Feature == "time_hours":
#        n = np.array(gaussian_filter1d(enrich_vector_to_length(n, enriched_vector_length), 3))
#    else:
#        n = np.array(gaussian_filter1d(enrich_vector_to_length(n, enriched_vector_length), 3))            
    exp_ = A_exp*np.exp(beta_exp*x)
    pl_ = A_pl*x**alpha_pl
    assert len(x) == len(n) == len(exp_) == len(pl_), f"The vectors must have the same length: x {len(x)}, n {len(n)}, exp_ {len(exp_)}, pl_ {len(pl_)}"
    if error_exp < error_pl:
        return x,n,x_mean,A_exp,beta_exp,exp_,True 
    else:
        return x,n,x_mean,A_pl,alpha_pl,pl_,False

