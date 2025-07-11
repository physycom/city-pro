import numpy as np
from fit_dictionaries import *
from fit_tricks_to_compare_different_conditions import *
from FittingProcedures import *
import polars as pl
from typing import Dict,List,Any

### ----------- FILTER FUNCTIONS ON DAtAFRAME ----------- ###
def extract_class_data(dataframe: pl.DataFrame, class_idx: int, feature: str) -> list:
    """
    Extract data for a specific class and feature from a dataframe.
    
    Args:
        dataframe: Polars DataFrame containing the data
        class_idx: Class index to filter by
        feature: Feature name to extract (e.g., 'time_hours', 'lenght_km')
    
    Returns:
        List of values for the specified feature and class
    """
    # Filter dataframe by class
    filtered_df = dataframe.filter(pl.col("class") == class_idx)
    
    # Extract the feature values
    if filtered_df.height == 0:
        return []  # Return empty list if no data for this class
    
    return filtered_df[feature].to_list()



def prepare_data_for_fitting(data, Feature, bins, bin_range, cut_class=None,class_idx = None):
    """
    Prepare data for fitting by computing histogram and applying class-specific cuts if needed.
    
    Args:
        data (list): Input data values
        Feature (str): Feature name ("time_hours" or "lenght_km")
        bins (int): Number of bins for histogram
        bin_range (tuple): Range for binning
        cut_class (dict, optional): Dictionary mapping class indices to cut thresholds
        
    Returns:
        tuple: x and y values for fitting, and x and y values for visualization
    """
    # Get histogram
    x, n = from_data_to_cut_distribution(data, bins, bin_range)
    
    # Apply cuts for length data based on class
    if "len" in Feature and cut_class is not None and class_idx in cut_class:
        cut_threshold = cut_class[class_idx]
        mask = np.array(x) > cut_threshold
        x_for_fit = np.array(x)[mask]
        y_for_fit = np.array(n)[mask]
    else:
        x_for_fit = np.array(x)
        y_for_fit = np.array(n)
        
    return x_for_fit, y_for_fit, x, n


#####################
# FITTING FUNCTIONS
#####################

def fit_distribution(x, y, class_idx=None):
    """
    Fit both exponential and power law models to data and return parameters and metrics.
    
    Args:
        x (array): X values
        y (array): Y values
        class_idx (int, optional): Class index (needed for special handling of class 0)
        
    Returns:
        tuple: Fit parameters, errors, and fitted curves
    """
    max_size = max(len(x), len(y))
    x = enrich_vector_to_length(x, max_size)
    y = enrich_vector_to_length(y, max_size)
    mask = np.array(x) > 0 
    mask1 = np.array(y) > 0
    mask = mask & mask1  # Ensure both x and y are positive 
    x = np.array(x)[mask]
    y = np.array(y)[mask] 
    # Fit both models
    A_exp, beta_exp, exp_, error_exp, R2_exp, A_pl, alpha_pl, pl_, error_pl, R2_pl, bins_plot = compare_exponential_power_law_from_xy(x, y)
    
    # Special case for class 0 - force exponential fit
    if class_idx == 0:
        error_pl = float('inf')
    
    # Select best fit based on error
    if error_exp < error_pl:
        best_model = "exp"
        A_fit = A_exp
        b_fit = beta_exp
        y_fit = exp_
    else:
        best_model = "pl"
        A_fit = A_pl
        b_fit = alpha_pl
        y_fit = pl_
    
    return {
        "exp": {"A": A_exp, "beta": beta_exp, "y_fit": exp_, "error": error_exp, "R2": R2_exp},
        "pl": {"A": A_pl, "alpha": alpha_pl, "y_fit": pl_, "error": error_pl, "R2": R2_pl},
        "best_model": best_model,
        "A_fit": A_fit,
        "b_fit": b_fit,
        "y_fit": y_fit
    }


def process_single_fit(data, Feature, range_time_hours, bin_size_time_hours, 
                     range_length_km, bin_size_length_km, enriched_vector_length, 
                     class_idx=None, cut_thresholds=None):
    """
    Process data to create a distribution, fit models, and return enhanced results.
    
    Args:
        data (list): Input data values
        Feature (str): Feature name ("time_hours" or "lenght_km")
        range_time_hours (tuple): Range for time data
        bin_size_time_hours (float): Bin size for time data
        range_length_km (tuple): Range for length data
        bin_size_length_km (float): Bin size for length data
        enriched_vector_length (int): Target length for enriched vectors
        class_idx (int, optional): Class index
        cut_thresholds (dict, optional): Dictionary mapping class indices to cut thresholds
        
    Returns:
        tuple: Enriched x and y values, fit parameters, and fitted curve
    """
    # Set up bins and range
    bins, bin_range = from_feature_to_nin_bin_range(
        Feature,
        range_time_hours, 
        bin_size_time_hours, 
        range_length_km, 
        bin_size_length_km
    )
    
    # Get raw histogram data
    x_for_fit, y_for_fit, x_raw, n_raw = prepare_data_for_fitting(
        data, Feature, bins, bin_range, cut_thresholds
    )
    
    # Perform fitting
    fit_results = fit_distribution(x_for_fit, y_for_fit, class_idx)
    
    # Enrich vectors for visualization
    n_enriched = np.array(gaussian_filter1d(enrich_vector_to_length(n_raw, enriched_vector_length), 3))
    x_enriched = np.array(enrich_vector_to_length(x_raw, enriched_vector_length))
    
    # Regenerate fit curves on enriched x values
    if fit_results["best_model"] == "exp":
        y_fit_enriched = fit_results["exp"]["A"] * np.exp(fit_results["exp"]["beta"] * x_enriched)
        b_fit = fit_results["exp"]["beta"]
        is_exp = True
    else:
        y_fit_enriched = fit_results["pl"]["A"] * x_enriched ** fit_results["pl"]["alpha"]
        b_fit = fit_results["pl"]["alpha"]
        is_exp = False
    
    # Calculate mean
    x_mean = np.nanmean(data)
    
    return x_enriched, n_enriched, fit_results["A_fit"], b_fit, y_fit_enriched, is_exp, x_mean


def update_fit_info(fit_info, dict_Lkclass, class_idx, x_mean, b_fit, is_exp, date_str):
    """
    Update the fit information dictionaries with new fit results.
    
    Args:
        fit_info (dict): Dictionary storing fit information
        dict_Lkclass (dict): Dictionary storing Lk class information
        class_idx (int): Class index
        x_mean (float): Mean value of x
        b_fit (float): Best fit parameter (beta or alpha)
        is_exp (bool): Whether exponential model was best
        date_str (str): Date string
        
    Returns:
        tuple: Updated fit_info and dict_Lkclass dictionaries
    """
    fit_info["class"].append(class_idx)
    fit_info["day"].append(date_str)
    
    if is_exp:
        fit_info["alpha"].append(None)
        fit_info["beta"].append(b_fit)
        # Update Lk class info only for exponential fits
        dict_Lkclass["class"].append(class_idx + 1)
        dict_Lkclass["day"].append(date_str)
        dict_Lkclass["Lk"].append(x_mean)
    else:
        fit_info["alpha"].append(b_fit)
        fit_info["beta"].append(None)
        
    fit_info["fit_name"].append("exp" if is_exp else "pl")
    fit_info["<x>"].append(x_mean)

    return fit_info, dict_Lkclass


def compute_average_distribution(feature, dates, class_idx, feature_data, target_length):
    """
    Compute average distribution across multiple dates for a given feature and class.
    
    Args:
        feature (str): Feature name
        dates (list): List of date strings
        class_idx (int): Class index
        feature_data (dict): Dictionary containing x, y, and mean values for each date/class
        target_length (int): Target length for result vector
        
    Returns:
        tuple: Average y values, x values, and average mean
    """
    avg_y = np.zeros(target_length)
    avg_mean = 0
    x_values = None
    
    for day in dates:
        # Get x values (we'll use the last day's x values)
        x = feature_data["x"][feature][day][class_idx]
        x_values = x
        
        # Get y values and add to average
        y = feature_data["y"][feature][day][class_idx]
        avg_y += np.array(y)
        
        # Add mean to average
        avg_mean += feature_data["mean"][feature][day][class_idx]
    
    # Calculate averages
    avg_y /= len(dates)
    avg_mean /= len(dates)
    
    return avg_y, x_values, avg_mean


def compute_single_fit_from_xy(x,n,Feature,bin_size_time_hours,bin_size_length_km,range_time_hours = (0.1,2),range_length_km = (0.1,10),enriched_vector_length = 50,class_idx = 0):
    A_exp,beta_exp,exp_,error_exp,R2_exp,A_pl,alpha_pl,pl_,error_pl,R2_pl,bins_plot = compare_exponential_power_law_from_xy(x,n)
    # 6) enrich the bins for better figures
    n = np.array(gaussian_filter1d(enrich_vector_to_length(n, enriched_vector_length), 3))
    x = np.array(enrich_vector_to_length(x, enriched_vector_length))
    # 7) generate the fitted distributions
    exp_ = A_exp*np.exp(beta_exp*x)
    pl_ = A_pl*x**alpha_pl
    # 8) add ad hoc condition on class 0 to force the exponential fit
    if class_idx == 0:
        error_pl = 1000000
    # 9) choose the best fit 
    assert len(x) == len(n) == len(exp_) == len(pl_), f"The vectors must have the same length: x {len(x)}, n {len(n)}, exp_ {len(exp_)}, pl_ {len(pl_)}"
    if error_exp < error_pl:
        A_fit = A_exp
        b_fit = beta_exp
        y_fit = exp_
        is_exp = True
    else:
        A_fit = A_pl
        b_fit = alpha_pl
        y_fit = pl_
        is_exp = False
    return x,n,A_fit,b_fit,y_fit,is_exp


#####################
# MAIN PROCESSING FUNCTIONS  
#####################

def process_feature_by_class_and_date(feature, dates, classes, data_source, params):
    """
    Process a feature across multiple dates and classes.
    
    Args:
        feature (str): Feature name
        dates (list): List of date strings
        classes (list): List of class indices
        data_source (callable): Function to get data for a given date/class
        params (dict): Processing parameters
        
    Returns:
        dict: Dictionary of processed results
    """
    # Initialize data structures
    results = {
        "x": {date: {cls: None for cls in classes} for date in dates},
        "y": {date: {cls: None for cls in classes} for date in dates},
        "y_fit": {date: {cls: None for cls in classes} for date in dates},
        "mean": {date: {cls: None for cls in classes} for date in dates},
        "fit_params": {date: {cls: None for cls in classes} for date in dates},
        "fit_info": init_fit_info()
    }
    
    # Process each date
    for date in dates:
        # Get data for this date
        date_data = data_source(date)
        
        # Process each class
        for cls in classes:
            # Get data for this class
            class_data = extract_class_data(date_data, cls, feature)
            
            # Process the data
            x, y, A, b, y_fit, is_exp, x_mean = process_single_fit(
                class_data, feature, 
                params["range_time_hours"], params["bin_size_time_hours"],
                params["range_length_km"], params["bin_size_length_km"],
                params["enriched_vector_length"], cls, params.get("cut_thresholds")
            )
            
            # Store results
            results["x"][date][cls] = x
            results["y"][date][cls] = y
            results["y_fit"][date][cls] = y_fit
            results["mean"][date][cls] = x_mean
            results["fit_params"][date][cls] = {
                "A": A, "b": b, "is_exp": is_exp
            }
            
            # Update fit info
            results["fit_info"][0], results["fit_info"][2] = update_fit_info(
                results["fit_info"][0], results["fit_info"][2], 
                cls, x_mean, b, is_exp, date
            )
            
    return results


def aggregate_results_across_dates(feature, dates, classes, results, enriched_length, cut_thresholds=None):
    """
    Aggregate results across multiple dates for each class.
    
    Args:
        feature (str): Feature name
        dates (list): List of date strings
        classes (list): List of class indices
        results (dict): Results dictionary from process_feature_by_class_and_date
        enriched_length (int): Target length for enriched vectors
        cut_thresholds (dict, optional): Dictionary mapping class indices to cut thresholds
        
    Returns:
        dict: Aggregated results
    """
    aggregated = {
        "x": {cls: None for cls in classes},
        "y": {cls: None for cls in classes},
        "y_fit": {cls: None for cls in classes},
        "mean": {cls: 0 for cls in classes},
        "variance": {cls: 0 for cls in classes}
    }
    
    # Use the first date's first class x values as a reference
    ref_date = dates[0]
    ref_x = results["x"][ref_date][classes[0]]
    size_x = len(ref_x)
    
    # Initialize arrays
    for cls in classes:
        aggregated["x"][cls] = ref_x
        aggregated["y"][cls] = np.zeros(size_x)
    
    # First pass: collect normalized distributions
    for date in dates:
        for cls in classes:
            # Get the original distribution
            n = results["y"][date][cls]
            # Normalize
            norm_factor = np.sum(n)
            if norm_factor > 0:
                normalized_n = n / norm_factor
            else:
                normalized_n = n
            # Ensure length matches
            normalized_n = enrich_vector_to_length(normalized_n, size_x)
            # Add to aggregate (with equal weight for each day)
            aggregated["y"][cls] += normalized_n / len(dates)
    
    # Second pass: fit models to aggregated data
    for cls in classes:
        x = aggregated["x"][cls]
        y = aggregated["y"][cls]
        
        # Apply class-specific cuts if needed
        if "len" in feature and cut_thresholds and cls in cut_thresholds:
            mask = x > cut_thresholds[cls]
            x_fit = x[mask]
            y_fit = y[mask]
        else:
            x_fit = x
            y_fit = y
        
        # Fit models
        fit_results = fit_distribution(x_fit, y_fit, cls)
        
        # Store results
        aggregated["y_fit"][cls] = fit_results["y_fit"]
        
        # Calculate statistics
        normalized_y = y / np.sum(y)
        aggregated["mean"][cls] = np.sum(normalized_y * x)
        aggregated["variance"][cls] = np.sqrt(np.sum(normalized_y * (x - aggregated["mean"][cls])**2))
    
    return aggregated



### AVERAGE OVER ALL DAYS

def extract_result_fit_from_x_y(feature: str,
                        x: np.ndarray, 
                        average_y_over_day: np.ndarray,
                        average_x_mean_over_day: float,
                        cut_length: float = 4) -> Dict[str, Any]:
    """
    Process a feature and determine the best fit (exponential or power law).
    
    Args:
        feature: Feature name ('time_hours' or 'lenght_km')
        x: x values
        average_y_over_day: y values averaged over all days
        average_x_mean_over_day: Mean value averaged over all days
        cut_length: Maximum length to consider for length distributions
        
    Returns:
        Dictionary containing fit results
    """
    try:
        # Make sure x and y have the same dimensions
        if len(x) != len(average_y_over_day):
            print(f"Warning: x and y dimensions don't match. x: {len(x)}, y: {len(average_y_over_day)}")
            # Match the lengths by truncating to the shorter one
            max_len = max(len(x), len(average_y_over_day))
            x = enrich_vector_to_length(x, max_len)
            average_y_over_day = enrich_vector_to_length(average_y_over_day)
            print(f"Adjusted to length {max_len} for both x and y.")
        
        if "leng" in feature:
            # Apply length cutoff for length feature
            mask = np.array(x) <= cut_length
            x_masked = np.array(x)[mask]
            average_y_masked = np.array(average_y_over_day)[mask]
            
            print(f"Feature: {feature}, size x: {len(x_masked)}, size average_y_over_day: {len(average_y_masked)}")
            
            # Fit exponential only for length
            A_exp, beta_exp, exp_, error_exp, R2_exp, bins_plot = fit_expo(x_masked, average_y_masked)
            error_pl = 1e10  # Force exponential fit for length
            pl_ = None
            alpha_pl = None
        else:
            # Compare exponential and power law fits for time
            A_exp, beta_exp, exp_, error_exp, R2_exp, A_pl, alpha_pl, pl_, error_pl, R2_pl, bins_plot = compare_exponential_power_law_from_xy(np.array(x), np.array(average_y_over_day))
        
        # Determine the best fit
        if error_exp < error_pl:
            # Exponential is better
            fit_result = {
                "fit_type": "exponential",
                "y_fit": exp_,
                "parameter": beta_exp,
                "parameter_name": "β",
                "fit_param": {"expo": beta_exp, "pl": None}
            }
        else:
            # Power law is better
            fit_result = {
                "fit_type": "power law",
                "y_fit": pl_,
                "parameter": alpha_pl,
                "parameter_name": "α",
                "fit_param": {"expo": None, "pl": alpha_pl}
            }
        
        # Add common attributes
        fit_result.update({
            "average_y": average_y_over_day,
            "x_mean": average_x_mean_over_day,
            "A_exp": A_exp,
            "exp_": exp_,
            "A_pl": A_pl if "A_pl" in locals() else None,
            "pl_": pl_
        })
        
        return fit_result
        
    except Exception as e:
        print(f"Error processing feature fit for {feature}: {e}")
        return {"fit_type": None, "error": str(e)}
