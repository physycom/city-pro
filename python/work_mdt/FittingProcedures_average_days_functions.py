import polars as pl
import numpy as np
from typing import List,Tuple,Dict
# Project
from fit_plots import *

def filter_users_if_one_point_out_of_polygon(Fcm,df_cut_traj,col_user_fcm,col_user_cut):
    """
        Filter the trajectories that are not completely inside the polygon.

    """
    print(df_cut_traj)
    df_cut_traj = df_cut_traj.filter(pl.col("size_traj")>2)
    users_2_filter = df_cut_traj.select(pl.col(col_user_cut)).unique().to_series().to_list()
    print("size filter",len(users_2_filter))
    return Fcm.filter(pl.col(col_user_fcm).is_in(users_2_filter))   
def load_data(fcm_dir: str, date_str: str, filter_space: pl.Expr, filter_time: pl.Expr) -> Optional[pl.DataFrame]:
    """
    Load FCM data for a specific date.
    
    Args:
        fcm_dir: Directory containing FCM data files
        date_str: Date string in format YYYY-MM-DD
        filter_space: Polars expression for filtering spatial data
        filter_time: Polars expression for filtering temporal data
        
    Returns:
        Filtered DataFrame or None if loading fails
    """
    file_path = os.path.join(fcm_dir, f"bologna_mdt_{date_str}_{date_str}_fcm.csv")
    try:
        fcm = pl.read_csv(file_path)
        filtered_fcm = fcm.filter(filter_space, filter_time)
        
        if filtered_fcm.height == 0:
            print(f"No data for {date_str} after applying filters")
            return None
        
        return filtered_fcm
    except Exception as e:
        print(f"Error reading or filtering data for {date_str}: {e}")
        return None




def compute_distributions(features: List[str],
                        dates: List[str],
                        fcm_dir: str,
                        filter_space: pl.Expr,
                        filter_time: pl.Expr,
                        bin_size_time_hours: float,
                        bin_size_length_km: float,
                        range_time_hours: List[float] = [0.1, 2],
                        range_length_km: List[float] = [0.1, 10],
                        enriched_vector_length: int = 50) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Compute distributions of time and length for each feature and each day.
    
    Args:
        features: List of features to analyze ('time_hours', 'lenght_km')
        dates: List of date strings to process
        fcm_dir: Directory containing FCM data files
        filter_space: Polars expression for filtering spatial data
        filter_time: Polars expression for filtering temporal data
        bin_size_time_hours: Bin size for time histograms
        bin_size_length_km: Bin size for length histograms
        range_time_hours: Range for time distributions [min, max]
        range_length_km: Range for length distributions [min, max]
        enriched_vector_length: Target length for interpolated vectors
        
    Returns:
        Tuple of dictionaries: (feature_2_day_2_x, feature_2_day_2_y, 
                               feature_2_day_2_y_fit, feature_2_day_2_mean)
    """
    # Initialize result dictionaries
    feature_2_day_2_x = {feature: {day: [] for day in dates} for feature in features}
    feature_2_day_2_y = {feature: {day: [] for day in dates} for feature in features}
    feature_2_day_2_y_fit = {feature: {day: [] for day in dates} for feature in features}
    feature_2_day_2_mean = {feature: {day: 0 for day in dates} for feature in features}
    
    # Process each feature
    for feature in features:
        print(f"Feature: {feature}")
        
        # Process each date
        for date_str in dates:
            print(f"Date: {date_str}")
            
            # Read FCM data and apply filters
            fcm_data = load_data(fcm_dir, date_str, filter_space, filter_time)
            if fcm_data is None:
                continue
            
            # Extract feature data
            feature_data = fcm_data[feature].to_list()
            
            # Compute fit and distribution
            try:
                # Choose the range for the bins
                bins, bin_range = from_feature_to_nin_bin_range(
                    feature, 
                    range_time_hours, 
                    bin_size_time_hours, 
                    range_length_km, 
                    bin_size_length_km
                )
                
                # Get histogram data
                x, n = from_data_to_cut_distribution(feature_data, bins, bin_range)
                
                # Calculate mean
                x_mean = np.nanmean(feature_data)
                
                # Generate enriched vectors for better visualization
                x_enriched = enrich_vector_to_length(x, enriched_vector_length)
                n_enriched = enrich_vector_to_length(n, enriched_vector_length)
                
                # Try fitting
                try:
                    A_exp, beta_exp, exp_, error_exp, R2_exp, A_pl, alpha_pl, pl_, error_pl, R2_pl, bins_plot = compare_exponential_power_law_from_xy(x, n)
                        
                    # Choose best fit
                    if error_exp < error_pl:
                        y_fit = enrich_vector_to_length(exp_, enriched_vector_length)
                        is_exp = True
                    else:
                        y_fit = enrich_vector_to_length(pl_, enriched_vector_length)
                        is_exp = False
                except Exception as e:
                    print(f"Error fitting distributions for {date_str}, feature {feature}: {e}")
                    y_fit = np.zeros_like(n_enriched)
                
                print(f"size x: {len(x_enriched)}, size n: {len(n_enriched)}, size y_fit: {len(y_fit)}, <x>: {x_mean:.4f}")
                
                # Store results
                feature_2_day_2_y[feature][date_str] = n_enriched
                feature_2_day_2_x[feature][date_str] = x_enriched
                feature_2_day_2_mean[feature][date_str] = x_mean
                feature_2_day_2_y_fit[feature][date_str] = y_fit
                
            except Exception as e:
                print(f"Error computing distributions for {date_str}, feature {feature}: {e}")
                
    return feature_2_day_2_x, feature_2_day_2_y, feature_2_day_2_y_fit, feature_2_day_2_mean


def run_analysis(config: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Run the full analysis pipeline.
    
    Args:
        config: Dictionary containing configuration parameters
            - features: List of features to analyze
            - dates: List of date strings
            - fcm_dir: Directory containing FCM data files
            - plot_dir: Directory to save plots
            - filter_params: Dictionary of filter parameters
            - distribution_params: Dictionary of distribution parameters
        
    Returns:
        Dictionary of fit parameters for each feature
    """
    # Extract configuration parameters
    features = config.get('features', ["time_hours", "lenght_km"])
    dates = config.get('dates', [])
    fcm_dir = config.get('fcm_dir', '')
    plot_dir = config.get('plot_dir', '')
    
    # Filter parameters
    filter_params = config.get('filter_params', {})
    filter_length_km = filter_params.get('filter_length_km', 10)
    filter_time_hours = filter_params.get('filter_time_hours', 1.5)
    filter_space = filter_params.get('filter_space', pl.col("lenght_km") < filter_length_km)
    filter_time = filter_params.get('filter_time', pl.col("time_hours") < filter_time_hours)
    
    # Distribution parameters
    dist_params = config.get('distribution_params', {})
    bin_size_time_hours = dist_params.get('bin_size_time_hours', 0.05)
    bin_size_length_km = dist_params.get('bin_size_length_km', 1)
    range_time_hours = dist_params.get('range_time_hours', [0.1, filter_time_hours])
    range_length_km = dist_params.get('range_length_km', [0.1, filter_length_km])
    enriched_vector_length = dist_params.get('enriched_vector_length', 50)
    cut_length = dist_params.get('cut_length', 4)
    case = dist_params.get('case', "")
    
    # Create output directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Starting time and length distribution analysis...")
    
    # Compute distributions
    feature_2_day_2_x, feature_2_day_2_y, feature_2_day_2_y_fit, feature_2_day_2_mean = compute_distributions(
                                                                                                            features,
                                                                                                            dates,
                                                                                                            fcm_dir,
                                                                                                            filter_space,
                                                                                                            filter_time,
                                                                                                            bin_size_time_hours,
                                                                                                            bin_size_length_km,
                                                                                                            range_time_hours,
                                                                                                            range_length_km,
                                                                                                            enriched_vector_length
                                                                                                        )
    
    # Plot distributions and get fit parameters
    fit_params = plot_distributions_averaged_over_days(
        features,
        dates,
        feature_2_day_2_x,
        feature_2_day_2_y,
        feature_2_day_2_mean,
        enriched_vector_length,
        plot_dir,
        range_time_hours,
        range_length_km,
        cut_length,
        case
    )
    
    print("Analysis complete. Results saved to:", plot_dir)
    
    return fit_params
