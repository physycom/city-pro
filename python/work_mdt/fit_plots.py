from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import matplotlib.pyplot as plt
from fit_tricks_to_compare_different_conditions import *
from typing import List,Dict,Any
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
import json
from typing import Dict, List, Tuple, Any, Optional
import os
from Fit_spcific_cases import *
#####################
# VISUALIZATION FUNCTIONS
#####################

def plot_distribution(ax,
                      class_idx,
                      x,
                      y,
                      y_fit,
                      labels, 
                      feature, 
                      x_lim_min, 
                      x_lim_max, 
                      marker_style=None, 
                      color_dict=None):
    """
    Plot a distribution with its fit on a given axis.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        class_idx (int): Class index
        x (array): X values
        y (array): Y values
        y_fit (array): Fitted Y values
        labels (list): Class labels
        feature (str): Feature name ("time_hours" or "lenght_km")
        x_lim_min (float): Minimum x limit
        x_lim_max (float): Maximum x limit
        marker_style (dict, optional): Dictionary mapping class indices to marker styles
        color_dict (dict, optional): Dictionary mapping class indices to colors
        
    Returns:
        matplotlib.axes.Axes: Updated axes
    """
    # Default markers if not provided
    if marker_style is None:
        marker_style = {0: "o", 1: "s", 2: "^", 3: "v"}
        
    # Default colors if not provided
    if color_dict is None:
        color_dict = {0: "blue", 1: "orange", 2: "green", 3: "red"}
    
    # Plot points
    marker = marker_style.get(class_idx, "o")
    color = color_dict.get(class_idx, None)
    
    if color:
        ax.scatter(x, y, label=labels[class_idx], marker=marker, color=color)
    else:
        ax.scatter(x, y, label=labels[class_idx], marker=marker)
    
    # Plot fit line (except for class 3 which might have special handling)
    if class_idx != 3:
        ax.plot(x, y_fit, label="")
    
    # Set axis properties
    ax.set_yscale("log")
    if feature == "time_hours":
        ax.set_xlabel(r"$t (h)$", fontsize=25)
        ax.set_ylabel(r"$P(t)$", fontsize=25)
    else:
        ax.set_xlabel(r"$l (km)$", fontsize=25)
        ax.set_ylabel(r"$P(l)$", fontsize=25)
    
    ax.set_xlim(x_lim_min, x_lim_max)
    
    return ax


def add_inset_plot(fig, ax, classes, means_data, dates, feature, position="upper center"):
    """
    Add an inset plot showing mean values by class.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to add inset to
        ax (matplotlib.axes.Axes): Main axes
        classes (list): List of class indices
        means_data (dict): Dictionary of mean values by date/class
        dates (list): List of date strings
        feature (str): Feature name
        position (str, optional): Position for inset
        
    Returns:
        matplotlib.axes.Axes: Inset axes
    """
    
    # Create inset
    ax_inset = inset_axes(ax, width="20%", height="20%", loc=position)
    
    # Calculate average means across dates
    avg_means = np.zeros(len(classes))
    
    # Plot individual date means
    for date in dates:
        day_means = np.zeros(len(classes))
        for class_idx in classes:
            mean_val = means_data[feature][date][class_idx]
            day_means[class_idx] = mean_val
            avg_means[class_idx] += mean_val
        
        ax_inset.scatter(classes, day_means, s=25)
    
    # Calculate and plot average
    avg_means /= len(dates)
    
    # Fit line to average points
    fit = np.polyfit(classes, avg_means, 1)
    a, b = fit
    
    # Plot average points and fit line
    ax_inset.scatter(classes, avg_means, s=25)
    ax_inset.plot(classes, a * np.array(classes) + b)
    
    # Set axis properties
    ax_inset.set_xticks(classes)
    if feature == "time_hours":
        ax_inset.set_xlabel("class", fontsize=12)
        ax_inset.set_ylabel(r"$\langle t \rangle$ (h)", fontsize=12)
    else:
        ax_inset.set_xlabel("class", fontsize=12)
        ax_inset.set_ylabel(r"$\langle l \rangle$ (km)", fontsize=12)
    
    return ax_inset



## AVERAGED OVER ALL DAYS ##
def plot_inset_means(day_2_mean: Dict[str, float], 
                    dates: List[str],
                    feature: str,
                    ax_inset: plt.Axes) -> plt.Axes:
    """
    Plot an inset showing the mean values for each day and the average across days.
    
    Args:
        day_2_mean: Dictionary mapping days to mean values
        dates: List of date strings
        feature: Feature name ('time_hours' or 'lenght_km')
        ax_inset: Matplotlib axes object for the inset plot
        
    Returns:
        Updated matplotlib axes object
    """
    # Initialize array for average calculation
    average_mean_over_day = 0.0
    valid_days = 0
    
    # Plot each day's mean
    for date_str in dates:
        if date_str in day_2_mean and day_2_mean[date_str] != 0:
            x_means_day = day_2_mean[date_str]
            average_mean_over_day += x_means_day
            valid_days += 1
            
            # Plot daily value
            ax_inset.scatter([0], [x_means_day], s=25)
    
    # Calculate and plot average
    if valid_days > 0:
        average_mean_over_day /= valid_days
        ax_inset.scatter([0], [average_mean_over_day], marker="*", s=45, color='red')
    
    # Configure axes
    ax_inset.set_xticks([0])
    ax_inset.set_xticklabels([""])
    
    # Set appropriate y-label based on feature
    if feature == "time_hours":
        ax_inset.set_ylabel(r"$\langle t \rangle$ (h)", fontsize=12)        
    else:
        ax_inset.set_ylabel(r"$\langle l \rangle$ (km)", fontsize=12)
    
    return ax_inset



def plot_feature_distribution(feature: str, 
                            fit_result: Dict[str, Any],
                            x: np.ndarray,
                            dates: List[str],
                            feature_2_day_2_x: Dict,
                            feature_2_day_2_y: Dict,
                            feature_2_day_2_mean: Dict,
                            enriched_vector_length: int,
                            plot_dir: str,
                            range_time_hours: List[float],
                            range_length_km: List[float],
                            case: str = "") -> None:
    """
    Plot the distribution for a feature with its best fit.
    
    Args:
        feature: Feature name ('time_hours' or 'lenght_km')
        fit_result: Dictionary containing fit results
        x: x values for plotting
        dates: List of date strings
        feature_2_day_2_x: Dictionary of x values by day
        feature_2_day_2_y: Dictionary of y values by day
        feature_2_day_2_mean: Dictionary of mean values by day
        enriched_vector_length: Target length for enriched vectors
        plot_dir: Directory to save plots
        range_time_hours: Range for time hours [min, max]
        range_length_km: Range for length km [min, max]
        case: Suffix for output filename
    """
    if fit_result.get("fit_type") is None:
        print(f"No valid fit for {feature}, skipping plot.")
        return
        
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot each day's data points
    for day in dates:            
        try:
            day_x = feature_2_day_2_x[feature][day]
            day_y = feature_2_day_2_y[feature][day]
            
            # Normalize
            day_y_normalized = day_y / np.sum(day_y) if np.sum(day_y) > 0 else day_y
            ax.scatter(day_x, day_y_normalized, label=day, alpha=0.5, s=20)
        except Exception as e:
            print(f"Error plotting data for day {day}: {e}")
    
    # Plot the fit curve
    if fit_result["fit_type"] == "exponential" and "exp_" in fit_result:
        # Normalize for plotting
        exp_norm = fit_result["exp_"] / np.sum(fit_result["exp_"]) if np.sum(fit_result["exp_"]) > 0 else fit_result["exp_"]
        
        # Scale exponential for length feature
        if "len" in feature:
            avg_y = fit_result["average_y"]
            if len(avg_y) > 0 and len(exp_norm) > 0 and exp_norm[0] != 0:
                mult_factor = avg_y[0] / exp_norm[0]
                exp_norm = exp_norm * mult_factor
        
        # Plot fit
        ax.plot(x, enrich_vector_to_length(exp_norm,enriched_vector_length), label="Exponential fit", linewidth=2, color='red')
        
    elif fit_result["fit_type"] == "power law" and "pl_" in fit_result:
        # Normalize for plotting
        pl_norm = fit_result["pl_"] / np.sum(fit_result["pl_"]) if np.sum(fit_result["pl_"]) > 0 else fit_result["pl_"]
        
        # Plot fit
        ax.plot(x, enrich_vector_to_length(pl_norm,enriched_vector_length), label="Power law fit", linewidth=2, color='blue')
    
    # Set plot properties
    ax.set_yscale("log")
    
    if feature == "time_hours":
        ax.set_xlabel(r"$t$ (h)", fontsize=25)
        ax.set_ylabel(r"$P(t)$", fontsize=25)
        ax.set_xlim(range_time_hours[0], range_time_hours[1])
    else:
        ax.set_xlabel(r"$l$ (km)", fontsize=25)
        ax.set_ylabel(r"$P(l)$", fontsize=25)
        ax.set_xlim(range_length_km[0], range_length_km[1])
        
    # Add inset with mean values
    ax_inset = inset_axes(ax, width="30%", height="25%", loc="upper right")
    ax_inset = plot_inset_means(
        feature_2_day_2_mean[feature], 
        dates,
        feature, 
        ax_inset
    )
    
    # Add title with fit information
    ax.yaxis.set_major_locator(ticker.LogLocator(numticks=4))  # Request 4 ticks (usually gives at least 3)
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs='auto', numticks=10))  # Add minor ticks

    # Optional: Format the tick labels to be more readable
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=False))
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='upper right', bbox_to_anchor=(0.98, 0.98),
             fontsize=12, framealpha=0.8)
    
    # Save figure
#    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{feature}_distribution_averaged_over_days{case}.png"), dpi=300)
    plt.close(fig)





def plot_distributions_averaged_over_days(features: List[str],
                      dates: List[str],
                      feature_2_day_2_x: Dict,
                      feature_2_day_2_y: Dict,
                      feature_2_day_2_mean: Dict,
                      enriched_vector_length: int,
                      plot_dir: str,
                      range_time_hours: List[float],
                      range_length_km: List[float],
                      cut_length: float = 4,
                      case: str = "") -> Dict[str, Dict]:
    """
    Plot time and length distributions averaged over all days.
    
    Args:
        features: List of features to analyze
        dates: List of date strings
        feature_2_day_2_x: Dictionary of x values by feature and day
        feature_2_day_2_y: Dictionary of y values by feature and day
        feature_2_day_2_mean: Dictionary of mean values by feature and day
        enriched_vector_length: Target length for enriched vectors
        plot_dir: Directory to save plots
        range_time_hours: Range for time hours [min, max]
        range_length_km: Range for length km [min, max]
        cut_length: Maximum length to consider for length distributions
        case: Suffix for output filenames
        
    Returns:
        Dictionary of fit parameters for each feature
    """
    # Initialize result dictionaries
    feature_2_fit_param = {feature: {"expo": None, "pl": None} for feature in features}
    
    # Process each feature
    for feature in features:
        # Initialize average arrays
        average_y_over_day = np.zeros(enriched_vector_length)
        average_x_mean_over_day = 0
        valid_days = 0
        x = None
        
        # Calculate average across days
        for day in dates:
            if not feature_2_day_2_x[feature].get(day) or not feature_2_day_2_y[feature].get(day):
                continue
            
            try:
                day_x = feature_2_day_2_x[feature][day] 
                day_y = feature_2_day_2_y[feature][day]
                
                if x is None:
                    x = day_x
                    
                # Make sure data is properly aligned
                if len(day_x) == len(day_y) == enriched_vector_length:
                    average_y_over_day += np.array(day_y)
                    average_x_mean_over_day += feature_2_day_2_mean[feature][day]
                    valid_days += 1
                else:
                    print(f"Warning: Skipping day {day} due to mismatched dimensions. x: {len(day_x)}, y: {len(day_y)}")
            except Exception as e:
                print(f"Error processing day {day}: {e}")
        
        # Skip if no valid days found
        if valid_days == 0:
            print(f"No valid data found for {feature}")
            continue
        
        # Calculate averages
        average_y_over_day /= valid_days
        average_x_mean_over_day /= valid_days
        
        # Process fit for this feature
        fit_result = extract_result_fit_from_x_y(
            feature,
            x,
            average_y_over_day,
            average_x_mean_over_day,
            cut_length
        )
        
        # Store fit parameters
        if fit_result.get("fit_param"):
            feature_2_fit_param[feature] = fit_result["fit_param"]
        
        # Plot the distribution
        plot_feature_distribution(
            feature, 
            fit_result, 
            x,
            dates,
            feature_2_day_2_x,
            feature_2_day_2_y,
            feature_2_day_2_mean,
            enriched_vector_length,
            plot_dir,
            range_time_hours,
            range_length_km,
            case
        )
    
    # Save fit parameters
    try:
        with open(os.path.join(plot_dir, f"fit_param_aggregated_length_and_time{case}.json"), "w") as f:
            json.dump(feature_2_fit_param, f, indent=4)
    except Exception as e:
        print(f"Error saving fit parameters: {e}")
        
    return feature_2_fit_param

