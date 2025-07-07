from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
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
