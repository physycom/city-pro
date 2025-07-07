#####################
# DATA PREPARATION
#####################
def init_data_structures(StrDates=None, Classes=None, Features=None):
    """
    Initialize dictionary data structures based on provided parameters.
    
    Args:
        StrDates (list): List of date strings
        Classes (list): List of class identifiers
        Features (list): List of feature names
    
    Returns:
        tuple: Data structures for storing results based on provided dimensions
    """
    if Features is None:
        raise ValueError("Features must be provided")
        
    # Case 1: Feature -> Day
    if Classes is None and StrDates is not None:
        feature_2_day_2_data = {"x":{Feature: {day: [] for day in StrDates} for Feature in Features},
                                "y":{Feature: {day: [] for day in StrDates} for Feature in Features},
                                "y_fit":{Feature: {day: [] for day in StrDates} for Feature in Features},
                                "x_fit":{Feature: {day: [] for day in StrDates} for Feature in Features},
                                "y_avg":{Feature: {day: 0 for day in StrDates} for Feature in Features}}
        return feature_2_day_2_data
    
    # Case 2: Feature -> Class
    elif Classes is not None and StrDates is None:
        feature_2_class_2_data = {"x":{Feature: {class_: [] for class_ in Classes} for Feature in Features},
                                 "y":{Feature: {class_: [] for class_ in Classes} for Feature in Features},
                                 "y_fit":{Feature: {class_: [] for class_ in Classes} for Feature in Features},
                                 "x_fit":{Feature: {class_: [] for class_ in Classes} for Feature in Features},
                                 "y_avg":{Feature: {class_: 0 for class_ in Classes} for Feature in Features}}
        return feature_2_class_2_data
    
    # Case 3: Feature -> Day -> Class
    elif Classes is not None and StrDates is not None:
        feature_2_day_2_class_2_data = {"x":{Feature: {day: {class_: [] for class_ in Classes} for day in StrDates} for Feature in Features},
                                            "y":{Feature: {day: {class_: [] for class_ in Classes} for day in StrDates} for Feature in Features},
                                            "y_fit":{Feature: {day: {class_: [] for class_ in Classes} for day in StrDates} for Feature in Features},
                                            "x_fit":{Feature: {day: {class_: [] for class_ in Classes} for day in StrDates} for Feature in Features},
                                            "y_avg":{Feature: {day: {class_: 0 for class_ in Classes} for day in StrDates} for Feature in Features}}
        return feature_2_day_2_class_2_data
    else:
        raise ValueError("Either Classes or StrDates (or both) must be provided")



def _init_data_structures(StrDates=None, Classes=None, Features=None):
    """
    Initialize dictionary data structures based on provided parameters.
    
    Args:
        StrDates (list): List of date strings
        Classes (list): List of class identifiers
        Features (list): List of feature names
    
    Returns:
        tuple: Data structures for storing results based on provided dimensions
    """
    if Features is None:
        raise ValueError("Features must be provided")
        
    # Case 1: Feature -> Day
    if Classes is None and StrDates is not None:
        feature_2_day_2_x = {Feature: {day: [] for day in StrDates} for Feature in Features}
        feature_2_day_2_y = {Feature: {day: [] for day in StrDates} for Feature in Features}
        feature_2_day_2_y_fit = {Feature: {day: [] for day in StrDates} for Feature in Features}
        feature_2_day_2_x_fit = {Feature: {day: [] for day in StrDates} for Feature in Features}
        feature_2_day_2_mean = {Feature: {day: 0 for day in StrDates} for Feature in Features}
        return feature_2_day_2_x, feature_2_day_2_y, feature_2_day_2_y_fit, feature_2_day_2_x_fit, feature_2_day_2_mean
    
    # Case 2: Feature -> Class
    elif Classes is not None and StrDates is None:
        feature_2_class_x = {Feature: {class_: [] for class_ in Classes} for Feature in Features}
        feature_2_class_y = {Feature: {class_: [] for class_ in Classes} for Feature in Features}
        feature_2_class_y_fit = {Feature: {class_: [] for class_ in Classes} for Feature in Features}
        feature_2_class_x_fit = {Feature: {class_: [] for class_ in Classes} for Feature in Features}
        feature_2_class_mean = {Feature: {class_: 0 for class_ in Classes} for Feature in Features}
        return feature_2_class_x, feature_2_class_y, feature_2_class_y_fit, feature_2_class_x_fit, feature_2_class_mean
    
    # Case 3: Feature -> Day -> Class
    elif Classes is not None and StrDates is not None:
        feature_2_day_2_class_2_x = {Feature: {day: {class_: [] for class_ in Classes} for day in StrDates} for Feature in Features}
        feature_2_day_2_class_2_y = {Feature: {day: {class_: [] for class_ in Classes} for day in StrDates} for Feature in Features}
        feature_2_day_2_class_2_x_fit = {Feature: {day: {class_: [] for class_ in Classes} for day in StrDates} for Feature in Features}
        feature_2_day_2_class_2_y_fit = {Feature: {day: {class_: [] for class_ in Classes} for day in StrDates} for Feature in Features}
        feature_2_day_2_class_2_mean = {Feature: {day: {class_: 0 for class_ in Classes} for day in StrDates} for Feature in Features}
        return feature_2_day_2_class_2_x, feature_2_day_2_class_2_y, feature_2_day_2_class_2_y_fit, feature_2_day_2_class_2_x_fit, feature_2_day_2_class_2_mean
    else:
        raise ValueError("Either Classes or StrDates (or both) must be provided")


def init_fit_info():
    """
    Initialize dictionaries for storing fit information.
    
    Returns:
        tuple: Dictionaries for storing fit information
    """
    fit_info = {"class": [], "day": [], "alpha": [], "beta": [], "fit_name": [], "<x>": []}
    fit_info_concat = {"class": [], "day": [], "alpha": [], "beta": [], "fit_name": [], "<x>": []}
    dict_Lkclass = {"class": [], "day": [], "Lk": []}
    dict_Lkclass_concat = {"class": [], "day": [], "Lk": []}
    return fit_info, fit_info_concat, dict_Lkclass, dict_Lkclass_concat
