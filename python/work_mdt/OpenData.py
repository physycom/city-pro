import os
import polars as pl
import geopandas as gpd
from EstimatePenetration import GetTimeRangesFromDfOpenData

def GenerateTimeRanges(DfTrafficOpenData):
    """
        Generate the Time Ranges from the DataFrame of the Traffic Open Data ["00:00-01:00",...]
        These are the columns of the OpenData that need to be mapped to time of the format of the other otuput.
        @param DfTrafficOpenData: DataFrame with the Traffic Open Data
        @return TimeRanges: List of Time Ranges
    """
    TimeRanges = GetTimeRangesFromDfOpenData(DfTrafficOpenData)
    Column2ConsiderGdfTrafficOpenData = TimeRanges.copy()
    Column2ConsiderGdfTrafficOpenData.append("data")
    Column2ConsiderGdfTrafficOpenData.append("geometry")
    return Column2ConsiderGdfTrafficOpenData,TimeRanges


def OpenDataFrameSpireAndTransformGeoData(DirOpenDataBologna,Days,StrFile = "rilevazione-flusso-veicoli-tramite-spire-anno-2022.csv"):
    """
        Open the DataFrame of the Traffic Open Data and transform it to a GeoDataFrame
        @param DirOpenDataBologna: Directory where the Open Data is stored
        @param Days: List of days to consider
        @return GdfTrafficOpenData: GeoDataFrame with the traffic data
        It is already filtered without the columns of speed.
    """
    DfTrafficOpenData = pl.read_csv(os.path.join(DirOpenDataBologna, StrFile),separator=';')
    DfTrafficOpenData = DfTrafficOpenData.filter(pl.col("data").is_in(Days))
    DfTrafficOpenData_pd = DfTrafficOpenData.to_pandas()
    GdfTrafficOpenData = gpd.GeoDataFrame(
        DfTrafficOpenData_pd, geometry=gpd.points_from_xy(DfTrafficOpenData_pd["longitudine"], DfTrafficOpenData_pd["latitudine"]))
    GdfTrafficOpenData.set_crs(epsg=4326, inplace=True)
    Column2ConsiderGdfTrafficOpenData,TimeRanges = GenerateTimeRanges(DfTrafficOpenData)
    GdfTrafficOpenData = GdfTrafficOpenData[Column2ConsiderGdfTrafficOpenData]

    return GdfTrafficOpenData,Column2ConsiderGdfTrafficOpenData,TimeRanges