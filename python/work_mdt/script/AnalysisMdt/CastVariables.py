import numpy as np
import datetime
def NormalizeWidthForPlot(arr,min_val,max_val,min_width = 1, max_width = 10):
    '''
        Description:
            Normalizes the width for road fluxes
    '''
    if not isinstance(arr,np.ndarray):
        arr = np.array(arr)
    else:
        pass
    if not np.isnan(min_val):
        pass
    else:
        min_val = 0
    if not np.isnan(max_val):
        pass
    else:
        max_val = 130
    if (max_val - min_val) == 0:
        print("Max value {0} Is equal to the minimum {1}.".format(max_val,min_val))
        return 1
    return (arr - min_val) / (max_val - min_val) * (max_width - min_width) + min_width
# CAST
def CastString2Int(Road):
    try:
        int(Road)
        return int(Road),True
    except:
        print("Road exception: ",Road)
        return Road,False

# CONVERSIONE
def ms2kmh(v):
    return v*3.6
def kmh2ms(v):
    return v/3.6
def s2h(t):
    return t/3600
def h2s(t):
    return t*3600
def m2km(x):
    return x/1000
def km2m(x):
    return x*1000
# TIME
def StrDate2DateFormatLocalProject(StrDate):
    return StrDate.split("-")[0],StrDate.split("-")[1],StrDate.split("-")[2]

def Timestamp2Datetime(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)

def Timestamp2Date(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).date()

def Datetime2Timestamp(datetime_):
    return datetime_.timestamp()

def InInterval(start_time,end_time,TimeStampDate,t,dt):
    if (int(start_time)> int(TimeStampDate)+t*dt and int(start_time)<int(TimeStampDate)+(t+1)*dt) and (int(end_time)> int(TimeStampDate)+t*dt and int(end_time)<int(TimeStampDate)+(t+1)*dt):
        return True
    else:
        return False
    

## USEFUL FUNCTIONS FOR TIME
def BinTimeTimestampGivenDay(TimeStampDate,dt,iterations):
    """
        @brief: Bin the TimeStampDate in iterations
        @param TimeStampDate: Timestamp
        @param dt: int
        @param iterations: int
        @return: List with the bins [Timestamp, Timestamp +dt, ..., Timestamp + iterations*dt]
    """
    BinTimestamp = [int(TimeStampDate) + t*dt for t in range(iterations)]
    BinStringDayHour = [Timestamp2Datetime(BinTimestamp[t]).strftime('%Y-%m-%d %H:%M:%S') for t in range(iterations)]
    BinStringHour = [Timestamp2Datetime(BinTimestamp[t]).strftime('%Y-%m-%d %H:%M:%S').split(" ")[1] for t in range(iterations)]
    return BinTimestamp,BinStringDayHour,BinStringHour