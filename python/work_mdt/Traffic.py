
import numpy as np
import polars as pl
import datetime
def ComputeGammaAllDays(ListDailyNetwork,
                        Class2CriticalTraffic,
                        TimeIntervalsDt,
                        Class2Idx,
                        CutIndexTime,
                        Classes):
    """
        @params:
            ListDailyNetwork: List of DailyNetwork objects.
            Class2CriticalTraffic: Dictionary with the critical traffic index for each class.
            Class2Idx: Dictionary with the indexes of the subplots for each class.
            CutIndexTime: Index to cut the time intervals.
            Classes: List of classes.
        @return:
            Class2CriticalTraffic: Dictionary with the critical traffic index for each class.
            Day2TrafficIndex: Dictionary with the traffic index for each day.
            TrafficIdx: Dictionary with the  index for each day.
        Compute \Gamma(t) for all days.
    """
    
    for MobDate in ListDailyNetwork:
        CountDays += 1
        for Class in Classes:
            TimeIntervals = list(MobDate.OrderedClass2TimeDeparture2UserId[Class].keys())
            if type(TimeIntervals[0]) == int:
                TimeIntervalsDt = [datetime.datetime.fromtimestamp(int(t)).strftime("%Y-%m-%d %H:%M:%S").split(" ")[1] for t in TimeIntervals]
            else:
                TimeIntervalsDt = TimeIntervals
                pass
            # Critical Value Traffic Index
            if Class2CriticalTraffic[Class] is None: 
                Class2CriticalTraffic[Class] = np.array(MobDate.Class2traffickIndex[Class])
            else:
                Class2CriticalTraffic[Class] += np.array(MobDate.Class2traffickIndex[Class])
            Day2TrafficIndex["Time"] = TimeIntervalsDt
            Day2TrafficIndex[MobDate.StrDate + "_" + str(Class)] = list(MobDate.Class2traffickIndex[Class][CutIndexTime:])
            TrafficIdx["Day"] = np.full(MobDate.StrDate, len(MobDate.Class2traffickIndex[Class][CutIndexTime:]))
            TrafficIdx[Class] = list(MobDate.Class2traffickIndex[Class][CutIndexTime:])
            TrafficIdx["Time"] = TimeIntervalsDt[CutIndexTime:]
    DfGamma = pl.DataFrame(TrafficIdx)

