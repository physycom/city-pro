def AddMessageToLog(Message,LogFile):
    with open(LogFile,'a') as f:
        f.write(Message+'\n')

def MessagePlotDailyDistr(CountFunctionsCalled,LogFile,Upload = False):
    print("++++++++++++++++++++")
    if Upload:
        CountFunctionsCalled += 1
        Message = "{} Plot Daily Speed Distr: True\n".format(CountFunctionsCalled)
        Message += "\tUpload self.InfoFittedParameters,self.DictFittedData"
        AddMessageToLog(Message,LogFile)
    else:
        CountFunctionsCalled += 1
        Message = "{} Plot Daily Speed Distr: True\n".format(CountFunctionsCalled)
        Message += "\tComputed Fitted Data"
        AddMessageToLog(Message,LogFile)

def MessagePlotTimePercorrenceDistributionAllClasses(CountFunctionsCalled,LogFile,Upload):
    if Upload:
        Message = "{} Plotting TimePercorrence Distribution: True\n".format(CountFunctionsCalled)
        Message += "\tUpload Class2Time2Distr, Class2AvgTimePercorrence"
        AddMessageToLog(Message,LogFile)
    else:
        Message = "{} Plotting TimePercorrence Distribution: True\n".format(CountFunctionsCalled)
        Message += "\tComputed Class2Time2Distr, Class2AvgTimePercorrence"
        AddMessageToLog(Message,LogFile)
