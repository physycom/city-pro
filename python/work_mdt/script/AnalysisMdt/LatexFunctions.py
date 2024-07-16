# CONSTANTS
DoubleBackslash = "\\\\"
NewLine = "\n"
LeftCurl = "{"
RightCurl = "}"
Vespar = "[\\vsepar]"


def InitTable(Cols):
    BeginCenter = "\\begin{center}\n"
    BeginTabular = "\\begin{tabular}{c|"
    for c in range(len(Cols)):
        if c < len(Cols) - 1:
            BeginTabular += "c|"
        else:
            BeginTabular += "c"
    BeginTabular += "}\n"
    return BeginCenter + BeginTabular
def FancyInitTable(Cols):
    BeginTable = "	{\setlength{\extrarowheight}{\\vsepar}\n"
    BeginTable += "\\begin{table}\n"
    BeginTable += "\\centering\n"
    BeginTable += "\pgfplotstabletypeset[normal]{\n"
    return BeginTable


def FillTable(Dict):
    """
        Example:
            {Class0:{Day0: Message0,
                    Day1: Message1,
                    Day2: Message2},
             Class1:{Day0: Message0,
                     Day1: Message1,
                     Day2: Message2},
            ...}
    """
    Table = ""
    DoubleBackslash = "\\\\"
    NewLine = "\n"
    for i,Row in enumerate(Dict.keys()):
        if i == 0:
            Table += "\hline\n"    
        for j,Col in enumerate(Dict[Row].keys()):
            if i == 0:
                if j == 0:
                    cell = " & {} ".format(Col)
                elif j == len(Dict[Row]) - 1:
                    cell = "& {} ".format(Col)
                    cell += DoubleBackslash + NewLine
                    cell += "\hline\n"
                else:
                    cell = "& {} ".format(Col)
                Table += cell
            else:
                if j == 0:
                    cell = "{} & ".format(Row)
                elif j == len(Dict[Row]) - 1:
                    cell = "{} ".format(Dict[Row][Col])
                    cell += DoubleBackslash + NewLine
                    cell += "\hline\n"
                else:
                    cell = "{} & ".format(Dict[Row][Col])
                Table += cell
    return Table

def FillTableFancy(Dict):
    Table = ""
    for i,Row in enumerate(Dict.keys()):
        if i == 0:
            Table += "\hline\n"    
        for j,Col in enumerate(Dict[Row].keys()):
            if i == 0:
                if j == 0:
                    cell = LeftCurl + RightCurl
                elif j == len(Dict[Row]) - 1:
                    cell = "& " + LeftCurl + "{}".format(Col) + RightCurl
                    cell += DoubleBackslash + Vespar + NewLine
                else:
                    cell = "& {} ".format(Col)
                Table += cell
            else:
                if j == 0:
                    cell = LeftCurl + "{}".format(Row) + RightCurl + " & "
                elif j == len(Dict[Row]) - 1:
                    cell = LeftCurl + "{}".format(Dict[Row][Col]) + RightCurl
                    cell += DoubleBackslash + Vespar + NewLine
                else:
                    cell = "{} & ".format(Dict[Row][Col])
                Table += cell
    return Table


def EndTable():
    EndTabular = "\end{tabular}\n"
    EndCenter = "\end{center}\n"
    return EndTabular + EndCenter

def EndFancyTable():
    EndTable = "\end{table}\n"
    return EndTable

def TableFromDict(Dict):
    """
        Input:
            - Dict: dict -> Dictionary with the data
        Output:
            - Table: str -> Latex Table
        Description:
            The Latex table will be created with the data from the dictionary. {Name Row: {Name Col: Value} }
    """
    Rows = list(Dict.keys())
    Cols = list(Dict[Rows[0]].keys())
    Table = InitTable(Cols)
    Table += FillTable(Dict)
    print("Fill Table:\n",Table)
    Table += EndTable()
    print("End Table:\n",Table)
    return Table

def FancyTableFromDict(Dict):
    """
        Input:
            - Dict: dict -> Dictionary with the data
        Output:
            - Table: str -> Latex Table
        Description:
            The Latex table will be created with the data from the dictionary. {Name Row: {Name Col: Value} }
    """
    Rows = list(Dict.keys())
    Cols = list(Dict[Rows[0]].keys())
    Table = FancyInitTable(Cols)
    Table += FillTableFancy(Dict)
    print("Fill Table:\n",Table)
    Table += EndFancyTable()
    return Table