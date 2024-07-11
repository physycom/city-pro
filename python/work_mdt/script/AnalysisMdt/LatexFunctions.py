def InitTable(Cols):
    BeginCenter = "\begin{center}\n"
    BeginTabular = "\begin{tabular}{"
    for c in range(len(Cols)):
        if c < len(Cols) - 1:
            BeginTabular += "c|"
        else:
            BeginTabular += "c"
    BeginTabular += " }\n"
    return BeginCenter + BeginTabular

def FillTable(Col,Rows,Dict):
    Table = ""
    for i in range(len(Rows)):
        if i == 0:
            Table += "\hline\n"    
        for j in range(len(Col)):
            if i == 0:
                if j == 0:
                    cell = "& {} ".format(Col[j])
                elif j == len(Col) - 1:
                    cell = "& {} \\".format(Col[j])
                else:
                    cell = "& {} ".format(Col[j])
                Table += cell
            else:
                if j == 0:
                    cell = "{} & ".format(Rows[i])
                elif j == len(Col) - 1:
                    cell = "{} \\".format(Dict[Rows[i]][Col[j]])
                else:
                    cell = "{} & ".format(Dict[Rows[i]][Col[j]])
                Table += cell

def EndTable():
    EndTabular = "\end{tabular}\n"
    EndCenter = "\end{center}\n"
    return EndTabular + EndCenter

def TableFromDict(Dict):
    """
        Input:
            - Dict: dict -> Dictionary with the data
        Output:
            - Table: str -> Latex Table
        Description:
            The Latex table will be created with the data from the dictionary. {Name Row: {Name Col: Value} }
    """
    Cols = list(Dict.keys())
    Rows = list(Dict[Cols[0]].keys())
    Table = InitTable(Cols)
    Table += FillTable(Cols,Rows,Dict)
    Table += EndTable()
    return Table