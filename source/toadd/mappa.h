#pragma once

#include "stdafx.h"
#include "polyline.h"
#include "dato.h"

//----------------------------------------------------------------------------
void CreaMappaTraj(void);
int  CercaTrajStartVicine(Dato dato, list <int> &vicini);
int  CercaTrajEndVicine(double lon, double lat, list <int> &vicini);
//----------------------------------------------------------------------------


