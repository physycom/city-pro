#pragma once

#include "stdafx.h"
#include "record.h"
#include "carto.h"
#include "data_reading.h"
#include "config.h"

//------------------------------------------------------------------------------------------------------------------
void load_poly(config config_,std::vector<poly_base> &poly,std::map<unsigned long long int, int> &poly_cid2lid);
void load_data(std::vector<activity_base> &activity,config &config_);
void load_polygon(std::vector<polygon_base> &polygon,config &config_);
void load_subnet(config &config_,map<string, vector<int>> &subnets);
//------------------------------------------------------------------------------------------------------------------
