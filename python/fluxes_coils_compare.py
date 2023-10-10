import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
import pytz
import geopandas as gpd
import folium
import base64

pd.options.mode.chained_assignment = None  # default='warn'


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-mf', '--mtsfile', help='file of fluxes')
  parser.add_argument('-cf', '--coilsave', help='file of average coils trand')
  parser.add_argument('-sf', '--simfile', help='file of simulated poly trand')
  parser.add_argument('-co', '--cartodata', help='coils match with carto file', required=True)
  parser.add_argument('-ca', '--carto', help='cartography file', required=True)
  parser.add_argument('-m', '--mode', choices=['sim-coils','sim-mts', 'mts-coils'], default='all')
  parser.add_argument ('-v', '--validation', help='dirmap defines coils of validation')
  #parser.add_argument('-d', '--direction', choices=["total","direction"], required=True) //future: edit for FT e TF analysis
  args = parser.parse_args()

  outdir = 'figure_compare'
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  first_N = 20
  cartofile = args.carto
  carto = gpd.read_file(cartofile)
  carto = carto.set_index('poly_cid')

  if (args.mode == 'mts-coils'):
    ### read mts data ###
    df_mts = pd.read_csv(args.mtsfile, sep=';')
    startdate_sim = df_mts.iloc[0].time
    stopdate_sim = df_mts.iloc[-1].time
    df_mts.time = pd.to_datetime(df_mts.time)

    ### read coils data ###
    ### rimini coils-poly corrisp
    coils_id = [182, 184, 186,188, 350, 352, 454]

    coilsdatafile = args.cartodata
    coilsd = pd.read_csv(coilsdatafile, sep=';', encoding='ISO-8859-1').sort_values(by='tag')
    coilsd = coilsd.set_index('tag')
    polylid_coil = [coilsd.loc[i, 'poly_lid'] for i in coils_id]
    polycid_coil = [carto[carto.poly_lid == i].index.values[0] for i in polylid_coil]
    map_corrisp = {i:j for i,j in zip(coils_id, polycid_coil)}
    #print(map_corrisp)

    df_coils = pd.read_csv(args.coilsave, sep=';')
    coils_dir=[]
    for i in coils_id:
      coils_dir.append(str(i)+'_1')
      coils_dir.append(str(i)+'_0')
    df_coils.set_index('datetime', inplace=True)
    #check: the coils are not always active
    coils_dir = list(set(coils_dir).intersection(df_coils.columns.to_list()))
    df_coils = df_coils[coils_dir]
    df_coils_time = df_coils[(df_coils.index>=startdate_sim) & (df_coils.index<stopdate_sim)]
    local_timezone = pytz.timezone("Europe/Rome")
    df_coils_time['hour'] = [int(i.split(' ')[-1].split(':')[0]) for i in df_coils_time.index]
    datetime_list = [i for i in df_coils_time.index.to_list() if "00:00+00:00" in i]
    hourly_df_coils = df_coils_time.groupby(by='hour').sum()
    hourly_df_coils['datetime'] = datetime_list
    hourly_df_coils.set_index('datetime', inplace=True)

    #######################
    #### mts vs COIL plot #####
    for k,v in map_corrisp.items():
      dfw_c = hourly_df_coils[[col for col in hourly_df_coils.columns if str(k) in col]]
      dfw_c['total'] = dfw_c.sum(axis=1)
      dfw_c.index = pd.to_datetime(dfw_c.index)
      #convert coils time from utc to local time
      dfw_c['local_time'] = [i.astimezone(local_timezone) for i in dfw_c.index]
      dfw_c['time']=dfw_c["local_time"].dt.tz_localize(None)
      fig,ax = plt.subplots()
      ax.plot(dfw_c.time.to_numpy(), dfw_c.total.to_numpy(), color='C0')
      plt.xticks(rotation=45)
      ax.set_ylabel('N coil', color='C0')
      ax.set_xlabel('Hour (M-D H)')
      dfw_f = df_mts[df_mts.cid == v]
      dfw_f.time = pd.to_datetime(dfw_f.time)
      ax2=ax.twinx()
      ax2.plot(dfw_f.time.to_numpy(), dfw_f.total_fluxes.to_numpy(),'--',color='C1')
      ax2.set_ylabel('N mts', color='C1')
      #plt.title(f'Coil id {k}')
      plt.tight_layout()
      fig.savefig(f'{outdir}/mts_coil_{k}.png', dpi=80)
      plt.clf()
      plt.close()

  ### read simulation data ###
  if (args.mode == 'sim-mts' or args.mode =='sim-coils'):
    df_sim = pd.read_csv(args.simfile,sep=';')
    df_sim_total = df_sim.groupby('poly_cid').sum()
    df_lid_cid= df_sim_total[['poly_lid']]
    df_sim_total = df_sim_total.drop('poly_lid', axis=1)
    startdate_sim = df_sim_total.columns[1]
    stopdate_sim = df_sim_total.columns[-1]
    df_sim_total.columns = pd.to_datetime(df_sim_total.columns)

  if (args.mode == 'sim-mts' or args.mode == 'all'):
    ### read mts data ###
    df_mts = pd.read_csv(args.mtsfile, sep=';')
    #select first N poly for total occurrances from mts data
    df_mts_time = df_mts[(df_mts.time >= startdate_sim) & (df_mts.time < stopdate_sim) ]
    poi = df_mts_time.groupby('cid')['total_fluxes'].sum().nlargest(first_N).index.to_list()
    df_mts_time.time = pd.to_datetime(df_mts_time.time)
    #### mts vs SIM plot in most popular poly (for mts) #####
    for p in poi:
      df_sim_w = df_sim_total[df_sim_total.index == p]
      df_mts_w = df_mts_time[df_mts_time.cid == p]
      fig,ax = plt.subplots()
      ax.plot(df_sim_w.columns.to_numpy(), df_sim_w.iloc[0].to_numpy(), color="C0")
      ax.set_ylabel('N sim', color="C0")
      plt.xticks(rotation=45)
      ax.set_xlabel('Hour (M-D H)')
      ax2=ax.twinx()
      ax2.plot(df_mts_w.time.to_numpy(), df_mts_w.total_fluxes.to_numpy(), '--', label='mts', color="C1")
      ax2.set_ylabel('N mts', color="C1")
      plt.title(f'Poly cid {p}')
      plt.tight_layout()
      fig.savefig(f'{outdir}/{p}.png', dpi=80)
      plt.clf()
      plt.close()

  if (args.mode == 'sim-coils' or args.mode == 'all'):

    ### read list of validation coil
    list_coil_valid=[]
    if args.validation:
      df_valid = pd.read_csv(args.validation, sep=';')
      list_coil_valid = df_valid.coil_id.to_list()

    ### coils of FORLI-CESENA
    coils_id=[]
    #coils_id = [9, 116, 170, 171, 173, 174, 175, 176, 258, 259, 260,
    #261, 262, 339, 340, 342, 343, 344, 345, 348, 349, 425, 435, 436, 607, 608, 629, 630, 631, 650, 666]
    coilsdatafile = args.cartodata
    coilsd = pd.read_csv(coilsdatafile, sep=';', encoding='ISO-8859-1').sort_values(by='tag')
    coilsd = coilsd.set_index('tag')
    if len(coils_id)==0:
      coils_id = coilsd.index.to_list()

    polylid_coil = [coilsd.loc[i, 'poly_lid'] for i in coils_id]
    polycid_coil = [carto[carto.poly_lid == i].index.values[0] for i in polylid_coil]
    map_corrisp = {i:j for i,j in zip(coils_id, polycid_coil)}

    df_coils = pd.read_csv(args.coilsave, sep=';')
    df_coils_dir = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'covid_hep','work_spire','coils-dir-mapping_template_1.00.csv'), sep=';')
    df_coils_dir = df_coils_dir[df_coils_dir.coil_id.isin(coils_id)]
    coils_dir=[]
    for i in coils_id:
      coils_dir.append(str(i)+'_1')
      coils_dir.append(str(i)+'_0')
    df_coils.set_index('datetime', inplace=True)
    #check: the coils are not always active
    coils_dir = list(set(coils_dir).intersection(df_coils.columns.to_list()))
    df_coils = df_coils[coils_dir]
    df_coils_time = df_coils[(df_coils.index>=startdate_sim) & (df_coils.index<stopdate_sim)]
    local_timezone = pytz.timezone("Europe/Rome")
    df_coils_time['hour'] = [int(i.split(' ')[-1].split(':')[0]) for i in df_coils_time.index]
    datetime_list = [i for i in df_coils_time.index.to_list() if "00:00+00:00" in i]
    hourly_df_coils = df_coils_time.groupby(by='hour').sum()
    hourly_df_coils['datetime'] = datetime_list
    hourly_df_coils.set_index('datetime', inplace=True)

    if args.mode == 'all':
      #######################
      #### mts vs COIL plot #####
      for k,v in map_corrisp.items():
        dfw_c = hourly_df_coils[[col for col in hourly_df_coils.columns if str(k) in col]]
        dfw_c['total'] = dfw_c.sum(axis=1)
        dfw_c.index = pd.to_datetime(dfw_c.index)
        #convert coils time from utc to local time
        dfw_c['local_time'] = [i.astimezone(local_timezone) for i in dfw_c.index]
        dfw_c['time']=dfw_c["local_time"].dt.tz_localize(None)
        fig,ax = plt.subplots()
        ax.plot(dfw_c.time.to_numpy(), dfw_c.total.to_numpy(), color='C0')
        plt.xticks(rotation=45)
        ax.set_ylabel('N coil', color='C0')
        ax.set_xlabel('Hour (M-D H)')
        dfw_f = df_mts_time[df_mts_time.cid == v]
        dfw_f.time = pd.to_datetime(dfw_f.time)
        ax2=ax.twinx()
        ax2.plot(dfw_f.time.to_numpy(), dfw_f.total_fluxes.to_numpy(),'--',color='C1')
        ax2.set_ylabel('N mts', color='C1')
        plt.title(f'Coil id {k}')
        plt.tight_layout()
        fig.savefig(f'{outdir}/mts_coil_{k}.png', dpi=80)
        plt.clf()
        plt.close()

      #### mts vs SIM plot in coil-poly#####
      for c, cid in map_corrisp.items():
        df_sim_w = df_sim_total[df_sim_total.index == cid]
        df_mts_w = df_mts_time[df_mts_time.cid == cid]
        fig,ax = plt.subplots()
        ax.plot(df_sim_w.columns.to_numpy(), df_sim_w.iloc[0].to_numpy(), color="C0")
        ax.set_ylabel('N sim', color="C0")
        plt.xticks(rotation=45)
        ax.set_xlabel('Hour (M-D H)')
        ax2=ax.twinx()
        ax2.plot(df_mts_w.time.to_numpy(), df_mts_w.total_fluxes.to_numpy(), '--', label='mts', color="C1")
        ax2.set_ylabel('N mts', color="C1")
        plt.title(f'Poly cid {cid}')
        plt.tight_layout()
        fig.savefig(f'{outdir}/sim_mts_{cid}.png', dpi=80)
        plt.clf()
        plt.close()

    #### COIL vs SIM plot in coil-poly#####
    for c, cid in map_corrisp.items():
      #print(c)
      dfw_c = hourly_df_coils[[col for col in hourly_df_coils.columns if str(c) in col]]
      dfw_c['total'] = dfw_c.sum(axis=1)
      dfw_c.index = pd.to_datetime(dfw_c.index)
      #convert coils time from utc to local time
      dfw_c['local_time'] = [i.astimezone(local_timezone) for i in dfw_c.index.copy()]
      dfw_c['time']=dfw_c["local_time"].dt.tz_localize(None)
      fig,ax = plt.subplots()
      ax.plot(dfw_c.time.to_numpy(), dfw_c.total.to_numpy(), color='C0')
      plt.xticks(rotation=45)
      ax.set_ylabel('N coil', color='C0')
      ax.set_xlabel('Hour (M-D H)')
      df_sim_w = df_sim_total[df_sim_total.index == cid]
      ax2=ax.twinx()
      ax2.plot(df_sim_w.columns.to_numpy(), df_sim_w.iloc[0].to_numpy(), '--',color="C1")
      ax2.set_ylabel('N sim', color="C1")
      plt.title(f'Coil {c}')
      plt.tight_layout()
      fig.savefig(f'{outdir}/sim_coil_{cid}.png', dpi=80)
      plt.clf()
      plt.close()

  day_start = startdate_sim.split(' ')[0]
  if args.mode == 'sim-mts':
    list_cid_box = poi
    name_output = f'dirmap_poly_{day_start}_mts_{first_N}.html'
  elif args.mode =='sim-coils' or args.mode == 'mts-coils':
    list_cid_box = polycid_coil
    name_output = f'dirmap_poly_{day_start}_coils.html'
  elif args.mode=='all':
    list_cid_box = poi
    name_output = f'dirmap_poly_{day_start}_{first_N}.html'

  carto_red = carto[ carto.index.isin(list_cid_box) ]
  bbox = carto_red.bounds
  s = bbox.miny.min()
  e = bbox.minx.min()
  n = bbox.maxy.max()
  w = bbox.maxx.max()



  ############ CREATE FOLIUM #################
  ############################################
  m = folium.Map(control_scale=True, zoom_start=12)
  layerlabel = '<span style="color: {col};">{txt}</span>'
  ## create poly for mts-sim compare
  if args.mode=='sim-mts' or args.mode == 'all':
    flayer_poly_vs = folium.FeatureGroup(name=layerlabel.format(col='blue', txt='mts-sim'))
    for p in poi:
      encoded = base64.b64encode(open(f'{outdir}/{p}.png', 'rb').read()).decode()
      html = '<img src="data:image/jpeg;base64,{}">'.format
      iframe = folium.IFrame(html(encoded), width=580, height=420)
      polyline = carto.loc[p, 'geometry']
      pt = folium.PolyLine(
        locations=[ [y,x] for x,y in polyline.coords ],
        popup=folium.Popup(iframe, show=False, sticky=True, max_width=580, max_height = 420),
        color='blue'
      )
      flayer_poly_vs.add_child(pt)
    m.add_child(flayer_poly_vs)
    ## nodes for better comprension of poly shape
    for p in poi:
      Flon, Flat = carto.loc[p, 'geometry'].coords[0]
      Tlon, Tlat = carto.loc[p, 'geometry'].coords[-1]
      folium.CircleMarker(
        location=[Flat, Flon],
        radius=1,
        color='black',
        fill_color='black'
      ).add_to(m)
      folium.CircleMarker(
        location=[Tlat, Tlon],
        radius=1,
        color='black',
        fill_color='black'
      ).add_to(m)

  if args.mode =='mts-coils':
    ## create red circle on coils for sim-coil compare
    flayer_poly_sc = folium.FeatureGroup(name=layerlabel.format(col='red', txt='mts-coil'))
    for c in coils_id:
      encoded = base64.b64encode(open(f'{outdir}/mts_coil_{c}.png', 'rb').read()).decode()
      html = '<img src="data:image/jpeg;base64,{}">'.format
      iframe = folium.IFrame(html(encoded), width=580, height=420)
      pt=folium.CircleMarker(
        location=[coilsd.loc[c].lat, coilsd.loc[c].lon],
        popup=folium.Popup(iframe, show=False, sticky=False, max_width=580, max_height = 420),
        radius=5,
        color='red',
        fill_color='red'
      )
      flayer_poly_sc.add_child(pt)
    m.add_child(flayer_poly_sc)

  if args.mode=='sim-coils' or args.mode == 'all':
    ## create red circle on coils for sim-coil compare
    flayer_poly_sc = folium.FeatureGroup(name=layerlabel.format(col='red', txt='sim-coil test'))
    flayer_poly_sc_v = folium.FeatureGroup(name=layerlabel.format(col='red', txt='sim-coil valid'))
    for c, cid in map_corrisp.items():
      encoded = base64.b64encode(open(f'{outdir}/sim_coil_{cid}.png', 'rb').read()).decode()
      html = '<img src="data:image/jpeg;base64,{}">'.format
      iframe = folium.IFrame(html(encoded), width=580, height=420)
      if c in list_coil_valid:
        pt=folium.CircleMarker(
          location=[coilsd.loc[c].lat, coilsd.loc[c].lon],
          popup=folium.Popup(iframe, show=False, sticky=False, max_width=580, max_height = 420),
          radius=5,
          color='blue',
          fill_color='blue'
        )
        flayer_poly_sc_v.add_child(pt)
      else:
        pt=folium.CircleMarker(
          location=[coilsd.loc[c].lat, coilsd.loc[c].lon],
          popup=folium.Popup(iframe, show=False, sticky=False, max_width=580, max_height = 420),
          radius=5,
          color='red',
          fill_color='red'
        )
        flayer_poly_sc.add_child(pt)
    m.add_child(flayer_poly_sc)
    m.add_child(flayer_poly_sc_v)

    if args.mode == 'all':
      ## create orange circle near coils for mts-coil compare
      flayer_poly_vc = folium.FeatureGroup(name=layerlabel.format(col='orange', txt='mts-coil'))
      for c in coils_id:
        encoded = base64.b64encode(open(f'{outdir}/mts_coil_{c}.png', 'rb').read()).decode()
        html = '<img src="data:image/jpeg;base64,{}">'.format
        iframe = folium.IFrame(html(encoded), width=580, height=420)
        pt = folium.CircleMarker(
          location=[coilsd.loc[c].lat+0.001, coilsd.loc[c].lon+0.001],
          popup=folium.Popup(iframe, show=False, sticky=True, max_width=580, max_height = 420),
          radius=4,
          color='orange',
          fill_color='orange'
        )
        flayer_poly_vc.add_child(pt)
      m.add_child(flayer_poly_vc)

      ## create poly for mts-sim compare
      flayer_poly_vs_c = folium.FeatureGroup(name=layerlabel.format(col='green', txt='mts-sim'))
      for c,cid in map_corrisp.items():
        encoded = base64.b64encode(open(f'{outdir}/sim_mts_{cid}.png', 'rb').read()).decode()
        html = '<img src="data:image/jpeg;base64,{}">'.format
        iframe = folium.IFrame(html(encoded), width=580, height=420)
        polyline = carto.loc[cid, 'geometry']
        pt=folium.PolyLine(
          locations=[ [y,x] for x,y in polyline.coords ],
          popup=folium.Popup(iframe, show=False, sticky=True, max_width=580, max_height = 420),
          color='green'
        )
        flayer_poly_vs_c.add_child(pt)
      m.add_child(flayer_poly_vs_c)
      ## nodes for better comprension of poly shape
      for c, cid in map_corrisp.items():
        Flon, Flat = carto.loc[cid, 'geometry'].coords[0]
        Tlon, Tlat = carto.loc[cid, 'geometry'].coords[-1]
        folium.CircleMarker(
          location=[Flat, Flon],
          radius=1,
          color='black',
          fill_color='black'
        ).add_to(m)
        folium.CircleMarker(
          location=[Tlat, Tlon],
          radius=1,
          color='black',
          fill_color='black'
        ).add_to(m)
      folium.map.LayerControl(collapsed=False).add_to(m)

  m.fit_bounds([ [s,w], [n,e] ])

  m.save(name_output)
