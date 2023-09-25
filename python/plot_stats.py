import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib as mpl
import os
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

sns.set_style("ticks")

def func_pl(x, a, b):
  return a * x**(-b)

def func_exp(x,a,b):
  return a * np.exp(-b*x)

pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-di', '--dir', help='input directory with mfd val', required=True)
  parser.add_argument('-cn', '--city_name', help='name of city', required=True)
  parser.add_argument('-c', '--classif', help='select for class comparison', action='store_true')
  parser.add_argument('-j', '--join', help='select for joining figures', action='store_true')
  args = parser.parse_args()

  MAX_SIZE = 25

  # Rimini
  xlim_lengh = 10.0
  xlim_time  = 60.0
  xlim_speed = 150.0
  bins_length = 75
  bins_time = 75
  bins_speed = 80
  default_class_value=10

  #rimini parameters
  thresh_time = 5.0
  thresh_length = 10.0

  name_file = args.city_name
  list_fname=[]
  for fname in os.listdir(args.dir):
    if fname.startswith(name_file) and fname.endswith('fcm.csv'):
      list_fname.append(fname)

  df=pd.DataFrame()
  for fn in list_fname:
    input_file = os.path.join(args.dir, fn)
    dfw=pd.read_csv(input_file, sep=';')
    if args.classif:
      input_file_centers = os.path.join(args.dir, fn.replace('fcm.csv','fcm_centers.csv'))
      df_centers = pd.read_csv(input_file_centers, sep=';', names=['num_class','av_speed','v_max','v_min','sinuosity','occur'])
      n_class = len(df_centers)
      df_centers.sort_values(by='av_speed', ignore_index=True, inplace=True)
      map_class={i:j for i,j in zip(df_centers.num_class.to_list(), df_centers.index.to_list())}
      dfw = dfw[dfw['class']!=default_class_value]
      dfw['new_class'] = [map_class[i] for i in dfw['class']]

    df = df.append(dfw, ignore_index=True)

  #df = pd.read_csv(args.input, sep=';')
  df = df[df.time >= 60] #prendi solo viaggi che superano il minuto

  df.lenght = df.lenght.div(1000)
  df.time = df.time.div(60)
  df.av_speed = df.av_speed.multiply(3.6)

  mpl.rc('text', usetex = True)

  if args.join:
    fig = plt.figure(figsize=(14,8), constrained_layout=True)

    gs = gridspec.GridSpec(2, 4, figure=fig)
    gs.update(wspace=0.5)
    ax1 = plt.subplot(gs[0, :2], )
    ax2 = plt.subplot(gs[0, 2:])
    ax3 = plt.subplot(gs[1, 1:3])


  #### GLOBAL L ####
  df_l = df[df.lenght < thresh_length]
  list_x_l = np.arange(0,thresh_length,0.1)
  histo_l = pd.DataFrame()
  histo_l['val'] =pd.cut(df_l['lenght'], bins=80).value_counts()
  histo_l['bins_mean'] = [i.mid for i in histo_l.index]
  histo_l.sort_values(by='bins_mean', inplace=True)
  histo_l_fitt = histo_l[histo_l.bins_mean > 1.5]
  popt_l, pcov = curve_fit(func_exp, histo_l_fitt['bins_mean'].to_list(), histo_l_fitt['val'].to_list())
  #print(f'L global interpolation: a = {popt_l[0]}, b={popt_l[1]}')
  if args.join:
    histo_l.plot.scatter(x='bins_mean',y='val', ax=ax1, edgecolors='black', s=40)
    ax1.plot(list_x_l, func_exp(list_x_l, *popt_l),'--', color='black')
    ax1.set_yscale('log')
    #ax1.set_ylim(min(func_exp(list_x_l, *popt_l)), max(func_exp(list_x_l, *popt_l)))
    ax1.set_ylim(50, 100000)
    ax1.set_xlabel('length (km)', fontsize=MAX_SIZE)
    ax1.axvline(x=df_l.lenght.mean(), color='k', linestyle='--')
    ax1.tick_params(axis='x', labelsize=MAX_SIZE)
    ax1.tick_params(axis='y', labelsize=MAX_SIZE)
    ax1.set_ylabel('N of activities', fontsize=MAX_SIZE)
  else:
    fig, ax = plt.subplots()
    histo_l.plot.scatter(x='bins_mean',y='val',ax=ax, edgecolors='black', s=40)
    ax.plot(list_x_l, func_exp(list_x_l, *popt_l),'--', color='black')
    ax.set_yscale('log')
    ax.set_ylim(50, 100000)
    ax.set_xlabel('length (km)', fontsize=MAX_SIZE)
    ax.axvline(x=df_l.lenght.mean(), color='k', linestyle='--')
    ax.tick_params(axis='x', labelsize=MAX_SIZE)
    ax.tick_params(axis='y', labelsize=MAX_SIZE)
    ax.set_ylabel('N of activities', fontsize=MAX_SIZE)
    plt.xticks(np.arange(int(min(df_l.lenght)), int(max(df_l.lenght))+2, 2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.savefig(name_file+'_global_stats_L.png', dpi=150, bbox_inches='tight')
    plt.clf()
    plt.close()

  #### GLOBAL T ####
  df_t = df[df.time > thresh_time]
  list_x_t = np.arange(thresh_time,xlim_time, 1.0)
  histo_t = pd.DataFrame()
  histo_t['val'] =pd.cut(df_t['time'], bins=80).value_counts()
  histo_t['bins_mean'] = [i.mid for i in histo_t.index]
  histo_t.sort_values(by='bins_mean', inplace=True)
  histo_t_fitt = histo_t[histo_t['bins_mean']>=10.0]
  popt_t, pcov = curve_fit(func_pl, histo_t_fitt['bins_mean'].to_list(), histo_t_fitt['val'].to_list())
  #print(f'T global interpolation: a = {popt_t[0]}, b={popt_t[1]}')
  if args.join:
    histo_t.plot.scatter(x='bins_mean',y='val', ax=ax2, edgecolors='black', s=40)
    ax2.plot(list_x_t, func_pl(list_x_t, *popt_t),'--', color='black')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim(50, 100000)
    ax2.set_xlabel('time (min)', fontsize=MAX_SIZE)
    ax2.axvline(x=df_t.time.mean(), color='k', linestyle='--')
    ax2.tick_params(axis='x', labelsize=MAX_SIZE)
    ax2.tick_params(axis='y', labelsize=MAX_SIZE)
    ax2.set_ylabel('N of activities', fontsize=MAX_SIZE)
  else:
    fig, ax = plt.subplots()
    histo_t.plot.scatter(x='bins_mean',y='val', ax=ax, edgecolors='black', s=40)
    #equation_t = r'${} * x^{{{}}}$'.format(f'{popt_t[0]:.2e}',f'-{popt_t[1]:.2f}')
    ax.plot(list_x_t, func_pl(list_x_t, *popt_t),'--', color='black')
    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(50, 100000)
    ax.set_xlabel('time (min)', fontsize=MAX_SIZE)
    plt.xticks(np.arange(int(min(df_t.time)), int(max(df_t.time))+10, 10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.axvline(x=df_t.time.mean(), color='k', linestyle='--')
    ax.tick_params(axis='x', labelsize=MAX_SIZE)
    ax.tick_params(axis='y', labelsize=MAX_SIZE)
    ax.set_ylabel('N of activities', fontsize=MAX_SIZE)
    plt.savefig(name_file+'_global_stats_T.png', dpi=150, bbox_inches='tight')
    plt.clf()
    plt.close()

  # dump parameters of interpolation
  df_param_int = pd.DataFrame(list(zip(popt_l, popt_t)), columns =['L_exp', 'T_pl'], index=['a','b'])
  df_param_int.to_csv(name_file+'_global_stats_param.csv')


  #### GLOBAL V ####
  if args.join:
    df['av_speed'].hist(bins=bins_speed, ax=ax3)
    #df['av_speed'].hist(bins=bins_speed, ax=ax3, density=True)

    #histo_v = pd.DataFrame()
    #histo_v['val'] =pd.cut(df['av_speed'], bins=bins_speed).value_counts()
    #histo_v['bins_mean'] = [i.mid for i in histo_v.index]
    #histo_v.sort_values(by='bins_mean', inplace=True)
    #histo_v['val'] = histo_v['val'] /histo_v['val'].abs().max()
    #histo_v.plot.bar(x='bins_mean',y='val', ax=ax3)

    ax3.set_xlim(0,xlim_speed)
    ax3.set_xlabel('Average Velocity (km/h)', fontsize=MAX_SIZE)
    ax3.axvline(x=df.av_speed.mean(), color='k', linestyle='--')
    ax3.tick_params(axis='x', labelsize=MAX_SIZE)
    ax3.tick_params(axis='y', labelsize=MAX_SIZE)
    ax3.set_ylabel('N of activities', fontsize=MAX_SIZE)
    plt.savefig(name_file+'_global_stats.png', dpi=150, bbox_inches='tight')
    plt.clf()
    plt.close()

  else:
    fig, ax = plt.subplots()
    df['av_speed'].hist(bins=bins_speed, ax=ax, density=True)
    ax.set_xlim(0,xlim_speed)
    ax.set_xlabel('Average Velocity (km/h)', fontsize=MAX_SIZE)
    ax.axvline(x=df.av_speed.mean(), color='k', linestyle='--')
    ax.tick_params(axis='x', labelsize=MAX_SIZE)
    ax.tick_params(axis='y', labelsize=MAX_SIZE)
    ax.set_ylabel('N of activities', fontsize=MAX_SIZE)
    plt.savefig(name_file+'_global_stats_V.png', dpi=150, bbox_inches='tight')
    plt.clf()
    plt.close()

  df_average_value=pd.DataFrame(columns=['L(km)','T(min)','V(km/h)'])
  df_average_value.loc['global']=[df.lenght.mean(),df.time.mean(),df.av_speed.mean()]

  if args.classif:
    cmap = plt.get_cmap("tab10")
    alpha_val=0.6

    if args.join:
      fig = plt.figure(figsize=(14,8), constrained_layout=True)

      gs = gridspec.GridSpec(2, 4, figure=fig)
      gs.update(wspace=0.5)
      ax1 = plt.subplot(gs[0, :2], )
      ax2 = plt.subplot(gs[0, 2:])
      ax3 = plt.subplot(gs[1, 1:3])

    #exclude class for higway
    excluded_class=3

    if args.join ==False:
      fig, ax = plt.subplots()

    list_columns=['value','class','a','b']
    df_fit_class =pd.DataFrame(columns=list_columns)


    marker_list = ['o','s','v','p']
    for i in np.arange(0,n_class):
      dfw = df[df['new_class'] == i]
      #collect average value
      df_average_value.loc[f'class {i}']=[dfw.lenght.mean(),dfw.time.mean(),dfw.av_speed.mean()]

      if i==excluded_class:
        continue

      #print(f'class = {i}, lenght mean = {dfw.lenght.mean()}')
      dfw['lenght'] = dfw.lenght.div(dfw.lenght.mean())
      histo_l = pd.DataFrame()
      histo_l['val'] =pd.cut(dfw['lenght'], bins=bins_length).value_counts()
      histo_l['bins_mean'] = [i.mid for i in histo_l.index]
      histo_l['x_range'] = [(i.right-i.left) for i in histo_l.index]
      histo_l['temp'] = histo_l['x_range']*histo_l['val']
      area_uc = histo_l['temp'].sum()
      histo_l['val'] = histo_l['val'].div(area_uc)
      histo_l.sort_values(by='bins_mean', inplace=True)
      histo_l_fitt = histo_l[histo_l.bins_mean>0.5]
      popt_l, pcov = curve_fit(func_exp, histo_l_fitt['bins_mean'].to_list(), histo_l_fitt['val'].to_list())
      df_temp = pd.DataFrame([['L',i,f'{popt_l[0]:.2f}',f'{popt_l[1]:.2f}']],columns=list_columns)
      df_fit_class = df_fit_class.append(df_temp)

      if args.join:
        histo_l.plot.scatter(x='bins_mean',y='val', ax=ax1, marker=marker_list[i], edgecolors='black', s=40, color=cmap(i),label=f'class {i}')
        ax1.set_yscale('log')
        ax1.set_ylim(0.001, 1)
        ax1.set_xlim(0,4.0)
        ax1.set_xlabel(r'$L/L_m$', fontsize=MAX_SIZE)
        ax1.set_ylabel('')
        ax1.tick_params(axis='x', labelsize=MAX_SIZE)
        ax1.tick_params(axis='y', labelsize=MAX_SIZE)
        ax1.set_ylabel('N of activities', fontsize=MAX_SIZE)
        ax1.legend(prop={"size":20}, ncol=2)
      else:
        histo_l.plot.scatter(x='bins_mean',y='val', ax=ax, marker=marker_list[i],edgecolors='black', s=40, color=cmap(i),label=f'class {i}')
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
        ax.set_xlim(0,4.0)
        ax.set_xlabel(r'$L/L_m$', fontsize=MAX_SIZE)
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=MAX_SIZE)
        ax.tick_params(axis='y', labelsize=MAX_SIZE)
        ax.set_ylabel('N of activities', fontsize=MAX_SIZE)
        ax.legend(prop={"size":20}, ncol=2)

    if args.join == False:
      plt.savefig(name_file+'_classes_L.png', dpi=150, bbox_inches='tight')
      plt.clf()
      plt.close()
      fig, ax = plt.subplots()

    for i in np.arange(0,n_class):
      if i==excluded_class:
        continue
      dfw = df[df['new_class'] == i]

      #print(f'class = {i}, time mean = {dfw.time.mean()}')
      dfw['time'] = dfw['time'].div(dfw.time.mean())
      histo_t = pd.DataFrame()
      histo_t['val'] =pd.cut(dfw['time'], bins=bins_time).value_counts()
      histo_t['bins_mean'] = [i.mid for i in histo_t.index]
      histo_t['x_range'] = [(i.right-i.left) for i in histo_t.index]
      histo_t['temp'] = histo_t['x_range']*histo_t['val']
      area_uc = histo_t['temp'].sum()
      histo_t['val'] = histo_t['val'].div(area_uc)
      histo_t.sort_values(by='bins_mean', inplace=True)
      histo_t_fitt = histo_t[histo_t.bins_mean>0.5]
      popt_t, pcov = curve_fit(func_exp, histo_t_fitt['bins_mean'].to_list(), histo_t_fitt['val'].to_list())
      df_temp = pd.DataFrame([['T',i,f'{popt_t[0]:.2f}',f'{popt_t[1]:.2f}']],columns=list_columns)
      df_fit_class = df_fit_class.append(df_temp)

      if args.join:
        histo_t.plot.scatter(x='bins_mean',y='val', ax=ax2, marker=marker_list[i], edgecolors='black', s=40, color=cmap(i), label=f'class {i}')
        ax2.set_yscale('log')
        ax2.set_ylim(0.001, 1)
        ax2.set_xlim(0, 4.0)
        ax2.set_xlabel(r'$T/T_m$', fontsize=MAX_SIZE)
        ax2.set_ylabel('')
        ax2.tick_params(axis='x', labelsize=MAX_SIZE)
        ax2.tick_params(axis='y', labelsize=MAX_SIZE)
        ax2.set_ylabel('N of activities', fontsize=MAX_SIZE)
        ax2.legend(prop={"size":20}, ncol=2)
      else:
        histo_t.plot.scatter(x='bins_mean',y='val', ax=ax, marker=marker_list[i], edgecolors='black', s=40, color=cmap(i), label=f'class {i}')
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
        ax.set_xlim(0, 4.0)
        ax.set_xlabel(r'$T/T_m$', fontsize=MAX_SIZE)
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=MAX_SIZE)
        ax.tick_params(axis='y', labelsize=MAX_SIZE)
        ax.set_ylabel('N of activities', fontsize=MAX_SIZE)
        ax.legend(prop={"size":20}, ncol=2)

    if args.join == False:
      plt.savefig(name_file+'_classes_T.png', dpi=150, bbox_inches='tight')
      plt.clf()
      plt.close()
      fig, ax = plt.subplots()

    for i in np.arange(0,n_class):
      if i==excluded_class:
        continue
      dfw = df[df['new_class'] == i]

      if args.join:
        dfw['av_speed'].hist(bins=bins_speed, ax=ax3, label = f'class {i}', alpha=alpha_val, density=True, color=cmap(i))
        ax3.set_xlabel('Average Velocity (km/h)', fontsize=MAX_SIZE)
        ax3.tick_params(axis='x', labelsize=MAX_SIZE)
        ax3.tick_params(axis='y', labelsize=MAX_SIZE)
        ax3.axvline(x=dfw.av_speed.mean(), linestyle='--', color=cmap(i))
        ax3.set_ylabel('Density of N activities', fontsize=MAX_SIZE)
        ax3.legend(prop={"size":20}, ncol=2)
      else:
        dfw['av_speed'].hist(bins=bins_speed, ax=ax, label = f'class {i}', alpha=alpha_val, density=True, color=cmap(i))
        ax.set_xlabel('Average Velocity (km/h)', fontsize=MAX_SIZE)
        ax.tick_params(axis='x', labelsize=MAX_SIZE)
        ax.tick_params(axis='y', labelsize=MAX_SIZE)
        ax.axvline(x=dfw.av_speed.mean(), linestyle='--', color=cmap(i))
        ax.set_ylabel('Density of N activities', fontsize=MAX_SIZE)
        ax.legend(prop={"size":20}, ncol=2)

    if args.join:
      plt.savefig(name_file+'_classes.png', dpi=150, bbox_inches='tight')
      plt.clf()
    else:
      plt.savefig(name_file+'_classes_V.png', dpi=150, bbox_inches='tight')
      plt.clf()

  df_average_value.to_csv(name_file+'_average_value.csv')
  df_fit_class.to_csv(name_file+'_param_interp.csv',index=False)
