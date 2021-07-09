import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib as mpl
from sympy import latex, sympify

sns.set_style("ticks")

def func_pl(x, a, b):
  return a * x**(-b)

def func_exp(x,a,b):
  return a * np.exp(-b*x)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', help='input file', required=True)
  parser.add_argument('-c', '--centers', help='centers input file')
  args = parser.parse_args()

  MAX_SIZE = 25

  # Rimini
  xlim_lengh = 10.0
  xlim_time  = 60.0
  xlim_speed = 150.0
  bins_length = 75
  bins_time = 75
  bins_speed = 80

  thresh_time = 5.0
  thresh_length = 10.0

  name_file =args.input.split('\\')[-1].split('.')[0]

  df = pd.read_csv(args.input, sep=';')
  df = df[df.time >= 60] #prendi solo viaggi che superano il minuto

  df.lenght = df.lenght.div(1000)
  df.time = df.time.div(60)
  df.av_speed = df.av_speed.multiply(3.6)

  mpl.rc('text', usetex = True)

  fig = plt.figure(figsize=(14,8), constrained_layout=True)

  gs = gridspec.GridSpec(2, 4, figure=fig)
  gs.update(wspace=0.5)
  ax1 = plt.subplot(gs[0, :2], )
  ax2 = plt.subplot(gs[0, 2:])
  ax3 = plt.subplot(gs[1, 1:3])

  print(df.mean())

  #### GLOBAL L ####
  df_l = df[df.lenght < thresh_length]
  histo_l = pd.DataFrame()
  histo_l['val'] =pd.cut(df_l['lenght'], bins=80).value_counts()
  histo_l['bins_mean'] = [i.mid for i in histo_l.index]
  histo_l.sort_values(by='bins_mean', inplace=True)

  histo_l.plot.scatter(x='bins_mean',y='val', ax=ax1, edgecolors='black', s=40)

  ax1.set_yscale('log')
  ax1.set_ylim(10, 1000)
  ax1.set_xlabel('length (km)', fontsize=MAX_SIZE)
  ax1.axvline(x=df.lenght.mean(), color='k', linestyle='--')
  ax1.tick_params(axis='x', labelsize=MAX_SIZE)
  ax1.tick_params(axis='y', labelsize=MAX_SIZE)
  ax1.set_ylabel('')


  #### GLOBAL T ####
  df_t = df[df.time > thresh_time]
  histo_t = pd.DataFrame()
  histo_t['val'] =pd.cut(df_t['time'], bins=80).value_counts()
  histo_t['bins_mean'] = [i.mid for i in histo_t.index]
  histo_t.sort_values(by='bins_mean', inplace=True)
  histo_t_fitt = histo_t[histo_t['bins_mean']>=10.0]
  popt_t, pcov = curve_fit(func_pl, histo_t_fitt['bins_mean'].to_list(), histo_t_fitt['val'].to_list())

  histo_t.plot.scatter(x='bins_mean',y='val', ax=ax2, edgecolors='black', s=40)
  equation_t = r'${} * x^{{{}}}$'.format(f'{popt_t[0]:.2e}',f'-{popt_t[1]:.2f}')
  ax2.plot(histo_t.bins_mean, func_pl(histo_t.bins_mean, *popt_t),'--', label=equation_t)
  ax2.set_xscale('log')
  ax2.set_yscale('log')
  ax2.set_ylim(0.1, 10000)
  ax2.set_xlabel('time (min)', fontsize=MAX_SIZE)
  ax2.axvline(x=df.time.mean(), color='k', linestyle='--')
  ax2.tick_params(axis='x', labelsize=MAX_SIZE)
  ax2.tick_params(axis='y', labelsize=MAX_SIZE)
  ax2.set_ylabel('')
  ax2.legend(fontsize=MAX_SIZE)


  #### GLOBAL V ####
  df['av_speed'].hist(bins=bins_speed, ax=ax3, density=True)

  #histo_v = pd.DataFrame()
  #histo_v['val'] =pd.cut(df['av_speed'], bins=bins_speed).value_counts()
  #histo_v['bins_mean'] = [i.mid for i in histo_v.index]
  #histo_v.sort_values(by='bins_mean', inplace=True)
  #histo_v['val'] = histo_v['val'] /histo_v['val'].abs().max()
  #histo_v.plot.bar(x='bins_mean',y='val', ax=ax3)

  ax3.set_xlim(0,xlim_speed)
  ax3.set_xlabel('Velocity (km/h)', fontsize=MAX_SIZE)
  ax3.axvline(x=df.av_speed.mean(), color='k', linestyle='--')
  ax3.tick_params(axis='x', labelsize=MAX_SIZE)
  ax3.tick_params(axis='y', labelsize=MAX_SIZE)
  plt.savefig(name_file+'_global_stats.png', dpi=150, bbox_inches='tight')
  plt.clf()

  if args.centers:
    df_centers = pd.read_csv(args.centers, sep=';', header=None)
    n_class = len(df_centers)
    for i in np.arange(0,n_class):
      feat_i = []
      for j in df_centers.columns:
        feat_i.append(df_centers.at[i,j])
      print(f'class {i} {feat_i[1]*3.6:.2f} km/h & {feat_i[2]*3.6:.2f} km/h & {feat_i[3]*3.6:.2f} km/h & {feat_i[4]:.2f} & {feat_i[5]}')

    cmap = plt.get_cmap("tab10")

    alpha_val=0.6

    fig = plt.figure(figsize=(14,8), constrained_layout=True)

    gs = gridspec.GridSpec(2, 4, figure=fig)
    gs.update(wspace=0.5)
    ax1 = plt.subplot(gs[0, :2], )
    ax2 = plt.subplot(gs[0, 2:])
    ax3 = plt.subplot(gs[1, 1:3])

    excluded_class=0

    for i in np.arange(0,n_class):
      if i==excluded_class:
        continue
      dfw = df[df['class'] == i]
      print(f'class = {i}, lenght mean = {dfw.lenght.mean()}')
      dfw['lenght'] = dfw.lenght.div(dfw.lenght.mean())
      histo_l = pd.DataFrame()
      histo_l['val'] =pd.cut(dfw['lenght'], bins=bins_length).value_counts()
      histo_l['bins_mean'] = [i.mid for i in histo_l.index]
      histo_l.sort_values(by='bins_mean', inplace=True)

      histo_l.plot.scatter(x='bins_mean',y='val', ax=ax1, edgecolors='black', s=40, color=cmap(i),label=f'class {i}')
      ax1.set_yscale('log')
      ax1.set_ylim(0.1, 10000)
      ax1.set_xlim(0,4.0)
      ax1.set_xlabel(r'$L/L_m$', fontsize=MAX_SIZE)
      ax1.set_ylabel('')
      ax1.tick_params(axis='x', labelsize=MAX_SIZE)
      ax1.tick_params(axis='y', labelsize=MAX_SIZE)
      ax1.legend(prop={"size":20}, ncol=2)

    for i in np.arange(0,n_class):
      if i==excluded_class:
        continue
      dfw = df[df['class'] == i]

      print(f'class = {i}, time mean = {dfw.time.mean()}')
      dfw['time'] = dfw['time'].div(dfw.time.mean())
      histo_t = pd.DataFrame()
      histo_t['val'] =pd.cut(dfw['time'], bins=bins_time).value_counts()
      histo_t['bins_mean'] = [i.mid for i in histo_t.index]
      histo_t.sort_values(by='bins_mean', inplace=True)
      histo_t.plot.scatter(x='bins_mean',y='val', ax=ax2, edgecolors='black', s=40, color=cmap(i), label=f'class {i}')

      ax2.set_yscale('log')
      ax2.set_ylim(0.1, 10000)
      ax2.set_xlim(0, 4.0)
      ax2.set_xlabel(r'$T/T_m$', fontsize=MAX_SIZE)
      ax2.set_ylabel('')
      ax2.tick_params(axis='x', labelsize=MAX_SIZE)
      ax2.tick_params(axis='y', labelsize=MAX_SIZE)
      ax2.legend(prop={"size":20}, ncol=2)

    for i in np.arange(0,n_class):
      if i==excluded_class:
        continue
      dfw = df[df['class'] == i]
      #dfw['av_speed '] = dfw['av_speed'].copy().div(dfw.av_speed.mean())

      dfw['av_speed'].hist(bins=bins_speed, ax=ax3, label = f'class {i}', alpha=alpha_val, density=True, color=cmap(i))
      #ax3.set_xlim(0,70.0)
      ax3.set_xlabel('Velocity (km/h)', fontsize=MAX_SIZE)
      ax3.tick_params(axis='x', labelsize=MAX_SIZE)
      ax3.tick_params(axis='y', labelsize=MAX_SIZE)
      print(f'class = {i}, speed mean = {dfw.av_speed.mean()}')
      ax3.axvline(x=dfw.av_speed.mean(), linestyle='--', color=cmap(i))
      ax3.legend(prop={"size":20}, ncol=2)

    plt.savefig(name_file+'_classes.png', dpi=150, bbox_inches='tight')
    plt.clf()