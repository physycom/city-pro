import seaborn as sns

'''
Plots ->   (time,[number of visit per coil_A,
                  number of visit per poly coil_A mdt data]
            )
      ->   (time,[(number of visit per coil_A) / (maximum of the data),
                  (number of visit per poly coil_A mdt data )/ maximum of the data]
            )
'''
def handle_opendata(df_opendata,start,end,nododa,nodoa):
    '''
    Input:
        dataframe opendata
        node from
        node to
        time start analysis
        time stop analysis
    Output:
        Total number of people passed in the poly
    '''
    
    
    df_opendata['data'] = convert2datetime(df_opendata['data'])
    df_opendata = df_opendata.sort_values(by = 'data',ascending = False)
    df_ocomp = df_opendata.loc[df_opendata['data'] >= start]
    df_ocomp = df_ocomp.loc[df_ocomp['data'] <= end]
    dftemp = df_opendata.loc[df_opendata['Nodo da'] == nododa]
    dftemp = dftemp.loc[dftemp['Nodo a'] == nodoa]
    total_number = 0
    for h in list_hours_opendata:
        total_number += df_ocomp[list_hours_opendata].to_numpy().sum()
    print(total_number)


def handle_dfcoils(df_coils,coil0,coil1,day,start,end,old = False):

    if old:
        df_ = df_coils.loc[df_coils['wday'] == day]
    else:
        df_coils['time'] = eliminapiù00(df_coils['datetime'])
        df_coils['time'] = convert2datetimecity(df_coils['time'])
        df_ = df_coils.loc[df_coils['time'] >= start]
        df_ = df_.loc[df_['time'] <= end]
    debug = True    
    c0 = df_[coil0].to_numpy()
    c1 = df_[coil1].to_numpy()
    aggregated1h_flux = []
    for i in range(len(c1)):
        if i%4 == 0:
            aggregated1h_flux.append(sum(c0[i:i+4]) + sum(c1[i:i+4]))
    debug = False
    if debug:
        print('aggregated fluxes from coils:\n',aggregated1h_flux)
    
    return aggregated1h_flux
    
def create_listsuccessive_start_stop_days(start,stop,number_days):
    '''
    Creates a list of number_days start and stop days in datetime format:
    "%Y-%m-%d %H:%M:%S"
    '''
    list_start = []
    list_stop = []
    for n in range(number_days):
        list_start.append(start + datetime.timedelta(days = n))
        list_stop.append(stop + datetime.timedelta(days = n))
    return list_start,list_stop

def eliminapiù00(dt_string):
    return [c[:19] for c in dt_string] 

def convert2datetimecity(x):
    return [datetime.datetime.strptime(x.iloc[i],"%Y-%m-%d %H:%M:%S") for i in range(len(x))]

def process_df_tf(df_tf,poly_cid):
    debug = False
    df_tf = df_tf.sort_values(by = 'time',ascending = False)
    df_tf['time'] = convert2datetimecity(df_tf['time'])
    maskcid = [True if x == poly_cid else False for x in df_tf['cid'].to_numpy()]
    temp = df_tf.loc[maskcid]
    if debug:
        print('type of the timestamp:\t',type(df_tf.iloc[0]['time']))
    maskend = [True if x < end else False for x in temp['time']]
    temp = temp.loc[maskend]
    maskstart = [True if x > start else False for x in temp['time']]    
    temp = temp.loc[maskstart]
    if debug:
        print('dataframe masked with cid and time:\n',temp)
        print('True in mask start\t',True in maskstart,'\nTrue in mask end:\t',True in maskend)
    temp = temp.sort_values(by = 'time')
    if debug:
        print('dataframe fluxes mobile phone data of the chosen poly:\n',temp)
    return temp

def plot_tot_flux(temp,aggregated1h_flux,save_path,day,via):
    fig,ax = plt.subplots(1,1,figsize = (10,8))
    if len(temp) == 22:
        plt.plot(temp['time'],np.roll(temp['total_fluxes'],1))
        plt.plot(temp['time'],aggregated1h_flux[:-2])        
    elif len(temp) == 23:
        plt.plot(temp['time'],np.roll(temp['total_fluxes'],1))
        plt.plot(temp['time'],aggregated1h_flux[:-1])
    plt.xticks(rotation = 90)
    ax.set_xlabel('time (h)')
    ax.set_ylabel('total flux')
    ax.set_title('poly {0}, {1}, {2} '.format(np.array(temp['cid'])[0],day,via))
    ax.legend(['mdt','coil regione'])
    plt.savefig(os.path.join(save_path,'total_flux_{0}_{1}'.format(day,via)))
    plt.show()

def plot_penetration(temp,aggregated1h_flux,save_path,day,via):
    fig,ax = plt.subplots(1,1,figsize = (10,8))
    if len(temp) == 22:
        plt.plot(temp['time'],np.roll(temp['total_fluxes'].to_numpy(),1)/np.array(aggregated1h_flux[:-2])*100)
    elif len(temp) == 23:
        plt.plot(temp['time'],np.roll(temp['total_fluxes'].to_numpy(),1)/np.array(aggregated1h_flux[:-1])*100)
    plt.xticks(rotation = 90)
    ax.set_xlabel('time (h)')
    ax.set_ylabel('penetration (%)')
    ax.set_title('poly {0}, {1}, {2} '.format(np.array(temp['cid'])[0],day,via))
#    ax.legend(['mdt','coil regione'])
    plt.savefig(os.path.join(save_path,'penetration_{0}_{1}'.format(day,via)))
    plt.show()

def create_dfpenflux(temp,aggregated1h_flux,save_path,day,via):
    df_pen_flux = pd.DataFrame()
    df_pen_flux['time'] = temp['time']
    if len(aggregated1h_flux)-len(temp['total_fluxes'].to_numpy()) == 1:        
        df_pen_flux['penetration'] = np.roll(temp['total_fluxes'].to_numpy(),1)/np.array(aggregated1h_flux[:-1])*100
        df_pen_flux['rescaled_max_mdt'] = np.array(aggregated1h_flux[:-1])/max(np.array(aggregated1h_flux[:-1]))
        df_pen_flux['rescaled_max_coils'] = np.roll(temp['total_fluxes'].to_numpy(),1)/max(temp['total_fluxes'].to_numpy()[4:])
        df_pen_flux['flux_coils'] = np.roll(temp['total_fluxes'].to_numpy(),1)
        df_pen_flux['flux_mdt'] = np.array(aggregated1h_flux[:-1])
    elif len(aggregated1h_flux)-len(temp['total_fluxes'].to_numpy()) == 2:        
        df_pen_flux['penetration'] = np.roll(temp['total_fluxes'].to_numpy(),1)/np.array(aggregated1h_flux[:-2])*100
        df_pen_flux['rescaled_max_mdt'] = np.array(aggregated1h_flux[:-2])/max(np.array(aggregated1h_flux[:-2]))
        df_pen_flux['rescaled_max_coils'] = np.roll(temp['total_fluxes'].to_numpy(),1)/max(temp['total_fluxes'].to_numpy()[4:])
        df_pen_flux['flux_coils'] = np.roll(temp['total_fluxes'].to_numpy(),1)
        df_pen_flux['flux_mdt'] = np.array(aggregated1h_flux[:-2])

    elif len(aggregated1h_flux)==len(temp['total_fluxes'].to_numpy()):
        df_pen_flux['penetration'] = np.roll(temp['total_fluxes'].to_numpy(),1)/np.array(aggregated1h_flux[:])*100
        df_pen_flux['rescaled_max_mdt'] = np.array(aggregated1h_flux[:])/max(np.array(aggregated1h_flux[:]))
        df_pen_flux['rescaled_max_coils'] = np.roll(temp['total_fluxes'].to_numpy(),1)/max(temp['total_fluxes'].to_numpy()[4:])
        df_pen_flux['flux_coils'] = np.roll(temp['total_fluxes'].to_numpy(),1)
        df_pen_flux['flux_mdt'] = np.array(aggregated1h_flux[:])
    else:
        print('dataframe flussi telefonici più lunghi di quelli su coil, di solito non accede. Errore che serve controllare')
    df_pen_flux.to_csv(os.path.join(save_path,'df_mdt_coil.csv'),';')
    return df_pen_flux

    
def plot_rescaled(df_pen_flux,day,via):
    fig,ax = plt.subplots(1,1,figsize = (10,8))
    plt.plot(df_pen_flux['time'],df_pen_flux['rescaled_max_mdt'])
    plt.plot(df_pen_flux['time'],df_pen_flux['rescaled_max_coils'])
    plt.xticks(rotation = 90)
    ax.set_xlabel('time (h)')
    ax.set_ylabel('flux / max flux')
    ax.set_title('poly {0}, {1}, {2} '.format(np.array(temp['cid'])[0],day,via))
    ax.legend(['mdt','coil regione'])
    plt.savefig(os.path.join(save_path,'rescaled_bymax_{0}_{1}'.format(day,via)))
    plt.show()

def plot_avgtime_over_max(df_pen_flux,mdt,coil,day,via,poly):
    '''
    Plots the the average flux rescaled by max:
    mdt, coil are arrays that contain the averaged fluxed in days
    '''
    fig,ax = plt.subplots(1,1,figsize = (10,8))
    plt.plot(df_pen_flux['time'],mdt/max(mdt))
    plt.plot(df_pen_flux['time'],coil/max(coil[4:]))
    plt.xticks(rotation = 90)
    ax.set_xlabel('time (h)')
    ax.set_ylabel('average (in time) flux / max flux')
    ax.set_title('poly {0}, {1}, {2} '.format(poly,day,via))
    ax.legend(['mdt','coil regione'])
    plt.savefig(os.path.join(save_path,'average_in_time_rescaled_bymax_{0}_{1}'.format(day,via)))
    plt.show()
    
                                       
                                       
def handleplot_time_fluxes(df_tf,start,end,aggregated1h_flux,day,via,poly_cid = 232080,plotta = False):
    '''
    Input:
        dataframe timed fluxes
        poly of confront
        time start analysis
        time stop analysis
    Output:
        Figure of timed fluxes
    '''
    temp = process_df_tf(df_tf,poly_cid)
    if plotta:
        plot_tot_flux(temp,aggregated1h_flux,save_path,day,via)
        plot_penetration(temp,aggregated1h_flux,save_path,day,via)
    df_pen_flux = create_dfpenflux(temp,aggregated1h_flux,save_path,day,via)
    if plotta:
        plot_rescaled(df_pen_flux,day,via)
    return df_pen_flux


bologna = True
if bologna:
    list_days = ['Fri','Sat']
    number_days = 2
    list_start,list_stop = create_listsuccessive_start_stop_days(start,end,number_days)
    via = ['via stradelli Guelfi','via Bruno Tosarelli','via Mezzanotte']
    list_poly_cid = [232080,179909,214686]
    coils = [['279_0','279_1'],['156_0','156_1'],['282_0','282_1']]
    for k in range(len(via)):
        array_avg = np.zeros((2,23))
        for i in range(len(list_days)):
            aggregated1h_flux = handle_dfcoils(df_coils,coils[k][0],coils[k][1],list_days[i],list_start[i],list_stop[i])
            df_pen_flux = handleplot_time_fluxes(df_tf,list_start[i],list_stop[i],aggregated1h_flux,list_days[i],via[k],list_poly_cid[k]) 
            array_avg[0] += df_pen_flux['flux_coils'].to_numpy()
            array_avg[1] += df_pen_flux['flux_mdt'].to_numpy()
        array_avg[0] = array_avg[0]/len(list_days)
        array_avg[1] = array_avg[1]/len(list_days)
        print(array_avg[0])
        plot_avgtime_over_max(df_pen_flux,array_avg[0],array_avg[1],list_days[i],via[k],list_poly_cid[k])                   