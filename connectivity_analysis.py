import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
from respi_analysis import analyse_resp


path_savefig = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Analyses\\pilote\\FC'
path_savefig_PSD = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Analyses\\pilote\\PSD_Coh'
path_save_precompute = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Analyses\\precompute'
path_data = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Analyses\\preprocessing\\data'

plot_token = False
save_fig_PSD = True



############################
######## PARAMETERS ########
############################





conditions = ['CV', 'FV', 'SV']
srate = 1000
nb_point_by_cycle = srate*2




# for PSD

nwind = int( 20*srate ) # window length in seconds*srate
nfft = nwind*5 # if no zero padding nfft = nwind
noverlap = np.round(nwind/2) # number of points of overlap here 50%
hannw = scipy.signal.windows.hann(nwind) # hann window



################################
######## LOAD DATA ########
################################



os.chdir(path_data)

raw_allcond = {}
data_allcond = {}

for cond_i in range(np.size(conditions)):

    raw_allcond[conditions[cond_i]] = mne.io.read_raw_fif('EEG_pilote_' + conditions[cond_i] + '.fif')
    data_allcond[conditions[cond_i]] = raw_allcond[conditions[cond_i]].get_data()


if srate != list(raw_allcond.values())[0].info['sfreq']:
    print('ERROR srate DIFFERENT')

chan_list = list(raw_allcond.values())[0].info['ch_names']

if plot_token == True :
    duration = 20.
    start = 0.
    n_chan = 1
    list(raw_allcond.values())[0].plot(scalings='auto',duration=duration,start=start,n_channels=n_chan)


########################################
######## COMPUTE RESPI FEATURES ########
########################################

respi_allcond = {}
for cond_i in range(np.size(conditions)):

    respi_allcond[conditions[cond_i]] = list(data_allcond.values())[cond_i][-1,:]



t_start = 0
resp_features_allcond = {}
fig_respi1_allcond = {}
fig_respi2_allcond = {}
for cond_i in range(np.size(conditions)):
    
    if cond_i == 0:
        condition_title = 'Comfort Ventilation'
    elif cond_i == 1:
        condition_title = 'Fast Ventilation'
    elif cond_i == 2:
        condition_title = 'Slow ventilation' 

    resp_features_allcond[conditions[cond_i]], fig_respi1_allcond[conditions[cond_i]], fig_respi2_allcond[conditions[cond_i]] = analyse_resp(list(respi_allcond.values())[cond_i], srate, t_start, condition=condition_title)
    plt.close(), plt.close()

if plot_token == True :
    cond_i = 0
    list(fig_respi1_allcond.values())[cond_i].show()
    list(fig_respi2_allcond.values())[cond_i].show()


#####################
######## PLI ########
#####################



# frequency parameters
min_freq =  2
max_freq = 40
num_frex = 50

srate = 1000
wavetime = np.arange(-2,2,1/srate) 
frex  = np.linspace(min_freq,max_freq,num_frex)
wavelets = np.zeros((num_frex,len(wavetime)) ,dtype=complex)
ncycle = 7 

for fi in range(0,num_frex):
    s = ncycle / (2*np.pi*frex[fi])
    gw = np.exp(-wavetime**2/ (2*s**2)) 
    sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
    mw =  gw * sw
    wavelets[fi,:] = mw





data = data_allcond.get('CV')
seed = 'Fp1'
#for nchan in range(len(chan_list)-2) :
for nchan in range(5) :

    if nchan == chan_list.index(seed):
        continue
    else :

        # initialize output time-frequency data
        ispc = np.zeros((num_frex))
        pli  = np.zeros((num_frex))

        x = data[chan_list.index(seed),:]
        y = data[nchan,:]

        # convolution per frequency
        for fi in range(0,num_frex):
            
            as1 = scipy.signal.fftconvolve(x, wavelets[fi], 'same')
            as2 = scipy.signal.fftconvolve(y, wavelets[fi], 'same')

            # collect "eulerized" phase angle differences
            cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
            
            # compute ISPC and PLI (and average over trials!)
            ispc[fi] = np.abs(np.mean(cdd))
            pli[fi] = np.abs(np.mean(np.sign(np.imag(cdd))))


        plt.plot(frex,ispc, label='ISPC')
        plt.plot(frex,pli, label='PLI')
        plt.ylim(0,1.1)
        plt.title(seed+'_'+chan_list[nchan])
        plt.xlabel('Frequency (Hz)'), plt.ylabel('Synchronization strength')
        plt.legend()
        plt.show()










####################################################
############# FOR ONE FREQ #########################
####################################################

# frequency parameters
min_freq =  8
max_freq = 12
num_frex = 5

srate = 1000
wavetime = np.arange(-2,2,1/srate) 
frex  = np.linspace(min_freq,max_freq,num_frex)
wavelets = np.zeros((num_frex,len(wavetime)) ,dtype=complex)
ncycle = 7 

for fi in range(0,num_frex):
    s = ncycle / (2*np.pi*frex[fi])
    gw = np.exp(-wavetime**2/ (2*s**2)) 
    sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
    mw =  gw * sw
    wavelets[fi,:] = mw

mat_con_pli_allcond = []
mat_con_itpc_allcond = []
for condi, cond in enumerate(conditions) :

    data = data_allcond.get(conditions[condi])
    mat_con_pli = np.zeros((len(chan_list)-2,len(chan_list)-2))
    mat_con_itpc = np.zeros((len(chan_list)-2,len(chan_list)-2))

    for seed in range(len(chan_list)-2) :
    #for seed in range(5) :

        x = data[0,:]

        as1_mat = np.zeros((num_frex, len(x)), dtype='complex')
        for fi in range(0,num_frex):
            as1_mat[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi], 'same')


        for nchan in range(len(chan_list)-2) :

            print('{:.2f} {:.2f}'.format((seed/(len(chan_list)-2)),nchan/((len(chan_list)-2))))
            if nchan == seed : 
                continue
                
            else :

                # initialize output time-frequency data
                ispc = np.zeros((num_frex))
                pli  = np.zeros((num_frex))

                y = data[nchan,:]

                # convolution per frequency
                for fi in range(0,num_frex):
                    
                    as1 = as1_mat[fi,:]
                    as2 = scipy.signal.fftconvolve(y, wavelets[fi], 'same')

                    # collect "eulerized" phase angle differences
                    cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
                    
                    # compute ISPC and PLI (and average over trials!)
                    ispc[fi] = np.abs(np.mean(cdd))
                    pli[fi] = np.abs(np.mean(np.sign(np.imag(cdd))))

                mat_con_pli[seed,nchan] = np.mean(ispc,0)
                mat_con_itpc[seed,nchan] = np.mean(pli,0)

    mat_con_pli_allcond.append(mat_con_pli)
    mat_con_itpc_allcond.append(mat_con_itpc)



########################
######## SAVE FIG ########
########################


os.chdir(path_savefig)


fig = plt.figure(facecolor='black')
for i in range(3):
    mne.viz.plot_connectivity_circle(mat_con_pli_allcond[i], node_names=chan_list[0:-2], n_lines=None, title=conditions[i], show=False, fig=fig, subplot=(1, 3, i+1))
plt.suptitle('PLI 8-12Hz', color='w')
plt.show()


fig = plt.figure(facecolor='black')
for i in range(3):
    mne.viz.plot_connectivity_circle(mat_con_itpc_allcond[i], node_names=chan_list[0:-2], n_lines=None, title=conditions[i], show=False, fig=fig, subplot=(1, 3, i+1))
plt.suptitle('PC 8-12Hz', color='w')
plt.show()


mne.viz.plot_connectivity_circle(mat_con_itpc, node_names=chan_list[0:-2], n_lines=None, title='ISPC', show=True)























