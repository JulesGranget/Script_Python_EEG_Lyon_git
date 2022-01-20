
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
import mne
import pandas as pd
import joblib

from n0_config import *
from n0bis_analysis_functions import *

debug = False




########################################
######## COMPUTE BASELINE ######## 
########################################

#sujet_i, session_i = 'Pilote', 1
def compute_and_save_baseline(sujet_i, session_i):

    print('#### COMPUTE BASELINES ####')

    #### verify if already computed
    if os.path.exists(os.path.join(path_prep, sujet, 'baseline', f'{sujet}_{session_i}_baselines.npy')):
        print(f'{sujet}_{session_i} : BASELINES ALREADY COMPUTED')
        return

    #### open raw
    os.chdir(os.path.join(path_data, sujet_i, f'{sujet_i.lower()}_sub01', f'ses_0{str(session_i+1)}'))

    raw = mne.io.read_raw_brainvision(f'{sujet_i}01_session{session_i+1}.vhdr', preload=True)

    #### Data vizualisation
    if debug == True :
        duration = 4.
        n_chan = 20
        raw.plot(scalings='auto',duration=duration,n_channels=n_chan)# verify

    #### remove unused chan
    drop_chan = ['36', '37', '38', '39', '40']
    raw.drop_channels(drop_chan)
    #raw.info # verify

    #### identify EOG and rename chans
    mne.rename_channels(raw.info, {EOG_chan[sujet_i]['HEOG'] : 'HEOG', EOG_chan[sujet_i]['VEOG'] : 'VEOG'})
    
    #### select raw_eeg
    raw_eeg = raw.copy()
    drop_chan = ['ECG','GSR','Respi']
    raw_eeg.info['ch_names']
    raw_eeg.drop_channels(drop_chan)

    #### select aux chan
    raw_aux = raw.copy()
    select_chan = ['ECG','GSR','Respi']
    raw_aux = raw_aux.pick_channels(select_chan)

    #### generate triggers
    trig = pd.DataFrame(raw.annotations)
    trig_time = (trig['onset'].values[1:] * raw.info['sfreq']).astype(int)

    trig_names = []
    for trig_i in trig['description'].values:
        if trig_i == 'New Segment/':
            continue
        else:
            trig_names.append(trig_i[10:].replace(' ', ''))

    trig = {'time' : trig_time, 'name' : trig_names}

    #raw_eeg.info # verify
    #raw_aux.info # verify
    
    del raw

    #### remove EOG
    if 'HEOG' in raw_eeg.info['ch_names'] or 'VEOG' in raw_eeg.info['ch_names']: 
        drop_chan = []
        if 'HEOG' in raw_eeg.info['ch_names']:
            drop_chan.append('HEOG')
        if 'VEOG' in raw_eeg.info['ch_names']:
            drop_chan.append('VEOG')
        raw_eeg.drop_channels(drop_chan)

    #### get raw params
    data = raw_eeg.get_data()
    srate = raw_eeg.info['sfreq']
    
    #### generate all wavelets to conv
    wavelets_to_conv = {}
    
    #band_prep, band_prep_i = 'lf', 0
    for band_prep_i, band_prep in enumerate(band_prep_list):
        
        #### select wavelet parameters
        if band_prep == 'lf':
            wavetime = np.arange(-2,2,1/srate)
            nfrex = nfrex_lf
            ncycle_list = np.linspace(ncycle_list_lf[0], ncycle_list_lf[1], nfrex) 

        if band_prep == 'hf':
            wavetime = np.arange(-.5,.5,1/srate)
            nfrex = nfrex_hf
            ncycle_list = np.linspace(ncycle_list_hf[0], ncycle_list_hf[1], nfrex)

        #band, freq = 'theta', [2, 10]
        for band, freq in freq_band_list[band_prep_i].items():

            #### compute wavelets
            frex  = np.linspace(freq[0],freq[1],nfrex)
            wavelets = np.zeros((nfrex,len(wavetime)) ,dtype=complex)

            # create Morlet wavelet family
            for fi in range(0,nfrex):
                
                s = ncycle_list[fi] / (2*np.pi*frex[fi])
                gw = np.exp(-wavetime**2/ (2*s**2)) 
                sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
                mw =  gw * sw

                wavelets[fi,:] = mw
                
            # plot all the wavelets
            if debug == True:
                plt.pcolormesh(wavetime,frex,np.real(wavelets))
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Real part of wavelets')
                plt.show()

            wavelets_to_conv[band] = wavelets

    # plot all the wavelets
    if debug == True:
        for band in list(wavelets_to_conv.keys()):
            wavelets2plot = wavelets_to_conv[band]
            plt.pcolormesh(np.arange(wavelets2plot.shape[1]),np.arange(wavelets2plot.shape[0]),np.real(wavelets2plot))
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(band)
            plt.show()

    #### compute convolutions

        #### count frequencies to compute
    n_fi2conv = 0
    for band in list(wavelets_to_conv.keys()):
        n_fi2conv += wavelets_to_conv[band].shape[0]

    os.chdir(path_memmap)
    baseline_allchan = np.memmap(f'{sujet}_{session_i}_baseline_convolutions.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], n_fi2conv))

        #### compute
    #n_chan = 0
    def baseline_convolutions(n_chan):

        if n_chan/np.size(data,0) % .2 <= .01:
            print("{:.2f}".format(n_chan/np.size(data,0)))

        x = data[n_chan,:]

        baseline_coeff = np.array(())

        for band in list(wavelets_to_conv.keys()):

            for fi in range(wavelets_to_conv[band].shape[0]):
                
                fi_conv = abs(scipy.signal.fftconvolve(x, wavelets_to_conv[band][fi,:], 'same'))**2
                baseline_coeff = np.append(baseline_coeff, np.median(fi_conv))
        
        baseline_allchan[n_chan,:] = baseline_coeff

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(baseline_convolutions)(n_chan) for n_chan in range(np.size(data,0)))

    #### save baseline
    os.chdir(os.path.join(path_prep, sujet, 'baseline'))
    np.save(f'{sujet}_{session_i}_baselines.npy', baseline_allchan)

    #### remove memmap
    os.chdir(path_memmap)
    os.remove(f'{sujet}_{session_i}_baseline_convolutions.dat')




################################
######## EXECUTE ########
################################


if __name__== '__main__':


    #### params
    sujet = 'Pilote'
    session_i = 1

    #### compute
    #compute_and_save_baseline(sujet, session_i)
    
    #### slurm execution
    for session_i in range(3): 
        execute_function_in_slurm_bash('n2_baseline_computation', 'compute_and_save_baseline', [sujet, session_i+1])





