
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import joblib

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False




########################################
######## COMPUTE BASELINE ######## 
########################################



#sujet, odor_i, band_prep = 'P01', 'o', 'wb'
def compute_and_save_baseline(sujet, odor_i, band_prep):

    print('#### COMPUTE BASELINES ####')

    #### verify if already computed
    verif_band_compute = []
    for band in list(freq_band_dict[band_prep].keys()):
        if os.path.exists(os.path.join(path_precompute, sujet, 'Baselines', f'{sujet}_{odor_i}_{band}_baselines.npy')):
            verif_band_compute.append(True)

    if np.sum(verif_band_compute) > 0:
        print(f'{sujet}_{odor_i} : BASELINES ALREADY COMPUTED')
        return
            
    #### open raw
    data_allcond = load_data_sujet(sujet, band_prep, 'allcond', odor_i, 0)

    #### Data vizualisation
    if debug == True :
        plt.plot(data_allcond[-1,:])
        plt.show()

    #### remove unused chan
    data_allcond = data_allcond[:len(chan_list_eeg),:]
    
    #### generate all wavelets to conv
    wavelets_to_conv = {}
        
    #band, freq = 'theta', [2, 10]
    for band, freq in freq_band_dict[band_prep].items():

        #### compute the wavelets
        wavelets_to_conv[band], nfrex = get_wavelets(band_prep, freq)  

    # plot all the wavelets
    if debug == True:
        for band in list(wavelets_to_conv.keys()):
            wavelets2plot = wavelets_to_conv[band]
            plt.pcolormesh(np.arange(wavelets2plot.shape[1]),np.arange(wavelets2plot.shape[0]),np.real(wavelets2plot))
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(band)
            plt.show()

    #### get trig values
    os.chdir(os.path.join(path_prep, sujet, 'info'))
    trig = pd.read_excel(f'{sujet}_{odor_i}_trig_allcond.xlsx')

    #### compute convolutions
    n_band_to_compute = len(list(freq_band_dict[band_prep].keys()))

    os.chdir(path_memmap)
    baseline_allchan = np.memmap(f'{sujet}_{odor_i}_baseline_convolutions.dat', dtype=np.float64, mode='w+', shape=(n_band_to_compute, data_allcond.shape[0], nfrex))

        #### compute
    #n_chan = 0
    def baseline_convolutions(n_chan):

        print_advancement(n_chan, data_allcond.shape[0], steps=[25, 50, 75])

        x = data_allcond[n_chan,:]
        #band_i, band = 0, 'theta'
        for band_i, band in enumerate(list(wavelets_to_conv.keys())):

            baseline_coeff_band = np.array(())
            #fi = 0
            for fi in range(nfrex):
                
                fi_conv = abs(scipy.signal.fftconvolve(x, wavelets_to_conv[band][fi,:], 'same'))**2

                #### chunk data
                fi_conv_chunked = np.array([])

                for cond in conditions_allsession:

                    _start = int(trig['time'][trig['name'] == cond].values[0].split(',')[0][1:])
                    _stop = int(trig['time'][trig['name'] == cond].values[0].split(',')[-1][1:-1])
                    fi_conv_chunked_i = fi_conv[_start:_stop]

                    fi_conv_chunked = np.append(fi_conv_chunked, fi_conv_chunked_i)

                baseline_coeff_band = np.append(baseline_coeff_band, np.median(fi_conv_chunked))
        
            baseline_allchan[band_i, n_chan,:] = baseline_coeff_band

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(baseline_convolutions)(n_chan) for n_chan in range(np.size(data_allcond,0)))

    #### save baseline
    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))

    for band_i, band in enumerate(list(freq_band_dict[band_prep].keys())):
    
        np.save(f'{sujet}_{odor_i}_{band}_baselines.npy', baseline_allchan[band_i, :, :])

    #### remove memmap
    os.chdir(path_memmap)
    os.remove(f'{sujet}_{odor_i}_baseline_convolutions.dat')









################################
######## EXECUTE ########
################################


if __name__== '__main__':


    #### params
    sujet = 'P01'
    band_prep = 'wb'
    odor_i = 'o'

    #### compute
    #compute_and_save_baseline(sujet, odor_i, band_prep)
    
    #### slurm execution
    #sujet = sujet_list[0]
    for sujet in sujet_list:
        for odor_i in odor_list:
            for band_prep in band_prep_list:
                execute_function_in_slurm_bash('n3_precompute_baselines', 'compute_and_save_baseline', [sujet, odor_i, band_prep])





