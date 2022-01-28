

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib

from n0_config import *
from n0bis_analysis_functions import *


debug = False




################################
######## EXECUTE ########
################################

if __name__ == '__main__':


    #### Params
    session_eeg = 0

    

    

    local_computing = False
    
    ########################################
    ######## LOCAL COMPUTING ######## 
    ########################################

    if local_computing:
    
        #### Params
        session_eeg = 0

        #### Pxx Cxy Cyclefreq
        conditions, chan_list, chan_list_ieeg, srate, respfeatures_allcond, respi_ratio_allcond, nwind, nfft, noverlap, hannw = get_all_info(session_eeg)
        respi_mean_allcond = compute_respi_mean(respfeatures_allcond)
        
        surrogates_allcond = load_surrogates_session(session_eeg)

        Pxx_allcond, Cxy_allcond, cyclefreq_allcond = compute_all_PxxCxyCyclefreq(session_eeg)

        Pxx_allcond, cyclefreq_allcond, Cxy_allcond, surrogates_allcond = reduce_PxxCxy_cyclefreq(Pxx_allcond, Cxy_allcond, cyclefreq_allcond, surrogates_allcond)

        os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'summary'))

        print('######## PLOT & SAVE PSD AND COH ########')

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(plot_save_PSD_Coh)(n_chan) for n_chan in range(len(chan_list_ieeg)))

        #### TF ITPC
        conditions, chan_list, chan_list_ieeg, srate, respfeatures_allcond, respi_ratio_allcond, nwind, nfft, noverlap, hannw = get_all_info(session_eeg)
        
        #tf_mode = 'TF'
        for tf_mode in ['TF', 'ITPC']:
            
            tf_stretch_allcond = load_TF_ITPC(session_eeg, tf_mode)

            if tf_mode == 'TF':
                print('######## PLOT & SAVE TF ########')
            if tf_mode == 'ITPC':
                print('######## PLOT & SAVE ITPC ########')
            
            for band_prep in band_prep_list: 

                joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_TF_ITPC_n_chan)(n_chan) for n_chan in range(len(chan_list_ieeg)))




