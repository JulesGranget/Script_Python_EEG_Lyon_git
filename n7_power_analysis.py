

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
######## LOAD DATA ########
################################

def get_all_info():

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(conditions_allsubjects)
    respfeatures_allcond = load_respfeatures(conditions)
    respi_ratio_allcond = get_all_respi_ratio(conditions, respfeatures_allcond)

    dict_loca = get_electrode_loca()

    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    return conditions, chan_list, chan_list_ieeg, srate, respfeatures_allcond, dict_loca, nwind, nfft, noverlap, hannw





################################################
######## PSD & COH WHOLE COMPUTATION ########
################################################


def load_surrogates_session(session_eeg):

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    surrogates_allcond = {'Cxy' : {}}

    for cond in conditions:

        if len(respfeatures_allcond[f's{session_eeg+1}'][cond]) == 1:

            surrogates_allcond['Cxy'][cond] = [np.load(f'{sujet}_s{session_eeg+1}_{cond}_1_Coh.npy')]

            for band_prep in band_prep_list:
                
                surrogates_allcond[f'cyclefreq_{band_prep}'] = {}
                surrogates_allcond[f'cyclefreq_{band_prep}'][cond] = [np.load(f'{sujet}_s{session_eeg+1}_{cond}_1_cyclefreq_{band_prep}.npy')]

            
        elif len(respfeatures_allcond[f's{session_eeg+1}'][cond]) > 1:

            data_load = {'Cxy' : []}

            for session_i in range(len(respfeatures_allcond[f's{session_eeg+1}'][cond])):

                data_load['Cxy'].append(np.load(f'{sujet}_s{session_eeg+1}_{cond}_{str(session_i+1)}_Coh.npy'))

                for band_prep in band_prep_list:
                    
                    surrogates_allcond[f'cyclefreq_{band_prep}'] = {}
                    surrogates_allcond[f'cyclefreq_{band_prep}'][cond] = [np.load(f'{sujet}_s{session_eeg+1}_{cond}_{str(session_i+1)}_cyclefreq_{band_prep}.npy')]                 


    return surrogates_allcond



#### compute Pxx & Cxy & Cyclefreq
def compute_PxxCxyCyclefreq_for_cond(band_prep, session_eeg, cond, session_i, nb_point_by_cycle):
    
    print(cond)

    #### extract data
    chan_i = chan_list.index('nasal')
    respi = load_data(band_prep, session_eeg, cond, session_i)[chan_i,:]
    data_tmp = load_data(band_prep, session_eeg, cond, session_i)

    #### prepare analysis
    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    Cxy_for_cond = np.zeros(( np.size(data_tmp,0), len(hzCxy)))
    Pxx_for_cond = np.zeros(( np.size(data_tmp,0), len(hzPxx)))
    cyclefreq_for_cond = np.zeros(( np.size(data_tmp,0), nb_point_by_cycle))

    for n_chan in range(np.size(data_tmp,0)):

        #### script avancement
        if n_chan/np.size(data_tmp,0) % .2 <= 0.01:
            print('{:.2f}'.format(n_chan/np.size(data_tmp,0)))

        x = data_tmp[n_chan,:]
        hzPxx, Pxx = scipy.signal.welch(x,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)

        y = respi
        hzPxx, Cxy = scipy.signal.coherence(x, y, fs=srate, window=hannw, nperseg=None, noverlap=noverlap, nfft=nfft)

        x_stretch, trash = stretch_data(respfeatures_allcond.get(cond)[session_i], nb_point_by_cycle, x, srate)
        x_stretch_mean = np.mean(x_stretch, 0)

        Cxy_for_cond[n_chan,:] = Cxy[mask_hzCxy]
        Pxx_for_cond[n_chan,:] = Pxx
        cyclefreq_for_cond[n_chan,:] = x_stretch_mean

    return Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond

        


def compute_all_PxxCxyCyclefreq(session_eeg):

    #### initiate dict
    Cxy_allcond = {}
    Pxx_allcond = {}
    cyclefreq_allcond = {}
    for band_prep in band_prep_list:
        Pxx_allcond[band_prep] = {}
        cyclefreq_allcond[band_prep] = {}

    #band_prep = band_prep_list[0]
    for band_prep in band_prep_list:

        print(band_prep)

        for cond in conditions:

            if ( len(respfeatures_allcond[f's{session_eeg+1}'][cond]) == 1 ) & (band_prep == 'lf' or band_prep == 'wb'):

                Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(band_prep, session_eeg, cond, 0, stretch_point_surrogates)

                Pxx_allcond[band_prep][cond] = [Pxx_for_cond]
                Cxy_allcond[cond] = [Cxy_for_cond]
                cyclefreq_allcond[band_prep][cond] = [cyclefreq_for_cond]

            elif ( len(respfeatures_allcond[f's{session_eeg+1}'][cond]) == 1 ) & (band_prep == 'hf') :

                Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(band_prep, session_eeg, cond, 0, stretch_point_surrogates)

                Pxx_allcond[band_prep][cond] = [Pxx_for_cond]
                cyclefreq_allcond[band_prep][cond] = [cyclefreq_for_cond]

            elif (len(respfeatures_allcond[f's{session_eeg+1}'][cond]) > 1) & (band_prep == 'lf' or band_prep == 'wb'):

                Pxx_load = []
                Cxy_load = []
                cyclefreq_load = []

                for session_i in range(len(respfeatures_allcond[f's{session_eeg+1}'][cond])):

                    Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(band_prep, session_eeg, cond, session_i, stretch_point_surrogates)

                    Pxx_load.append(Pxx_for_cond)
                    Cxy_load.append(Cxy_for_cond)
                    cyclefreq_load.append(cyclefreq_for_cond)

                Pxx_allcond[band_prep][cond] = Pxx_load
                Cxy_allcond[cond] = Cxy_load
                cyclefreq_allcond[band_prep][cond] = cyclefreq_load

            elif (len(respfeatures_allcond[f's{session_eeg+1}'][cond]) > 1) & (band_prep == 'hf'):

                Pxx_load = []
                cyclefreq_load = []

                for session_i in range(len(respfeatures_allcond[f's{session_eeg+1}'][cond])):

                    Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(band_prep, session_eeg, cond, session_i, stretch_point_surrogates)

                    Pxx_load.append(Pxx_for_cond)
                    cyclefreq_load.append(cyclefreq_for_cond)

                Pxx_allcond[band_prep][cond] = Pxx_load
                cyclefreq_allcond[band_prep][cond] = cyclefreq_load

    return Pxx_allcond, Cxy_allcond, cyclefreq_allcond




def reduce_PxxCxy_cyclefreq(Pxx_allcond, Cxy_allcond, cyclefreq_allcond, surrogates_allcond):

    #### reduce data to one session
    respfeatures_allcond_adjust = {} # to conserve respi_allcond for TF

    for cond in conditions:

        if len(respfeatures_allcond[f's{session_eeg+1}'][cond]) == 1:

            respfeatures_allcond_adjust[f's{session_eeg+1}'][cond] = respfeatures_allcond[f's{session_eeg+1}'][cond].copy()

        elif len(respfeatures_allcond[f's{session_eeg+1}'][cond]) > 1:
            
            data_to_short = {
            'respfeatures' : {'data' : []},
            'Cxy' : {'data' : {}, 'surrogates' : {}}, 
            'Pxx' : {'data' : {}},
            'cyclefreq' : {'data' : {}, 'surrogates' : {}},
            }

            #### Cxy & Respfeatures
            data_to_short['respfeatures']['data'] = respfeatures_allcond[f's{session_eeg+1}'][cond]
            data_to_short['Cxy']['data'] = Cxy_allcond[cond]
            data_to_short['Cxy']['surrogates'] = surrogates_allcond['Cxy'][cond]

            #### Pxx & Cyclefreq
            for band_prep in band_prep_list:

                data_to_short['Pxx']['data'][band_prep] = Pxx_allcond[band_prep][cond]
                data_to_short['cyclefreq']['data'][band_prep] = cyclefreq_allcond[band_prep][cond]
                data_to_short['cyclefreq']['surrogates'][band_prep] = surrogates_allcond[f'cyclefreq_{band_prep}'][cond]









            for data_short_i in range(len(respfeatures_allcond[f's{session_eeg+1}'][cond])):

                if data_short_i == 0:
                    for session_i in range(len(respfeatures_allcond[f's{session_eeg+1}'][cond])):
                        if session_i == 0:
                            _short = data_to_short[data_short_i][session_i]
                        else:
                            _short = (_short + data_to_short[data_short_i][session_i])/2
                    data_to_short[data_short_i] = [_short]

                else:    
                    for session_i in range(len(respfeatures_allcond[f's{session_eeg+1}'][cond])):
                        if session_i == 0:
                            _short = data_to_short[data_short_i][session_i]
                        else:
                            _short = (_short + data_to_short[data_short_i][session_i])/2
                    data_to_short[data_short_i] = [_short]

            #### fill values
            respfeatures_allcond_adjust[f's{session_eeg+1}'][cond] = data_to_short[0]

            Pxx_allcond['lf'][cond] = data_to_short[1]
            Pxx_allcond['hf'][cond] = data_to_short[2]

            Cxy_allcond[cond] = data_to_short[3]
            surrogates_allcond['Cxy'][cond] = data_to_short[4]

            cyclefreq_allcond['lf'][cond] = data_to_short[5]
            cyclefreq_allcond['hf'][cond] = data_to_short[6]
            surrogates_allcond['cyclefreq_lf'][cond] = data_to_short[7]
            surrogates_allcond['cyclefreq_hf'][cond] = data_to_short[8]
                

    #### verif if one session only
    for cond in conditions :

        verif_size = []

        verif_size.append(len(respfeatures_allcond_adjust[f's{session_eeg+1}'][cond]) == 1)
        verif_size.append(len(Pxx_allcond['lf'][cond]) == 1)
        verif_size.append(len(Pxx_allcond['hf'][cond]) == 1)
        verif_size.append(len(Cxy_allcond[cond]) == 1)
        verif_size.append(len(surrogates_allcond['Cxy'][cond]) == 1)
        verif_size.append(len(cyclefreq_allcond['lf'][cond]) == 1)
        verif_size.append(len(cyclefreq_allcond['hf'][cond]) == 1)
        verif_size.append(len(surrogates_allcond['cyclefreq_lf'][cond]) == 1)

        if verif_size.count(False) != 0 :
            raise ValueError('!!!! PROBLEM VERIF !!!!')

        elif verif_size.count(False) == 0 :
            print('Verif OK')






################################################
######## PLOT & SAVE PSD AND COH ########
################################################

os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'summary'))

print('######## PLOT & SAVE PSD AND COH ########')

#### def functions
def plot_save_PSD_Coh_lf(n_chan):

    session_i = 0       
    
    chan_name = chan_list_ieeg[n_chan]

    if n_chan/len(chan_list_ieeg) % .2 <= 0.01:
        print('{:.2f}'.format(n_chan/len(chan_list_ieeg)))

    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### plot

    if len(conditions) == 1:

        fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
        plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

        cond = conditions[0]

        #### supress NaN
        keep = np.invert(np.isnan(respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values))
        cycle_for_mean = respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values[keep]
        respi_mean = round(np.mean(cycle_for_mean), 2)
        
        #### plot
        ax = axs[0]
        ax.set_title(cond, fontweight='bold', rotation=0)
        ax.semilogy(hzPxx, Pxx_allcond['lf'].get(cond)[session_i][n_chan,:], color='k')
        ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond['lf'].get(cond)[session_i][n_chan,:]), color='r')
        ax.set_xlim(0,60)

        ax = axs[1]
        ax.plot(hzPxx[remove_zero_pad:],Pxx_allcond['lf'].get(cond)[session_i][n_chan,:][remove_zero_pad:], color='k')
        ax.set_xlim(0, 2)
        ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond['lf'].get(cond)[session_i][n_chan,:]), color='r')

        ax = axs[2]
        ax.plot(hzCxy,Cxy_allcond.get(cond)[session_i][n_chan,:], color='k')
        ax.plot(hzCxy,surrogates_allcond['Cxy'].get(cond)[session_i][n_chan,:], color='c')
        ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

        ax = axs[3]
        ax.plot(cyclefreq_allcond['lf'].get(cond)[session_i][n_chan,:], color='k')
        ax.plot(surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][0, n_chan,:], color='b')
        ax.plot(surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][1, n_chan,:], color='c', linestyle='dotted')
        ax.plot(surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][2, n_chan,:], color='c', linestyle='dotted')
        if stretch_TF_auto:
            ax.vlines(respi_ratio_allcond.get(cond)[session_i]*stretch_point_surrogates, ymin=np.min( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][2, n_chan,:] ), ymax=np.max( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][1, n_chan,:] ), colors='r')
        else:
            ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=np.min( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][2, n_chan,:] ), ymax=np.max( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][1, n_chan,:] ), colors='r')

    else:

        fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
        plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))
        
        for c, cond in enumerate(conditions):

            #### supress NaN
            keep = np.invert(np.isnan(respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values))
            cycle_for_mean = respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values[keep]
            respi_mean = round(np.mean(cycle_for_mean), 2)
            
            #### plot
            ax = axs[0,c]
            ax.set_title(cond, fontweight='bold', rotation=0)
            ax.semilogy(hzPxx,Pxx_allcond['lf'].get(cond)[session_i][n_chan,:], color='k')
            ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond['lf'].get(cond)[session_i][n_chan,:]), color='r')
            ax.set_xlim(0,60)

            ax = axs[1,c]
            ax.plot(hzPxx[remove_zero_pad:],Pxx_allcond['lf'].get(cond)[session_i][n_chan,:][remove_zero_pad:], color='k')
            ax.set_xlim(0, 2)
            ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond['lf'].get(cond)[session_i][n_chan,:]), color='r')

            ax = axs[2,c]
            ax.plot(hzCxy,Cxy_allcond.get(cond)[session_i][n_chan,:], color='k')
            ax.plot(hzCxy,surrogates_allcond['Cxy'].get(cond)[session_i][n_chan,:], color='c')
            ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

            ax = axs[3,c]
            ax.plot(cyclefreq_allcond['lf'].get(cond)[session_i][n_chan,:], color='k')
            ax.plot(surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][0, n_chan,:], color='b')
            ax.plot(surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][1, n_chan,:], color='c', linestyle='dotted')
            ax.plot(surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][2, n_chan,:], color='c', linestyle='dotted')
            if stretch_TF_auto:
                ax.vlines(respi_ratio_allcond.get(cond)[session_i]*stretch_point_surrogates, ymin=np.min( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][2, n_chan,:] ), ymax=np.max( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][1, n_chan,:] ), colors='r')
            else:
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=np.min( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][2, n_chan,:] ), ymax=np.max( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][1, n_chan,:] ), colors='r') 
    #plt.show()
    
    #### save
    fig.savefig(sujet + '_' + chan_name + '_lf.jpeg', dpi=600)
    plt.close()

    return



def plot_save_PSD_Coh_hf(n_chan):    
    
    session_i = 0

    chan_name = chan_list_ieeg[n_chan]

    if n_chan/len(chan_list_ieeg) % .2 <= 0.01:
        print('{:.2f}'.format(n_chan/len(chan_list_ieeg)))

    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))

    #### plot

    fig, axs = plt.subplots(nrows=2, ncols=len(conditions))
    plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

    if len(conditions) == 1 :

        for c, cond in enumerate(conditions):

            #### supress NaN
            keep = np.invert(np.isnan(respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values))
            cycle_for_mean = respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values[keep]
            respi_mean = round(np.mean(cycle_for_mean), 2)
            
            #### plot
            ax = axs[0]
            ax.set_title(cond, fontweight='bold', rotation=0)
            ax.semilogy(hzPxx,Pxx_allcond['hf'].get(cond)[session_i][n_chan,:], color='k')
            ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond['hf'].get(cond)[session_i][n_chan,:]), color='r')
            ax.set_xlim(45,120)

            ax = axs[1]
            ax.plot(cyclefreq_allcond['hf'].get(cond)[session_i][n_chan,:], color='k')
            ax.plot(surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][0, n_chan,:], color='b')
            ax.plot(surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][1, n_chan,:], color='c', linestyle='dotted')
            ax.plot(surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][2, n_chan,:], color='c', linestyle='dotted')
            if stretch_TF_auto:
                ax.vlines(respi_ratio_allcond.get(cond)[session_i]*stretch_point_surrogates, ymin=np.min( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][2, n_chan,:] ), ymax=np.max( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][1, n_chan,:] ), colors='r')
            else:
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=np.min( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][2, n_chan,:] ), ymax=np.max( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][1, n_chan,:] ), colors='r')

    else:

        for c, cond in enumerate(conditions):

            #### supress NaN
            keep = np.invert(np.isnan(respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values))
            cycle_for_mean = respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values[keep]
            respi_mean = round(np.mean(cycle_for_mean), 2)
            
            #### plot
            ax = axs[0,c]
            ax.set_title(cond, fontweight='bold', rotation=0)
            ax.semilogy(hzPxx,Pxx_allcond['hf'].get(cond)[session_i][n_chan,:], color='k')
            ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond['hf'].get(cond)[session_i][n_chan,:]), color='r')
            ax.set_xlim(45,120)

            ax = axs[1,c]
            ax.plot(cyclefreq_allcond['hf'].get(cond)[session_i][n_chan,:], color='k')
            ax.plot(surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][0, n_chan,:], color='b')
            ax.plot(surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][1, n_chan,:], color='c', linestyle='dotted')
            ax.plot(surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][2, n_chan,:], color='c', linestyle='dotted')
            if stretch_TF_auto:
                ax.vlines(respi_ratio_allcond.get(cond)[session_i]*stretch_point_surrogates, ymin=np.min( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][2, n_chan,:] ), ymax=np.max( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][1, n_chan,:] ), colors='r')
            else:
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=np.min( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][2, n_chan,:] ), ymax=np.max( surrogates_allcond['cyclefreq_lf'].get(cond)[session_i][1, n_chan,:] ), colors='r')


    #### save
    fig.savefig(sujet + '_' + chan_name + '_hf.jpeg', dpi=600)
    plt.close()

    return





#### compute joblib

joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(plot_save_PSD_Coh_lf)(n_chan) for n_chan in range(len(chan_list_ieeg)))
joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(plot_save_PSD_Coh_hf)(n_chan) for n_chan in range(len(chan_list_ieeg)))







################################
######## LOAD TF ########
################################

print('######## LOAD TF ########')

#### load and reduce to all cond
os.chdir(os.path.join(path_precompute, sujet, 'TF'))

#### generate str to search file
freq_band_str = {}

for band_prep_i, band_prep in enumerate(band_prep_list):

    freq_band = freq_band_list[band_prep_i]

    for band, freq in freq_band.items():
        freq_band_str[band] = str(freq[0]) + '_' + str(freq[1])


#### load file with reducing to one TF

tf_stretch_allcond = {}

for cond in conditions:

    tf_stretch_onecond = {}

    if len(respfeatures_allcond.get(cond)) == 1:

        #### generate file to load
        load_file = []
        for file in os.listdir(): 
            if file.find(cond) != -1:
                load_file.append(file)
            else:
                continue

        #### impose good order in dict
        for freq_band in freq_band_list:
            for band, freq in freq_band.items():
                tf_stretch_onecond[band] = 0

        #### file load
        for file in load_file:

            for freq_band in freq_band_list:

                for i, (band, freq) in enumerate(freq_band.items()):

                    if file.find(freq_band_str.get(band)) != -1:
                        tf_stretch_onecond[band] = np.load(file)
                    else:
                        continue
                    
        tf_stretch_allcond[cond] = tf_stretch_onecond

    elif len(respfeatures_allcond.get(cond)) > 1:

        #### generate file to load
        load_file = []
        for file in os.listdir(): 
            if file.find(cond) != -1:
                load_file.append(file)
            else:
                continue

        #### implement count
        for freq_band in freq_band_list:
            for band, freq in freq_band.items():
                tf_stretch_onecond[band] = 0

        #### load file
        for file in load_file:

            for freq_band in freq_band_list:

                for i, (band, freq) in enumerate(freq_band.items()):

                    if file.find(freq_band_str.get(band)) != -1:

                        if np.sum(tf_stretch_onecond.get(band)) != 0:

                            session_load_tmp = ( np.load(file) + tf_stretch_onecond.get(band) ) /2
                            tf_stretch_onecond[band] = session_load_tmp

                        else:
                            
                            tf_stretch_onecond[band] = np.load(file)

                    else:

                        continue

        tf_stretch_allcond[cond] = tf_stretch_onecond




#### verif

for cond in conditions:
    if len(tf_stretch_allcond.get(cond)) != 6:
        print('ERROR COND : ' + cond)

    for freq_band in freq_band_list:

        for band, freq in freq_band.items():
            if len(tf_stretch_allcond.get(cond).get(band)) != len(chan_list_ieeg) :
                print('ERROR FREQ BAND : ' + band)
            






################################
######## PLOT & SAVE TF ########
################################




print('######## SAVE TF ########')

#n_chan = 0
#freq_band_i, freq_band = 1, freq_band_list[1]
def save_TF_n_chan(n_chan, freq_band_i, freq_band):

    os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))
    
    chan_name = chan_list_ieeg[n_chan]

    if n_chan/len(chan_list_ieeg) % .2 <= .01:
        print('{:.2f}'.format(n_chan/len(chan_list_ieeg)))

    time = range(stretch_point_TF)
    frex = np.size(tf_stretch_allcond.get(conditions[0]).get(list(freq_band.keys())[0]),1)

    #### determine plot scale
    scales = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}

    for c, cond in enumerate(conditions):

        for i, (band, freq) in enumerate(freq_band.items()) :

            if band == 'whole' or band == 'l_gamma':
                continue

            data = tf_stretch_allcond[cond][band][n_chan, :, :]
            frex = np.linspace(freq[0], freq[1], np.size(data,0))

            scales['vmin_val'] = np.append(scales['vmin_val'], np.min(data))
            scales['vmax_val'] = np.append(scales['vmax_val'], np.max(data))
            scales['median_val'] = np.append(scales['median_val'], np.median(data))

    median_diff = np.max([np.abs(np.min(scales['vmin_val']) - np.median(scales['median_val'])), np.abs(np.max(scales['vmax_val']) - np.median(scales['median_val']))])

    vmin = np.median(scales['median_val']) - median_diff
    vmax = np.median(scales['median_val']) + median_diff

    #### plot
    if freq_band_i == 0:
        fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
    else:
        fig, axs = plt.subplots(nrows=2, ncols=len(conditions))
    
    plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

    #### for plotting l_gamma down
    if freq_band_i == 1:
        keys_list_reversed = list(freq_band.keys())
        keys_list_reversed.reverse()
        freq_band_reversed = {}
        for key_i in keys_list_reversed:
            freq_band_reversed[key_i] = freq_band[key_i]
        freq_band = freq_band_reversed

    for c, cond in enumerate(conditions):

        #### plot
        for i, (band, freq) in enumerate(freq_band.items()) :

            data = tf_stretch_allcond[cond][band][n_chan, :, :]
            frex = np.linspace(freq[0], freq[1], np.size(data,0))
        
            if len(conditions) == 1:
                ax = axs[i]
            else:
                ax = axs[i,c]

            if i == 0 :
                ax.set_title(cond, fontweight='bold', rotation=0)

            ax.pcolormesh(time, frex, data, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))

            if c == 0:
                ax.set_ylabel(band)

            if stretch_TF_auto:
                ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')
            else:
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')
    #plt.show()

    #### save
    if freq_band_i == 0:
        fig.savefig(sujet + '_' + chan_name + '_lf.jpeg', dpi=600)
    else:
        fig.savefig(sujet + '_' + chan_name + '_hf.jpeg', dpi=600)
    plt.close()



#### compute
#freq_band_i, freq_band = 0, freq_band_list[0]
for freq_band_i, freq_band in enumerate(freq_band_list): 

    print(band_prep_list[freq_band_i])
    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_TF_n_chan)(n_chan, freq_band_i, freq_band) for n_chan in range(len(chan_list_ieeg)))






################################
######## LOAD ITPC ########
################################

print('######## LOAD ITPC ########')

#### load and reduce to all cond
os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

#### generate str to search file
freq_band_str = {}

for band_prep_i, band_prep in enumerate(band_prep_list):

    freq_band = freq_band_list[band_prep_i]

    for band, freq in freq_band.items():
        freq_band_str[band] = str(freq[0]) + '_' + str(freq[1])

#### load file with reducing to one TF

tf_itpc_allcond = {}

for cond in conditions:

    tf_itpc_onecond = {}

    if len(respfeatures_allcond.get(cond)) == 1:

        #### generate file to load
        load_file = []
        for file in os.listdir(): 
            if file.find(cond) != -1:
                load_file.append(file)
            else:
                continue

        #### impose good order in dict
        for freq_band in freq_band_list:
            for band, freq in freq_band.items():
                tf_itpc_onecond[band] = 0

        #### file load
        for file in load_file:

            for freq_band in freq_band_list:

                for i, (band, freq) in enumerate(freq_band.items()):
                    if file.find(freq_band_str.get(band)) != -1:
                        tf_itpc_onecond[ band ] = np.load(file)
                    else:
                        continue
                    
        tf_itpc_allcond[cond] = tf_itpc_onecond

    elif len(respfeatures_allcond.get(cond)) > 1:

        #### generate file to load
        load_file = []
        for file in os.listdir(): 
            if file.find(cond) != -1:
                load_file.append(file)
            else:
                continue

        #### implement count
        for freq_band in freq_band_list:
            for band, freq in freq_band.items():
                tf_itpc_onecond[band] = 0

        #### load file
        for file in load_file:

            for freq_band in freq_band_list:

                for i, (band, freq) in enumerate(freq_band.items()):

                    if file.find(freq_band_str.get(band)) != -1:

                        if np.sum(tf_itpc_onecond.get(band)) != 0:

                            session_load_tmp = ( np.load(file) + tf_itpc_onecond.get(band) ) /2
                            tf_itpc_onecond[band] = session_load_tmp

                        else:
                            
                            tf_itpc_onecond[band] = np.load(file)

                    else:

                        continue

        tf_itpc_allcond[cond] = tf_itpc_onecond


#### verif

for cond in conditions:
    if len(tf_itpc_allcond.get(cond)) != 6:
        print('ERROR COND : ' + cond)

    for freq_band in freq_band_list:

        for band, freq in freq_band.items():
            if len(tf_itpc_allcond.get(cond).get(band)) != len(chan_list_ieeg) :
                print('ERROR FREQ BAND : ' + band)
            






########################################
######## PLOT & SAVE ITPC ########
########################################

print('######## SAVE ITPC ########')

#n_chan = 16
#freq_band_i, freq_band = 0, freq_band_list[0]
def save_itpc_n_chan(n_chan, freq_band_i, freq_band):       
    
    os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))

    chan_name = chan_list_ieeg[n_chan]

    if n_chan/len(chan_list_ieeg) % .2 <= .01:
        print('{:.2f}'.format(n_chan/len(chan_list_ieeg)))

    time = range(stretch_point_TF)
    frex = np.size(tf_itpc_allcond.get(conditions[0]).get(list(freq_band.keys())[0]),1)

    #### determine plot scale
    scales = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}

    for c, cond in enumerate(conditions):

        for i, (band, freq) in enumerate(freq_band.items()) :

            if band == 'whole' or band == 'l_gamma':
                continue
            
            data = tf_itpc_allcond[cond][band][n_chan, :, :]
            frex = np.linspace(freq[0], freq[1], np.size(data,0))

            scales['vmin_val'] = np.append(scales['vmin_val'], np.min(data))
            scales['vmax_val'] = np.append(scales['vmax_val'], np.max(data))
            scales['median_val'] = np.append(scales['median_val'], np.median(data))

    median_diff = np.max([np.abs(np.min(scales['vmin_val']) - np.median(scales['median_val'])), np.abs(np.max(scales['vmax_val']) - np.median(scales['median_val']))])

    vmin = np.median(scales['median_val']) - median_diff
    vmax = np.median(scales['median_val']) + median_diff

    #### plot
    if freq_band_i == 0:
        fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
    else:
        fig, axs = plt.subplots(nrows=2, ncols=len(conditions))
    
    plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

    #### for plotting l_gamma down
    if freq_band_i == 1:
        keys_list_reversed = list(freq_band.keys())
        keys_list_reversed.reverse()
        freq_band_reversed = {}
        for key_i in keys_list_reversed:
            freq_band_reversed[key_i] = freq_band[key_i]
        freq_band = freq_band_reversed

    for c, cond in enumerate(conditions):
        
        #### plot
        for i, (band, freq) in enumerate(freq_band.items()) :

            data = tf_itpc_allcond[cond][band][n_chan, :, :]
            frex = np.linspace(freq[0], freq[1], np.size(data,0))
        
            if len(conditions) == 1:
                ax = axs[i]
            else:
                ax = axs[i,c]

            if i == 0 :
                ax.set_title(cond, fontweight='bold', rotation=0)

            ax.pcolormesh(time, frex, data, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))

            if c == 0:
                ax.set_ylabel(band)

            if stretch_TF_auto:
                ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')
            else:
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')
    #plt.show()     

    #### save
    if freq_band_i == 0:
        fig.savefig(sujet + '_' + chan_name + '_lf.jpeg', dpi=600)
    else:
        fig.savefig(sujet + '_' + chan_name + '_hf.jpeg', dpi=600)
    plt.close()


for freq_band_i, freq_band in enumerate(freq_band_list): 

    print(band_prep_list[freq_band_i])
    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_itpc_n_chan)(n_chan, freq_band_i, freq_band) for n_chan in range(len(chan_list_ieeg)))









################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #### load data infos
    conditions, chan_list, chan_list_ieeg, srate, respfeatures_allcond, dict_loca, nwind, nfft, noverlap, hannw = get_all_info()

    #### define session_eeg
    session_eeg = 0

    ########################################
    ######## Pxx Cxy CycleFreq ########
    ########################################
    

    print('######## COMPUTE PxxCxyCyclefreq ########')

    #### import and compute
    surrogates_allcond = load_surrogates_session(session_eeg) 

    Pxx_allcond, Cxy_allcond, cyclefreq_allcond = compute_all_PxxCxyCyclefreq()









