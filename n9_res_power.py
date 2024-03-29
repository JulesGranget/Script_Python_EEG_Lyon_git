
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib

import pickle
import gc

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False







########################################
######## PSD & COH PRECOMPUTE ########
########################################



#dict2reduce = cyclefreq_binned_allcond
def reduce_data(dict2reduce, prms):

    #### identify count
    dict_count = {}
        #### for cyclefreq & Pxx
    if list(dict2reduce.keys())[0] in band_prep_list:

        for cond in prms['conditions']:
            dict_count[cond] = len(dict2reduce[band_prep_list[0]][cond])
        #### for surrogates
    elif len(list(dict2reduce.keys())) == 4 and list(dict2reduce.keys())[0] not in prms['conditions']:

        for cond in prms['conditions']:
            dict_count[cond] = len(dict2reduce[list(dict2reduce.keys())[0]][cond])
        #### for Cxy & MVL
    else:

        for cond in prms['conditions']:
            dict_count[cond] = len(dict2reduce[cond])    

    #### for Pxx & Cyclefreq reduce
    if np.sum([True for i in list(dict2reduce.keys()) if i in band_prep_list]) > 0:
    
        #### generate dict
        dict_reduced = {}

        for band_prep in band_prep_list:
            dict_reduced[band_prep] = {}

            for cond in prms['conditions']:
                dict_reduced[band_prep][cond] = np.zeros(( dict2reduce[band_prep][cond][0].shape ))

        #### fill
        for band_prep in band_prep_list:

            for cond in prms['conditions']:

                for session_i in range(dict_count[cond]):

                    dict_reduced[band_prep][cond] += dict2reduce[band_prep][cond][session_i]

                dict_reduced[band_prep][cond] /= dict_count[cond]

    #### for Cxy & MVL reduce
    elif np.sum([True for i in list(dict2reduce.keys()) if i in prms['conditions']]) > 0:

        #### generate dict
        dict_reduced = {}

        for cond in prms['conditions']:

            dict_reduced[cond] = np.zeros(( dict2reduce[cond][0].shape ))

        #### fill
        for cond in prms['conditions']:

            for session_i in range(dict_count[cond]):

                dict_reduced[cond] += dict2reduce[cond][session_i]

            dict_reduced[cond] /= dict_count[cond]

    #### for surrogates
    else:
        
        #### generate dict
        dict_reduced = {}
        for key in list(dict2reduce.keys()):
            dict_reduced[key] = {}
            for cond in prms['conditions']:
                dict_reduced[key][cond] = np.zeros(( dict2reduce[key][cond][0].shape ))

        #### fill
        #key = 'Cxy'
        for key in list(dict2reduce.keys()):

            for cond in prms['conditions']:

                for session_i in range(dict_count[cond]):

                    dict_reduced[key][cond] += dict2reduce[key][cond][session_i]

                dict_reduced[key][cond] /= dict_count[cond]

    #### verify
        #### for cyclefreq & Pxx
    if list(dict2reduce.keys())[0] in band_prep_list:

        for band_prep in band_prep_list:
            for cond in prms['conditions']:
                try: 
                    _ = dict_reduced[band_prep][cond].shape
                except:
                    raise ValueError('reducing wrong')
        
        #### for surrogates
    elif len(list(dict2reduce.keys())) == 4 and list(dict2reduce.keys())[0] not in prms['conditions']:

        list_surr = list(dict2reduce.keys())

        for surr_i in list_surr:
        
            for cond in prms['conditions']:
                try: 
                    _ = dict_reduced[surr_i][cond].shape
                except:
                    raise ValueError('reducing wrong')
    
        #### for Cxy & MVL
    else:

        for cond in prms['conditions']:
            try: 
                _ = dict_reduced[cond].shape
            except:
                raise ValueError('reducing wrong')

    return dict_reduced






def load_surrogates(sujet):

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    surrogates_allcond = {}

    for data_type in ['Cxy', 'cyclefreq_wb', 'MVL']:

        surrogates_allcond[data_type] = {}

        for cond in conditions:

            surrogates_allcond[data_type][cond] = {}

            for odor_i in odor_list:

                if data_type == 'Cxy':
                    surrogates_allcond['Cxy'][cond][odor_i] = np.load(f'{sujet}_{cond}_{odor_i}_Coh.npy')
                if data_type == 'cyclefreq_wb':
                    surrogates_allcond['cyclefreq_wb'][cond][odor_i] = np.load(f'{sujet}_{cond}_{odor_i}_cyclefreq_wb.npy')
                if data_type == 'MVL':
                    surrogates_allcond['MVL'][cond][odor_i] = np.load(f'{sujet}_{cond}_{odor_i}_MVL_wb.npy')

    return surrogates_allcond







#### compute Pxx & Cxy & Cyclefreq
def compute_PxxCxyCyclefreq_for_cond_session(sujet, cond, odor_i, band_prep):
    
    print(cond, odor_i)

    #### extract data
    respfeatures_allcond = load_respfeatures(sujet)
    prms = get_params()
    chan_i = prms['chan_list'].index('PRESS')
    respi = load_data_sujet(sujet, band_prep, cond, odor_i, session_i)[chan_i,:]
    data_tmp = load_data_sujet(sujet, band_prep, cond, odor_i, session_i)[:len(chan_list_eeg),:]

    #### prepare analysis
    hzPxx = np.linspace(0,prms['srate']/2,int(prms['nfft']/2+1))
    hzCxy = np.linspace(0,prms['srate']/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### compute
    Cxy_for_cond = np.zeros(( data_tmp.shape[0], len(hzCxy)))
    Pxx_for_cond = np.zeros(( data_tmp.shape[0], len(hzPxx)))
    cyclefreq_for_cond = np.zeros(( data_tmp.shape[0], stretch_point_surrogates))
    # MI_for_cond = np.zeros(( data_tmp.shape[0] ))
    MVL_for_cond = np.zeros(( data_tmp.shape[0] ))
    # cyclefreq_binned_for_cond = np.zeros(( data_tmp.shape[0], MI_n_bin))

    # MI_bin_i = int(stretch_point_surrogates / MI_n_bin)

    #n_chan = 0
    for n_chan in range(data_tmp.shape[0]):

        #### Pxx, Cxy, CycleFreq
        x = data_tmp[n_chan,:]
        hzPxx, Pxx = scipy.signal.welch(x, fs=prms['srate'], window=prms['hannw'], nperseg=prms['nwind'], noverlap=prms['noverlap'], nfft=prms['nfft'])

        y = respi
        hzPxx, Cxy = scipy.signal.coherence(x, y, fs=prms['srate'], window=prms['hannw'], nperseg=prms['nwind'], noverlap=prms['noverlap'], nfft=prms['nfft'])

        x_stretch, trash = stretch_data(respfeatures_allcond[cond][odor_i], stretch_point_surrogates, x, prms['srate'])
        x_stretch_mean = np.mean(x_stretch, 0)
        x_stretch_mean = x_stretch_mean - x_stretch_mean.mean() 

        Cxy_for_cond[n_chan,:] = Cxy[mask_hzCxy]
        Pxx_for_cond[n_chan,:] = Pxx
        cyclefreq_for_cond[n_chan,:] = x_stretch_mean

        #### MVL
        x_zscore = zscore(x)
        x_stretch, trash = stretch_data(respfeatures_allcond[cond][odor_i], stretch_point_surrogates, x_zscore, prms['srate'])

        MVL_for_cond[n_chan] = get_MVL(np.mean(x_stretch,axis=0)-np.mean(x_stretch,axis=0).min())

        if debug:

            plt.plot(zscore(x))
            plt.plot(zscore(y))
            plt.show()

            plt.plot(hzPxx, Pxx)
            plt.show()

            plt.plot(hzPxx, Cxy)
            plt.show()

            plt.plot(x_stretch_mean)
            plt.show()

            plt.plot(np.mean(x_stretch,axis=0)-np.mean(x_stretch,axis=0).min())
            plt.show()

        # #### MI
        # x = x_stretch_mean

        # x_bin = np.zeros(( MI_n_bin ))

        # for bin_i in range(MI_n_bin):
        #     x_bin[bin_i] = np.mean(x[MI_bin_i*bin_i:MI_bin_i*(bin_i+1)])

        # cyclefreq_binned_for_cond[n_chan,:] = x_bin

        # x_bin += np.abs(x_bin.min())*2 #supress zero values
        # x_bin = x_bin/np.sum(x_bin) #transform into probabilities
            
        # MI_for_cond[n_chan] = Shannon_MI(x_bin)

    if debug:

        for nchan in range(data_tmp.shape[0]):

            # plt.plot(hzPxx, Pxx_for_cond[nchan,:])
            # plt.plot(hzPxx[mask_hzCxy], Cxy_for_cond[nchan,:])
            plt.plot(zscore(cyclefreq_for_cond[nchan,:]))

        plt.show()


    return Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond, MVL_for_cond

        





def compute_all_PxxCxyCyclefreq(sujet, band_prep):

    data_allcond = {}

    for data_type in ['Pxx', 'Cxy', 'cyclefreq', 'MVL']:

        data_allcond[data_type] = {}

        for cond in conditions:

            data_allcond[data_type][cond] = {}

            for odor_i in odor_list:

                data_allcond[data_type][cond][odor_i] = []

    for cond in conditions:

        for odor_i in odor_list:

            data_allcond[data_type][cond][odor_i] = []        

            Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond, MVL_for_cond = compute_PxxCxyCyclefreq_for_cond_session(sujet, cond, odor_i, band_prep)

            data_allcond['Pxx'][cond][odor_i] = Pxx_for_cond
            data_allcond['Cxy'][cond][odor_i] = Cxy_for_cond
            data_allcond['cyclefreq'][cond][odor_i] = cyclefreq_for_cond
            data_allcond['MVL'][cond][odor_i] = MVL_for_cond

    return data_allcond




def compute_PxxCxyCyclefreqSurrogates(sujet, band_prep):

    #### load params
    surrogates_allcond = load_surrogates(sujet)

    compute_token = False
        
    if os.path.exists(os.path.join(path_precompute, sujet, 'PSD_Coh', f'allcond_{sujet}_Pxx.pkl')) == False:

        compute_token = True

    if compute_token:
    
        #### compute metrics
        data_allcond = compute_all_PxxCxyCyclefreq(sujet, band_prep)

        #### save 
        os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

        with open(f'allcond_{sujet}_Pxx.pkl', 'wb') as f:
            pickle.dump(data_allcond['Pxx'], f)

        with open(f'allcond_{sujet}_Cxy.pkl', 'wb') as f:
            pickle.dump(data_allcond['Cxy'], f)

        with open(f'allcond_{sujet}_surrogates.pkl', 'wb') as f:
            pickle.dump(surrogates_allcond, f)

        with open(f'allcond_{sujet}_cyclefreq.pkl', 'wb') as f:
            pickle.dump(data_allcond['cyclefreq'], f)

        with open(f'allcond_{sujet}_MVL.pkl', 'wb') as f:
            pickle.dump(data_allcond['MVL'], f)

    else:

        print('ALREADY COMPUTED')

    print('done') 











################################################
######## PLOT & SAVE PSD AND COH ########
################################################




def get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet):

    source_path = os.getcwd()
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
                
    with open(f'allcond_{sujet}_Pxx.pkl', 'rb') as f:
        Pxx_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_Cxy.pkl', 'rb') as f:
        Cxy_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_surrogates.pkl', 'rb') as f:
        surrogates_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_cyclefreq.pkl', 'rb') as f:
        cyclefreq_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_MVL.pkl', 'rb') as f:
        MVL_allcond = pickle.load(f)

    os.chdir(source_path)

    return Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond



#n_chan, chan_name = 0, chan_list_eeg[0]
def plot_save_PSD_Cxy_CF_MVL(n_chan, chan_name, band_prep):

    #### load data
    Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond = get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet)
    prms = get_params()
    respfeatures_allcond = load_respfeatures(sujet)
    
    #### plot
    print_advancement(n_chan, len(chan_list_eeg), steps=[25, 50, 75])

    hzPxx = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #odor_i = odor_list[0]
    for odor_i in odor_list:

        fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
        plt.suptitle(f'{sujet}_{chan_name}_{odor_i}')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(conditions):

            #### identify respi mean
            respi_mean = np.round(respfeatures_allcond[cond][odor_i]['cycle_freq'].median(), 3)
                    
            #### plot
            ax = axs[0, c]
            ax.set_title(cond, fontweight='bold', rotation=0)
            ax.semilogy(hzPxx, Pxx_allcond[cond][odor_i][n_chan,:], color='k')
            ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond[cond][odor_i][n_chan,:].max(), color='r')
            ax.set_xlim(0,60)
 
            ax = axs[1, c]
            Pxx_sel_min = Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:][np.where(hzPxx[remove_zero_pad:] < 2)[0]].min()
            Pxx_sel_max = Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:][np.where(hzPxx[remove_zero_pad:] < 2)[0]].max()
            ax.semilogy(hzPxx[remove_zero_pad:], Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:], color='k')
            ax.set_xlim(0, 2)
            ax.set_ylim(Pxx_sel_min, Pxx_sel_max)
            ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:].max(), color='r')

            ax = axs[2, c]
            ax.plot(hzCxy,Cxy_allcond[cond][odor_i][n_chan,:], color='k')
            ax.plot(hzCxy,surrogates_allcond['Cxy'][cond][odor_i][n_chan,:], color='c')
            ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

            ax = axs[3, c]
            MVL_i = np.round(MVL_allcond[cond][odor_i][n_chan], 5)
            MVL_surr = np.percentile(surrogates_allcond['MVL'][cond][odor_i][n_chan,:], 99)
            if MVL_i > MVL_surr:
                MVL_p = f'MVL : {MVL_i}, *** {int(MVL_i * 100 / MVL_surr)}%'
            else:
                MVL_p = f'MVL : {MVL_i}, NS {int(MVL_i * 100 / MVL_surr)}%'
            # ax.set_title(MVL_p, rotation=0)
            ax.set_xlabel(MVL_p)

            ax.plot(cyclefreq_allcond[cond][odor_i][n_chan,:], color='k')
            ax.plot(surrogates_allcond['cyclefreq_wb'][cond][odor_i][0, n_chan,:], color='b')
            ax.plot(surrogates_allcond['cyclefreq_wb'][cond][odor_i][1, n_chan,:], color='c', linestyle='dotted')
            ax.plot(surrogates_allcond['cyclefreq_wb'][cond][odor_i][2, n_chan,:], color='c', linestyle='dotted')
            ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=surrogates_allcond['cyclefreq_wb'][cond][odor_i][2, n_chan,:].min(), ymax=surrogates_allcond['cyclefreq_wb'][cond][odor_i][1, n_chan,:].max(), colors='r')
            #plt.show() 

        #### save
        os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'summary', 'odor'))
        fig.savefig(f'{sujet}_{chan_name}_{odor_i}_{band_prep}.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #cond = 'FR_CV_1'
    for cond in conditions:

        fig, axs = plt.subplots(nrows=4, ncols=len(odor_list))
        plt.suptitle(f'{sujet}_{chan_name}_{cond}')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, odor_i = 0, odor_list[0]
        for c, odor_i in enumerate(odor_list):

            #### identify respi mean
            respi_mean = np.round(respfeatures_allcond[cond][odor_i]['cycle_freq'].median(), 3)
                    
            #### plot
            ax = axs[0, c]
            ax.set_title(odor_i, fontweight='bold', rotation=0)
            ax.semilogy(hzPxx, Pxx_allcond[cond][odor_i][n_chan,:], color='k')
            ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond[cond][odor_i][n_chan,:].max(), color='r')
            ax.set_xlim(0,60)
 
            ax = axs[1, c]
            Pxx_sel_min = Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:][np.where(hzPxx[remove_zero_pad:] < 2)[0]].min()
            Pxx_sel_max = Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:][np.where(hzPxx[remove_zero_pad:] < 2)[0]].max()
            ax.semilogy(hzPxx[remove_zero_pad:], Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:], color='k')
            ax.set_xlim(0, 2)
            ax.set_ylim(Pxx_sel_min, Pxx_sel_max)
            ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:].max(), color='r')

            ax = axs[2, c]
            ax.plot(hzCxy,Cxy_allcond[cond][odor_i][n_chan,:], color='k')
            ax.plot(hzCxy,surrogates_allcond['Cxy'][cond][odor_i][n_chan,:], color='c')
            ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

            ax = axs[3, c]
            MVL_i = np.round(MVL_allcond[cond][odor_i][n_chan], 5)
            MVL_surr = np.percentile(surrogates_allcond['MVL'][cond][odor_i][n_chan,:], 99)
            if MVL_i > MVL_surr:
                MVL_p = f'MVL : {MVL_i}, *** {int(MVL_i * 100 / MVL_surr)}%'
            else:
                MVL_p = f'MVL : {MVL_i}, NS {int(MVL_i * 100 / MVL_surr)}%'
            # ax.set_title(MVL_p, rotation=0)
            ax.set_xlabel(MVL_p)

            ax.plot(cyclefreq_allcond[cond][odor_i][n_chan,:], color='k')
            ax.plot(surrogates_allcond['cyclefreq_wb'][cond][odor_i][0, n_chan,:], color='b')
            ax.plot(surrogates_allcond['cyclefreq_wb'][cond][odor_i][1, n_chan,:], color='c', linestyle='dotted')
            ax.plot(surrogates_allcond['cyclefreq_wb'][cond][odor_i][2, n_chan,:], color='c', linestyle='dotted')
            ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=surrogates_allcond['cyclefreq_wb'][cond][odor_i][2, n_chan,:].min(), ymax=surrogates_allcond['cyclefreq_wb'][cond][odor_i][1, n_chan,:].max(), colors='r')
            #plt.show() 

        #### save
        os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'summary', 'condition'))
        fig.savefig(f'{sujet}_{chan_name}_{cond}_{band_prep}.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

        








    

################################
######## TOPOPLOT ########
################################



#n_chan, chan_name = 0, chan_list_eeg[0]
def plot_save_PSD_Cxy_CF_MVL_TOPOPLOT(sujet, band_prep):

    #### create montage
    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    #### load data
    Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond = get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet)
    prms = get_params()
    respfeatures_allcond = load_respfeatures(sujet)

    #### params
    hzPxx = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### reduce data
    topoplot_data = {}

    for cond in conditions:

        topoplot_data[cond] = {}

        for odor_i in odor_list:

            topoplot_data[cond][odor_i] = {}

            mean_resp = respfeatures_allcond[cond][odor_i]['cycle_freq'].mean()
            hzCxy_mask = (hzCxy > (mean_resp - around_respi_Cxy)) & (hzCxy < (mean_resp + around_respi_Cxy))

            Cxy_allchan_i = Cxy_allcond[cond][odor_i][:,hzCxy_mask].mean(axis=1)
            topoplot_data[cond][odor_i]['Cxy'] = Cxy_allchan_i
            MVL_allchan_i = MVL_allcond[cond][odor_i]
            topoplot_data[cond][odor_i]['MVL'] = MVL_allchan_i

            Cxy_allchan_surr_i = surrogates_allcond['Cxy'][cond][odor_i][:len(chan_list_eeg),hzCxy_mask].mean(axis=1)
            topoplot_data[cond][odor_i]['Cxy_surr'] = np.array(Cxy_allchan_surr_i > Cxy_allchan_i)*1
            MVL_allchan_surr_i = np.array([np.percentile(surrogates_allcond['MVL'][cond][odor_i][nchan,:], 99) for nchan, _ in enumerate(chan_list_eeg)])
            topoplot_data[cond][odor_i]['MVL_surr'] = np.array(MVL_allchan_surr_i > MVL_allchan_i)*1

    for cond in conditions:

        for odor_i in odor_list:

            #band, freq = 'theta', [4, 8]
            for band, freq in freq_band_fc_analysis.items():

                hzPxx_mask = (hzPxx >= freq[0]) & (hzPxx <= freq[-1])
                Pxx_mean_i = Pxx_allcond[cond][odor_i][:,hzPxx_mask].mean(axis=1)
                topoplot_data[cond][odor_i][f'Pxx_{band}'] = Pxx_mean_i

    #### scales
    scales_cond = {}
    scales_odor = {}
    scales_allband = {}

    data_type_list = [f'Pxx_{band}' for band in freq_band_fc_analysis.keys()] + ['MVL']

    #data_type = data_type_list[0]
    for data_type in data_type_list:

        if data_type == 'MVL':

            scales_cond[data_type] = {}
            scales_odor[data_type] = {}

            for odor_i in odor_list:

                scales_cond[data_type][odor_i] = {}

                val = np.array([])

                for cond in conditions:

                    val = np.append(val, topoplot_data[cond][odor_i][data_type])

                scales_cond[data_type][odor_i]['min'] = val.min()
                scales_cond[data_type][odor_i]['max'] = val.max()

            for cond in conditions:

                scales_odor[data_type][cond] = {}

                val = np.array([])

                for odor_i in odor_list:

                    val = np.append(val, topoplot_data[cond][odor_i][data_type])

                scales_odor[data_type][cond]['min'] = val.min()
                scales_odor[data_type][cond]['max'] = val.max()

        else:

            scales_allband[data_type] = {}

            val = np.array([])

            for odor_i in odor_list:

                for cond in conditions:

                    val = np.append(val, topoplot_data[cond][odor_i][data_type])

            scales_allband[data_type]['min'] = val.min()
            scales_allband[data_type]['max'] = val.max()


    #### plot Cxy MVL
    #odor_i = odor_list[0]
    for odor_i in odor_list:

        fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
        plt.suptitle(f'{sujet}_{odor_i}')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(conditions):
                    
            #### plot
            ax = axs[0, c]
            ax.set_title(cond, fontweight='bold', rotation=0)
            if c == 0:
                ax.set_ylabel(f'Cxy [0-1]')
            mne.viz.plot_topomap(topoplot_data[cond][odor_i]['Cxy'], info, axes=ax, vmin=0, vmax=1, show=False)
 
            ax = axs[1, c]
            if c == 0:
                ax.set_ylabel(f'Cxy_surr [0-1]')
            mne.viz.plot_topomap(topoplot_data[cond][odor_i]['Cxy_surr'], info, axes=ax, vmin=0, vmax=1, show=False)

            ax = axs[2, c]
            vmin = np.round(scales_odor['MVL'][cond]['min'], 2)
            vmax = np.round(scales_odor['MVL'][cond]['max'], 2)
            if c == 0:
                ax.set_ylabel(f'MVL [{vmin}-{vmax}]')
            mne.viz.plot_topomap(topoplot_data[cond][odor_i]['MVL'], info, axes=ax, vmin=vmin, vmax=vmax, show=False)

            ax = axs[3, c]
            if c == 0:
                ax.set_ylabel(f'MVL_surr [0-1]')
            mne.viz.plot_topomap(topoplot_data[cond][odor_i]['MVL_surr'], info, axes=ax, vmin=0, vmax=1, show=False)
            #plt.show() 

        #### save
        os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'topoplot', 'odor'))
        fig.savefig(f'{sujet}_{odor_i}_{band_prep}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #cond = 'FR_CV_1'
    for cond in conditions:

        fig, axs = plt.subplots(nrows=4, ncols=len(odor_list))
        plt.suptitle(f'{sujet}_{cond}')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, odor_i = 0, odor_list[0]
        for c, odor_i in enumerate(odor_list):

            #### plot
            ax = axs[0, c]
            ax.set_title(odor_i, fontweight='bold', rotation=0)
            if c == 0:
                ax.set_ylabel(f'Cxy [0-1]')
            mne.viz.plot_topomap(topoplot_data[cond][odor_i]['Cxy'], info, axes=ax, vmin=0, vmax=1, show=False)
 
            ax = axs[1, c]
            if c == 0:
                ax.set_ylabel(f'Cxy_surr [0-1]')
            mne.viz.plot_topomap(topoplot_data[cond][odor_i]['Cxy_surr'], info, axes=ax, vmin=0, vmax=1, show=False)

            ax = axs[2, c]
            vmin = np.round(scales_odor['MVL'][cond]['min'], 2)
            vmax = np.round(scales_odor['MVL'][cond]['max'], 2)
            if c == 0:
                ax.set_ylabel(f'MVL [{vmin}-{vmax}]')
            mne.viz.plot_topomap(topoplot_data[cond][odor_i]['MVL'], info, axes=ax, vmin=vmin, vmax=vmax, show=False)

            ax = axs[3, c]
            if c == 0:
                ax.set_ylabel(f'MVL_surr [0-1]')
            mne.viz.plot_topomap(topoplot_data[cond][odor_i]['MVL_surr'], info, axes=ax, vmin=0, vmax=1, show=False)
            #plt.show() 

        #### save
        os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'topoplot', 'condition'))
        fig.savefig(f'{sujet}_{cond}_{band_prep}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #### plot Pxx
    #band, freq = 'theta', [4, 8]
    for band, freq in freq_band_fc_analysis.items():

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
        plt.suptitle(f'{sujet}_{band}')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(conditions):

            #r, odor_i = 0, odor_list[0]
            for r, odor_i in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(cond, fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor_i}')
                
                mne.viz.plot_topomap(topoplot_data[cond][odor_i][f'Pxx_{band}'], info, axes=ax, vmin=scales_allband[f'Pxx_{band}']['min'], 
                                     vmax=scales_allband[f'Pxx_{band}']['max'], show=False)

        # plt.show() 

        #### save
        os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'topoplot'))
        fig.savefig(f'{sujet}_{band}_{band_prep}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()








################################
######## LOAD TF & ITPC ########
################################


def compute_TF_ITPC(sujet):

    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:
    
        if tf_mode == 'TF':
            print('######## LOAD TF ########')
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            if os.path.exists(os.path.join(path_precompute, sujet, 'TF', f'allcond_{sujet}_tf_stretch.pkl')):
                print('ALREADY COMPUTED')
                continue
            
        elif tf_mode == 'ITPC':
            print('######## LOAD ITPC ########')
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

            if os.path.exists(os.path.join(path_precompute, sujet, 'ITPC', f'allcond_{sujet}_itpc_stretch.pkl')):
                print('ALREADY COMPUTED')
                continue

        #### load file with reducing to one TF
        tf_stretch_allcond = {}

        #band_prep = 'wb'
        for band_prep in band_prep_list:

            tf_stretch_allcond[band_prep] = {}

            #### chose nfrex
            _, nfrex = get_wavelets(band_prep, list(freq_band_dict[band_prep].values())[0])  

            #cond = 'FR_CV'
            for cond in conditions:

                tf_stretch_allcond[band_prep][cond] = {}

                for odor_i in odor_list:

                    tf_stretch_allcond[band_prep][cond][odor_i] = {}

                    #### impose good order in dict
                    for band, freq in freq_band_dict[band_prep].items():
                        tf_stretch_allcond[band_prep][cond][odor_i][band] = np.zeros(( len(chan_list_eeg), nfrex, stretch_point_TF ))

                    #### load file
                    for band, freq in freq_band_dict[band_prep].items():
                        
                        for file_i in os.listdir(): 

                            if file_i.find(f'{freq[0]}_{freq[1]}_{cond}_{odor_i}') != -1 and file_i.find('STATS') == -1:
                                file_to_load = file_i
                            else:
                                continue
                        
                        tf_stretch_allcond[band_prep][cond][odor_i][band] += np.load(file_to_load)
               
        #### save
        if tf_mode == 'TF':
            with open(f'allcond_{sujet}_tf_stretch.pkl', 'wb') as f:
                pickle.dump(tf_stretch_allcond, f)
        elif tf_mode == 'ITPC':
            with open(f'allcond_{sujet}_itpc_stretch.pkl', 'wb') as f:
                pickle.dump(tf_stretch_allcond, f)

    print('done')








########################################
######## PLOT & SAVE TF & ITPC ########
########################################


def get_tf_stats(tf, nchan, pixel_based_distrib, nfrex):

    tf_thresh = tf.copy()
    #wavelet_i = 0
    for wavelet_i in range(nfrex):
        mask = np.logical_or(tf_thresh[wavelet_i, :] >= pixel_based_distrib[nchan, wavelet_i, 0], tf_thresh[wavelet_i, :] <= pixel_based_distrib[nchan, wavelet_i, 1])
        tf_thresh[wavelet_i, mask] = 1
        tf_thresh[wavelet_i, np.logical_not(mask)] = 0

    return tf_thresh




def get_tf_itpc_stretch_allcond(sujet, tf_mode):

    source_path = os.getcwd()

    if tf_mode == 'TF':

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        with open(f'allcond_{sujet}_tf_stretch.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)


    elif tf_mode == 'ITPC':
        
        os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

        with open(f'allcond_{sujet}_itpc_stretch.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)

    os.chdir(source_path)

    return tf_stretch_allcond






#n_chan = 0
def save_TF_ITPC_n_chan(n_chan, tf_mode, band_prep):

    if tf_mode == 'TF':
        os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))
    elif tf_mode == 'ITPC':
        os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))

    print_advancement(n_chan, len(chan_list_eeg), steps=[25, 50, 75])

    chan_name = chan_list_eeg[n_chan]

    #band_prep_plot = 'lf'
    for band_prep_plot in ['lf', 'hf']:

        freq_band = freq_band_dict[band_prep_plot]

        #odor_i = odor_list[0]
        for odor_i in odor_list:

            #### scale
            # vmaxs = {}
            # vmins = {}
            # for cond in prms['conditions']:

            #     scales = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}

            #     for i, (band, freq) in enumerate(freq_band.items()) :

            #         if band == 'whole' or band == 'l_gamma':
            #             continue

            #         data = get_tf_itpc_stretch_allcond(sujet, tf_mode)[band_prep][cond][band][n_chan, :, :]
            #         frex = np.linspace(freq[0], freq[1], np.size(data,0))

            #         scales['vmin_val'] = np.append(scales['vmin_val'], np.min(data))
            #         scales['vmax_val'] = np.append(scales['vmax_val'], np.max(data))
            #         scales['median_val'] = np.append(scales['median_val'], np.median(data))

            #         del data

            #     median_diff = np.max([np.abs(np.min(scales['vmin_val']) - np.median(scales['median_val'])), np.abs(np.max(scales['vmax_val']) - np.median(scales['median_val']))])

            #     vmin = np.median(scales['median_val']) - median_diff
            #     vmax = np.median(scales['median_val']) + median_diff

            #     vmaxs[cond] = vmax
            #     vmins[cond] = vmin

            #### plot
            fig, axs = plt.subplots(nrows=len(freq_band), ncols=len(conditions))

            plt.suptitle(f'{sujet}_{chan_name}')

            fig.set_figheight(10)
            fig.set_figwidth(10)

            #### for plotting l_gamma down
            if band_prep == 'hf':
                keys_list_reversed = list(freq_band.keys())
                keys_list_reversed.reverse()
                freq_band_reversed = {}
                for key_i in keys_list_reversed:
                    freq_band_reversed[key_i] = freq_band[key_i]
                freq_band = freq_band_reversed

            #c, cond = 1, conditions[1]
            for c, cond in enumerate(conditions):

                #### plot
                #i, (band, freq) = 0, list(freq_band.items())[0] 
                for i, (band, freq) in enumerate(freq_band.items()) :

                    data = get_tf_itpc_stretch_allcond(sujet, tf_mode)[band_prep][cond][odor_i][band][n_chan, :, :]
                    frex = np.linspace(freq[0], freq[1], np.size(data,0))
                
                    ax = axs[i,c]

                    if i == 0 :
                        ax.set_title(cond, fontweight='bold', rotation=0)

                    time = range(stretch_point_TF)

                    ax.pcolormesh(time, frex, rscore_mat(data), vmin=-rscore_mat(data).max(), vmax=rscore_mat(data).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))

                    if tf_mode == 'TF' and cond != 'FR_CV_1':
                        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

                        pixel_based_distrib = np.load(f'{sujet}_tf_STATS_{cond}_{odor_i}_intra_{str(freq[0])}_{str(freq[1])}.npy')
                        
                        _, nfrex = get_wavelets(band_prep, freq)
                        if get_tf_stats(rscore_mat(data), n_chan, pixel_based_distrib, nfrex).sum() != 0:
                            ax.contour(time, frex, get_tf_stats(rscore_mat(data), n_chan, pixel_based_distrib, nfrex), levels=0, colors='g')

                    if c == 0:
                        ax.set_ylabel(band)

                    ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')

                    del data

            #plt.show()

            if tf_mode == 'TF':
                os.chdir(os.path.join(path_results, sujet, 'TF', 'summary', 'odor'))
            elif tf_mode == 'ITPC':
                os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary', 'odor'))

            #### save
            fig.savefig(f'{sujet}_{chan_name}_{odor_i}_{band_prep}.jpeg', dpi=150)
                
            fig.clf()
            plt.close('all')
            gc.collect()


        #cond = conditions[0]
        for cond in conditions:

            #### scale
            # vmaxs = {}
            # vmins = {}
            # for cond in prms['conditions']:

            #     scales = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}

            #     for i, (band, freq) in enumerate(freq_band.items()) :

            #         if band == 'whole' or band == 'l_gamma':
            #             continue

            #         data = get_tf_itpc_stretch_allcond(sujet, tf_mode)[band_prep][cond][band][n_chan, :, :]
            #         frex = np.linspace(freq[0], freq[1], np.size(data,0))

            #         scales['vmin_val'] = np.append(scales['vmin_val'], np.min(data))
            #         scales['vmax_val'] = np.append(scales['vmax_val'], np.max(data))
            #         scales['median_val'] = np.append(scales['median_val'], np.median(data))

            #         del data

            #     median_diff = np.max([np.abs(np.min(scales['vmin_val']) - np.median(scales['median_val'])), np.abs(np.max(scales['vmax_val']) - np.median(scales['median_val']))])

            #     vmin = np.median(scales['median_val']) - median_diff
            #     vmax = np.median(scales['median_val']) + median_diff

            #     vmaxs[cond] = vmax
            #     vmins[cond] = vmin

            #### plot
            fig, axs = plt.subplots(nrows=len(freq_band), ncols=len(odor_list))

            plt.suptitle(f'{sujet}_{chan_name}')

            fig.set_figheight(10)
            fig.set_figwidth(10)

            #### for plotting l_gamma down
            if band_prep == 'hf':
                keys_list_reversed = list(freq_band.keys())
                keys_list_reversed.reverse()
                freq_band_reversed = {}
                for key_i in keys_list_reversed:
                    freq_band_reversed[key_i] = freq_band[key_i]
                freq_band = freq_band_reversed

            #c, odor_i = 1, odor_list[1]
            for c, odor_i in enumerate(odor_list):

                #### plot
                #i, (band, freq) = 0, list(freq_band.items())[0] 
                for i, (band, freq) in enumerate(freq_band.items()) :

                    data = get_tf_itpc_stretch_allcond(sujet, tf_mode)[band_prep][cond][odor_i][band][n_chan, :, :]
                    frex = np.linspace(freq[0], freq[1], np.size(data,0))
                
                    ax = axs[i,c]

                    if i == 0 :
                        ax.set_title(odor_i, fontweight='bold', rotation=0)

                    time = range(stretch_point_TF)

                    ax.pcolormesh(time, frex, rscore_mat(data), vmin=-rscore_mat(data).max(), vmax=rscore_mat(data).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))

                    if tf_mode == 'TF' and odor_i != 'o':
                        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

                        pixel_based_distrib = np.load(f'{sujet}_tf_STATS_{cond}_{odor_i}_inter_{str(freq[0])}_{str(freq[1])}.npy')
                        
                        _, nfrex = get_wavelets(band_prep, freq)
                        if get_tf_stats(rscore_mat(data), n_chan, pixel_based_distrib, nfrex).sum() != 0:
                            ax.contour(time, frex, get_tf_stats(rscore_mat(data), n_chan, pixel_based_distrib, nfrex), levels=0, colors='g')

                    if c == 0:
                        ax.set_ylabel(band)

                    ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')

                    del data

            #plt.show()

            if tf_mode == 'TF':
                os.chdir(os.path.join(path_results, sujet, 'TF', 'summary', 'condition'))
            elif tf_mode == 'ITPC':
                os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary', 'condition'))

            #### save
            fig.savefig(f'{sujet}_{chan_name}_{cond}_{band_prep}.jpeg', dpi=150)
                
            fig.clf()
            plt.close('all')
            gc.collect()




########################################
######## COMPILATION FUNCTION ########
########################################

def compilation_compute_Pxx_Cxy_Cyclefreq_MVL(sujet, band_prep):

    #### compute & reduce surrogates
    print('######## COMPUTE PSD AND COH ########')
    compute_PxxCxyCyclefreqSurrogates(sujet, band_prep)
    
    #### compute joblib
    print('######## PLOT & SAVE PSD AND COH ########')
    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(plot_save_PSD_Cxy_CF_MVL)(n_chan, chan_name, band_prep) for n_chan, chan_name in enumerate(chan_list_eeg))

    print('######## PLOT & SAVE TOPOPLOT ########')
    plot_save_PSD_Cxy_CF_MVL_TOPOPLOT(sujet, band_prep)

    print('done')

    


def compilation_compute_TF_ITPC(sujet):

    compute_TF_ITPC(sujet)
    
    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:
        
        if tf_mode == 'TF':
            print('######## PLOT & SAVE TF ########')
        if tf_mode == 'ITPC':
            continue
            print('######## PLOT & SAVE ITPC ########')
        
        #band_prep = 'wb'
        for band_prep in band_prep_list: 

            print(band_prep)

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_TF_ITPC_n_chan)(n_chan, tf_mode, band_prep) for n_chan, tf_mode, band_prep in zip(range(len(chan_list_eeg)), [tf_mode]*len(chan_list_eeg), [band_prep]*len(chan_list_eeg)))

    print('done')










################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    band_prep = 'wb'

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        print(sujet)

        #### Pxx Cxy CycleFreq
        compilation_compute_Pxx_Cxy_Cyclefreq_MVL(sujet, band_prep)
        # execute_function_in_slurm_bash_mem_choice('n9_res_power', 'compilation_compute_Pxx_Cxy_Cyclefreq_MVL', [sujet], 15)

        #### TF & ITPC
        compilation_compute_TF_ITPC(sujet)
        # execute_function_in_slurm_bash_mem_choice('n9_res_power', 'compilation_compute_TF_ITPC', [sujet], 15)


