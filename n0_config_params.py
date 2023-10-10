
import numpy as np
import scipy.signal

################################
######## MODULES ########
################################

# anaconda (numpy, scipy, pandas, matplotlib, glob2, joblib, xlrd)
# neurokit2 as nk
# respirationtools
# mne
# neo
# bycycle
# pingouin

################################
######## GENERAL PARAMS ######## 
################################

enable_big_execute = False
perso_repo_computation = False

#sujet = 'DEBUG'

conditions = ['FR_CV_1', 'RD_CV', 'RD_SV', 'RD_FV', 'FR_CV_2']
conditions_allsession = ['FR_CV_1', 'RD_CV_1', 'RD_CV_2', 'RD_SV_1', 'RD_SV_2', 'RD_SV_3', 'RD_FV_1', 'RD_FV_2', 'FR_CV_2']

n_session_condition = {'FR_CV_1' : 1, 'RD_CV' : 2, 'RD_SV' : 3, 'RD_FV' : 2, 'FR_CV_2' : 1}

sujet_list = np.array(['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09',
'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18',
'P19', 'P20', 'P21', 'P23', 'P24', 'P25', 'P26', 'P27',
'P28', 'P29', 'P30', 'P31'])

band_prep_list = ['wb']

freq_band_dict = {'wb' : {'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120], 'whole' : [2,50]},
                'lf' : {'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40], 'whole' : [2,50]},
                'hf' : {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]} }

freq_band_list_precompute = {'wb' : {'theta_1' : [2,10], 'theta_2' : [4,8], 'alpha_1' : [8,12], 'alpha_2' : [8,14], 
                                    'beta_1' : [12,40], 'beta_2' : [10,40], 'whole_1' : [2,50], 'l_gamma_1' : [50, 80], 
                                    'h_gamma_1' : [80, 120]} }

freq_band_dict_FC = {'wb' : {'theta' : [4,8], 'alpha' : [8,12], 'beta' : [12,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]},
                'lf' : {'theta' : [4,8], 'alpha' : [8,12], 'beta' : [12,40]},
                'hf' : {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]} }

odor_list = ['o', '+', '-']

phase_list = ['whole', 'inspi', 'expi']

srate = 500

chan_list = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 
            'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 
            'ECG', 'GSR', 'RespiNasale', 'RespiVentrale', 'ECG_cR']

chan_list_aux = ['ECG', 'GSR', 'RespiNasale', 'RespiVentrale', 'ECG_cR']
            
chan_list_eeg = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 
            'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2']

chan_list_eeg_fc = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8','Fz', 'FC1', 'FC2', 'FC5', 'FC6', 'FT9', 'FT10', 'Cz', 'C3', 'C4',
                    'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8', 'T7', 'T8', 'TP9', 'TP10', 'Oz', 'O1', 'O2']



########################
######## TRIG ########
########################

correspondance_trig = {'FR_CV' : [61, 62], 'RD_CV' : [31, 32], 'RD_SV' : [11, 12], 'RD_FV' : [51, 52]}


################################
######## ODOR ORDER ########
################################

odor_order = {

'P01' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   'P02' : {'ses02' : 'o', 'ses03' : '+', 'ses04' : '-'},   'P03' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   
'P04' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},   'P05' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   'P06' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   
'P07' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},   'P08' : {'ses02' : 'o', 'ses03' : '+', 'ses04' : '-'},   'P09' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   
'P10' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},   'P11' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   'P12' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},   
'P13' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   'P14' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   'P15' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},
'P16' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   'P17' : {'ses02' : '+', 'ses03' : '-', 'ses04' : 'o'},   'P18' : {'ses02' : '+', 'ses03' : '-', 'ses04' : 'o'},   
'P19' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   'P20' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   'P21' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},   
'P22' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   'P23' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   'P24' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   
'P25' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},   'P26' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   'P27' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   
'P28' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},   'P29' : {'ses02' : 'o', 'ses03' : '+', 'ses04' : '-'},   'P30' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},
'P31' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},
}




########################################
######## PATH DEFINITION ########
########################################

import socket
import os
import platform
 
PC_OS = platform.system()
PC_ID = socket.gethostname()

if PC_ID == 'LAPTOP-EI7OSP7K':

    PC_working = 'Jules_Home'
    if perso_repo_computation:
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon\\EEG'
    else:    
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon\\EEG'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Lyon\\EEG'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Lyon\\EEG\\Mmap'
    n_core = 4

elif PC_ID == 'DESKTOP-3IJUK7R':

    PC_working = 'Jules_Labo_Win'
    if perso_repo_computation:
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    else:    
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Mmap'
    n_core = 2

elif PC_ID == 'pc-jules':

    PC_working = 'Jules_Labo_Linux'
    if perso_repo_computation:
        path_main_workdir = '/home/jules/Bureau/perso_repo_computation/Script_Python_EEG_Lyon_VJ_git'
    else:    
        path_main_workdir = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/Emosens/NBuonviso2022_Emosens1_Jules_Valentin/EEG_Lyon_VJ/Script_Python_EEG_Lyon_git'
    path_general = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/Emosens/NBuonviso2022_Emosens1_Jules_Valentin/EEG_Lyon_VJ'
    path_memmap = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/Emosens/NBuonviso2022_Emosens1_Jules_Valentin/EEG_Lyon_VJ/Mmap'
    n_core = 4

elif PC_ID == 'pc-valentin':

    PC_working = 'Valentin_Labo_Linux'
    if perso_repo_computation:
        path_main_workdir = '/home/valentin/Bureau/perso_repo_computation/Script_Python_EEG_Lyon_git'
    else:    
        path_main_workdir = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/Emosens/NBuonviso2022_Emosens1_Jules_Valentin/EEG_Lyon_VJ/Script_Python_EEG_Lyon_git'
    path_general = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/Emosens/NBuonviso2022_Emosens1_Jules_Valentin/EEG_Lyon_VJ'
    path_memmap = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/Emosens/NBuonviso2022_Emosens1_Jules_Valentin/EEG_Lyon_VJ/Mmap'
    n_core = 6

elif PC_ID == 'nodeGPU':

    PC_working = 'nodeGPU'
    path_main_workdir = '/crnldata/cmo/Projets/Emosens/NBuonviso2022_Emosens1_Jules_Valentin/EEG_Lyon_VJ/Script_Python_EEG_Lyon_git'
    path_general = '/crnldata/cmo/Projets/Emosens/NBuonviso2022_Emosens1_Jules_Valentin/EEG_Lyon_VJ'
    path_memmap = '/mnt/data/julesgranget'
    n_core = 15

else:

    PC_working = 'crnl_cluster'
    path_main_workdir = '/crnldata/cmo/Projets/Emosens/NBuonviso2022_Emosens1_Jules_Valentin/EEG_Lyon_VJ/Script_Python_EEG_Lyon_git'
    path_general = '/crnldata/cmo/Projets/Emosens/NBuonviso2022_Emosens1_Jules_Valentin/EEG_Lyon_VJ'
    path_memmap = '/mnt/data/julesgranget'
    n_core = 10
    

path_data = os.path.join(path_general, 'Data')
path_prep = os.path.join(path_general, 'Analyses', 'preprocessing')
path_precompute = os.path.join(path_general, 'Analyses', 'precompute') 
path_results = os.path.join(path_general, 'Analyses', 'results') 
path_respfeatures = os.path.join(path_general, 'Analyses', 'results') 
path_anatomy = os.path.join(path_general, 'Analyses', 'anatomy')
path_slurm = os.path.join(path_general, 'Script_slurm')

#### slurm params
mem_crnl_cluster = '10G'
n_core_slurms = 10







################################
######## RESPI PARAMS ########
################################ 

#### INSPI DOWN
sujet_respi_adjust = {
'P01':'inverse',   'P02':'inverse',   'P03':'inverse',   'P04':'inverse',   'P05':'inverse',
'P06':'inverse',   'P07':'inverse',   'P08':'inverse',   'P09':'inverse',   'P10':'inverse',
'P11':'inverse',   'P12':'inverse',   'P13':'inverse',   'P14':'inverse',   'P15':'inverse',
'P16':'inverse',   'P17':'inverse',   'P18':'inverse',   'P19':'inverse',   'P20':'inverse',
'P21':'inverse',   'P22':'inverse',   'P23':'inverse',   'P24':'inverse',   'P25':'inverse',
'P26':'inverse',   'P27':'inverse',   'P28':'inverse',   'P29':'inverse',   'P30':'inverse',
'P31':'inverse',
}


cycle_detection_params = {
'exclusion_metrics' : 'med',
'metric_coeff_exclusion' : 3,
'inspi_coeff_exclusion' : 2,
'respi_scale' : [0.1, 0.35], #Hz
}



################################
######## ECG PARAMS ########
################################ 

sujet_ecg_adjust = {
'P01':'inverse',   'P02':'inverse',   'P03':'inverse',   'P04':'inverse',   'P05':'inverse',
'P06':'inverse',   'P07':'inverse',   'P08':'inverse',   'P09':'inverse',   'P10':'inverse',
'P11':'inverse',   'P12':'inverse',   'P13':'inverse',   'P14':'inverse',   'P15':'inverse',
'P16':'inverse',   'P17':'inverse',   'P18':'inverse',   'P19':'inverse',   'P20':'inverse',
'P21':'inverse',   'P22':'inverse',   'P23':'inverse',   'P24':'inverse',   'P25':'inverse',
'P26':'inverse',   'P27':'inverse',   'P28':'inverse',   'P29':'inverse',   'P30':'inverse',
'P31':'inverse',
}


hrv_metrics_short_name = ['HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2']




################################
######## PREP PARAMS ########
################################ 


prep_step_debug = {
'reref' : {'execute': True, 'params' : ['TP9']}, #chan = chan to reref
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': True},
'csd_computation' : {'execute': True},
}

prep_step_wb = {
'reref' : {'execute': True, 'params' : ['TP9', 'TP10']}, #chan = chan to reref
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'csd_computation' : {'execute': False},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': False},
}

prep_step_lf = {
'reref' : {'execute': False, 'params' : ['chan']}, #chan = chan to reref
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': False},
'csd_computation' : {'execute': True},
}

prep_step_hf = {
'reref_mastoide' : {'execute': False},
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : 55, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : None, 'h_freq': None}},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': False},
'csd_computation' : {'execute': True},
}





########################################
######## PARAMS SURROGATES ########
########################################

#### Pxx Cxy

zero_pad_coeff = 15

def get_params_spectral_analysis(srate):
    nwind = int( 20*srate ) # window length in seconds*srate
    nfft = nwind*zero_pad_coeff # if no zero padding nfft = nwind
    noverlap = np.round(nwind/2) # number of points of overlap here 50%
    hannw = scipy.signal.windows.hann(nwind) # hann window

    return nwind, nfft, noverlap, hannw

#### plot Pxx Cxy  
if zero_pad_coeff - 5 <= 0:
    remove_zero_pad = 0
remove_zero_pad = zero_pad_coeff - 5

#### stretch
stretch_point_surrogates = 500

#### coh
n_surrogates_coh = 1000
freq_surrogates = [0, 2]
percentile_coh = .95

#### cycle freq
n_surrogates_cyclefreq = 1000
percentile_cyclefreq_up = .99
percentile_cyclefreq_dw = .01






################################
######## PRECOMPUTE TF ########
################################

#### stretch
stretch_point_TF = 500
stretch_TF_auto = False
ratio_stretch_TF = 0.5

#### TF & ITPC
nfrex_hf = 50
nfrex_lf = 50
nfrex_wb = 50
ncycle_list_lf = [7, 15]
ncycle_list_hf = [20, 30]
ncycle_list_wb = [7, 30]
srate_dw = 10


#### STATS
n_surrogates_tf = 1000



################################
######## POWER ANALYSIS ########
################################

#### analysis
coh_computation_interval = .02 #Hz around respi


################################
######## FC ANALYSIS ########
################################


#### band to remove
freq_band_fc_analysis = {'theta' : [4, 8], 'alpha' : [9,12], 'beta' : [15,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}

percentile_thresh = 90

#### for DFC
slwin_dict = {'theta' : 5, 'alpha' : 3, 'beta' : 1, 'l_gamma' : .3, 'h_gamma' : .3} # seconds
slwin_step_coeff = .1  # in %, 10% move

band_name_fc_dfc = ['theta', 'alpha', 'beta', 'l_gamma', 'h_gamma']

#### cond definition
cond_FC_DFC = ['FR_CV', 'AL', 'SNIFF', 'AC']

#### down sample for AL
dw_srate_fc_AL = 10

#### down sample for AC
dw_srate_fc_AC = 50

#### n points for AL interpolation
n_points_AL_interpolation = 10000
n_points_AL_chunk = 1000

#### for df computation
percentile_graph_metric = 25



################################
######## TOPOPLOT ########
################################

around_respi_Cxy = 0.025


################################
######## HRV ANALYSIS ########
################################



srate_resample_hrv = 10
nwind_hrv = int( 128*srate_resample_hrv )
nfft_hrv = nwind_hrv
noverlap_hrv = np.round(nwind_hrv/10)
win_hrv = scipy.signal.windows.hann(nwind_hrv)
f_RRI = (.1, .5)




################################
######## HRV TRACKER ########
################################

cond_label_tracker = {'FR_CV_1' : 1, 'MECA' : 2, 'CO2' : 3, 'FR_CV_2' : 1}








