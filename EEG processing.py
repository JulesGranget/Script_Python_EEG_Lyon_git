
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.fftpack
from scipy import *
import math
import mne

# Change dir to load data 
os.chdir('C:\\Users\\jules\\Desktop\\Codage Informatique\\Training Data')


    ### Load mne object

# Extract .mat data
matdat = sio.loadmat('EEG_resting_EO.mat')  #File read into dictionary
data = matdat['dataRest']  #Extract the numpy array in the dictionary
chanlist = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7',
    'P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz',
    'Fp2','AF8','AF4','Afz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz',
    'C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2','EOG1','EOG2','EOG3','TRIG']  #Channel name
srate = 256  #Sampling Rate 
chan_types = ['eeg'] * 64 + ['eog'] * 3 + ['misc'] # chan types for mne

# Add bad chan
#noisy_chan = max(data[0,:])/20 * np.random.rand(np.size(data[0,:]))
#chan2switch = 'F1'
#data[chanlist.index(chan2switch),:] = noisy_chan

# Load data into mne
data_info = mne.create_info(chanlist, sfreq=srate, ch_types=chan_types)  #Create signal information 
raw = mne.io.RawArray(data, data_info) # generate mne object 
mne.datasets.eegbci.standardize(raw) # standardize channel names
raw.set_montage('standard_1020') # add dig position according to standard 10_20
raw.info # verify mne info

# Data vizualisation 
events = None
duration = 10.
start = 0.
n_chan = 20
raw.plot(scalings='auto',events=events,duration=duration,start=start,n_channels=n_chan) # selecting a channel mark it as bad channel

fmin = 0
fmax = srate/2
nfft = srate*20
n_overlap = nfft/2
chan_picks = chanlist[:]
raw.plot_psd(fmin=fmin, fmax=fmax, n_fft=nfft, n_overlap=n_overlap, picks=chan_picks, dB=True, estimate='power')

fig = plt.figure()
ax2d = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection='3d')
raw.plot_sensors(ch_type='eeg', axes=ax2d) # here bad chan will be marked on red
raw.plot_sensors(ch_type='eeg', axes=ax3d, kind='3d') # same here for bad chan


    ### Manipulate mne

# How to extract np.array from mne object
mne_select = raw_eeg.get_data()[:,:] 
mne_info = raw.info['sfreq'] # we go into info and then indicate what field we want to extract

# Remove chan
chan_names = raw.info['ch_names'] # identify which chan to drop
raw_drop = raw.copy()
drop_chan = ['EOG1','EOG2','EOG3','TRIG']
raw_drop = raw_drop.drop_channels(drop_chan)
raw_drop.info # verify
raw_select.plot(scalings='auto') # verify

# Select chan
chan_names = raw.info['ch_names'] # identify which chan to drop
raw_select = raw.copy()
select_chan = ['EOG1','EOG2','EOG3']
raw_select = raw.pick_channels(select_chan)
raw_select.info # verify
raw_select.plot(scalings='auto') # verify

# Crop data
raw_crop = raw.copy()
t_min = 0
t_max = 30
raw_crop.crop(t_min,t_max)


    # Analysis

# Bad chan
bad_chan = raw.info['bads']
print(bad_chan)
#raw.info['bads'].append('EEG 050')               # add a single channel
#raw.info['bads'].extend(['EEG 051', 'EEG 052'])  # add a list of channels
#raw.info['bads'].pop(-1)                         # remove the last entry in the list

# Interpolate bad chan
raw_inter = raw.copy()
raw_inter = raw_inter.copy().interpolate_bads(reset_bads=True) # with reset_bads=True, bad chan will be automaticaly removed and replace by interpolation 

# Data vizualisation 
events = None
duration = 10.
start = 0.
n_chan = 20
raw_inter.plot(scalings='auto',events=events,duration=duration,start=start,n_channels=n_chan) # selecting a channel mark it as bad channel

# ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw_inter)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
ica.plot_properties(raw_inter)

# Filter

l_freq = None
h_freq = None
filter_length = 'auto'
raw_inter.filter(l_freq, h_freq, filter_length=filter_length, l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=1, method='fir', iir_params=None, copy=True, phase='zero', fir_window='hamming', fir_design='firwin', pad='reflect_limited', verbose=None)[source]


order = srate*10
f_p = 40 # in Hz
transition_band = 0.25 * f_p
f_s = f_p + transition_band
freq = [0., f_p, f_s, srate / 2.]
gain = [1., 1., 0., 0.]
h = mne.filter.create_filter(raw_filter, srate, l_freq=None, h_freq=f_p, fir_design='firwin2')
mne.viz.plot_filter(h, srate, freq, gain, 'MNE-Python 0.14 default', compensate=True)







# Annotation
my_annot = mne.Annotations(onset=[3, 5, 7],
                           duration=[1, 0.5, 0.25],
                           description=['AAA', 'BBB', 'CCC'])
print(my_annot)
raw.set_annotations(my_annot)

# Events
events = mne.find_events(raw, stim_channel='STI 014')
events_from_file = mne.read_events(adress
raw.plot(events=events, start=5, duration=10, color='gray',
         event_color={1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'y', 32: 'k'}))
