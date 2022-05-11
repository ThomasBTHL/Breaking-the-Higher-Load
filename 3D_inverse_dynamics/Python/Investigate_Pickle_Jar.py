"""Import modules"""
import pickle
import functions as f
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import scipy.signal as sp
import pandas as pd

"""Investigate_pickle_jar is developed and written by Thomas van Hogerwou, master student TU-Delft

all this file does is give me a way to look at pickle data without running a ton of code
Contact E-Mail: thom.hogerwou@gmail.com

Version 1.5 (2021-05-18)"""

fs_EMG = 2000
fs_opti = 120
lead = .2
lag = .8
filter_state = 'Filtered'
pitcher = 'PP03'
inning = 'Inning_2'
EMG_markers = ['ACC','BIC','DM','FMP','LD','PC','PS','TRI']
EMG_downsampled = dict.fromkeys(EMG_markers)

"""
Load inning dictionary
"""
# --- Define path where pitch output data is stored --- #
path = "Results/Pitches/"+filter_state+"/"+pitcher+"/"+inning+"/"
filename = "Outputs"

# --- Load data from pickle --- #
filenameIn = path + filename
infile = open(filenameIn, 'rb')
Outputs = pickle.load(infile)
infile.close()


# --- Define path where EMG data is stored --- #
path = "data/EMG_Data/01 Raw Data/PP03/EMG_Cut/"
filename = "PP03_EMGCut_20"

# --- Load data from pickle --- #
filenameIn = path + filename
infile = open(filenameIn, 'rb')
EMG_data = pickle.load(infile)
infile.close()

"""
Downsample EMG data to new dictionary
"""
fs_scaling = fs_opti/fs_EMG
for key in EMG_markers:
    EMG_downsampled[key] = sp.resample(EMG_data[key], int(len(EMG_data[key]) * fs_scaling))

peaks = sp.find_peaks(-1*EMG_downsampled['ACC'], height= np.nanmax(EMG_downsampled['ACC'])*.6, distance= 100)

for segment in Outputs:
    for EMG_marker in EMG_markers:
        Outputs[segment][EMG_marker] = dict()
        i = 0
        for pitch in Outputs[segment]['max_abduction_moment']:
            Outputs[segment][EMG_marker][pitch] = EMG_downsampled[EMG_marker][int(peaks[0][i] - lead * fs_opti):int(peaks[0][i] + lag * fs_opti)]
            i = i + 1