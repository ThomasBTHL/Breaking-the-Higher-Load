"""Import modules"""
import copy
import pickle
import functions as f
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import scipy.signal as sp
import pandas as pd

"""
Add_EMG_data is developed and written by Thomas van Hogerwou, master student TU-Delft

It adds EMG data to the Outputs result dictionary based on peak accelerations
Contact E-Mail: thom.hogerwou@gmail.com
Version 1.5 (2021-05-18)
"""

"""
Input area
"""
# pitchers = ['PP01','PP02','PP03','PP04','PP05','PP06','PP07','PP08','PP12','PP14','PP15'] #PP01 - PP15\
fs_EMG = 2000
fs_opti = 120
lead = .2
lag = .8
fs_scaling = fs_opti / fs_EMG
EMG_markers = ['ACC','BIC','DM','FMP','LD','PC','PS','TRI']
Wn = 20 #Hz of the lowpass filter
N = 2 #Order of lowpass filter

frequencies = ['120 Hz', '2000 Hz']
pitchers = ['PP01']

for pitcher in pitchers:
    for frequency in frequencies:
        # --- Define path where cumulative data is saved --- #
        path = "data/EMG_Data/02 Preprocessed data/"+frequency+"/" + pitcher + "/"
        filename = "Cumulative_EMG"

        # --- Load data from pickle --- #
        filenameIn = path + filename
        infile = open(filenameIn, 'rb')
        EMG_data = pickle.load(infile)
        infile.close()

        # --- Prepare rectified dictionary --- #
        Rectified_storage = copy.deepcopy(EMG_data)

        # --- Rectify EMG_data --- #
        for EMG_marker in EMG_data:
            for pitch in EMG_data[EMG_marker]:
                Rectified_storage[EMG_marker][pitch] = np.array(np.abs(EMG_data[EMG_marker][pitch]))

        # --- Prepare filtered dictionary --- #
        Filtered_storage = copy.deepcopy(Rectified_storage)

        if frequency == '120 Hz':
            fs = fs_opti
        else:
            fs = fs_EMG
        b, a = sp.butter(N, (Wn/(fs/2)))
        # --- Filter EMG_data --- #
        for EMG_marker in EMG_data:
            if EMG_marker != 'ACC':
                for pitch in EMG_data[EMG_marker]:
                    Filtered_storage[EMG_marker][pitch] = sp.filtfilt(b = b, a = a, x = np.array(Rectified_storage[EMG_marker][pitch]))

plt.figure()
for EMG_marker in Filtered_storage:
    if EMG_marker != 'ACC':
        plt.plot(Filtered_storage[EMG_marker]['pitch_1'],label = EMG_marker)
        plt.legend()

