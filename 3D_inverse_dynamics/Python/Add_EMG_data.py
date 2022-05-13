"""Import modules"""
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
pitchers = ['PP01','PP02','PP03','PP04','PP05','PP06','PP07','PP08','PP12','PP14','PP15'] #PP01 - PP15\
# pitchers =['PP14','PP15']
filter_state = 'Filtered' # Unfiltered or Filtered
fs_EMG = 2000
fs_opti = 120
lead = .2
lag = .8
fs_scaling = fs_opti / fs_EMG
EMG_markers = ['ACC','BIC','DM','FMP','LD','PC','PS','TRI']
Segments = ['hand','forearm','upperarm']

for pitcher in pitchers:
    """
    individual pitcher information
    """
    if pitcher == 'PP01':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_5','Inning_6','Inning_7','Inning_8'] # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = [6, 11, 17, 22, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 51, 58, 59, 62, 64, 78, 79]  # pitches to remove
        mean_hand_length = 13.591315171765809
        mean_forearm_length = 25.252098917741122
        mean_upperarm_length = 26.06705963822564

    if pitcher == 'PP02':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6'] # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = [1, 5, 8, 9, 10, 18, 21, 28, 32, 34, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 48, 49, 50, 53,
                           54, 55, 56, 57, 58, 59, 60] # pitches to remove
        mean_forearm_length = 26.4043074818835
        mean_upperarm_length = 27.336296427610584
        mean_hand_length = 18.422773876495867

    if pitcher == 'PP03':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9','Inning_10','Inning_11']  # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = [5,10,14,41,57,73,101]  # pitches to remove
        mean_forearm_length = 27.386957807443334
        mean_upperarm_length = 27.890377309454344
        mean_hand_length = 18.83490902787976

    if pitcher == 'PP04':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = [11,14,15,16,70,72,78]  # pitches to remove
        mean_forearm_length = 26.595358306227542
        mean_upperarm_length = 27.23938265717186
        mean_hand_length = 14.9429087641141

    if pitcher == 'PP05':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9']  # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = [6,21,32,39,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]  # pitches to remove
        mean_forearm_length = 29.752215647994383
        mean_upperarm_length = 28.121066436944993
        mean_hand_length = 22.955661063883426

    if pitcher == 'PP06':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = [62,63,64,65,66,73,75]  # pitches to remove
        mean_forearm_length = 27.521338884279118
        mean_upperarm_length = 23.763250943167908
        mean_hand_length = 20.429070804532

    if pitcher == 'PP07':
        Innings = ['Inning_1', 'Inning_2', 'Inning_3', 'Inning_4', 'Inning_5',
                   'Inning_6']  # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = [45]  # pitches to remove
        mean_forearm_length = []
        mean_upperarm_length = []
        mean_hand_length = []

    if pitcher == 'PP08':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = [21,55,56]  # pitches to remove
        mean_forearm_length = 24.77073677467198
        mean_upperarm_length = 25.480910546536542
        mean_hand_length = 19.558310090270854

    if pitcher == 'PP09':
        Innings = []  # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = []  # pitches to remove
        mean_forearm_length = []
        mean_upperarm_length = []
        mean_hand_length = []

    if pitcher == 'PP10':
        Innings = []  # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = []  # pitches to remove
        mean_forearm_length = []
        mean_upperarm_length = []
        mean_hand_length = []

    if pitcher == 'PP11':
        Innings = []  # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = []  # pitches to remove
        mean_forearm_length = []
        mean_upperarm_length = []
        mean_hand_length = []

    if pitcher == 'PP12':
        Innings = ['Inning_1', 'Inning_2', 'Inning_3', 'Inning_4', 'Inning_5',
                   'Inning_6']  # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = [60]  # pitches to remove
        mean_forearm_length = 28.56499792514336
        mean_upperarm_length = 26.624701431754023
        mean_hand_length = 21.201363791024246

    if pitcher == 'PP13':
        Innings = []  # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = []  # pitches to remove
        mean_forearm_length = []
        mean_upperarm_length = []
        mean_hand_length = []

    if pitcher == 'PP14':
        Innings = ['Inning_1', 'Inning_2','Inning_5','Inning_6',
                   'Inning_7','Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = [2,3,4,5,6,7,8,9,10,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,46,63,67,71,76]  # pitches to remove
        mean_forearm_length = 25.57505164468459
        mean_upperarm_length = 27.10506628400707
        mean_hand_length = 18.1430265386685

    if pitcher == 'PP15':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9','Inning_10','Inning_11']  # Inning where you want to look, for pitches gives all pitches in inning
        problem_pitches = [7,9,18,21,22,27,38,41,53,79,87,99,100,102]  # pitches to remove
        mean_forearm_length = 27.54292653447997
        mean_upperarm_length = 28.69856917848192
        mean_hand_length = 21.989473106624217

    Cumulative_storage = dict.fromkeys(Segments)
    for segment in Segments:
        Cumulative_storage[segment] = dict.fromkeys(EMG_markers)
        for EMG_marker in EMG_markers:
            Cumulative_storage[segment][EMG_marker] = dict()

    for inning in Innings:
        # --- Reset EMG_downsampled --- #
        EMG_downsampled = dict.fromkeys(Segments)
        for segment in Segments:
            EMG_downsampled[segment] = dict.fromkeys(EMG_markers)
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
        path = "data/EMG_Data/01 Raw Data/"+pitcher+"/EMG_Cut/"
        j = 10*int(inning[7:])
        filename = pitcher + "_EMGCut_"+str(j)

        # --- Load data from pickle --- #
        filenameIn = path + filename
        infile = open(filenameIn, 'rb')
        EMG_data = pickle.load(infile)
        infile.close()

        """
        Downsample EMG data to new dictionary
        """

        for segment in Outputs:
            for EMG_marker in EMG_markers:
                EMG_downsampled[segment][EMG_marker] = sp.resample(EMG_data[EMG_marker], int(len(EMG_data[EMG_marker]) * fs_scaling))
                Outputs[segment][EMG_marker] = dict()
                peaks = sp.find_peaks((-EMG_downsampled[segment]['ACC']),height=(np.nanmax((-EMG_downsampled[segment]['ACC'])) * .4), distance=800)
                peaks = list(peaks[0])


        """
        remove bad peaks
        """

        if len(peaks) > 10:
            print(len(peaks))
            print(peaks)
            remove_peaks = []
            distance = []
            plt.figure()
            plt.plot((-EMG_downsampled[segment]['ACC']))
            # Use ginput to manually select cut points
            tuples = plt.ginput(15, -1, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
            for i in range(len(tuples)):
                remove_peaks.append(np.round(tuples[i][0]))
            for remove in remove_peaks:
                distance = [abs(remove - peak) for peak in peaks]
                peaks.remove(peaks[np.argmin(distance)])

        for segment in Outputs:
            for EMG_marker in EMG_markers:
                for pitch in Outputs[segment]['max_abduction_moment']:
                    i = int(pitch[-1]) -1
                    if i == -1:
                        i = 9
                    Outputs[segment][EMG_marker][pitch] = EMG_downsampled[segment][EMG_marker][int(peaks[i] - lead * fs_opti):int(peaks[i] + lag * fs_opti)]
                    Cumulative_storage[segment][EMG_marker][pitch] = Outputs[segment][EMG_marker][pitch]


        # --- Define path where pitch output data is stored --- #
        path = "Results/Pitches/"+filter_state+"/"+pitcher+"/"+inning+"/"
        filename = "Outputs"

        # --- Save data --- #
        filenameIn = path + filename
        outfile = open(filenameIn, 'wb')
        # Write the dictionary into the binary file
        pickle.dump(Outputs, outfile)
        outfile.close()
        print('EMG data has been saved.')
        infile.close()


    # --- Define path where Results are stored --- #
    Last_inning = Innings[-1]
    path = 'Results/Pitches/' + filter_state +'/' + pitcher + '/' + Last_inning + '/'
    filename = "Cumulative_til_this_point"

    # --- Load data from pickle --- #
    filenameIn = path + filename
    infile = open(filenameIn, 'rb')
    cumulative_data = pickle.load(infile)
    infile.close()

    for segment in cumulative_data:
        cumulative_data[segment].update(Cumulative_storage[segment])

    # --- Save data --- #
    outfile = open(filenameIn, 'wb')
    # Write the dictionary into the binary file
    pickle.dump(cumulative_data, outfile)
    outfile.close()
    print('Cumulative EMG data has been saved.')
    infile.close()