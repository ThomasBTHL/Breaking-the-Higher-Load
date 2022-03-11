import functions
import pandas as pd
from tqdm import tqdm
from time import sleep
import xlrd
import c3d
import os
import re
from functions import *
import copy
from tkinter import Tk
from tkinter.filedialog import askdirectory
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

"""
This program is part of the preprocessing motion capture data.
1) Visual check markers you want to analyze with the functions visual_check_markers
2) Correct for switching markers using ginput an plots
3) Save the unfiltered corrected data in binary form
4) Use a filter to get rid of the peaks in the data, as an effect of different camera's that capture the marker. 
    We use a savgol filter for this
5) Save the filtered corrected data in binary form

In this function you can use three plot functions, these can be found in functions.py:
1) visual_check_markers('RUS', 'RRS', 'RMHE', 'RLHE', markers)
2)  visual_check_smoothing_effect(markerName, coordinateName, markers, markersNew)
3)  visual_check_markers_switching('LASIS', 'RASIS', 'LPSIS', 'RPSIS', markersNew, change=False)

Preprocessing_data.py is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam 
& Bart van Trigt, PhD-candidate TU Delft
Contact E-Mail: a.j.r.leenen@vu.nl; b.vantrigt@tudelft.nl

Logbook:
Version 1.0 (2020-05-19)
Version 2.0 (2020-07-14) Added the save file and make it interactive for plotting using matplotlib.use('macosx')
Version 3.0 (2020-09-04) Added the save with pickle
"""

# Define the path to load the data
path = os.path.abspath(os.path.join("data/Optitrack_Data/Preprocessed_data/pp01_c3d/"))
## Load the c3d files to xlsx
markers_c3d = load_c3d(path)
markers = markers_c3d['PITCH_0']

""" 
1) Visual check marker data
"""
print('1) do you want the check visual marker data?')
yes = {'yes', 'y', 'ye', ''}
no = {'no', 'n'}

choice = input().lower()
if choice in yes:
    # Check right Forearm and for switching
    visual_check_markers('VU_Baseball_R_RUS', 'VU_Baseball_R_RRS', 'VU_Baseball_R_RMHE', 'VU_Baseball_R_RLHE', markers)
    #visual_check_markers_switching('VU_Baseball_R_RUS', 'VU_Baseball_R_RRS', 'VU_Baseball_R_RMHE', 'VU_Baseball_R_RLHE', markers,title='ForeArm')

    # Check right shoulder and for switching
    visual_check_markers('VU_Baseball_R_RAC', [], [], [], markers)
    #visual_check_markers_switching('VU_Baseball_R_RAC', [], [], [], markers, title='Shoulder')

    # Check Trunk markers
    visual_check_markers('VU_Baseball_R_IJ', 'VU_Baseball_R_PX', 'VU_Baseball_R_C7', 'VU_Baseball_R_T8', markers)
    #visual_check_markers_switching('VU_Baseball_R_IJ', 'VU_Baseball_R_PX', 'VU_Baseball_R_C7', 'VU_Baseball_R_T8', markers,title='Trunk')

    # Check Pelvic markers
    visual_check_markers('VU_Baseball_R_LASIS', 'VU_Baseball_R_RASIS', 'VU_Baseball_R_LPSIS', 'VU_Baseball_R_RPSIS', markers)
    #visual_check_markers_switching('VU_Baseball_R_LASIS', 'VU_Baseball_R_RASIS', 'VU_Baseball_R_LPSIS', 'VU_Baseball_R_RPSIS', markers, title='Pelvic')
else:
    print('no markers will be plotted.')


"""
2) change switching markers
"""
markersNew = copy.deepcopy(markers)
#%% right  foreArm
print('2) do you want to change ForeArm marker data?')
yes = {'yes', 'y', 'ye', ''}
no = {'no', 'n'}
choice = input().lower()
if choice in yes:

    index, value, changeMarker1, changeMarker2 = visual_check_markers_switching('RUS', 'RRS', 'RMHE', 'RLHE', markersNew, change=True,title='ForeArm')

    for i in range(len(index) - 1):
        markersNew[changeMarker1].iloc[[int(index[i] - 1), int(index[i + 1] + 1)], :], markersNew[changeMarker2].iloc[
                                                                                       [int(index[i]), int(index[i + 1])],
                                                                                       :] = np.nan, np.nan
        markersNew[changeMarker1].iloc[int(index[i]):int(index[i + 1]), :], markersNew[changeMarker2].iloc[
                                                                            int(index[i]):int(index[i + 1]), :] = \
        markersNew[changeMarker2].iloc[int(index[i]):int(index[i + 1]), :].copy(), markersNew[changeMarker1].iloc[
                                                                                   int(index[i]):int(index[i + 1]),
                                                                                   :].copy()
        i = i + 1

#%%  right Shoulder
print('2)  do you want to change Shoulder marker data?')
yes = {'yes', 'y', 'ye', ''}
no = {'no', 'n'}
choice = input().lower()
if choice in yes:

    index, value, changeMarker1, changeMarker2 = visual_check_markers_switching('RAC1', 'RAC2', 'RAC3', [], markersNew, change=True,title='Shoulder')

    for i in range(len(index) - 1):
        markersNew[changeMarker1].iloc[[int(index[i] - 1), int(index[i + 1] + 1)], :], markersNew[changeMarker2].iloc[
                                                                                       [int(index[i]),
                                                                                        int(index[i + 1])],
                                                                                       :] = np.nan, np.nan
        markersNew[changeMarker1].iloc[int(index[i]):int(index[i + 1]), :], markersNew[changeMarker2].iloc[
                                                                            int(index[i]):int(index[i + 1]), :] = \
            markersNew[changeMarker2].iloc[int(index[i]):int(index[i + 1]), :].copy(), markersNew[changeMarker1].iloc[
                                                                                       int(index[i]):int(index[i + 1]),
                                                                                       :].copy()
        i = i + 1

#%% thorax
print('2) do you want to change Thorax marker data?')
yes = {'yes', 'y', 'ye', ''}
no = {'no', 'n'}
choice = input().lower()
if choice in yes:

    index, value, changeMarker1, changeMarker2 = visual_check_markers_switching('IJ', 'PX', 'C7', 'T8', markersNew, change=True,title='Thorax')

    for i in range(len(index) - 1):
        markersNew[changeMarker1].iloc[[int(index[i] - 1), int(index[i + 1] + 1)], :], markersNew[changeMarker2].iloc[
                                                                                       [int(index[i]),
                                                                                        int(index[i + 1])],
                                                                                       :] = np.nan, np.nan
        markersNew[changeMarker1].iloc[int(index[i]):int(index[i + 1]), :], markersNew[changeMarker2].iloc[
                                                                            int(index[i]):int(index[i + 1]), :] = \
            markersNew[changeMarker2].iloc[int(index[i]):int(index[i + 1]), :].copy(), markersNew[changeMarker1].iloc[
                                                                                       int(index[i]):int(index[i + 1]),
                                                                                       :].copy()
        i = i + 1

#%% Pelvic
print('3) do you want to change Pelvic marker data?')
yes = {'yes', 'y', 'ye', ''}
no = {'no', 'n'}
choice = input().lower()
if choice in yes:
    index, value, changeMarker1, changeMarker2 = visual_check_markers_switching('LASIS', 'RASIS', 'LPSIS', 'RPSIS',
                                                                                markersNew, change=True,title='Pelvis')

    for i in range(len(index) - 1):
        markersNew[changeMarker1].iloc[[int(index[i] - 1), int(index[i + 1] + 1)], :], markersNew[changeMarker2].iloc[
                                                                                       [int(index[i]), int(index[i + 1])],
                                                                                       :] = np.nan, np.nan
        markersNew[changeMarker1].iloc[int(index[i]):int(index[i + 1]), :], markersNew[changeMarker2].iloc[
                                                                            int(index[i]):int(index[i + 1]), :] = \
        markersNew[changeMarker2].iloc[int(index[i]):int(index[i + 1]), :].copy(), markersNew[changeMarker1].iloc[
                                                                                   int(index[i]):int(index[i + 1]),
                                                                                   :].copy()
        i = i + 1


"""
3) Save the unfiltered corrected data in binary form
"""
# Implement the markersNew back into the original data set
markers_unfiltered = markers_c3d.copy()
markers_unfiltered["PITCH_0"] = markersNew

# Path where the pickle will be saved. Last part will be the name of the file
# "data/Optitrack_Data/Preprocessed_data/pp01_c3d/"
filename = 'data/Unfiltered_Grouped/PP01/PP01_10_Pickle_unfiltered' # Change to automatic naming

# Initialize the pickle file
outfile = open(filename,'wb')

# Write the dictionary into the binary file
pickle.dump(markers_unfiltered,outfile)
outfile.close()

print('Unfiltered dictionary has been saved.')


"""
4) smoothing the signal
This part smooth the signal with the Savgol filter by the least squares method
With the function visual_check_smoothing_effect you can investigate its effect compared to the raw signal
"""
print('4) do you want to smooth the data?')
yes = {'yes', 'y', 'ye', ''}
no = {'no', 'n'}
choice = input().lower()
if choice in yes:
    markersFilter = copy.deepcopy(markersNew)
    from scipy.signal import savgol_filter

    window_length = 11
    poly_order = 3

    for key in markers.keys():
        markersFilter[key]['X'] = savgol_filter(markersNew[key]['X'], window_length, poly_order)
        markersFilter[key]['Y'] = savgol_filter(markersNew[key]['Y'], window_length, poly_order)
        markersFilter[key]['Z'] = savgol_filter(markersNew[key]['Z'], window_length, poly_order)

    markerName = 'RRS'
    coordinateName = 'Y'
    visual_check_smoothing_effect(markerName, coordinateName, markersNew, markersFilter)

    """
    5) Save the filtered corrected data in binary form 
    """
    # Implement the filtered markers into the original data set
    markers_filtered = markers_c3d.copy()
    markers_filtered["PITCH_0"] = markersFilter  # Gives error if choice == no for smoothing, move to within loop?

    # Path where the pickle will be saved. Last part will be the name of the file
    filename = 'data/Pitch/PP01_Pickle_filtered'

    # Initialize the pickle file
    outfile = open(filename, 'wb')

    # Write the dictionary into the binary file
    pickle.dump(markers_c3d, outfile)
    outfile.close()

    print('Filtered dictionary has been saved.')
else:
    print('the data has not been smoothed')

"""
6) Cut data
"""
print('1) Do you want to cut the data?')
yes = {'yes', 'y', 'ye', ''}
no = {'no', 'n'}

choice = input().lower()
if choice in yes:
    ball_pickups = ball_pickup_indexs('VU_Baseball_R_RUS', 'VU_Baseball_R_RRS', 'VU_Baseball_R_RMHE', 'VU_Baseball_R_RLHE', markers)
    print(ball_pickups)
else:
    print('Data will not be cut')