"""Import modules"""
import copy
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import functions as f
import pickle
import pandas as pd
from tqdm import tqdm
import time
import xlrd
import c3d
import os
import re

"""3D inverse dynamic model is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
Contact E-Mail: a.j.r.leenen@vu.nl
"" Changes made by Thomas van Hogerwou, Master student TU Delft: Thom.hogerwou@gmail.com

Version 1.5 (2020-07-15)"""

"""
Input area
"""
length = 'Pitches' # Pitches or Innings
filter_state = 'Unfiltered' # Unfiltered or Filtered
pitcher = 'PP04' #PP01 - PP15
Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8'] # Inning where you want to look, for pitches gives all pitches in inning

fs = 120
problem_pitches = [11,14,15,16,70,72,78] # pitches to remove

# Selection based on right or left-handed pitchers
if pitcher == ('PP09' or 'PP10' or 'PP11' or 'PP13'):

    side = 'left'
else:
    side = 'right'

forearm_length = []
upperarm_length = []
hand_length = []

mean_forearm_length = 26.595358306227542
mean_upperarm_length = 27.23938265717186
mean_hand_length = 14.9429087641141

for Inning in Innings:

    """
    Output setup
    """
    Fatigue_dictionary = {}

    segments = ['hand', 'forearm', 'upperarm']
    outputs = ['max_norm_moment', 'max_abduction_moment']
    for segment in segments:
        Fatigue_dictionary[segment] = {}
        for output in outputs:
            Fatigue_dictionary[segment][output] = {}

    """
    Inning setup
    """
    j = 0 # used for time sync of inning data
    Inning_MER_events = []
    Inning_cross_corr_events = []
    Inning_max_normM_events = []
    Inning_max_M_events = []
    Inning_seg_M_joint = dict()
    Inning_F_joint = dict()

    """
    Load inning data and remove unwanted pitches
    """
    # Path where the pitching dictionary is saved
    filename = 'data' + '/' + length + '/' + filter_state + '/' + pitcher + '/' + Inning

    # Read the dictionary as a new variable
    infile = open(filename, 'rb')
    inning_data_raw = pickle.load(infile)
    infile.close()

    # remove unwanted problem pitches
    pitches_to_remove = []
    for i in range(len(problem_pitches)):
        pitches_to_remove.append("pitch_{0}".format(problem_pitches[i]))

    if len(inning_data_raw) == 14:
        inning_data = dict()
        inning_data['whole inning'] = inning_data_raw
    else:
        inning_data = copy.deepcopy(inning_data_raw)
        for pitch in inning_data_raw:
            if pitch in pitches_to_remove:
                inning_data.pop(pitch)

    for pitch_number in inning_data:
        # Subdivide dictionary into separate variables
        pitch = inning_data[pitch_number]

        # Calculate the segment parameters
        # --- Pelvis Segment --- #
        pelvis_motion = f.calc_pelvis(pitch['VU_Baseball_R_RASIS'], pitch['VU_Baseball_R_LASIS'], pitch['VU_Baseball_R_RPSIS'], pitch['VU_Baseball_R_LPSIS'], gender='male',sample_freq=fs)

        # --- Thorax Segment --- #
        thorax_motion = f.calc_thorax(pitch['VU_Baseball_R_IJ'], pitch['VU_Baseball_R_PX'], pitch['VU_Baseball_R_C7'], pitch['VU_Baseball_R_T8'], gender='male',sample_freq=fs)

        # --- Upperarm Segment --- #
        upperarm_motion = f.calc_upperarm(pitch['VU_Baseball_R_RLHE'], pitch['VU_Baseball_R_RMHE'], pitch['VU_Baseball_R_RAC'], side, gender='male',sample_freq=fs, mean_seg_length = mean_upperarm_length)

        # --- Forearm Segment --- #
        forearm_motion = f.calc_forearm(pitch['VU_Baseball_R_RLHE'], pitch['VU_Baseball_R_RMHE'], pitch['VU_Baseball_R_RUS'], pitch['VU_Baseball_R_RRS'], side, gender='male',sample_freq=fs, mean_seg_length= mean_forearm_length)

        # --- Hand Segment --- #
        hand_motion = f.calc_hand(pitch['VU_Baseball_R_RUS'], pitch['VU_Baseball_R_RRS'], pitch['VU_Baseball_R_RHIP3'], side, gender='male',sample_freq=fs, mean_seg_length = mean_hand_length)

        # Combine all the referenced segment dictionaries into dictionary in order to loop through the keys for net force and moment calculations
        model = f.segments2combine(pelvis_motion, thorax_motion, upperarm_motion, forearm_motion, hand_motion)

        # Rearrange model to have the correct order of segments for 'top-down' method
        model = f.rearrange_model(model, 'top-down')

        hand_length.append(model['hand']['seg_length'])
        forearm_length.append(model['forearm']['seg_length'])
        upperarm_length.append(model['upperarm']['seg_length'])

        if (j == 0):
            model_1 = copy.deepcopy(model)
            j = 1

        # Save model as pickle
        # Path where the pickle will be saved. Last part will be the name of the file
        filename = 'Models' + '/' + length + '/' + filter_state + '/' + pitcher + '/' + Inning + '/' + pitch_number
        # Initialize the pickle file
        outfile = open(filename, 'wb')
        # Write the dictionary into the binary file
        pickle.dump(model, outfile)
        outfile.close()
        print('model has been saved.')

        # Calculate the net forces according the newton-euler method
        F_joint = f.calc_net_reaction_force(model)

        # Calculate the net moments according the newton-euler method
        M_joint = f.calc_net_reaction_moment(model, F_joint)

        if (np.isnan(np.nanmean(M_joint['hand']['M_proximal'])) == False) and (
                np.isnan(np.nanmean(M_joint['forearm']['M_proximal'])) == False) and (
                np.isnan(np.nanmean(M_joint['upperarm']['M_proximal'])) == False) and (
                np.isnan(np.nanmean(M_joint['thorax']['M_proximal'])) == False) and (
                np.isnan(np.nanmean(M_joint['pelvis']['M_proximal'])) == False):

            # Project the calculated net moments according the newton-euler method to local coordination system to be anatomically meaningful
            joints = {'hand': 'wrist', 'forearm': 'elbow', 'upperarm': 'shoulder', 'thorax': 'spine', 'pelvis': 'hip'}  # Joints used to calculate the net moments according the newton-euler method

            # Initialise parameters
            seg_M_joint = dict()
            for segment in model:
                seg_M_joint[segment] = f.moments2segment(model[segment]['gRseg'], M_joint[segment]['M_proximal'])

            if side == 'left':
                seg_M_joint[segment][0:2, :] = -seg_M_joint[segment][0:2, :]

            """
            Time syncing methods
            """
            # Determine MER index
            [pitch_MER, pitch_index_MER] = f.MER_event(model)
            Inning_MER_events.append(pitch_index_MER)
            # Determine cross correlation index
            cross_corr_s , cross_corr_index = f.Cross_correlation_sync_event(model_1, model)
            Inning_cross_corr_events.append(cross_corr_index)

            # Determine norm max moment index
            max_normM_index = np.nanargmax([np.linalg.norm(seg_M_joint['forearm'][:, index]) for index in range(len(seg_M_joint['forearm'][0,:]))])
            Inning_max_normM_events.append(max_normM_index)

            # Determine max abduction moment [0] correlation index
            max_M_index = np.nanargmax(seg_M_joint['forearm'][0,:])
            Inning_max_M_events.append(max_M_index)

            # Select which delay method to use for making variability graphs <--- Choose here!!!
            # sync_type = "MER"
            # delay = pitch_index_MER - Inning_MER_events[0] # MER
            # sync_type = "Corr"
            # delay = cross_corr_index # cross corr
            # sync_type = "norm Moment"
            # delay = max_normM_index - Inning_max_normM_events[0] # max norm moment
            sync_type = "abduction Moment"
            delay = max_M_index - Inning_max_M_events[0] # max moment

            # Visualisation of the global and local net moments
            synced_seg_M_joint = f.time_sync_moment_data(seg_M_joint, delay)
            Inning_seg_M_joint[pitch_number] = synced_seg_M_joint

            synced_seg_F_joint = f.time_sync_force_data(F_joint, delay)
            Inning_F_joint[pitch_number] = synced_seg_F_joint

            """
            Post proccessing of results
            """

            # Visualisation of the global and local net moments
            f.plot_inning_segment_moments(synced_seg_M_joint,pitch_number,figure_number = 2)

            # Max moment data for fatigue study
            for segment in segments:
                Fatigue_dictionary[segment]['max_norm_moment'][pitch_number] = np.nanmax([np.linalg.norm(seg_M_joint[segment][:, index]) for index in range(len(seg_M_joint[segment][0, :]))])
                Fatigue_dictionary[segment]['max_abduction_moment'][pitch_number] = np.nanmax([(seg_M_joint[segment][0, index]) for index in range(len(seg_M_joint[segment][0, :]))])

            """
            Save the max moment data to results folder
            """
            # Path where the pickle will be saved. Last part will be the name of the file
            filename = 'Results/Pitches/Unfiltered/' + pitcher + '/' + Inning + '/' + 'Max_norm_moments'
            # Initialize the pickle file
            outfile = open(filename, 'wb')
            # Write the dictionary into the binary file
            pickle.dump(Fatigue_dictionary, outfile)
            outfile.close()
            print('Fatigue dictionary has been saved.')

    """
    Making pretty variability plots
    """
    # Inning_mean_seg_M_joint, Inning_var_seg_M_joint, Inning_mean_pos_var_seg_M_joint, Inning_mean_neg_var_seg_M_joint = f.calc_variability_seg_M_joint(Inning_seg_M_joint)
    # time = np.linspace(0, len(Inning_mean_neg_var_seg_M_joint['forearm'][0, :]) / 120, len(Inning_mean_neg_var_seg_M_joint['forearm'][0, :]))
    # f.plot_inning_mean_moments(time, Inning_mean_seg_M_joint, Inning_mean_pos_var_seg_M_joint, Inning_mean_neg_var_seg_M_joint, figure_number=1)

    """
    Checking different sync methods
    """
    # print('MER')
    # print(Inning_MER_events)
    # print('Cross correlation')
    # print(Inning_cross_corr_events)
    # print('norm M')
    # print(Inning_max_normM_events)
    # print('M')
    # print(Inning_max_M_events)
    plt.show()
"""
Interpreting pitcher data
"""
# for segment in segments:
#     ser_segment = pd.Series(data = Fatigue_dictionary[segment]['max_abduction_moment'], index = Fatigue_dictionary[segment]['max_abduction_moment'].keys())
#     rolling_var = ser_segment.rolling(10).var()
#
#     plt.figure()
#     plt.subplot(2,1,1)
#     plt.plot(ser_segment)
#
#     plt.subplot(2,1,2)
#     plt.plot(rolling_var)
#
#     plt.show()
mean_hand_length = np.nanmean(hand_length)
print('hand length is')
print(mean_hand_length)

mean_forearm_length = np.nanmean(forearm_length)
print('forearm length is')
print(mean_forearm_length)

mean_upperarm_length = np.nanmean(upperarm_length)
print('upperarm length is')
print(mean_upperarm_length)

plt.figure()

plt.subplot(3,1,1)
plt.plot(hand_length)

plt.subplot(3,1,2)
plt.plot(forearm_length)

plt.subplot(3,1,3)
plt.plot(upperarm_length)
plt.show()