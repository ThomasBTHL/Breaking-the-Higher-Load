"""Import modules"""
import copy

import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy
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

filter_state = 'Unfiltered'
pitcher = 'PP08'
Inning = 'Inning_1'
fs = 120

# Selection based on right or left-handed pitchers
if pitcher == ('PP09' or 'PP10' or 'PP11' or 'PP13'):
    side = 'left'
else:
    side = 'right'

# Path where the pitching dictionary is saved
filename = 'data' + '/' + 'Pitches' + '/' + filter_state + '/' + pitcher + '/' + Inning

# Read the dictionary as a new variable
infile = open(filename, 'rb')
inning_data = pickle.load(infile)
infile.close()

Inning_MER_events = []
Inning_seg_M_joint = dict.fromkeys(inning_data.keys())
Inning_F_joint = dict.fromkeys(inning_data.keys())

for pitch_number in inning_data:
    # Subdivide dictionary into separate variables
    pitch = inning_data[pitch_number]

    # Calculate the segment parameters
    # --- Pelvis Segment --- #
    pelvis_motion = f.calc_pelvis(pitch['VU_Baseball_R_RASIS'], pitch['VU_Baseball_R_LASIS'], pitch['VU_Baseball_R_RPSIS'], pitch['VU_Baseball_R_LPSIS'], gender='male',sample_freq=fs)

    # --- Thorax Segment --- #
    thorax_motion = f.calc_thorax(pitch['VU_Baseball_R_IJ'], pitch['VU_Baseball_R_PX'], pitch['VU_Baseball_R_C7'], pitch['VU_Baseball_R_T8'], gender='male',sample_freq=fs)

    # --- Upperarm Segment --- #
    upperarm_motion = f.calc_upperarm(pitch['VU_Baseball_R_RLHE'], pitch['VU_Baseball_R_RMHE'], pitch['VU_Baseball_R_RAC'], side, gender='male',sample_freq=fs)

    # --- Forearm Segment --- #
    forearm_motion = f.calc_forearm(pitch['VU_Baseball_R_RLHE'], pitch['VU_Baseball_R_RMHE'], pitch['VU_Baseball_R_RUS'], pitch['VU_Baseball_R_RRS'], side, gender='male',sample_freq=fs)

    # --- Hand Segment --- #
    hand_motion = f.calc_hand(pitch['VU_Baseball_R_RUS'], pitch['VU_Baseball_R_RRS'], pitch['VU_Baseball_R_RHIP3'], side, gender='male',sample_freq=fs)

    # Combine all the referenced segment dictionaries into dictionary in order to loop through the keys for net force and moment calculations
    model = f.segments2combine(pelvis_motion, thorax_motion, upperarm_motion, forearm_motion, hand_motion)

    # Rearrange model to have the correct order of segments for 'top-down' method
    model = f.rearrange_model(model, 'top-down')

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

        if (np.isnan(np.nanmean(M_joint['hand']['M_proximal'])) == False) and (
                np.isnan(np.nanmean(M_joint['forearm']['M_proximal'])) == False) and (
                np.isnan(np.nanmean(M_joint['upperarm']['M_proximal'])) == False) and (
                np.isnan(np.nanmean(M_joint['thorax']['M_proximal'])) == False) and (
                np.isnan(np.nanmean(M_joint['pelvis']['M_proximal'])) == False):

            # Determine MER index
            [pitch_MER, pitch_index_MER] = f.MER_event(model)
            Inning_MER_events.append(pitch_index_MER)

            # Visualisation of the global and local net moments
            synced_seg_M_joint = f.time_sync_moment_data(seg_M_joint, pitch_index_MER - Inning_MER_events[0])
            Inning_seg_M_joint[pitch_number] = synced_seg_M_joint

            synced_seg_F_joint = f.time_sync_force_data(F_joint, pitch_index_MER - Inning_MER_events[0])
            Inning_F_joint[pitch_number] = synced_seg_F_joint

            # Visualisation of the global and local net moments
            f.plot_inning_segment_moments(synced_seg_M_joint,pitch_number,figure_number = 2)
#            f.plot_inning_segment_moments(seg_M_joint,pitch_number,figure_number = 1)

print(Inning_MER_events)

Inning_mean_seg_M_joint, Inning_var_seg_M_joint, Inning_mean_pos_var_seg_M_joint, Inning_mean_neg_var_seg_M_joint = f.calc_variability_seg_M_joint(Inning_seg_M_joint)

f.plot_inning_segment_moments(Inning_mean_seg_M_joint, 'mean', figure_number=3)
f.plot_inning_segment_moments(Inning_mean_pos_var_seg_M_joint, 'pos_var', figure_number=3)
f.plot_inning_segment_moments(Inning_mean_neg_var_seg_M_joint, 'neg_var', figure_number=3)

plt.show()