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
pitchers = ['PP01','PP02','PP03','PP04','PP05','PP06','PP07','PP08','PP12','PP14','PP15'] #PP01 - PP15\
length = 'Innings' # Pitches or Innings
filter_state = 'unFiltered' # Unfiltered or Filtered
Cumulative_inning_state = False

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
                   'Inning_6','Inning_7']  # Inning where you want to look, for pitches gives all pitches in inning
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
        Innings = ['Inning_1', 'Inning_2', 'Inning_5',
                   'Inning_6','Inning_7','Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning
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

    """
    Same for all pitchers
    """
    fs = 120
    # Selection based on right or left-handed pitchers
    if pitcher == ('PP09' or 'PP10' or 'PP11' or 'PP13'):

        side = 'left'
    else:
        side = 'right'

    for Inning in Innings:
        """
        Load inning data and remove unwanted pitches
        """
        # Path where the pitching dictionary is saved
        filename = 'data' + '/' + length + '/' + 'Unfiltered' + '/' + pitcher + '/' + Inning

        # Read the dictionary as a new variable
        infile = open(filename, 'rb')
        inning_data_raw = pickle.load(infile)
        infile.close()

        inning_data = dict()
        inning_data['whole inning'] = inning_data_raw


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

            # Save model as pickle
            if Cumulative_inning_state == False:
                # Path where the pickle will be saved. Last part will be the name of the file
                filename = 'Models' + '/' + length + '/' + filter_state + '/' + pitcher + '/' + Inning + '/' + pitch_number
                # Initialize the pickle file
                outfile = open(filename, 'wb')
                # Write the dictionary into the binary file
                pickle.dump(model, outfile)
                outfile.close()
                print('model has been saved.')