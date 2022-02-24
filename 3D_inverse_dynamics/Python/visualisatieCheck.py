"""Import modules"""
import functions as f
import pickle
import os
import pandas as pd
import time
import copy
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

"""Visualisation is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
Contact E-Mail: a.j.r.leenen@vu.nl
​
Version 1.5 (2021-05-18)"""

# --- Select pitcher (and corresponding side) to be evaluated --- #
pitcher = 'PP04'
side = 'right'  # Options: 'right' or 'left'
method = 'Tape'  # Options: 'no tape' or 'tape'

# --- Define path where data is stored --- #
path = "//Users/bvantrigt/TU Delft/Shared/Python Inverse Dynamic Model/Bart/3d_inverse_dynamica/data/pitcher_information/" + method + "/"
filename = pitcher + "_info.pickle"

# --- Load data from pickle --- #
filenameIn = path + filename
infile = open(filenameIn, 'rb')
data = pickle.load(infile)
angularVelocityTimeSerie = data[2]
infile.close()

# --- Initialise pitch_number --- #
pitch_number = []
# --- Setup loop to evaluate all the data --- #
for i in range(1,26):
    if i < 10:
        # --- Select pitch --- #
        pitch_number = 'Pitch_0' + str(i)

        # --Plot Pelvis en Trunk
        plt.figure(1)
        plt.plot(angularVelocityTimeSerie['pelvis'][pitch_number], label='Pelvis', linewidth=1)
        plt.plot(angularVelocityTimeSerie['thorax'][pitch_number], label='Thorax', linewidth=1)
        # --- Limit the y-axis to pre-specified range --- #
        # plt.ylim(-20, 80)

        # --- Plot information --- #
        plt.title('Norm Angular Velocities \n' + pitch_number)
        plt.xlabel('Samples')
        plt.ylabel('Moment [Nm]')
        plt.legend()
        # --- Continue with the next plot when user pressed a key on the keyboard --- #
        plt.waitforbuttonpress()
        # --- Clear axes for next plot --- #
        plt.clf()
    else:
        # --- Select pitch --- #
        pitch_number = 'Pitch_' + str(i)

        # --Plot Pelvis en Trunk
        plt.figure(1)
        plt.plot(angularVelocityTimeSerie['pelvis'][pitch_number], label='Pelvis', linewidth=1)
        plt.plot(angularVelocityTimeSerie['thorax'][pitch_number], label='Pelvis', linewidth=1)
        # --- Limit the y-axis to pre-specified range --- #
        # plt.ylim(-20, 80)

        # --- Plot information --- #
        plt.title('Norm Angular Velocities \n' + pitch_number )
        plt.xlabel('Samples')
        plt.ylabel('deg/s')
        plt.legend()
        # --- Continue with the next plot when user pressed a key on the keyboard --- #
        plt.waitforbuttonpress()
        # --- Clear axes for next plot --- #
        plt.clf()
