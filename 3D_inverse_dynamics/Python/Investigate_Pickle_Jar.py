"""Import modules"""
import pickle
import functions as f
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import pandas as pd

"""Investigate_pickle_jar is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
all this file does is give me a way to look at pickle data without running a ton of code
Contact E-Mail: a.j.r.leenen@vu.nl
​
Version 1.5 (2021-05-18)"""


# --- Define path where EMG data is stored --- #
path = "E:/MSc/Thesis/Breaking the Higher Load/3D_inverse_dynamics/Python/data/previous work/"
filename = "pitch_25_pp_2.pickle"

# --- Load data from pickle --- #
filenameIn = path + filename
infile = open(filenameIn, 'rb')
model = pickle.load(infile)
infile.close()

# Quick thing for Ton
elbow_angles = f.euler_angles('yxz', model['forearm']['gRseg'], model['upperarm']['gRseg'])
plt.figure()
plt.plot(elbow_angles[0, :], label='y')
plt.plot(elbow_angles[1, :], label='x')
plt.plot(elbow_angles[2, :], label='z')
plt.legend()
plt.xlabel('samples')
plt.ylabel('Angles [degrees]')
plt.show()