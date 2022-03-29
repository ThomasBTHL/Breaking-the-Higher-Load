"""Import modules"""
import pickle
from functions import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import pandas as pd

"""Investigate_pickle_jar is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
all this file does is give me a way to look at pickle data without running a ton of code
Contact E-Mail: a.j.r.leenen@vu.nl
​
Version 1.5 (2021-05-18)"""


# --- Define path where data is stored --- #
path = "E:/MSc/Thesis/Breaking the Higher Load/3D_inverse_dynamics/Python/data/EMG_Data/01 Raw Data/PP03/EMG_Cut/"
filename = "PP03_EMGCut_10"

# --- Load data from pickle --- #
filenameIn = path + filename
infile = open(filenameIn, 'rb')
data = pickle.load(infile)
infile.close()

print(data.keys())

plt.figure()
plt.subplot(2,1,1)
plt.plot(data['time_s'],data['ACC'])

plt.subplot(2,1,2)
plt.plot(data['time_s'],data['FMP'])

plt.show()