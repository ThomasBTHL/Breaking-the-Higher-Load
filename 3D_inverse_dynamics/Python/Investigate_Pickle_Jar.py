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
path = "Results/Pitches/Unfiltered/PP03/Inning_11/"
filename = "Cumulative_til_this_point"

# --- Load data from pickle --- #
filenameIn = path + filename
infile = open(filenameIn, 'rb')
model = pickle.load(infile)
infile.close()