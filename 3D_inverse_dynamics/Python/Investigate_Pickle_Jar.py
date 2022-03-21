"""Import modules"""
import pickle

"""Investigate_pickle_jar is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
all this file does is give me a way to look at pickle data without running a ton of code
Contact E-Mail: a.j.r.leenen@vu.nl
​
Version 1.5 (2021-05-18)"""


# --- Define path where data is stored --- #
path = "E:/MSc/Thesis/Breaking the Higher Load/3D_inverse_dynamics/Python/data/Pitches/Unfiltered/PP03/"
filename = "Inning_1"

# --- Load data from pickle --- #
filenameIn = path + filename
infile = open(filenameIn, 'rb')
data = pickle.load(infile)
infile.close()