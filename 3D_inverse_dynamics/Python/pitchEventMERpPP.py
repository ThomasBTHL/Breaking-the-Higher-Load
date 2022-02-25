"""Import modules"""
import functions as f
import pickle
import pandas as pd
import numpy as np
import copy
from numpy import linalg as LNG
import matplotlib.pyplot as plt


# Ask which pitchers and which methods one wants to use
print('Which pitcher do you want to evaluate? Give number only, in 2 digits')
pitcher = "PP" + str(input())

method = 'No tape'
# Make a list of all desired pitches
Npitches = list()
for i in range(25):
    i = i+1
    if i < 10:
        Npitches.append('Pitch_0'+str(i))
    else:
        Npitches.append('Pitch_' + str(i))

# Initialize list
MER = list()

for i in range(len(Npitches)):
    Npitch = Npitches[i]
    # Path where the pitching dictionary is saved
    filename = 'data/Preprocessed/' + pitcher + '/' + method + '/' + Npitch + '_Pickle'

    # Read the dictionary as a new variable
    infile = open(filename, 'rb')
    marker_data = pickle.load(infile)
    infile.close()

    # Subdivide dictionary into separate variables
    pitch = marker_data['PITCH_0']

    # Determine rotation matrix of the upperarm
    upperarm_motion = f.calc_upperarm(pitch['RLHE'], pitch['RMHE'], pitch['RAC1'], gender='male')
    R_upperarm = upperarm_motion['gRseg']

    # Determine rotation matrix of the thorax
    thorax_motion = f.calc_thorax(pitch['IJ'], pitch['PX'], pitch['C7'], pitch['T8'], gender='male')
    R_thorax = thorax_motion['gRseg']

    # Euler angles humerus relative to the thorax
    GHeuler = f.euler_angles('yxy',R_upperarm,R_thorax)
    GH = GHeuler.to_numpy()     # change from pandas dataframe to numpy array
    SER = GH[:,5]   # Select the rotation of the humerus relative to the thorax in the z-direction

    MER.append(np.nanmax(SER))

# make a graph of the MER
plt.figure()
plt.plot(MER)