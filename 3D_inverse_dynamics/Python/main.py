"""Import modules"""
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

pitcher = 'PP06'
method = 'No tape'
pitch = 'Pitch_03'

# Selection based on right or left-handed pitchers
if pitcher == 'PP03' or 'PP11':
    side = 'left'
else:
    side = 'right'

# Path where the pitching dictionary is saved
filename = 'data' + '/' + 'preprocessed' + '/' + pitcher + '/' + method + '/' + pitch + '_Pickle'

# Read the dictionary as a new variable
infile = open(filename, 'rb')
marker_data = pickle.load(infile)
infile.close()

# Define path to load the data
# path = os.path.abspath(os.path.join(""))

# Read and load c3d data from path
# marker_data = f.load_c3d(path)

# Subdivide dictionary into separate variables
pitch = marker_data['PITCH_0']

# Interpolate the marker data
for keys in pitch:
    pitch[keys] = pitch[keys].interpolate(method='polynomial', order=2)

# Calculate the segment parameters

# --- Pelvis Segment --- #
start = time.time()
pelvis_motion = f.calc_pelvis(pitch['RASIS'], pitch['LASIS'], pitch['RPSIS'], pitch['LPSIS'], gender='male')
end = time.time()
print('Thorax segment parameters calculated in ' + str(end - start) + ' seconds!\n')

# --- Thorax Segment --- #
start = time.time()
thorax_motion = f.calc_thorax(pitch['IJ'], pitch['PX'], pitch['C7'], pitch['T8'], gender='male')
end = time.time()
print('Thorax segment parameters calculated in ' + str(end - start) + ' seconds!\n')

# --- Upperarm Segment --- #
start = time.time()
upperarm_motion = f.calc_upperarm(pitch['RLHE'], pitch['RMHE'], pitch['RAC1'], side, gender='male')
end = time.time()
print('Upperarm segment parameters calculated in ' + str(end - start) + ' seconds!\n')

# --- Forearm Segment --- #
start = time.time()
forearm_motion = f.calc_forearm(pitch['RLHE'], pitch['RMHE'], pitch['RUS'], pitch['RRS'], side, gender='male')
end = time.time()
print('Forearm segment parameters calculated in ' + str(end - start) + ' seconds!\n')

# --- Hand Segment --- #
start = time.time()
hand_motion = f.calc_hand(pitch['RUS'], pitch['RRS'], pitch['RHIP3'], side, gender='male')
end = time.time()
print('Hand segment parameters calculated in ' + str(end - start) + ' seconds!\n')

# Combine all the referenced segment dictionaries into dictionary in order to loop through the keys for net force and moment calculations
model = f.segments2combine(upperarm_motion, forearm_motion, hand_motion)
print('Segment parameters combined in one dictionary: model\n')

# Rearrange model to have the correct order of segments for 'top-down' method
model = f.rearrange_model(model, 'top-down')

# Calculate the net forces according the newton-euler method
F_joint = f.calc_net_reaction_force(model)
print('Segment forces are calculated and combined in one dictionary: F_joint\n')

# Calculate the net moments according the newton-euler method
M_joint = f.calc_net_reaction_moment(model, F_joint)
print('Segment moments are calculated and combined in one dictionary: M_joint\n')

# Project the calculated net moments according the newton-euler method to local coordination system to be anatomically meaningful
joints = {'hand': 'wrist', 'forearm': 'elbow', 'upperarm': 'shoulder'}  # Joints used to calculate the net moments according the newton-euler method

# Initialise parameters
seg_M_joint = dict()
for segment in model:
    seg_M_joint[segment] = f.moments2segment(model[segment]['gRseg'], M_joint[segment]['M_proximal'])

    if side == 'left':
        seg_M_joint[segment][0:2, :] = -seg_M_joint[segment][0:2, :]

    print('Global net moments of the ' + joints[segment] + ' are projected on the local coordination system of the ' + model[segment]['seg_name'] + '\n')

# Calculate Euler angles
# elbow_angles = R.from_matrix(R_elbow).as_euler('zxy', degrees=True)

print('Main Finished')

# Visualisation of the global and local net moments
plt.figure(1)
plt.subplot(211)
plt.plot(M_joint['forearm']['M_proximal'][0, :], label='Add(+)/Abd(-')
plt.plot(M_joint['forearm']['M_proximal'][1, :], label='Pro(+)/Sup(-)')
plt.plot(M_joint['forearm']['M_proximal'][2, :], label='Flex(+)/Ext(-)')
# plt.plot(normMoment, label='Norm')
plt.title('Moments Expressed in Global Coordination System')
plt.xlabel('Samples')
plt.ylabel('Moment [Nm]')
plt.legend()

plt.subplot(212)
plt.plot(seg_M_joint['forearm'][0, :], label='Add(+)/Abd(-)')
plt.plot(seg_M_joint['forearm'][1, :], label='Pro(+)/Sup(-)')
plt.plot(seg_M_joint['forearm'][2, :], label='Flex(+)/Ext(-)')
# plt.plot(normMoment, label='Norm')
plt.title('Moments Projected on Forearm Coordination System')
plt.xlabel('Samples')
plt.ylabel('Moment [Nm]')
plt.legend()
plt.show()




