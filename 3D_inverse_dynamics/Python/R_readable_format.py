"""Import modules"""
import pandas as pd

from functions import *
import numpy as np
import pickle
import sys
import os

# --- Initialise dataframe to emerge peak moments --- #
peak_moments = pd.DataFrame(np.nan, index=np.array(range(550)), columns=['Participant_Number', 'Pitch_Number', 'Condition',
                                                                         'Elbow_Adduction_MER', 'Elbow_Flexion_MER', 'Elbow_Norm_MER', 'Shoulder_Norm_MER'])

# --- Create array with pitch number --- #
pitch_number = np.array(range(25)) + 1

# --- Initialisation parameters --- #
method = ['No tape', 'Tape']
label_condition = []

# --- Initialise dataframes used to append --- #
participant_rows = []
pitch_rows = []
labels_rows = []

elbow_adduction_rows = []
elbow_flexion_rows = []
elbow_rows = []
shoulder_rows = []

for condition in method:

    # --- Create array with dichotomous labeling variable --- #
    if condition == 'No tape':
        label_condition = np.zeros((25, 1))

    elif condition == 'Tape':
        label_condition = np.ones((25, 1))

    # --- Specify path or folder to read pickle files --- #
    path = '/Users/ajrleenen/PycharmProjects/3d_inverse_dynamica/data/moment_peaks/' + condition + '/'

    # --- Load pickles from path or folder --- #
    for filename in sorted_alphanumeric(os.listdir(path)):

        # --- Load the .csv file(s) in the path or folder only --- #
        if filename.endswith('MER.pickle'):

            # --- Create array with participant number --- #
            participant_number = np.full((25, 1), int(filename[2:4]))

            # --- Load data from pickle --- #
            filenameIn = path + filename
            infile = open(filenameIn, 'rb')
            data_moments = pickle.load(infile)
            infile.close()

            if filename.endswith('elbow_adduction_MER.pickle'):

                # --- Emerge all rows in one array --- #
                elbow_adduction_rows = np.append(elbow_adduction_rows, data_moments)

                # --- Emerge all rows in one array only 1-in-4 times --- #
                participant_rows = np.append(participant_rows, participant_number)
                pitch_rows = np.append(pitch_rows, pitch_number)
                labels_rows = np.append(labels_rows, label_condition)

            if filename.endswith('elbow_flexion_MER.pickle'):
                # --- Emerge all rows in one array --- #
                elbow_flexion_rows = np.append(elbow_flexion_rows, data_moments)

            if filename.endswith('elbow_MER.pickle'):
                # --- Emerge all rows in one array --- #
                elbow_rows = np.append(elbow_rows, data_moments)

            if filename.endswith('elbow_MER.pickle'):
                # --- Emerge all rows in one array --- #
                shoulder_rows = np.append(shoulder_rows, data_moments)

# --- Emerge all in one dataframe ready to save in csv.format --- #
peak_moments['Participant_Number'] = participant_rows
peak_moments['Pitch_Number'] = pitch_rows
peak_moments['Condition'] = labels_rows
peak_moments['Elbow_Adduction_MER'] = elbow_adduction_rows
peak_moments['Elbow_Flexion_MER'] = elbow_flexion_rows
peak_moments['Elbow_Norm_MER'] = elbow_rows
peak_moments['Shoulder_Norm_MER'] = shoulder_rows

# --- Add manually two extra pitches from participant 9 to the dataframe --- #
# peak_moments.loc[495, 'Elbow_Adduction_MER'] =
# peak_moments.loc[496, 'Elbow_Adduction_MER'] =

# peak_moments.loc[495, 'Elbow_Flexion_MER'] =
# peak_moments.loc[496, 'Elbow_Flexion_MER'] =

# peak_moments.loc[495, 'Elbow_Norm_MER'] =
# peak_moments.loc[496, 'Elbow_Norm_MER'] =

# peak_moments.loc[495, 'Shoulder_Norm_MER'] =
# peak_moments.loc[496, 'Shoulder_Norm_MER'] =

