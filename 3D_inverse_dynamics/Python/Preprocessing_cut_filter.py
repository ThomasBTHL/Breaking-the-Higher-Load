from functions import *
import copy
import matplotlib
import pickle
from scipy.signal import savgol_filter
"""
This program is part of the preprocessing motion capture data.
1) Determine ball_pickups and save into markers
2) Filter the markers and save as markersFilter
3) Save all 4 configurations of filtered / cut

In this function you can use three plot functions, these can be found in functions.py:
1) visual_check_markers('RUS', 'RRS', 'RMHE', 'RLHE', markers)
2) visual_check_smoothing_effect(markerName, coordinateName, markers, markersNew)

Preprocessing_data.py is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam 
& Bart van Trigt, PhD-candidate TU Delft  & Thomas van Hogerwou, master student TU-Delft
Contact E-Mail: a.j.r.leenen@vu.nl; b.vantrigt@tudelft.nl; t.c.vanhogerwou@student.tudelft.nl

Logbook:
Version 1.0 (2022-03-18)
"""

subject_name = "PP03"

# Filter parameters
window_length = 11
poly_order = 3

# Define the path to load the data
path = os.path.abspath(os.path.join("data/Optitrack_Data/Preprocessed_data/" + subject_name + "_c3d/"))
## Load the c3d files to xlsx
markers_c3d = load_c3d_innings(path)

for Innings in markers_c3d:
    inning_name = Innings

    markers = markers_c3d[Innings]

    """
    1) Determine ball pickups
    """

    ball_pickups = ball_pickup_indexs('VU_Baseball_R_RUS', 'VU_Baseball_R_RRS', 'VU_Baseball_R_RMHE', 'VU_Baseball_R_RLHE', markers)
    markers_cut_unfiltered = cut_markers(markers,ball_pickups)  # Outputs dictionary of dictionaries contatining individual pitches, length = ballpickups + 1

    """
    2) smoothing the signal
    This part smooth the signal with the Savgol filter by the least squares method
    With the function visual_check_smoothing_effect you can investigate its effect compared to the raw signal
    """
    markersFilter = copy.deepcopy(markers)
    for key in markers.keys():
        markersFilter[key]['X'] = savgol_filter(markers[key]['X'], window_length, poly_order)
        markersFilter[key]['Y'] = savgol_filter(markers[key]['Y'], window_length, poly_order)
        markersFilter[key]['Z'] = savgol_filter(markers[key]['Z'], window_length, poly_order)

    markerName = 'VU_Baseball_R_RRS'
    coordinateName = 'Y'

    """
    3.1) Save the unfiltered inning data in binary form 
    """
    # Path where the pickle will be saved. Last part will be the name of the file
    filename = 'data/Innings/Unfiltered/'+ subject_name + '/' + inning_name

    # Initialize the pickle file
    outfile = open(filename, 'wb')

    # Write the dictionary into the binary file
    pickle.dump(markers, outfile)
    outfile.close()

    print('Filtered dictionary has been saved.')

    """
    3.2) Save the filtered inning data in binary form 
    """
    # Path where the pickle will be saved. Last part will be the name of the file
    filename = 'data/Innings/Filtered/'+ subject_name + '/' + inning_name

    # Initialize the pickle file
    outfile = open(filename, 'wb')

    # Write the dictionary into the binary file
    pickle.dump(markersFilter, outfile)
    outfile.close()

    print('Filtered dictionary has been saved.')


    """
    3.3) Save unfiltered pitch data
    """

    # Path where the pickle will be saved. Last part will be the name of the file
    filename = 'data/Pitches/Unfiltered/' + subject_name + '/' + inning_name

    # Initialize the pickle jar file
    outfile = open(filename, 'wb')

    # Write the dictionary into the binary file
    pickle.dump(markers_cut_unfiltered, outfile)
    outfile.close()

    print('Unfiltered pitches have been saved.')

    """
    3.4) Cut filtered data and save
    """

    markers_cut_filtered = cut_markers(markersFilter,
                                       ball_pickups)  # Outputs dictionary of dictionaries contatining individual pitches, length = ballpickups + 1

    # Path where the pickle will be saved. Last part will be the name of the file
    filename = 'data/Pitches/Filtered/' + subject_name + '/' + inning_name

    # Initialize the pickle jar file
    outfile = open(filename, 'wb')

    # Write the dictionary into the binary file
    pickle.dump(markers_cut_filtered, outfile)
    outfile.close()

    print('Filtered pitches have been saved as pickle')