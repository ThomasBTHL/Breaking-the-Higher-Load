import pickle
import seaborn as sns
import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

Outputs = ['Fatigue Reports', 'max_abduction_moment']
Pitchers = ['PP01','PP02','PP03','PP04','PP05','PP06','PP07','PP08','PP12','PP14','PP15']

ax = plt.figure(figsize=[8, 5])
for pitcher in Pitchers:
    """
    individual pitcher information
    """
    if pitcher == 'PP01':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_5','Inning_6','Inning_7','Inning_8'] # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP02':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6'] # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP03':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9','Inning_10','Inning_11']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP04':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP05':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP06':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP07':
        Innings = ['Inning_1', 'Inning_2', 'Inning_3', 'Inning_4', 'Inning_5',
                   'Inning_6']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP08':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP09':
        Innings = []  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP10':
        Innings = []  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP11':
        Innings = []  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP12':
        Innings = ['Inning_1', 'Inning_2', 'Inning_3', 'Inning_4', 'Inning_5',
                   'Inning_6','Inning_7']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP13':
        Innings = []  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP14':
        Innings = ['Inning_1', 'Inning_2', 'Inning_3', 'Inning_4', 'Inning_5',
                   'Inning_6','Inning_7','Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP15':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9','Inning_10','Inning_11']  # Inning where you want to look, for pitches gives all pitches in inning

    """"
    Generate plots
    """
    Last_inning = Innings[-1]
    Plotted_data = []

    # --- Define path where Results are stored --- #
    path = 'Results/Pitches/Unfiltered/' + pitcher + '/' + Last_inning + '/'
    filename = "Cumulative_til_this_point"

    # --- Load data from pickle --- #
    filenameIn = path + filename
    infile = open(filenameIn, 'rb')
    data = pickle.load(infile)
    infile.close()

    for pitch in data['forearm']['max_abduction_moment']:
        Plotted_data.append([data['forearm'][Outputs[0]][pitch],data['forearm'][Outputs[1]][pitch]])

    df = pd.DataFrame(Plotted_data)
    df.columns = Outputs



    # ax = sns.violinplot(y="Inning", x=output, data=df, inner='quartile', bw = 'scott') #bw = 1.059 for normal distribution
    ax = plt.scatter(df[Outputs[0]],df[Outputs[1]],label = pitcher)
    plt.legend()

    plt.ylabel(Outputs[1])
    plt.xlabel(Outputs[0])