import pickle
import seaborn as sns
import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

Outputs = ['max_abduction_moment']
filter_state = 'Unfiltered'
Pitchers = ['PP01','PP02','PP03','PP04','PP05','PP07','PP08','PP14','PP15']

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


    for output in Outputs:
        Max_moments = []
        for inning in Innings:
            # --- Define path where Results are stored --- #
            path = 'Results/Pitches/'+ filter_state +'/' + pitcher + '/' + inning + '/'
            filename = "Outputs"

            # --- Load data from pickle --- #
            filenameIn = path + filename
            infile = open(filenameIn, 'rb')
            data = pickle.load(infile)
            infile.close()

            for pitch in data['forearm'][output]:
                Max_moments.append([data['forearm'][output][pitch],inning])

        df = pd.DataFrame(Max_moments)
        df.columns = [output, 'Inning']

        ax = plt.figure(figsize=[8,5])

        # plt.xlim(10,50)
        ax = sns.violinplot(y="Inning", x=output, data=df, inner='quartile', bw = 'scott') #bw = 1.059 for normal distribution
        ax = sns.swarmplot(y="Inning", x=output, data=df, color=".3")

        plt.ylabel(None)
        plt.xlabel('Maximum Moment [Nm]')
        plt.title(pitcher + ' Maximum Elbow Abduction Moment Violin Plot')
        plt.savefig("Results/Violin/" + filter_state + "/Innings/"+pitcher)