import pickle
import seaborn as sns
import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
colors = sns.color_palette()
ax = plt.figure(figsize=[8, 5])
# 'Ball Speed', 'max_abduction_moment'
Outputs = ['Fatigue Reports', 'max_abduction_moment']
Pitchers = ['PP01','PP02','PP03','PP04','PP05','PP07','PP08','PP14','PP15']
filter_state = 'Filtered'
# Pitchers = ['PP01','PP02']

k = 0
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
    path = 'Results/Pitches/' + filter_state +'/' + pitcher + '/' + Last_inning + '/'
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
    ax = plt.scatter(df[Outputs[0]],df[Outputs[1]],label = pitcher, color = colors[k])
    # df[Outputs[0]] = [(df[Outputs[0]][l] + 0.001 *np.random.rand()) for l in range(len(df[Outputs[0]]))] # adds a tiny bit of noise
    df = df.sort_values(Outputs[0])

    z = np.polyfit(df[Outputs[0]], df[Outputs[1]], 1)
    y_hat = np.poly1d(z)(df[Outputs[0]])

    plt.plot(df[Outputs[0]], y_hat, lw=2, color = colors[k])
    plt.legend(facecolor = 'white')
    plt.ylabel(Outputs[1])
    plt.xlabel(Outputs[0])

    k += 1

plt.show()