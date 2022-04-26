import pickle
import seaborn as sns
import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

pitcher = 'PP03'
Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9','Inning_10','Inning_11'] # Inning where you want to look, for pitches gives all pitches in inning
# Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6']

Outputs = ['max_abduction_moment']

for output in Outputs:
    Max_moments = []
    for inning in Innings:
        # --- Define path where Results are stored --- #
        path = 'Results/Pitches/Unfiltered/' + pitcher + '/' + inning + '/'
        filename = "Max_norm_moments"
        # filename = "Cumulative_til_this_point"

        # --- Load data from pickle --- #
        filenameIn = path + filename
        infile = open(filenameIn, 'rb')
        data = pickle.load(infile)
        infile.close()

        for pitch in data['forearm'][output]:
            Max_moments.append([data['forearm'][output][pitch],inning])

    df = pd.DataFrame(Max_moments)
    df.columns = [output, 'Inning']

    ax = sns.violinplot(x="Inning", y=output, data=df, inner='quartile', bw = 1,cut = 2)
    ax = sns.swarmplot(x="Inning", y=output, data=df, color=".3")

