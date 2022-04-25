import pickle
import seaborn as sns
import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

pitcher = 'PP07'
# Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9','Inning_10','Inning_11'] # Inning where you want to look, for pitches gives all pitches in inning
Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6']

Outputs = ['max_abduction_moment']

for output in Outputs:
    Max_moments = []
    for inning in Innings:
        # --- Define path where Results are stored --- #
        path = 'E:/MSc/Thesis/Breaking the Higher Load/3D_inverse_dynamics/Python/Results/Pitches/Unfiltered/' + pitcher + '/' + inning + '/'
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

    ax = sns.boxplot(x=output, y="Inning", data=df, whis=np.inf)

    ax = sns.swarmplot(x=output, y="Inning", data=df, color=".3")
    # ax = sns.scatterplot(x=output, y="Inning", data=df, color=".3")
    # ax = sns.stripplot(x=output, y="Inning", data=df, color=".3")

