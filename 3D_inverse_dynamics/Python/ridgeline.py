import pickle
import seaborn as sns
import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

pitcher = 'PP15'
Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9','Inning_10','Inning_11'] # Inning where you want to look, for pitches gives all pitches in inning
# Innings = ['Inning_1','Inning_2','Inning_5','Inning_6','Inning_7','Inning_8']
Outputs = ['max_abduction_moment']
filenames = ["Cumulative_til_this_point", "Max_norm_moments"]

for filename in filenames:
    for output in Outputs:
        Max_moments = []
        for inning in Innings:
            # --- Define path where Results are stored --- #
            path = 'Results/Pitches/Unfiltered/' + pitcher + '/' + inning + '/'

            # --- Load data from pickle --- #
            filenameIn = path + filename
            infile = open(filenameIn, 'rb')
            data = pickle.load(infile)
            infile.close()

            for pitch in data['forearm'][output]:
                Max_moments.append([data['forearm'][output][pitch],inning])

        df = pd.DataFrame(Max_moments)
        df.columns = [output, 'Inning']

        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(df, row="Inning", hue="Inning", aspect=15, height=.5, palette=pal)

        # Draw the densities in a few steps
        # g.map(sns.stripplot, output, clip_on=False, color ="k", lw=2)
        g.map(sns.kdeplot, output, bw_adjust=1.059, clip_on=False, fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, output, clip_on=False, color="w", lw=2, bw_adjust=1.059)
        # g.map(sns.kdeplot, output, clip_on=False, color="k", lw=2, bw_adjust=1)



        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


        # Define and use a simple function to label the plot in axes coordinates
        def label(Moment, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, color='k',
                    ha="left", va="center", transform=ax.transAxes)


        g.map(label, output)
        g.axes[-1,0].set_xlabel('Maximum Moment [Nm]')

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-.25)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        if filename == "Cumulative_til_this_point":
            g.fig.suptitle(pitcher + ' Maximum Elbow Abduction Moment Cumulative Density Plot')
            """
            Save the figures to results folder
            """
            g.figure.savefig('Results/Ridgelines/Unfiltered/Cumulative'+pitcher)
            print('Figures have been saved.')

        if filename == "Max_norm_moments":
            g.fig.suptitle(pitcher + ' Maximum Elbow Abduction Moment Density Plot')
            """
            Save the figures to results folder
            """
            g.figure.savefig('Results/Ridgelines/Unfiltered/'+pitcher)
            print('Figures have been saved.')

