import pickle
import seaborn as sns
import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

"""
Ridgeline_groups makes ridgline plot of groupped inning data of size n pitches
Contact E-Mail: Thom.hogerwou@gmail.com
​
Version 1 (2022-04-26)
"""

pitcher = "PP03"
last_inning = 'Inning_11' # last inning of data available for pitcher. cumulative innings already contain all other data
n = 25

# --- Define path where dictonary data is stored --- #
path = "Results/Pitches/Unfiltered/" + pitcher + "/" + last_inning + "/"
filename = "Cumulative_til_this_point"

# --- Load data from pickle --- #
filenameIn = path + filename
infile = open(filenameIn, 'rb')
all_data = pickle.load(infile)
infile.close()

Outputs = ['max_abduction_moment']

i = 1
group = 'group_' + str(i)
groups = [group]
Max_moments = []

for output in Outputs:
    for pitch in all_data['forearm'][output]:
        Max_moments.append([all_data['forearm'][output][pitch], group])
        if len(Max_moments) == i * n:
            i += 1
            group = 'group_' + str(i)
            groups.append(group)

    df = pd.DataFrame(Max_moments)
    df.columns = [output, 'Group']

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="Group", hue="Group", aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    # g.map(sns.stripplot, output, clip_on=False, color ="k", lw=2)
    # g.map(sns.kdeplot, output, bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    # g.map(sns.kdeplot, output, clip_on=False, color="w", lw=2, bw_adjust=.5)
    
    g.map(sns.kdeplot, output, clip_on=False, color="k", lw=2, bw_adjust=1)



    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=.1, linestyle="-", color=None, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(Moment, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)


    g.map(label, output)
    g.set(xlim=(0, 60))

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)