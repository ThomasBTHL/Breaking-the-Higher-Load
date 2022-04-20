import pickle
import seaborn as sns
import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

pitcher = 'PP08'
# Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9','Inning_10','Inning_11'] # Inning where you want to look, for pitches gives all pitches in inning
Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8']
Max_moments = []

for inning in Innings:
    # --- Define path where Results are stored --- #
    path = 'E:/MSc/Thesis/Breaking the Higher Load/3D_inverse_dynamics/Python/Results/Pitches/Unfiltered/' + pitcher + '/' + inning + '/'
    filename = "Max_norm_moments"

    # --- Load data from pickle --- #
    filenameIn = path + filename
    infile = open(filenameIn, 'rb')
    data = pickle.load(infile)
    infile.close()

    for pitch in data['upperarm']['max_norm_moment']:
        Max_moments.append([data['upperarm']['max_norm_moment'][pitch],inning])

df = pd.DataFrame(Max_moments)
df.columns = ['Moment', 'Inning']

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="Inning", hue="Inning", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "Moment",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1    , linewidth=1.5)
g.map(sns.kdeplot, "Moment", clip_on=False, color="w", lw=2, bw_adjust=.5)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(Moment, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "Moment")
g.set(xlim=(30, 100))

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)