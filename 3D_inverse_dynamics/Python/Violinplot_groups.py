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
last_inning = 'Inning_11'  # last inning of data available for pitcher. cumulative innings already contain all other data
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
        if (len(all_data['forearm'][Outputs[0]]) - ((i-1) * n) < n/2):
            break

    df = pd.DataFrame(Max_moments)
    df.columns = [output, 'Group']

    ax = plt.figure()

    # plt.xlim(20,50)
    ax = sns.violinplot(x="Group", y=output, data=df, inner='quartile', bw = 'scott') #bw = 1.059 for normal distribution
    ax = sns.swarmplot(x="Group", y=output, data=df, color=".3")