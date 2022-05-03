"""Adds velocity pitch data from an excel to a dictionary

   Function is developed and written by Thomas van Hogerwou, master student TU-Delft
   Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

   Version 1.0 (2022-03-25)

   Arguments:
        Output_Dictionary: Dictonary of outputs that lack pitch velos
        data: Data coming from an excel file to be added to dictionary
        data_key: key(name) of new data
   Returns:
       Appended_dictionary: Dictionary containing added data
"""
pitcher = 'PP03'
inning = 'Inning_1'
data_key = 'Ball Speed'

"""
Results Dictionary
"""
path = 'Results/Pitches/Unfiltered/' + pitcher + '/' + inning + '/'
# --- Load data from pickle --- #
filenameIn = path + 'Max_norm_moments'
infile = open(filenameIn, 'rb')
Output_Dictionary = pickle.load(infile)
infile.close()



"""
xlsx data
"""
filename = "data/21_03_22_Overview UCL variability fatigue_graphs_BvT(1).xlsx"
xlsx_data = openpyxl.load_workbook(filename=filename,data_only=True)

df = pd.DataFrame(xlsx_data[pitcher].values) # make it a df


segments = Output_Dictionary.keys()


for segment in segments:
    Output_Dictionary[segment][data_key] = dict()
    Outputs = Output_Dictionary[segment].keys()
    for pitch in Output_Dictionary[segment]['max_abduction_moment']:
        xlsx_pitches = df[0].squeeze()
        rows = xlsx_pitches.str.find(pitch,end = len(pitch))
        for i in range(len(rows)):
            if rows[i] == 0:
                row = i
                break
        new_pitch_data = df[1][row]#grab from excel data
        Output_Dictionary[segment][data_key][pitch] = new_pitch_data