import pickle
import pandas as pd
import functions as f
import numpy as np

appended_data = dict()
participants= ['PP01','PP02','PP03','PP04','PP05','PP06','PP07','PP08','PP09','PP10','PP11']
method = 'No tape'
data=dict()

for pitcher in participants:
    fileName ='/Users/bvantrigt/TU Delft/Shared/Python Inverse Dynamic Model/Bart/3d_inverse_dynamica/data/separation_times/No tape/'+ pitcher+'_separation_times.pickle'
    dataImport = pickle.load(open(fileName,'rb'))
    globals()[f'data_{pitcher}'] = pd.DataFrame.from_dict(dataImport)

combined_dict = pd.concat([data_PP01, data_PP02,data_PP03,data_PP04,data_PP05,data_PP06,data_PP07,data_PP08,data_PP09,data_PP10,data_PP11], axis=0)


pathSave ='/Users/bvantrigt/TU Delft/Shared/Python Inverse Dynamic Model/Bart/3d_inverse_dynamica/data/'
f.save_dict2xlsx_together(combined_dict,pathSave,'dataLarisa.csv')



