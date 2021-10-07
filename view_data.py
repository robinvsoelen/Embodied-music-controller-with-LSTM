import pickle
import pandas as pd

infile = open('data/data.pckl','rb')
new_dict = pickle.load(infile)
infile.close()

variable = pd.read_pickle('data/data.pckl')

print(variable.to_string())