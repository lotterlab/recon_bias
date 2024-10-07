import pandas as pd 


file = '../../../data/UCSF-PDGM/metadata.csv'
data = pd.read_csv(file)
# filter for split 
data = data[data['split'] == 'test']

print(data["os"]/500)