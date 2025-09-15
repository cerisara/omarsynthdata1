import pandas as pd

df = pd.read_csv('dataset.csv')
for elt in df['text']:
    print(elt)
