'''Write a program to Transforming Nominal Features, Transforming Ordinal Features and
Encoding Categorical Features using one-hot Encoding Scheme'''

import pandas as pd
df = pd.read_csv("pokemon.csv",encoding='utf-8')

nominal_features = ['Generation']
df_nominal = pd.get_dummies(df[nominal_features])

gen_ord_map = {'Gen1':1,'Gen2':2,'Gen3':3,
                'Gen4':4,'Gen5':5,'Gen6':6}
df['GenerationLabel'] = df['Generation'].map(gen_ord_map)

categorical =['Generation','Legendary']
df_encoded = pd.get_dummies(df[categorical])

df_transformed = pd.concat([df,df_nominal,df_encoded],axis=1)
print(df_transformed)
