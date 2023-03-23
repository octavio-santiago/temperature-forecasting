import pandas as pd
import numpy as np

df = pd.read_csv('tabela.csv')

df['Data'] = df.apply(lambda row: row['Data'] + ' 0' + str(row['Hora (UTC)'])[0] if len(str(row['Hora (UTC)'])) == 3 else row['Data'] + ' ' + str(row['Hora (UTC)'])[:2] , axis=1)

df['Radiacao (KJ/m²)'] = df['Radiacao (KJ/m²)'].fillna(0)
df = df.loc[:,'Data':'Temp. Min. (C)']
df['Data'] = pd.to_datetime(df.Data, format="%d/%m/%Y %H")
df = df.drop(columns=['Hora (UTC)'])
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.groupby('Data').agg({'Temp. Max. (C)': 'max'
                            })
df = df.sort_values(by='Data')


df['Variation'] = df['Temp. Max. (C)'].diff().shift(periods=1, freq="H")
df['Avg.3D'] = df['Temp. Max. (C)'].rolling(3).mean().shift(periods=1, freq="H")
df['Avg.7D'] = df['Temp. Max. (C)'].rolling(7).mean().shift(periods=1, freq="H")
#df['Avg.14D'] = df['Temp. Max. (C)'].rolling(14).mean().shift(periods=1, freq="D")
df['Weekday'] =  list(df.reset_index()['Data'].apply(lambda x: x.weekday()))
df['Month'] =  list(df.reset_index()['Data'].apply(lambda x: x.month))
df = df.dropna()

df.to_csv('table_proc.csv')

