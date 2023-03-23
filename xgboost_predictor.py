import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from IPython.core.display import Image, display
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from numpy import array
#from keras.models import Sequential, load_model
#from keras.layers import LSTM, Dense, Bidirectional
#from keras import metrics
#import tensorflow as tf


df = pd.read_csv('table_proc.csv')
df['Data'] = pd.to_datetime(df["Data"])
df.index = df['Data']
df = df.drop(columns='Data')

#split
train_size = round(len(df)*0.8)
print(train_size)
tscv = TimeSeriesSplit(gap=0, max_train_size=train_size, n_splits=5, test_size=None)
print(tscv)
var_drop_list = ['Temp. Max. (C)']
X = df.drop(columns=var_drop_list)
y = df[['Temp. Max. (C)']]
print(X.columns)
for train_index, test_index in tscv.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]


def plot_history(history, val=False):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    for col in hist.columns:
        if col != 'epoch' and 'val' not in col:
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel(col)
            plt.plot(hist['epoch'] , hist[col], label= 'Train Error')
            if val == True:
                plt.plot(hist['epoch'] , hist['val_'+col], label= 'Validation Error')
                
            plt.ylim(0,max(hist[col]))    
            plt.legend()
            plt.show()
            
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# prediction
def predict_lstm(y, n_steps):
    y_pred = []
    y_true_list = []
    for i in range(0,len(y)-n_steps):
        x_input = y.to_numpy().flatten()
        y_true = x_input[i+n_steps]
        x_input = x_input[i:i+n_steps]
        #print(x_input)
        x_input = x_input.reshape((1, n_steps, n_features))
        y_p = model.predict(x_input, verbose=0)
        y_p = y_p[0][0]
        y_pred.append(y_p)
        y_true_list.append(y_true)
        #print(y_p, y_true)
        
    y_pred = np.array(y_pred)
    y_true_list = np.array(y_true_list)
    
    return y_true_list,y_pred


def predict_lstm_future(y, n_steps, days, model):
    y = y.reset_index().drop(columns='Data').iloc[-n_steps:,:]
    #y = y.iloc[-n_steps:,:]
    
    y_pred = []
    y_true_list = []
        
    x_input = y.to_numpy().flatten()
    y_true = x_input
    #x_input = x_input[i:i+n_steps]
    #print(x_input)
    x_input = x_input.reshape((1, n_steps, n_features))
    y_p = model.predict(x_input, verbose=0)
    y_p = y_p[0][0]
    y_pred.append(y_p)
    y_true_list.append(y_true)
    #print(y_p, y_true)
    new_y = list(y.copy().to_numpy().flatten())
    #print(new_y)
    for i in range(0,days):
        x_input = new_y.copy()
        x_input = np.array(x_input).flatten()
        x_input = x_input[-n_steps:]
        #print(x_input)
        x_input = x_input.reshape((1, n_steps, n_features))
        
        y_p = model.predict(x_input, verbose=0)
        y_p = y_p[0][0]
        
        y_pred.append(y_p)
        #y_true_list.append(y_true)
        new_y.append(y_p)       
    
    y_pred = np.array(y_pred)
    y_true_list = np.array(y_true_list)
    
    print(y_pred)
    
    return y_true_list,y_pred

def plot_evaluation(y_true, y_pred, ci):
    fig = plt.figure(figsize=(15,10))
    x_ax = range(len(y_true))
    x_ax2 = range(len(y_pred))
    plt.plot(x_ax, y_true, label="original")
    plt.plot(x_ax2, y_pred, label="predicted")
    #some confidence interval
    #ci = 1.95 * np.std(y_pred)/np.mean(y_pred)
    #ci = mse_t
    plt.ylim(0,50)
    plt.fill_between(x_ax2, (y_pred-ci), (y_pred+ci), color='b', alpha=.1)
    plt.title("Temperature test and predicted data")
    plt.legend()
    plt.show()


# create an xgboost regression model
xgbr = xgboost.XGBRegressor(n_estimators=50, max_depth=3, eta=0.1)
#train
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)
y_pred = xgbr.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
print("MAE: %.2f" % mae)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))
# test
y_pred = xgbr.predict(X_test)
#score = xgbr.score(X_test, y_test)  
#print("Training score: ", score)
mse_t = mean_squared_error(y_test, y_pred)
mae_t = mean_absolute_error(y_test, y_pred)
print("MAE: %.2f" % mae)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))

plot_evaluation(y_test, y_pred, mae)
#ultimas 96 horas
plot_evaluation(y_test[-96:], y_pred[-96:], mae)
#predict new day
days = 2
hours = days * 24
date_ = df.index.max() + dt.timedelta(days=-8)
df_new = df.copy().loc[date_:,:]
df_new['Variation'] = df_new['Temp. Max. (C)'].diff()
df_new['Avg.3D'] = df_new['Temp. Max. (C)'].rolling(3).mean()
df_new['Avg.7D'] = df_new['Temp. Max. (C)'].rolling(7).mean()
df_new['Weekday'] =  list(df_new.reset_index()['Data'].apply(lambda x: x.weekday()))
df_new['Month'] =  list(df_new.reset_index()['Data'].apply(lambda x: x.month))
temps = list(df_new.iloc[-14:,:]['Temp. Max. (C)'])
temps_or = temps.copy()
df_new = df_new.iloc[-1:,:]
max_date = df.index.max()
df_new.index = [df.index.max() + dt.timedelta(days=1)]
#create initial X - 23/03
X_new = df_new.drop(columns=var_drop_list)
temps = temps_or.copy()
for i in range(1,hours+1):
    date = max_date + dt.timedelta(hours=i)
#predict new temp
    new_temp = xgbr.predict(X_new.iloc[-1:,:])[0]
    print(new_temp)
    temps.append(new_temp)
diff = temps[-1] - temps[-2]
new_line = pd.DataFrame({
        'Variation':[diff],
        'Avg.3D':[np.mean(temps[-3:])],
        'Avg.7D':[np.mean(temps[-7:])],
        'Weekday':[date.weekday()],
        'Month':[date.month]
    })
new_line.index = [date]
X_new = X_new.append(new_line)
#print(X_new)
fig = plt.figure(figsize=(15,10))
#start = y_test.index.max()
#end = y_test.index.max() + dt.timedelta(days=days)
y_pred_new = np.array(temps[-hours:])
start_or = X_new.index.max() + dt.timedelta(hours=-len(temps)) 
end_or = X_new.index.max() + dt.timedelta(hours=-hours-1)
start_p = X_new.index.max() + dt.timedelta(hours=-hours+1) 
end_p = X_new.index.max()
x_ax = pd.date_range(start=start_or, end=end_or, freq='H')
x_ax2 = pd.date_range(start=start_p, end=end_p, freq='H')
plt.plot(x_ax, temps[-len(temps):-hours], label="original")
plt.plot(x_ax2, temps[-hours:], label="predicted")
#some confidence interval
#ci = 1.95 * np.std(y_pred)/np.mean(y_pred)
ci = mae
plt.ylim(0,50)
plt.fill_between(x_ax2, (y_pred_new-ci), (y_pred_new+ci), color='b', alpha=.1)
plt.title("Temperature test and predicted data")
plt.legend()
plt.show()
