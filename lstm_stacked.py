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
#import xgboost
from numpy import array
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional
from keras import metrics
import tensorflow as tf


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


# choose a number of time steps
n_steps = 18
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X_lstm,y_lstm = split_sequence(y_train.values, n_steps)
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], n_features))
X_test_lstm,y_test_lstm = split_sequence(y_test.values, n_steps)
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], n_features))




# define model
model = Sequential()
model.add(LSTM(20, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(20, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=[metrics.mean_absolute_error, 
                                                     metrics.mean_squared_error, 
                                                     metrics.mean_absolute_percentage_error,
                                                     metrics.categorical_accuracy
                                                    ])

# fit model
try:
    model = load_model('models\lstm_stacked_tmax_interlagos.h5')
except:
    history = model.fit(X_lstm, y_lstm, epochs=100, verbose=0)
    model.save("models\lstm_stacked_tmax_interlagos.h5")

    plot_history(history, val=False)
    
mse_loss,mae,mse,mape,acc = model.evaluate(X_lstm, y_lstm)
print("Train MSE loss, MAE, MSE, MAPE, acc: ", mse_loss,mae,mse,mape,acc)

mse_loss,mae,mse,mape,acc= model.evaluate(X_test_lstm, y_test_lstm)
print("Test MSE loss, MAE, MSE, MAPE, acc: ", mse_loss,mae,mse,mape,acc)

y_true_list,y_pred = predict_lstm(y_test, n_steps)
plot_evaluation(y_true_list, y_pred, mae)

#last 96 hours
plot_evaluation(y_test[-96:], y_pred[-96:], mae)

#future prediction
days = 2
hours = days * 24
y_true_list_new,y_pred_new = predict_lstm_future(y_test, n_steps, hours, model)

fig = plt.figure(figsize=(15,10))
start = y_test.index.max()
end = y_test.index.max() + dt.timedelta(hours=hours)

x_ax = pd.date_range(start=start, end=end, freq='H')
x_ax2 = y_test.reset_index()[['Data']].iloc[-n_steps:,:]

plt.plot(x_ax, y_pred_new, label="predicted")
plt.plot(x_ax2['Data'], y_test.reset_index().drop(columns='Data').iloc[-n_steps:,:], label="original")

#some confidence interval
#ci = 1.95 * np.std(y_pred)/np.mean(y_pred)
ci = mae
plt.ylim(0,50)
plt.fill_between(x_ax, (y_pred_new-ci), (y_pred_new+ci), color='b', alpha=.1)
plt.title("Temperature test and predicted data")
plt.legend()
plt.show()
