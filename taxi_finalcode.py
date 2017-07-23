# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
np.random.seed(7)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Merge, Embedding, Flatten
import datetime as dt
from keras.callbacks import Callback
import ml_metrics as metrics
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import StratifiedShuffleSplit


class prediction_history1(Callback):
    def __init__(self):
        self.predhis = []
        self.predhisval = []
        self.scores_train = []
        self.scores_validation = []
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0 or epoch==(epochs1-1):
            self.predhis.append(model1.predict([np.array(X_train[x]) for x in variables1]))
            self.predhisval.append(model1.predict([np.array(X_validation[x]) for x in variables1]))
            self.scores_train.append(metrics.quadratic_weighted_kappa(get_output(predictions1.predhis[-1]),np.array(X_train['revenue_class'])))
            self.scores_validation.append(metrics.quadratic_weighted_kappa(np.digitize(predictions1.predhisval[-1][:,0],get_offset(predictions1.predhis[-1]))+1,np.array(X_validation['revenue_class'])))
            print 'training : '+str(self.scores_train[-1])
            print 'validation : '+str(self.scores_validation[-1])            
        
predictions1=prediction_history1()

class prediction_history2(Callback):
    def __init__(self):
        self.predhis = []
        self.predhisval = []
        self.scores_train = []
        self.scores_validation = []
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0 or epoch==(epochs2-1):
            self.predhis.append(model2.predict([np.array(X_train[x]) for x in variables2]))
            self.predhisval.append(model2.predict([np.array(X_validation[x]) for x in variables2]))
            self.scores_train.append(metrics.quadratic_weighted_kappa(get_output(predictions2.predhis[-1]),np.array(X_train['revenue_class'])))
            self.scores_validation.append(metrics.quadratic_weighted_kappa(np.digitize(predictions2.predhisval[-1][:,0],get_offset(predictions2.predhis[-1]))+1,np.array(X_validation['revenue_class'])))
            print 'training : '+str(self.scores_train[-1])
            print 'validation : '+str(self.scores_validation[-1])            
        
predictions2=prediction_history2()

def get_output(prediction):
    if len(prediction.shape)!=1:
        pred = prediction[:,0]
    else:
        pred = prediction
    num = pred.shape[0]
    output = np.asarray([5]*num,dtype=int)
    rank = pred.argsort()
    output[rank[:int(num*cdf[0]-1)]] = 1
    output[rank[int(num*cdf[0]):int(num*cdf[1]-1)]] = 2
    output[rank[int(num*cdf[1]):int(num*cdf[2]-1)]] = 3
    output[rank[int(num*cdf[2]):int(num*cdf[3]-1)]] = 4
    return output

def getClfScore(preds):
    w = np.asarray(np.arange(1,6))
    preds = preds * w[np.newaxis,:]
    preds = np.sum(preds, axis=1)
    output = get_output(preds)
    output = np.asarray(output, dtype=int)
    return output
    
def get_one_vec(preds):
    w = np.asarray(np.arange(1,6))
    preds = preds * w[np.newaxis,:]
    preds = np.sum(preds, axis=1)
    return preds
    
def get_offset(prediction_on_train):
    if len(prediction_on_train.shape)==2:
        predict_one=prediction_on_train[:,0]       
    else:
        predict_one=get_one_vec(prediction_on_train)
    array=len(predict_one)*cdf[:-1]
    b=predict_one[predict_one.argsort()[array.astype(int)]]
    return b
    
def get_offsetstacked(prediction_on_train):
    predict_one=prediction_on_train
    array=len(predict_one)*cdf[:-1]
    b=predict_one[predict_one.argsort()[array.astype(int)]]
    return b
    
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6369 * c
    return km
    
haversine_vectorize = np.vectorize(haversine)    
    
df = pd.read_csv('data_train_competition.csv')

test = pd.read_csv('data_test_N_competition.csv')

geo_parameters=['starting_latitude','starting_longitude']

feriados_train=[dt.date(2015,01,01),
dt.date(2015,01,06),
dt.date(2015,02,23),
dt.date(2015,03,25)]

feriados_test=[dt.date(2015,04,10),
dt.date(2015,04,12),
dt.date(2015,05,01),
dt.date(2015,06,01)]

pts=np.array(df[geo_parameters])

bw = 0.001
ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5)
ms.fit(pts)

df['geolocation1']=ms.labels_
test['geolocation1'] = ms.predict(np.array(test[geo_parameters]))

bw = 0.0005
ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5)
ms.fit(pts)

df['geolocation2']=ms.labels_
test['geolocation2'] = ms.predict(np.array(test[geo_parameters]))

df['utc']=[dt.datetime.utcfromtimestamp(x) for x in df['starting_timestamp']]
df['utc']=df['utc']-pd.Timedelta(hours=2)

test['utc']=[dt.datetime.utcfromtimestamp(x) for x in test['starting_timestamp']]
test['utc']=test['utc']-pd.Timedelta(hours=2)

feriados_train_antes=[y-dt.timedelta(days=1) for y in feriados_train]
feriados_train_despues=[y+dt.timedelta(days=1) for y in feriados_train]
df['feriado']=[2 if x.date() in feriados_train else 1 if x.date() in feriados_train_antes else 3 if x.date() in feriados_train_despues else 0 for x in df['utc']]

df['sin_day'] = np.sin(2*np.pi*df.utc.dt.day/df.utc.dt.days_in_month)
df['cos_day'] = np.cos(2*np.pi*df.utc.dt.day/df.utc.dt.days_in_month)

df['dayofweek'] = df.utc.dt.dayofweek

df['qhour'] = df.utc.dt.hour * 4 + df.utc.dt.minute // 15

g=pd.concat([df,test])
g.reset_index(drop=True,inplace=True)

stack=pd.Series()
stack_distance=pd.Series()
for name, group in g.groupby(['taxi_id'],sort=False):
    stack=pd.concat([stack,pd.Series(group['utc'].diff(),index=group.index).shift(-1)])
    stack_distance=pd.concat([stack_distance,pd.Series(haversine_vectorize(
                            np.array(group['starting_latitude']),
                            np.array(group['starting_longitude']),
                            np.roll(np.array(group['starting_latitude']),-1),
                            np.roll(np.array(group['starting_longitude']),-1)),index=group.index)])
    stack_distance.iloc[-1]=0

stack=stack.dt.seconds
stack.sort_index(inplace=True)

df['taxi_last']=stack.iloc[:len(df)]
df['taxi_last'].fillna(0,inplace=True)
test['taxi_last']=stack.iloc[len(df):].reset_index(drop=True)
test['taxi_last'].fillna(0,inplace=True)

stack_distance.sort_index(inplace=True)

df['taxi_dist']=stack_distance.iloc[:len(df)]
test['taxi_dist']=stack_distance.iloc[len(df):].reset_index(drop=True)

minutes_offset=[15]

if len(minutes_offset)!=0:
    for elapsed in minutes_offset:
        start_dates = g['utc'] - pd.Timedelta(minutes=elapsed)
        g['start_index'] = g['utc'].values.searchsorted(start_dates, side='right')
        g['end_index'] = np.arange(len(g))
        
        g['demand'+str(elapsed)]=g['end_index']-g['start_index']
        
        df['demand'+str(elapsed)]=g['demand'+str(elapsed)].iloc[:len(df)]
        test['demand'+str(elapsed)]=g['demand'+str(elapsed)].iloc[len(df):].reset_index(drop=True)        

del g

variables_continuous=['taxi_last','taxi_dist']

if len(minutes_offset)!=0:
    variables_continuous+=['demand'+str(x) for x in minutes_offset]

if len(variables_continuous)!=0:
    scaler = StandardScaler()
    df[variables_continuous]=scaler.fit_transform(df[variables_continuous])

variables_otras=['sin_day','cos_day']

variables_categorical=['taxi_id','qhour','dayofweek','feriado','geolocation1','geolocation2']

categories={feature:pd.Categorical(df[feature]).categories for feature in variables_categorical}

for feature in variables_categorical:
    df[feature] = pd.Categorical(df[feature],categories=categories[feature]).codes

vc={x:len(categories[x]) for x in variables_categorical}

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

for train_index, test_index in sss.split(df, df['revenue_class']):
    X_train, X_validation = df.iloc[train_index], df.iloc[test_index]

###############test_phase
test['dayofweek'] = test.utc.dt.dayofweek

test['sin_day'] = np.sin(2*np.pi*test.utc.dt.day/test.utc.dt.days_in_month)
test['cos_day'] = np.cos(2*np.pi*test.utc.dt.day/test.utc.dt.days_in_month)

test['qhour'] = test.utc.dt.hour * 4 + test.utc.dt.minute // 15

feriados_test_antes=[y-dt.timedelta(days=1) for y in feriados_test]
feriados_test_despues=[y+dt.timedelta(days=1) for y in feriados_test]
test['feriado']=[2 if x.date() in feriados_test else 1 if x.date() in feriados_test_antes else 3 if x.date() in feriados_test_despues else 0 for x in test['utc']]

if len(variables_continuous)!=0:
    test[variables_continuous]=scaler.transform(test[variables_continuous])

for feature in variables_categorical:
    test[feature] = pd.Categorical(test[feature],categories=categories[feature]).codes
  
#####
variables1=variables_categorical+variables_otras+variables_continuous
variables1.remove('geolocation2')

hist = np.bincount(X_train['revenue_class']-1)
cdf = np.cumsum(hist) / float(sum(hist))

Y_train=np.array(X_train['revenue_class']-1)
Y_validation=np.array(X_validation['revenue_class']-1)

batchsize=200
epochs1=7

models1=[]

model_taxi_id = Sequential()
model_taxi_id.add(Embedding(vc['taxi_id'], 10, input_length=1))
model_taxi_id.add(Flatten())
models1.append(model_taxi_id)

model_qhour = Sequential()
model_qhour.add(Embedding(vc['qhour'], 10, input_length=1))
model_qhour.add(Flatten())
models1.append(model_qhour)

model_dayofweek = Sequential()
model_dayofweek.add(Embedding(vc['dayofweek'], 5, input_length=1))
model_dayofweek.add(Flatten())
models1.append(model_dayofweek)

model_feriado= Sequential()
model_feriado.add(Embedding(vc['feriado'], 3, input_length=1))
model_feriado.add(Flatten())
models1.append(model_feriado)

model_geolocation = Sequential()
model_geolocation.add(Embedding(vc['geolocation1'], 10, input_length=1))
model_geolocation.add(Flatten())
models1.append(model_geolocation)

model_sinday = Sequential()
model_sinday.add(Dense(1,input_dim=1))
models1.append(model_sinday)

model_cosday = Sequential()
model_cosday.add(Dense(1,input_dim=1))
models1.append(model_cosday)

if len(variables_continuous)!=0:
    for var in variables_continuous:
        model_var = Sequential()
        model_var.add(Dense(1,input_dim=1))
        models1.append(model_var)  
         
model1 = Sequential()
model1.add(Merge(models1, mode='concat'))
model1.add(Dense(500, kernel_initializer='uniform', activation='relu'))
model1.add(Dense(250, kernel_initializer='uniform', activation='relu'))
model1.add(Dense(1, kernel_initializer='glorot_uniform',activation='linear'))
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.fit([np.array(X_train[x]) for x in variables1],Y_train,epochs=epochs1, batch_size = batchsize,validation_data=([np.array(X_validation[x]) for x in variables1],Y_validation),callbacks=[predictions1],shuffle=True) 

plt.plot(predictions1.scores_train)
plt.plot(predictions1.scores_validation)
plt.show()
 
###########################
variables2=variables_categorical+variables_otras+variables_continuous
variables2.remove('geolocation1') 
variables2.remove('taxi_last') 
variables2.remove('taxi_dist') 
    
epochs2=19

models2=[]

model_taxi_id = Sequential()
model_taxi_id.add(Embedding(vc['taxi_id'], 10, input_length=1))
model_taxi_id.add(Flatten())
models2.append(model_taxi_id)

model_qhour = Sequential()
model_qhour.add(Embedding(vc['qhour'], 10, input_length=1))
model_qhour.add(Flatten())
models2.append(model_qhour)

model_dayofweek = Sequential()
model_dayofweek.add(Embedding(vc['dayofweek'], 5, input_length=1))
model_dayofweek.add(Flatten())
models2.append(model_dayofweek)

model_feriado= Sequential()
model_feriado.add(Embedding(vc['feriado'], 3, input_length=1))
model_feriado.add(Flatten())
models2.append(model_feriado)

model_geolocation = Sequential()
model_geolocation.add(Embedding(vc['geolocation2'], 10, input_length=1))
model_geolocation.add(Flatten())
models2.append(model_geolocation)

model_sinday = Sequential()
model_sinday.add(Dense(1,input_dim=1))
models2.append(model_sinday)

model_cosday = Sequential()
model_cosday.add(Dense(1,input_dim=1))
models2.append(model_cosday)
 
if len(minutes_offset)!=0:
    for var in minutes_offset:
        model_var = Sequential()
        model_var.add(Dense(1,input_dim=1))
        models2.append(model_var)  
        
model2 = Sequential()
model2.add(Merge(models2, mode='concat'))
model2.add(Dropout(0.2))
model2.add(Dense(200, kernel_initializer='uniform', activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(1, kernel_initializer='glorot_uniform',activation='linear'))

model2.compile(loss='mean_squared_error', optimizer='adam')

model2.fit([np.array(X_train[x]) for x in variables2],Y_train,epochs=epochs2, batch_size = batchsize,validation_data=([np.array(X_validation[x]) for x in variables2],Y_validation),callbacks=[predictions2],shuffle=True) 

plt.plot(predictions2.scores_train)
plt.plot(predictions2.scores_validation)
plt.show()
################
prediction_kaggle1=model1.predict([np.array(test[x]) for x in variables1])
prediction_kaggle2=model2.predict([np.array(test[x]) for x in variables2])

predict_train1=predictions1.predhis[-1]
test['revenue_class']=np.digitize(prediction_kaggle1[:,0],get_offset(predict_train1))+1 

predict_train2=predictions2.predhis[-1]
test['revenue_class_aux']=np.digitize(prediction_kaggle2[:,0],get_offset(predict_train2))+1 

test['revenue_class'][test['taxi_last']==test['taxi_last'].iloc[-1]]=test['revenue_class_aux'][test['taxi_last']==test['taxi_last'].iloc[-1]]

test[['ID','revenue_class']].to_csv('submission.csv',index=False,encoding='utf-8')