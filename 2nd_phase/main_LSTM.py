# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 12:38:00 2018

@author: Rajesh
"""
# importing relevant packages
from Input_Preprocessing import input_preprocessing
from Input_Preprocessing import input_preparation_LSTM
from Input_Parsing import input_parsing

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Bidirectional,TimeDistributed
import keras
from keras import optimizers

from sklearn.metrics import confusion_matrix


#Configuring GPU , soft placement, 40% occupancy
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True, device_count = {'GPU' : 4})
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.40
config.gpu_options.allocator_type = 'BFC'

Total_Label_TS=[]
Total_data1=[]
Total_data2=[]

Total_data1,Total_data2,Total_Label_TS = input_parsing()

Data_withLabel=[]
Data_withLabel1=[]

Data_withLabel,Data_withLabel1=input_preprocessing(Total_data1,Total_data2,Total_Label_TS)

X_train_LSTM,X_test_LSTM,Y_train,Y_test= input_preparation_LSTM(Data_withLabel,Data_withLabel1)

opt=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
time_steps=6
div_seq=1000
batch_size=32

#Model: LSTM
with tf.device('/gpu:0'):
    session = tf.Session(config=config)
    tf.keras.backend.set_session
    
    model = Sequential()
    model.add((LSTM(64, return_sequences=True,input_shape=((time_steps,div_seq))))) 
    model.add((LSTM(64,return_sequences=True)) ) 
    model.add(TimeDistributed(Dense(7, activation='softmax')))
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])
    model.summary()
    
    history=model.fit(X_train_LSTM, Y_train,batch_size=batch_size,shuffle=True, epochs=100, verbose=1,validation_data=(X_test_LSTM, Y_test))
    score,acc = model.evaluate(X_test_LSTM, Y_test, verbose=0)
    print('Test loss:', score)
    print('Test accuracy:', acc)


#Accuracy vs Epochs
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()

#Loss vs Epochs
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()

model.save('LSTM.h5')