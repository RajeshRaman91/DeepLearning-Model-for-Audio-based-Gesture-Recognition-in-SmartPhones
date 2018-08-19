# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 12:38:00 2018

@author: Rajesh
"""
# importing relevant packages
from Input_Preprocessing import input_preprocessing
from Input_Preprocessing import input_preparation
from Input_Parsing import input_parsing

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import GRU
from keras.layers.normalization import BatchNormalization

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

X_train_CNN,X_test_CNN,Y_train,Y_test= input_preparation(Data_withLabel,Data_withLabel1)

#Model: CNN-GRU
with tf.device('/gpu:0'):
    session = tf.Session(config=config)
    tf.keras.backend.set_session
    
    model = Sequential()

    model.add(Conv1D(32,3, activation='relu', input_shape=(6000, 1)))
    model.add(BatchNormalization())
    
    model.add(MaxPooling1D(3))
    
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(BatchNormalization())

    model.add(GRU(64,return_sequences=True))
    model.add(BatchNormalization())
    
    model.add(MaxPooling1D(35))
    model.add(Flatten())
    
    model.add(Dense(units = 128, activation='relu'))

    model.add(Dense(7, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    history=model.fit(X_train_CNN, Y_train, batch_size=64,shuffle=True, epochs=15,verbose=1,validation_data=(X_test_CNN, Y_test))

    score, acc = model.evaluate(X_test_CNN, Y_test, batch_size=16)
    print(acc)

#Evaluation of confusion matrix
pred_classes=model.predict_classes(X_test_CNN)
num_ytest=[ np.where(r==1)[0][0] for r in Y_test ]
confusion=confusion_matrix(num_ytest,pred_classes)
print(confusion)

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

model.save('CNN_GRU.h5')