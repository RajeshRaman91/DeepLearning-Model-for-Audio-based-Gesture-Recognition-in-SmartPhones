# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:34:00 2018

@author: Rajesh
"""
# importing relevant packages
import pandas as pd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt 
import tensorflow as tf

def input_preprocessing(Total_data1,Total_data2,Total_Label_TS):   
    """
    Circle  ---1
    Plus    ---2
    V       ---3
    Double  ---4
    One     ---5
    Divide  ---6
    Noise   ---7
    """
    #initially started with 11-classes, upon analysis and observation of results reduced to 7-classes
    label={'Circle':1,'Plus':2,'V':3,'Double':4,'One':5,'Divide':6,'Noise':7}
    invlabel=dict([[v,k] for k,v in label.items()])# inverse label function, can be used to perform inverse check   	
    Data_withLabel=[]
    Data_withLabel1=[]
    temp=[]
    temp1=[]
    data_temp=[]
    data_temp1=[]
    label_check=[]
    X_data_temp=[]
    X_data_temp1=[]
    X_data_circle=[]
    X_data_circle1=[]
    X_data_plus=[]
    X_data_plus1=[]
    X_data_double=[]
    X_data_double1=[]
    X_data_one=[]
    X_data_one1=[]
    X_data_divide=[]
    X_data_divide1=[]
    X_data_v=[]
    X_data_v1=[]
    X_data_noise=[]
    X_data_noise1=[]
    i=0
    j=0
    Case_Sw=None
    for t_data1,t_data2,t_Label_TS in zip(Total_data1,Total_data2,Total_Label_TS):
        for j in range(0,len(t_Label_TS),1):
            #time is adjusted as per sampling rate - 44,100 Hz
            start=float(t_Label_TS[j][1])*44.1  
            stop= float(t_Label_TS[j][2])*44.1
            Case_Sw=t_Label_TS[j][0]
            X_data_temp=t_data1[int(start+10000):int(stop-10000)]# trimming off of first and last 10000 samples
            X_data_temp1=t_data2[int(start+10000):int(stop-10000)]
    			
            for t in range (0, len(X_data_temp),16):# down sampling by a factor of 16
                data_temp.append(X_data_temp[t])
                data_temp1.append(X_data_temp1[t])
            X_data_temp=data_temp
            X_data_temp1=data_temp1
            data_temp=[]
            data_temp1=[]
            # Parsing and accumulation of identical gestures
            if Case_Sw == "Circle":
                X_data_circle.append(X_data_temp)
                X_data_circle1.append(X_data_temp1)
                temp=[X_data_temp,label[Case_Sw]]
                Data_withLabel.append(temp)
                label_check.append(Case_Sw)
                temp1=[X_data_temp1,label[Case_Sw]]
                Data_withLabel1.append(temp1)
                temp=[]
                temp1=[]
            elif Case_Sw=="Plus":
                X_data_plus.append(X_data_temp)
                X_data_plus1.append(X_data_temp1)
                temp=[X_data_temp,label[Case_Sw]]
                Data_withLabel.append(temp)
                label_check.append(Case_Sw)
                temp1=[X_data_temp1,label[Case_Sw]]
                Data_withLabel1.append(temp1)
                temp=[]
                temp1=[]
            elif Case_Sw=="V":
                X_data_v.append(X_data_temp)
                X_data_v1.append(X_data_temp1)
                temp=[X_data_temp,label[Case_Sw]]
                Data_withLabel.append(temp)
                label_check.append(Case_Sw)
                temp1=[X_data_temp1,label[Case_Sw]]
                Data_withLabel1.append(temp1)
                temp=[]
                temp1=[]
            elif Case_Sw=="Double":
                X_data_double.append(X_data_temp)
                X_data_double1.append(X_data_temp1)
                temp=[X_data_temp,label[Case_Sw]]
                Data_withLabel.append(temp)
                label_check.append(Case_Sw)
                temp1=[X_data_temp1,label[Case_Sw]]
                Data_withLabel1.append(temp1)
                temp=[]
                temp1=[]
            elif Case_Sw=="One":
                X_data_one.append(X_data_temp)
                X_data_one1.append(X_data_temp1)
                temp=[X_data_temp,label[Case_Sw]]
                Data_withLabel.append(temp)
                label_check.append(Case_Sw)
                temp1=[X_data_temp1,label[Case_Sw]]
                Data_withLabel1.append(temp1)
                temp=[]
                temp1=[]
            elif Case_Sw=="Divide":
                X_data_divide.append(X_data_temp)
                X_data_divide1.append(X_data_temp1)
                temp=[X_data_temp,label[Case_Sw]]
                Data_withLabel.append(temp)
                label_check.append(Case_Sw)
                temp1=[X_data_temp1,label[Case_Sw]]
                Data_withLabel1.append(temp1)
                temp=[]
                temp1=[]
            elif Case_Sw=="Noise":
                X_data_noise.append(X_data_temp)
                X_data_noise1.append(X_data_temp1)
                temp=[X_data_temp,label[Case_Sw]]
                Data_withLabel.append(temp)
                label_check.append(Case_Sw)
                temp1=[X_data_temp1,label[Case_Sw]]
                Data_withLabel1.append(temp1)
                temp=[]
                temp1=[]
    return Data_withLabel,Data_withLabel1
		
#Visualization of Gesture audio patterns, data: array of gestures, num: first n- numbers of gestures which will be displayed

def Gesture_Visualization(data,num):
	for j in range(0,num,1):
		plt.plot(data[j])
		plt.ylabel('Amplitude')
		plt.xlabel('time')
		plt.title("Circle_mic1"+'_'+str(j))#change the corresponding name
		plt.show()
		print(len(data[j]))

#Extracting MIC-1 data, one hot code vector generation for y-labels, splitting of training and testing samples, slicing upto 0:6000 units, 
#performing CNN based input dimension compatibility  
def input_preparation(Data_withLabel,Data_withLabel1):

	x_label_totmic1, y_label_tot = map(list, zip(*Data_withLabel))#Extracting MIC-1 data
	x_label_totmic2, y_label_tot_dummy = map(list, zip(*Data_withLabel1))
	
	y_label_tot=np.array(y_label_tot)
	y_train=pd.get_dummies(y_label_tot).values#one hot code vector generation for y-labels
	
	x_label_tot1=x_label_totmic1
	x_label_tot2=x_label_totmic2

	n_train=len(x_label_tot1)-1500 #splitting of training and testing samples, set the number of test samples, here :1500
	n_total=len(x_label_tot1)
	
	x_label_tot1_train=[]
	x_label_tot2_train=[]
	for j in range(0,n_train,1):
		temp_train1 = x_label_tot1[j][0:6000]
		temp_train2 = x_label_tot2[j][0:6000]
		x_label_tot1_train.append(temp_train1)
		x_label_tot2_train.append(temp_train2)#slicing upto 0:6000 units,

	x_label_tot1_test=[]
	x_label_tot2_test=[]
	for j in range(n_train,n_total,1):
		temp_test1 = x_label_tot1[j][0:6000]
		temp_test2 = x_label_tot2[j][0:6000]
		x_label_tot1_test.append(temp_test1)
		x_label_tot2_test.append(temp_test2)
	
	X_train1=x_label_tot1_train
	X_train2=x_label_tot2_train
	X_test1=x_label_tot1_test
	X_test2=x_label_tot2_test

	y_train1=y_train[0:n_train]
	y_test1=y_train[n_train:n_total]
	
	length = len(sorted(X_train1,key=len, reverse=True)[0])
	X_train=np.array([xi+[0]*(length-len(xi)) for xi in X_train1])# filling up with zeros
	print(X_train.shape)

	length = len(sorted(X_test1,key=len, reverse=True)[0])
	X_test=np.array([xi+[0]*(length-len(xi)) for xi in X_test1])
	X_test.shape
	
	X_train_CNN = np.expand_dims(X_train, axis=2)
	X_test_CNN = np.expand_dims(X_test, axis=2)#performing CNN based input dimension compatibility 
	Y_train=np.array(y_train1)
	Y_test=np.array(y_test1)
	
	return X_train_CNN,X_test_CNN,Y_train,Y_test

#Extracting MIC-1 data, one hot code vector generation for y-labels, splitting of training and testing samples, slicing upto 0:6000 units, 
#performing LSTM based input dimension compatibility  
def input_preparation_LSTM(Data_withLabel,Data_withLabel1):

	x_label_totmic1, y_label_tot = map(list, zip(*Data_withLabel))#Extracting MIC-1 data
	x_label_totmic2, y_label_tot_dummy = map(list, zip(*Data_withLabel1))
	
	y_label_tot=np.array(y_label_tot)
	y_train=pd.get_dummies(y_label_tot).values#one hot code vector generation for y-labels
	
	x_label_tot1=x_label_totmic1
	x_label_tot2=x_label_totmic2

	n_train=len(x_label_tot1)-1500 #splitting of training and testing samples, set the number of test samples, here :1500
	n_total=len(x_label_tot1)
	
	x_label_tot1_train=[]
	x_label_tot2_train=[]
	for j in range(0,n_train,1):
		temp_train1 = x_label_tot1[j][0:6000]
		temp_train2 = x_label_tot2[j][0:6000]
		x_label_tot1_train.append(temp_train1)
		x_label_tot2_train.append(temp_train2)#slicing upto 0:6000 units,

	x_label_tot1_test=[]
	x_label_tot2_test=[]
	for j in range(n_train,n_total,1):
		temp_test1 = x_label_tot1[j][0:6000]
		temp_test2 = x_label_tot2[j][0:6000]
		x_label_tot1_test.append(temp_test1)
		x_label_tot2_test.append(temp_test2)
	
	X_train1=x_label_tot1_train
	X_train2=x_label_tot2_train
	X_test1=x_label_tot1_test
	X_test2=x_label_tot2_test

	y_train1=y_train[0:n_train]
	y_test1=y_train[n_train:n_total]
	
	length = len(sorted(X_train1,key=len, reverse=True)[0])
	X_train=np.array([xi+[0]*(length-len(xi)) for xi in X_train1])# filling up with zeros
	print(X_train.shape)

	length = len(sorted(X_test1,key=len, reverse=True)[0])
	X_test=np.array([xi+[0]*(length-len(xi)) for xi in X_test1])
	X_test.shape
	Y_train=np.array(y_train1)
	Y_test=np.array(y_test1)
	
	time_steps=6
	div_seq=1000
	batch_size=32


	x_train_TD = [x.reshape((-1, time_steps, div_seq)) for x in X_train]
	x_train_TD = np.array(x_train_TD).reshape((-1,  time_steps,div_seq))
	print(x_train_TD.shape)

	x_test_TD = [x.reshape((-1, time_steps, div_seq)) for x in X_test]
	x_test_TD = np.array(x_test_TD).reshape((-1,  time_steps,div_seq))
	print(x_test_TD.shape)
	
	Y_train_stack= np.column_stack([Y_train, Y_train,Y_train,Y_train,Y_train,Y_train])
	Y_test_stack=np.column_stack([Y_test, Y_test,Y_test,Y_test,Y_test,Y_test])
	y_train_TD = [x.reshape((-1, time_steps, 7)) for x in Y_train_stack]
	y_train_TD = np.reshape(y_train_TD, (Y_train.shape[0],6, 7))
	y_test_TD = [x.reshape((-1, time_steps, 7)) for x in Y_test_stack]
	y_test_TD = np.reshape(y_test_TD, (Y_test.shape[0],6, 7))
	
	return x_train_TD,x_test_TD,y_train_TD,y_test_TD