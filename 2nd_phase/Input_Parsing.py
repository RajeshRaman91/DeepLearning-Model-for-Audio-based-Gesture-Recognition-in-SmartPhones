# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 12:34:00 2018

@author: Rajesh
"""
# importing relevant packages
from scipy.io import wavfile
import csv
import glob  

def input_parsing():
	#Accumulating , Categorizing and Sorting  .csv files - User data and Time stamp files
	T_sync=[]
	Ini_csv=[]
	path1 = '*.csv'
	files=glob.glob(path1)
	for f in files:
		if '_TimeSync.csv' in f:
			T_sync.append(f)
		elif '_TimeSync_1.csv' in f:
			T_sync.append(f)
		else:
			Ini_csv.append(f)
			
	T_sync.sort();
	Ini_csv.sort();

	#Start time and Stop time parsing for the gestures

	start_time=[]
	stop_time=[]
	for t in Ini_csv:
		with open(t, newline='') as f:
			reader = csv.reader(f)
			for row in reader:       
				print(row)
			start_time.append(row[4])
			stop_time.append(row[5])
			f.close()

	#Evaluating delta time and extracting corresponding Gesture label name

	Total_Gestures=[]
	Total_DeltaTime=[]
	for count,t in enumerate (T_sync):
		with open(t, newline='') as f:
			reader = csv.reader(f)
			Gestures=[]
			DeltaTimes=[]
			i=0
			for row in reader: 
				if i==1:
					Gesture=row[0]
					CurrentTime=row[1]
					DeltaTime=int(CurrentTime)-int(start_time[count])
					Gestures.append(Gesture)
					DeltaTimes.append(DeltaTime)
				i=1
			Total_Gestures.append(Gestures)
			Total_DeltaTime.append(DeltaTimes)
			f.close()

	#Gesture parsing and  accumulation		
	Total_Ges1=[]
	for Gest1 in Total_Gestures:    
		Ges1=[]
		Ges2=[]
		flg=0
		flg1=1
		for k in Gest1:
			if flg==1:
				a,b=k.split('_',1)
				if flg1==1:
					Ges1.append(a)
					flg1=0
				else:
					flg1=1
			flg=1        
		Total_Ges1.append(Ges1)

	#Time Stamp parsing and accumulation
	Total_Start_ti=[]
	Total_Stop_ti=[]
	for DeltaTimes in Total_DeltaTime:
		Start_ti=[]
		Stop_ti=[]
		flg=1
		for k in DeltaTimes:
			if flg==1:
				Start_ti.append(k)
				flg=0
			elif flg==0:
				Stop_ti.append(k)
				flg=1
		Total_Start_ti.append(Start_ti)
		Total_Stop_ti.append(Stop_ti)

	#Time Stamp, label Merging
	Total_Label_TS=[]
	Label_TS_temp=[]

	flg=0
	i=0

	for Gest_sample,t_start,t_stop in zip(Total_Ges1,Total_Start_ti,Total_Stop_ti):
		Label_TS=[]
		for k,t1,t2 in zip(Gest_sample,t_start,t_stop):
			if flg==0:
				Label_TS_temp=[k,t1,t2]
				Label_TS.append(Label_TS_temp)
		Total_Label_TS.append(Label_TS)

	# Audio wave file extracting and accumulation
	Total_data=[]
	path2 = '*.wav'
	files=glob.glob(path2)
	files.sort();
	for f in files:
		fs,data=wavfile.read(f);
		Total_data.append(data)

	Total_data1=[]
	Total_data2=[]
	for t_data in Total_data:
		Total_data1.append(t_data[:,0])
		Total_data2.append(t_data[:,1])
	
	return Total_data1,Total_data2,Total_Label_TS