# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:05:52 2022

@author: Owner
"""

from glob import glob
import pickle
import os
os.chdir("C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/basic classes")
import pandas as pd
import numpy as np
import re
import scipy.io
import scipy.stats as stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
from mat4py import loadmat
import sys
from neuron_class import cell_task,filtCells,load_cell_task,smooth_data
import time
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
import copy
import seaborn as sb
import imageio as iio
from PIL import Image
from PIL import ImageDraw
from scipy.io import savemat
import cv2

#This function gets as input the output of an anova test and calculates the omega_square
def omega_square(aov):
    SS_effect=aov_table.sum_sq['C(dir)']
    df_effect=aov_table.df['C(dir)']
    df_error=aov_table.df['Residual']
    SS_error=aov_table.sum_sq['Residual']
    N=sum(aov_table['df'])+1
    
    num=SS_effect-(df_effect/df_error)*SS_error
    den=SS_effect+(N-df_effect)*SS_error/df_error
    omega_partial=num/den
    return omega_partial

#This function gets as input the output of an anova test and calculates the omega_square
def omega_square_across_time(aov):
    SS_effect=aov_table.sum_sq['C(dir)']+aov_table.sum_sq['C(dir):C(bin_inx)']
    df_effect=aov_table.df['C(dir)']+aov_table.df['C(dir):C(bin_inx)']
    df_error=aov_table.df['Residual']
    SS_error=aov_table.sum_sq['Residual']
    N=sum(aov_table['df'])+1
    
    num=SS_effect-(df_effect/df_error)*SS_error
    den=SS_effect+(N-df_effect)*SS_error/df_error
    omega_partial=num/den
    return omega_partial

path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses"
os.chdir(path)


class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)

def shortest_distance_to_unity_line(x, y):
    # Coefficients for the unity line equation: Ax + By + C = 0
    A = -1
    B = 1
    C = 0

    # Calculate the distance
    distance = abs(A*x + B*y + C) / math.sqrt(A**2 + B**2)

    return distance

#%% 
path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses"
os.chdir(path)

SNR=scipy.io.loadmat('SNR.mat')
SNR=SNR['SNR_array']

cell_db_file='fiona_cells_db.xlsx'
#get list of cells
cell_task_py_folder="units_task_python_two_monkeys/"
cell_pursuit_list=os.listdir(cell_task_py_folder+'8dir_active_passive_interleaved_100_25') #list of strings
cell_saccade_list=os.listdir(cell_task_py_folder+'8dir_saccade_100_25') #list of strings
cell_list=[x for x in cell_saccade_list if x in cell_pursuit_list]
cell_list=[int(item) for item in cell_list] #list of ints



cutoff_cell=8229
yasmin_cell_list=[x for x in cell_list if x>cutoff_cell]
fiona_cell_list=[x for x in cell_list if x<=cutoff_cell]
#cell_list=fiona_cell_list
#cell_list=[9057]

#%%omega squares - scatter plot
task_list=['8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_saccade_100_25']
trial_type_list=['v20a','v20p','filterOff']
event_list=['motion_onset','motion_onset','motion_onset']

# average omega square across cells
window_begin=0
window_end=800
bin_width=100
bin_begins=np.arange(window_begin,window_end,bin_width)
bin_ends=bin_begins+bin_width

omega_sq_mean=np.empty([len(task_list),len(cell_list)])
omega_sq_mean[:]=np.NaN
SNR_array=np.empty([len(cell_list)])
SNR_array[:]=np.NaN

for task_inx,(task,trial_type,event) in enumerate(zip(task_list,trial_type_list,event_list)):
    n_cells=0
    for cell_inx,cell_ID in enumerate(cell_list):        
        cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
        dictFilterTrials = {'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}
        trial_df=cur_cell_task.filtTrials(dictFilterTrials)
        trial_df=trial_df.reset_index()
         
        if task_inx==0:
            #find inx of the cell in SNR array
            cell_SNR_inx=np.where(SNR[:,0]==cell_ID)[0][0]
            cur_SNR=SNR[cell_SNR_inx,1]
            SNR_array[cell_inx]=cur_SNR #for SNR
            #SNR_array[cell_inx]=int(cur_cell_task.getGrade()) # for grade
        
        if len(trial_df)<40:
            continue           
        n_cells=n_cells+1
        

        for bin_inx,(bin_begin,bin_end) in enumerate(zip(bin_begins,bin_ends)):
            #calculate average firing in curent bin
            FR_bin=cur_cell_task.get_mean_FR_event(dictFilterTrials,event,window_pre=bin_begin,window_post=bin_end)
            #add the serie with FR as a column in the trial_df data frame
            trial_df2 = trial_df.assign(FR_bin = FR_bin)
            trial_df2['bin_inx'] = bin_inx
            
            #omega caculation across bins
            if bin_inx==0:
                trial_df_bin_concat=trial_df2
            else:
                trial_df_bin_concat= pd.concat([trial_df_bin_concat,trial_df2])
                
        #2way anova with dir and bin_inx as independant variables        
        model = ols('FR_bin ~ C(dir) + C(bin_inx) + C(dir):C(bin_inx)', data=trial_df_bin_concat).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        omega_sq_mean[task_inx,cell_inx]=omega_square_across_time(aov_table)


#%% scatter with SNR as color
color_map='coolwarm'
up_lim=0.8
down_lim=-0.2
x=np.arange(down_lim,up_lim+0.05,0.05)
#scatter active vs passive
scatter=plt.scatter(omega_sq_mean[0,:],omega_sq_mean[1,:],c=SNR_array,cmap=color_map, alpha=0.7,vmax=10)
plt.plot(x,x,color='black')
plt.xlabel('active')
plt.ylabel('passive')
plt.title('active vs passive')
plt.xlim([down_lim,up_lim])
plt.ylim([down_lim,up_lim])
cbar = plt.colorbar()
cbar.set_label('SNR')
colors_AP = scatter.to_rgba(SNR_array)
plt.show()


#scatter active vs saccade
scatter=plt.scatter(omega_sq_mean[0,:],omega_sq_mean[2,:],c=SNR_array,cmap=color_map, alpha=0.7,vmax=10)
plt.plot(x,x,color='black')
plt.xlabel('active')
plt.ylabel('saccade')
plt.title('active vs saccade')
plt.xlim([down_lim,up_lim])
plt.ylim([down_lim,up_lim])
cbar = plt.colorbar()
cbar.set_label('SNR')
colors_AS = scatter.to_rgba(SNR_array)
plt.show()


#scatter saccade vs passive
scatter=plt.scatter(omega_sq_mean[2,:],omega_sq_mean[1,:],c=SNR_array,cmap=color_map, alpha=0.7,vmax=10)
plt.plot(x,x,color='black')
plt.xlabel('saccade')
plt.ylabel('passive')
plt.title('active vs saccade')
plt.xlim([down_lim,up_lim])
plt.ylim([down_lim,up_lim])
cbar = plt.colorbar()
cbar.set_label('SNR')
colors_AS = scatter.to_rgba(SNR_array)
plt.show()

save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figureR3_omega_SNR/"
mdic = {"scatter_omega_arrays":[omega_sq_mean[0,:],omega_sq_mean[1,:],omega_sq_mean[2,:],SNR_array]}
savemat(save_path+"scatters_omega"+ ".mat", mdic)




#%% calculate distance from unity line
omega_cut_off_value=0
omega_cutoff_inxs=np.logical_and(omega_sq_mean[0,:]>omega_cut_off_value,omega_sq_mean[1,:]>omega_cut_off_value) 
omega_cutoff_inxs2=np.logical_and(omega_sq_mean[0,:]>omega_cut_off_value,omega_sq_mean[2,:]>omega_cut_off_value) 
omega_cutoff_inxs3=np.logical_and(omega_sq_mean[1,:]>omega_cut_off_value,omega_sq_mean[2,:]>omega_cut_off_value) 


distance_active_passive=shortest_distance_to_unity_line(omega_sq_mean[0,:],omega_sq_mean[1,:])
inxs=np.logical_and(omega_cutoff_inxs,~np.isnan(SNR_array)) 
result=scipy.stats.pearsonr(distance_active_passive[inxs],SNR_array[inxs]) 


distance_active_saccade=shortest_distance_to_unity_line(omega_sq_mean[0,:],omega_sq_mean[2,:])
inxs2=np.logical_and(omega_cutoff_inxs2,~np.isnan(SNR_array)) 
result2=scipy.stats.pearsonr(distance_active_saccade[inxs2],SNR_array[inxs2]) 

distance_saccade_suppression=shortest_distance_to_unity_line(omega_sq_mean[2,:],omega_sq_mean[1,:])
inxs3=np.logical_and(omega_cutoff_inxs3,~np.isnan(SNR_array)) 
result3=scipy.stats.pearsonr(distance_saccade_suppression[inxs3],SNR_array[inxs3]) 

#%% SPlit cells into low and high SNR (above and below SNR median)
#remove cells with nan SNR:
SNR_array2=SNR_array[np.where(~np.isnan(SNR_array))[0]]
omega_sq_mean2=omega_sq_mean[:,np.where(~np.isnan(SNR_array))[0]]
inxs=np.argsort(SNR_array2)
omega_sq_mean_ordered=omega_sq_mean2[:,inxs]
SNR_array_ordered=SNR_array2[inxs]

#Cells with low SNR:
median_inx=int(np.floor(np.size(SNR_array_ordered)/2))
omega_sq_mean_low_SNR=omega_sq_mean_ordered[:,0:median_inx]
nonNanInxs=np.where(~np.isnan(omega_sq_mean_low_SNR).any(axis=0))[0].astype(int)
res=scipy.stats.pearsonr(omega_sq_mean_low_SNR[0,nonNanInxs],omega_sq_mean_low_SNR[2,nonNanInxs])
r=str(round(res[0],3)) 
plt.scatter(omega_sq_mean_low_SNR[0,:],omega_sq_mean_low_SNR[2,:])
plt.axline([0,0],[1,1])
plt.title('pursuit vs saccade - low SNR'+' r='+r)
plt.xlabel('active')
plt.ylabel('saccade')
plt.show()

nonNanInxs=np.where(~np.isnan(omega_sq_mean_low_SNR).any(axis=0))[0].astype(int)
res=scipy.stats.pearsonr(omega_sq_mean_low_SNR[0,nonNanInxs],omega_sq_mean_low_SNR[1,nonNanInxs])
r=str(round(res[0],3)) 
plt.scatter(omega_sq_mean_low_SNR[0,:],omega_sq_mean_low_SNR[1,:])
plt.axline([0,0],[1,1])
plt.title('pursuit vs suppression - low SNR'+' r='+r)
plt.xlabel('active')
plt.ylabel('suppression')
plt.show()

nonNanInxs=np.where(~np.isnan(omega_sq_mean_low_SNR).any(axis=0))[0].astype(int)
res=scipy.stats.pearsonr(omega_sq_mean_low_SNR[2,nonNanInxs],omega_sq_mean_low_SNR[1,nonNanInxs])
r=str(round(res[0],3)) 
plt.scatter(omega_sq_mean_low_SNR[2,:],omega_sq_mean_low_SNR[1,:])
plt.axline([0,0],[1,1])
plt.title('pursuit vs suppression - low SNR'+' r='+r)
plt.xlabel('saccade')
plt.ylabel('suppression')

#Cells with high SNR:
omega_sq_mean_high_SNR=omega_sq_mean_ordered[:,median_inx:]
nonNanInxs=np.where(~np.isnan(omega_sq_mean_high_SNR).any(axis=0))[0].astype(int)
res=scipy.stats.pearsonr(omega_sq_mean_high_SNR[0,nonNanInxs],omega_sq_mean_high_SNR[2,nonNanInxs])
r=str(round(res[0],3)) 
plt.scatter(omega_sq_mean_high_SNR[0,:],omega_sq_mean_high_SNR[2,:])
plt.axline([0,0],[1,1])
plt.title('pursuit vs saccade - high SNR'+' r='+r)
plt.xlabel('active')
plt.ylabel('saccade')
plt.show()

nonNanInxs=np.where(~np.isnan(omega_sq_mean_high_SNR).any(axis=0))[0].astype(int)
res=scipy.stats.pearsonr(omega_sq_mean_high_SNR[0,nonNanInxs],omega_sq_mean_high_SNR[1,nonNanInxs])
r=str(round(res[0],3)) 
plt.scatter(omega_sq_mean_high_SNR[0,:],omega_sq_mean_high_SNR[1,:])
plt.axline([0,0],[1,1])
plt.title('pursuit vs suppression - high SNR'+' r='+r)
plt.xlabel('active')
plt.ylabel('suppression')

nonNanInxs=np.where(~np.isnan(omega_sq_mean_high_SNR).any(axis=0))[0].astype(int)
res=scipy.stats.pearsonr(omega_sq_mean_high_SNR[2,nonNanInxs],omega_sq_mean_high_SNR[1,nonNanInxs])
r=str(round(res[0],3)) 
plt.scatter(omega_sq_mean_high_SNR[2,:],omega_sq_mean_high_SNR[1,:])
plt.axline([0,0],[1,1])
plt.title('pursuit vs suppression - high SNR'+' r='+r)
plt.xlabel('saccade')
plt.ylabel('suppression')