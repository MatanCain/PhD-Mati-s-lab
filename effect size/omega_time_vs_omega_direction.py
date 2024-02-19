# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:05:52 2022

@author: Owner
"""
#This script checks the correlation between spatial and temporal tuning

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


# temporal omega_square
def omega_square_across_time_temporal(aov): 
    SS_effect=aov_table.sum_sq['C(bin_inx)']
    df_effect=aov_table.df['C(bin_inx)']
    df_error=aov_table.df['Residual']
    SS_error=aov_table.sum_sq['Residual']
    N=sum(aov_table['df'])+1
    
    num=SS_effect-(df_effect/df_error)*SS_error
    den=SS_effect+(N-df_effect)*SS_error/df_error
    omega_partial=num/den
    return omega_partial

#  directional omega_square
def omega_square_across_time_direction(aov):
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

#%% List of cells
path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses"
os.chdir(path)

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
cell_list=cell_list
#cell_list=[9057]

#%%omega squares - scatter plot
task_list=['8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_saccade_100_25','8dir_active_passive_interleaved_100_25']
trial_type_list=['v20a','v20p','filterOff','v20a|v20p']
event_list=['motion_onset','motion_onset','motion_onset','cue_onset']
title_list=['pursuit','passive','saccade','cue']


# task_list=['8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_saccade_100_25']
# trial_type_list=['v20a','v20p','filterOff']
# event_list=['motion_onset','motion_onset','motion_onset']
# title_list=['pursuit','passive','saccade']

# average omega square across cells
window_begin=0
window_end=800
bin_width=100
bin_begins=np.arange(window_begin,window_end,bin_width)
bin_ends=bin_begins+bin_width

omega_sq_mean=np.empty([len(task_list)+1,len(cell_list),2])
omega_sq_mean[:]=np.NaN

for task_inx,(task,trial_type,event) in enumerate(zip(task_list,trial_type_list,event_list)):
    n_cells=0
    for cell_inx,cell_ID in enumerate(cell_list):        
        cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
        dictFilterTrials = {'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}
        trial_df=cur_cell_task.filtTrials(dictFilterTrials)
        trial_df=trial_df.reset_index()
        
        if len(trial_df)<40:
            continue
            
    
        n_cells=n_cells+1
           
        #cell coordinates        
        #if task_inx==0:
            #cell_coor[cell_inx,:]=cur_cell_task.XYZCoor
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
        omega_sq_mean[task_inx,cell_inx,0]=omega_square_across_time_temporal(aov_table)
        omega_sq_mean[task_inx,cell_inx,1]=omega_square_across_time_direction(aov_table)

        omega_sq_mean[-1,cell_inx]=cell_ID

#%% Plot the correlation in omega squeres for each task
for task_inx,title in enumerate(title_list):
    cur_omega_sq_mean=omega_sq_mean[task_inx, :,:]
    
    omega_sq_mean_noNan=cur_omega_sq_mean[~np.isnan(cur_omega_sq_mean).any(axis=1),:]
    temporal_omega_array=omega_sq_mean_noNan[:,0]
    directional_omega_array=omega_sq_mean_noNan[:,1]
    
    corr=stats.pearsonr(temporal_omega_array,directional_omega_array)
    r=round(corr[0],3)
    p=corr[1]    
    plt.scatter(temporal_omega_array,directional_omega_array)
    plt.axline([0,0],[1,1],color='black')
    plt.xlabel('temporal tuning')
    plt.ylabel('directional tuning')
    plt.title(title+' r='+str(r))
    plt.show()
    
#%%sanity check
task='8dir_saccade_100_25'
window_PSTH={"timePoint":'motion_onset','timeBefore':-100,'timeAfter':800}
dictFilterTrials = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}
directions=[0,45,90,135,180,225,270,315]


for cell_inx,cell_ID in enumerate(cell_list):
    cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)

    temporal_omega=omega_sq_mean[0,cell_inx,0]
    temporal_omega=round(temporal_omega,3)
    direction_omega=omega_sq_mean[0,cell_inx,1]
    direction_omega=round(direction_omega,3)
    
    if np.isnan(direction_omega) or np.isnan(temporal_omega):
        continue
    if temporal_omega<0.05 and direction_omega<0.05:

    
        for direction in directions: 
            dictFilterTrials['dir']=direction
            psth=cur_cell_task.PSTH(window_PSTH,dictFilterTrials)
            plt.plot(psth)
        plt.legend(directions)
        plt.title('t:'+str(temporal_omega)+' d:'+str(direction_omega))    
        plt.show()


#%%
task_inx=0
cur_omega_sq_mean=omega_sq_mean[task_inx, :,:]

omega_sq_mean_noNan=cur_omega_sq_mean[~np.isnan(cur_omega_sq_mean).any(axis=1),:]
temporal_omega_array=omega_sq_mean_noNan[:,0]
directional_omega_array=omega_sq_mean_noNan[:,1]

corr=stats.spearmanr(temporal_omega_array,directional_omega_array)
r=round(corr[0],3)
p=corr[1]    
plt.scatter(temporal_omega_array,directional_omega_array)
plt.axline([0,0],[1,1],color='black')
plt.xlabel('temporal tuning')
plt.ylabel('directional tuning')
plt.title(title+' r='+str(r))
plt.show()
