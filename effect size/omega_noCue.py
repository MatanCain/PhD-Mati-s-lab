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
from scipy.stats import kruskal


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

#%% 
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
#cell_list=cell_list
#cell_list=[9057]

#%% Cue cells OR NOT CUE CELLS
cur_task='8dir_active_passive_interleaved_100_25'
cur_trial_type='v20a|v20p'
cur_event='cue_onset'

cur_win_begin=0
cur_win_end=350
win_begin_PSTH=0
win_end_PSTH=350

SMOOTH_PSTH_EDGE=200


win_begin_baseline=-300 #for the baseline calculation of the cell
win_end_baseline=-100

PSTH_length=win_end_PSTH-win_begin_PSTH
directions=[0,45,90,135,180,225,270,315]

p_crit=0.05
#create an array task*n_tests*cells
#The first test check is there is a main effect event vs baseline. The second test checks if tuning during event is significant
sig_cue_array=np.zeros([2,len(cell_list)])
for cell_inx,cell_ID in enumerate(cell_list):    
     
     try:
         cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
     except:
         continue
     dictFilterTrials = {'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0}
    
     # Check whether cell is significant for the event relative to baseline       
     try:
         check_main_effect_cond=cur_cell_task.check_main_effect_motion_vs_baseline(cur_event,dictFilterTrials,crit_value=p_crit)
     except:
         check_main_effect_cond=0
     sig_cue_array[0,cell_inx]=check_main_effect_cond
    
    # Check whether cell is significant for the tuning curve          
     try:
        #tuning curve of the cell
        tuning_curve_list=[]
        for cur_dir_inx,cur_direction in enumerate(directions):
            dictFilterTrials['dir']=cur_direction
            try:
                FR_cell_bin_cond=cur_cell_task.get_mean_FR_event(dictFilterTrials,cur_event,window_pre=cur_win_begin,window_post=cur_win_end)
                tuning_curve_list.append(FR_cell_bin_cond)
            except:
                continue           
        #check if tuning is significant:
        try:
            test_result=kruskal(*tuning_curve_list)
            tuning_effect=test_result[1]<p_crit
        except:
            tuning_effect=0   
     except:
         continue
     sig_cue_array[1,cell_inx]=tuning_effect

cue_tuned_cells=sig_cue_array[1,:]
cell_list=np.array(cell_list)[np.where(cue_tuned_cells==0)[0]]
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
#The coordinate of each cell (x,y,z)
cell_coor=np.empty([len(cell_list),3])
cell_coor[:]=np.NaN
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
        omega_sq_mean[task_inx,cell_inx]=omega_square_across_time(aov_table)


up_lim=0.8
down_lim=-0.2
x=np.arange(down_lim,up_lim+0.05,0.05)
#scatter active vs passive
plt.scatter(omega_sq_mean[0,:],omega_sq_mean[1,:])
plt.plot(x,x,color='red')
plt.xlabel('active')
plt.ylabel('passive')
plt.title('active vs passive')
plt.xlim([down_lim,up_lim])
plt.ylim([down_lim,up_lim])
plt.show()


#scatter active vs saccade
plt.scatter(omega_sq_mean[0,:],omega_sq_mean[2,:])
plt.plot(x,x,color='red')
plt.xlabel('active')
plt.ylabel('saccade')
plt.title('active vs saccade')
plt.xlim([down_lim,up_lim])
plt.ylim([down_lim,up_lim])
plt.show()
[stat,pval]=scipy.stats.wilcoxon(omega_sq_mean[0,:],omega_sq_mean[2,:])


#scatter passive vs saccade
plt.scatter(omega_sq_mean[1,:],omega_sq_mean[2,:])
plt.plot(x,x,color='red')
plt.xlabel('passive')
plt.ylabel('saccade')
plt.title('passive vs saccade')
plt.xlim([down_lim,up_lim])
plt.ylim([down_lim,up_lim])
plt.show()
[stat,pval]=scipy.stats.wilcoxon(omega_sq_mean[1,:],omega_sq_mean[2,:])



#remove index of nans
omega_sq_mean_noNan=(omega_sq_mean[:, ~np.isnan(omega_sq_mean).any(axis=0)])
#corr_active_passive
corr1=stats.pearsonr(omega_sq_mean_noNan[0,:],omega_sq_mean_noNan[1,:])
r1=corr1[0]
pval1=corr1[1]
#corr saccade active
corr2=stats.pearsonr(omega_sq_mean_noNan[0,:],omega_sq_mean_noNan[2,:])
r2=corr2[0]
pval2=corr2[1]
#corr saccade passive
corr3=stats.pearsonr(omega_sq_mean_noNan[1,:],omega_sq_mean_noNan[2,:])
r3=corr3[0]
pval3=corr3[1]


# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figureR4_omega_noCue/"
# mdic = {"scatter_omega_arrays":[omega_sq_mean[0,:],omega_sq_mean[1,:],omega_sq_mean[2,:]]}
# savemat(save_path+"scatters_omega"+ ".mat", mdic)


#%% average omega square across cells-dynamics in time


task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
trial_type_array=['filterOff','v20a','v20p','v20a|v20p']
event_array=['motion_onset','motion_onset','motion_onset','cue_onset']
legend_array=['saccade','pursuit active','pursuit passive','cue']

# task_array=['8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
# trial_type_array=['v20a','v20p','v20a|v20p']
# event_array=['cue_onset','cue_onset','cue_onset']
# legend_array=['cue active','cue passive','cue all']


window_begin=-100
window_end=800
bin_width=100
bin_begins=np.arange(window_begin,window_end,bin_width)
bin_ends=bin_begins+bin_width

#each row of omega_sq is for a cell, each column is a bin
omega_sq=np.empty([len(bin_begins),len(cell_list),len(task_array)])
omega_sq[:]=np.NaN  
 
for cond_inx in np.arange(len(task_array)):
    event=event_array[cond_inx]
    task=task_array[cond_inx]
    trial_type=trial_type_array[cond_inx]
 

    n_cells=0
    for cell_inx,cell_ID in enumerate(cell_list):
        cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
        dictFilterTrials = {'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}
        trial_df=cur_cell_task.filtTrials(dictFilterTrials)
        trial_df=trial_df.reset_index()
        
        if len(trial_df)<40:
            continue
    
        n_cells=n_cells+1
        #initialize omega_sq
        omega_sq_cur_cell=np.empty([len(bin_begins)])
        omega_sq_cur_cell[:]=np.NaN
        
        for bin_inx,(bin_begin,bin_end) in enumerate(zip(bin_begins,bin_ends)):
            #calculate average firing in curent bin
            FR_bin=cur_cell_task.get_mean_FR_event(dictFilterTrials,event,window_pre=bin_begin,window_post=bin_end)
            #add the serie with FR as a column in the trial_df data frame
            trial_df2 = trial_df.assign(FR_bin = FR_bin)
            trial_df2['bin_inx'] = bin_inx
            
            #omega square for current bin
            #anova with dir as independant variable
            model = ols('FR_bin ~ C(dir)', data=trial_df2).fit()
            aov_table = sm.stats.anova_lm(model, typ=2)
            omega_sq_cur_cell[bin_inx]=omega_square(aov_table)
        omega_sq[:,cell_inx,cond_inx]=omega_sq_cur_cell
    
#omega square as a function of time
mean_omega=np.nanmean(omega_sq, axis=1)
sem_omega=np.round(stats.sem(omega_sq, axis=1,nan_policy='omit'),3)
plt.plot(bin_begins,mean_omega)
plt.title('mean omega ')
plt.xlabel('time from event')
plt.ylabel('omega^2')
plt.legend(legend_array)
plt.axvline(x=0,color='red')
#plt.ylim([-0.01,0.12])
plt.show()
    
    
# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure2_omega/"
# mdic = {"omega_dynamics":mean_omega,'bin_begin':bin_begin}
# savemat(save_path+"omega_dynamics"+ ".mat", mdic)

# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure6_cue/"
# mdic = {"omega_dynamics_cue":mean_omega,'bin_begin':bin_begin}
# savemat(save_path+"omega_dynamics_cue"+ ".mat", mdic)


        

