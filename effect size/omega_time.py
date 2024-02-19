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
def omega_square_across_time(aov):
    SS_effect=aov_table.sum_sq['C(bin_inx)']
    df_effect=aov_table.df['C(bin_inx)']
    df_error=aov_table.df['Residual']
    SS_error=aov_table.sum_sq['Residual']
    N=sum(aov_table['df'])+1
    
    num=SS_effect-(df_effect/df_error)*SS_error
    den=SS_effect+(N-df_effect)*SS_error/df_error
    omega_partial=num/den
    return omega_partial

# # #This function gets as input the output of an anova test and calculates the omega_square
# def omega_square_across_time(aov):
#     SS_effect=aov_table.sum_sq['C(bin_inx)']+aov_table.sum_sq['C(dir):C(bin_inx)']
#     df_effect=aov_table.df['C(bin_inx)']+aov_table.df['C(dir):C(bin_inx)']
#     df_error=aov_table.df['Residual']
#     SS_error=aov_table.sum_sq['Residual']
#     N=sum(aov_table['df'])+1
    
#     num=SS_effect-(df_effect/df_error)*SS_error
#     den=SS_effect+(N-df_effect)*SS_error/df_error
#     omega_partial=num/den
#     return omega_partial


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
cell_list=fiona_cell_list
#cell_list=[9057]

#%%omega squares - scatter plot
# task_list=['8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
# trial_type_list=['v20a','v20p','filterOff','v20a|v20p','v20a','v20p']
# event_list=['motion_onset','motion_onset','motion_onset','cue_onset','cue_onset','cue_onset']

task_list=['8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_saccade_100_25']
trial_type_list=['v20a','v20p','filterOff']
event_list=['motion_onset','motion_onset','motion_onset']

# average omega square across cells
window_begin=0
window_end=800
bin_width=100
bin_begins=np.arange(window_begin,window_end,bin_width)
bin_ends=bin_begins+bin_width

omega_sq_mean=np.empty([len(task_list)+1,len(cell_list)])
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
        omega_sq_mean[task_inx,cell_inx]=omega_square_across_time(aov_table)
        
        omega_sq_mean[-1,cell_inx]=cell_ID


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

# #scatter cue active vs saccade
# plt.scatter(omega_sq_mean[4,:],omega_sq_mean[2,:])
# plt.plot(x,x,color='red')
# plt.xlabel('cue')
# plt.ylabel('saccade')
# plt.title('cue vs saccade')
# plt.xlim([down_lim,up_lim])
# plt.ylim([down_lim,up_lim])
# plt.show()
# [stat,pval]=scipy.stats.wilcoxon(omega_sq_mean[2,:],omega_sq_mean[4,:])

# #scatter cue active vs pursuit
# plt.scatter(omega_sq_mean[4,:],omega_sq_mean[0,:])
# plt.plot(x,x,color='red')
# plt.xlabel('cue')
# plt.ylabel('pursuit')
# plt.title('cue vs pursuit')
# plt.xlim([down_lim,up_lim])
# plt.ylim([down_lim,up_lim])
# plt.show()
# [stat,pval]=scipy.stats.wilcoxon(omega_sq_mean[0,:],omega_sq_mean[4,:])

# #scatter cue active vs suppression
# plt.scatter(omega_sq_mean[4,:],omega_sq_mean[1,:])
# plt.plot(x,x,color='red')
# plt.xlabel('cue')
# plt.ylabel('suppression')
# plt.title('cue vs suppression')
# plt.xlim([down_lim,up_lim])
# plt.ylim([down_lim,up_lim])
# plt.show()
# [stat,pval]=scipy.stats.wilcoxon(omega_sq_mean[1,:],omega_sq_mean[4,:])

# #scatter cue active vs cue suppression
# plt.scatter(omega_sq_mean[4,:],omega_sq_mean[5,:])
# plt.plot(x,x,color='red')
# plt.xlabel('cue active')
# plt.ylabel('cue suppression')
# plt.title('cue active vs cue suppression')
# plt.xlim([down_lim,up_lim])
# plt.ylim([down_lim,up_lim])
# plt.show()
# [stat,pval]=scipy.stats.wilcoxon(omega_sq_mean[4,:],omega_sq_mean[5,:])

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
# #corr saccade cue active
# corr4=stats.pearsonr(omega_sq_mean_noNan[4,:],omega_sq_mean_noNan[2,:])
# r4=corr4[0]
# pval4=corr4[1]
# #corr pursuit cue active
# corr5=stats.pearsonr(omega_sq_mean_noNan[4,:],omega_sq_mean_noNan[0,:])
# r5=corr5[0]
# pval5=corr5[1]
# #corr supression cue active
# corr6=stats.pearsonr(omega_sq_mean_noNan[4,:],omega_sq_mean_noNan[1,:])
# r6=corr6[0]
# pval6=corr6[1]
# #corr cue active vs cue supression
# corr7=stats.pearsonr(omega_sq_mean_noNan[4,:],omega_sq_mean_noNan[5,:])
# r7=corr7[0]
# pval7=corr7[1]

# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figureR5_omega_time/"
# mdic = {"scatter_omega_arrays":[omega_sq_mean[0,:],omega_sq_mean[1,:],omega_sq_mean[2,:],omega_sq_mean[3,:]]}
# savemat(save_path+"scatters_omega"+ ".mat", mdic)

#%%sanity check
# cell_ID=7626
# task='8dir_active_passive_interleaved_100_25'
# cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
# dictFilterTrials = {'dir':'filterOff', 'trial_name':'v20p', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}
# window_PSTH={"timePoint":'motion_onset','timeBefore':-100,'timeAfter':800}
# psth=cur_cell_task.PSTH(window_PSTH,dictFilterTrials)
# plt.plot(psth)
# plt.show()