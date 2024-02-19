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
cell_list=cell_list
#cell_list=[9057]

#%%omega squares - scatter plot
task_list=['8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_saccade_100_25']
trial_type_list=['v20a','v20p','filterOff']
event_list=['motion_onset','motion_onset','motion_onset']

# task_list=['8dir_saccade_100_25']
# trial_type_list=['filterOff']
# event_list=['motion_onset']

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
        # cur_grade=int(cur_cell_task.getGrade())
        # if cur_grade<=7:
        #     continue
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

#remove index of nans
omega_sq_mean_noNan=(omega_sq_mean[:, ~np.isnan(omega_sq_mean).any(axis=0)])


up_lim=0.8
down_lim=-0.2
x=np.arange(down_lim,up_lim+0.05,0.05)
#scatter active vs passive
plt.scatter(omega_sq_mean_noNan[0,:],omega_sq_mean_noNan[1,:])
plt.plot(x,x,color='red')
plt.xlabel('active')
plt.ylabel('passive')
plt.title('active vs passive')
plt.xlim([down_lim,up_lim])
plt.ylim([down_lim,up_lim])
plt.show()


#scatter active vs saccade
plt.scatter(omega_sq_mean_noNan[0,:],omega_sq_mean_noNan[2,:])
plt.plot(x,x,color='red')
plt.xlabel('active')
plt.ylabel('saccade')
plt.title('active vs saccade')
plt.xlim([down_lim,up_lim])
plt.ylim([down_lim,up_lim])
plt.show()
[stat,pval]=scipy.stats.wilcoxon(omega_sq_mean_noNan[0,:],omega_sq_mean_noNan[2,:])


#scatter passive vs saccade
plt.scatter(omega_sq_mean_noNan[1,:],omega_sq_mean_noNan[2,:])
plt.plot(x,x,color='red')
plt.xlabel('passive')
plt.ylabel('saccade')
plt.title('passive vs saccade')
plt.xlim([down_lim,up_lim])
plt.ylim([down_lim,up_lim])
plt.show()
[stat,pval]=scipy.stats.wilcoxon(omega_sq_mean_noNan[1,:],omega_sq_mean_noNan[2,:])

# #scatter cue active vs saccade
# plt.scatter(omega_sq_mean_noNan[4,:],omega_sq_mean_noNan[2,:])
# plt.plot(x,x,color='red')
# plt.xlabel('cue')
# plt.ylabel('saccade')
# plt.title('cue vs saccade')
# plt.xlim([down_lim,up_lim])
# plt.ylim([down_lim,up_lim])
# plt.show()
# [stat,pval]=scipy.stats.wilcoxon(omega_sq_mean_noNan[2,:],omega_sq_mean_noNan[4,:])

# #scatter cue active vs pursuit
# plt.scatter(omega_sq_mean_noNan[4,:],omega_sq_mean_noNan[0,:])
# plt.plot(x,x,color='red')
# plt.xlabel('cue')
# plt.ylabel('pursuit')
# plt.title('cue vs pursuit')
# plt.xlim([down_lim,up_lim])
# plt.ylim([down_lim,up_lim])
# plt.show()
# [stat,pval]=scipy.stats.wilcoxon(omega_sq_mean_noNan[0,:],omega_sq_mean_noNan[4,:])

# #scatter cue active vs suppression
# plt.scatter(omega_sq_mean_noNan[4,:],omega_sq_mean_noNan[1,:])
# plt.plot(x,x,color='red')
# plt.xlabel('cue')
# plt.ylabel('suppression')
# plt.title('cue vs suppression')
# plt.xlim([down_lim,up_lim])
# plt.ylim([down_lim,up_lim])
# plt.show()
# [stat,pval]=scipy.stats.wilcoxon(omega_sq_mean_noNan[1,:],omega_sq_mean_noNan[4,:])

# #scatter cue active vs cue suppression
# plt.scatter(omega_sq_mean_noNan[4,:],omega_sq_mean_noNan[5,:])
# plt.plot(x,x,color='red')
# plt.xlabel('cue active')
# plt.ylabel('cue suppression')
# plt.title('cue active vs cue suppression')
# plt.xlim([down_lim,up_lim])
# plt.ylim([down_lim,up_lim])
# plt.show()
# [stat,pval]=scipy.stats.wilcoxon(omega_sq_mean_noNan[4,:],omega_sq_mean_noNan[5,:])


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
# corr4=stats.spearmanr(omega_sq_mean_noNan[4,:],omega_sq_mean_noNan[2,:])
# r4=corr4[0]
# pval4=corr4[1]
# #corr pursuit cue active
# corr5=stats.spearmanr(omega_sq_mean_noNan[4,:],omega_sq_mean_noNan[0,:])
# r5=corr5[0]
# pval5=corr5[1]
# #corr supression cue active
# corr6=stats.spearmanr(omega_sq_mean_noNan[4,:],omega_sq_mean_noNan[1,:])
# r6=corr6[0]
# pval6=corr6[1]
# #corr cue active vs cue supression
# corr7=stats.spearmanr(omega_sq_mean_noNan[4,:],omega_sq_mean_noNan[5,:])
# r7=corr7[0]
# pval7=corr7[1]

# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure3_omega/"
# mdic = {"scatter_omega_arrays":[omega_sq_mean[0,:],omega_sq_mean[1,:],omega_sq_mean[2,:],omega_sq_mean[3,:],omega_sq_mean[4,:],omega_sq_mean[5,:]]}
# savemat(save_path+"scatters_omega"+ ".mat", mdic)


    
#%% average omega square across cells-dynamics in time


task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
trial_type_array=['filterOff','v20a','v20p']
event_array=['motion_onset','motion_onset','motion_onset']
legend_array=['saccade','pursuit active','pursuit passive']

# task_array=['8dir_saccade_100_25']
# trial_type_array=['filterOff']
# event_array=['motion_onset']
# legend_array=['saccade']


nonNanInxs=np.where(~np.isnan(omega_sq_mean).any(axis=0))[0].astype(int)
cell_list2=[cell_list[index] for index in nonNanInxs]

window_begin=-100
window_end=800
bin_width=100
bin_begins=np.arange(window_begin,window_end,bin_width)
bin_ends=bin_begins+bin_width

#each row of omega_sq is for a cell, each column is a bin
omega_sq=np.empty([len(bin_begins),len(cell_list2),len(task_array)])
omega_sq[:]=np.NaN  
 
for cond_inx in np.arange(len(task_array)):
    event=event_array[cond_inx]
    task=task_array[cond_inx]
    trial_type=trial_type_array[cond_inx]
 

    n_cells=0
    for cell_inx,cell_ID in enumerate(cell_list2):
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
    
#check for each bin if effectsize in pursuit is different than saccade or supression
#remove index of nans
sac_pursuit=[]
sac_sup=[]
sup_pursuit=[]
for bin_inx,(bin_begin,bin_end) in enumerate(zip(bin_begins,bin_ends)):
    omega_sq2=omega_sq[bin_inx,:,:] 
    omega_sq2=omega_sq2[~np.isnan(omega_sq2).any(axis=1),:]
    pursuit_array=omega_sq2[:,1]
    saccade_array=omega_sq2[:,0]
    suppression_array=omega_sq2[:,2]
    res=stats.ttest_ind(pursuit_array,saccade_array,equal_var=False)
    sac_pursuit.append(res.pvalue<0.01)
    res2=stats.ttest_ind(pursuit_array,suppression_array,equal_var=False)
    sup_pursuit.append(res2.pvalue<0.01)
    res3=stats.ttest_ind(saccade_array,suppression_array,equal_var=False)
    sac_sup.append(res3.pvalue<0.01)

   
omega_sq2=omega_sq[0,:,:]
#omega_sq2=omega_sq2[~np.isnan(omega_sq2).any(axis=1),:]
pursuit_array=omega_sq2[:,1]
saccade_array=omega_sq2[:,0] 
plt.scatter(pursuit_array,saccade_array)
plt.axline([0,0],[1,1])
plt.show()
# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure2_omega/"
# mdic = {"omega_dynamics":medan_omega,'bin_begin':bin_begin}
# savemat(save_path+"omega_dynamics"+ ".mat", mdic)

# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure6_cue/"
# mdic = {"omega_dynamics_cue":mean_omega,'bin_begin':bin_begin}
# savemat(save_path+"omega_dynamics_cue"+ ".mat", mdic)


#%% find the omega dynamics of a given cell
# cell_ID=7720
# cell_inx=[inx for inx,x in enumerate(cell_list) if x==cell_ID ][0]

# #omega_cell_dynamics in time
# omega_cell=np.squeeze(omega_sq[:,cell_inx,:])

# #omega values in scatter
# cell_active_omega=omega_sq_mean[0,cell_inx]
# cell_passive_omega=omega_sq_mean[1,cell_inx]
# cell_saccade_omega=omega_sq_mean[2,cell_inx]

# up_lim=0.8
# down_lim=-0.2
# x=np.arange(down_lim,up_lim+0.05,0.05)
# #scatter active vs passive

# plt.scatter(omega_sq_mean[0,:],omega_sq_mean[1,:])
# plt.scatter(cell_active_omega,cell_passive_omega,color='red')
# plt.plot(x,x,color='red')
# plt.xlabel('active')
# plt.ylabel('passive')
# plt.title('active vs passive')
# plt.xlim([down_lim,up_lim])
# plt.ylim([down_lim,up_lim])
# plt.show()


# #scatter active vs saccade
# plt.scatter(omega_sq_mean[0,:],omega_sq_mean[2,:])
# plt.scatter(cell_active_omega,cell_saccade_omega,color='red')

# plt.plot(x,x,color='red')
# plt.xlabel('active')
# plt.ylabel('saccade')
# plt.title('active vs saccade')
# plt.xlim([down_lim,up_lim])
# plt.ylim([down_lim,up_lim])
# plt.show()
# [stat,pval]=scipy.stats.wilcoxon(omega_sq_mean[0,:],omega_sq_mean[2,:])



# omega_mean=np.nanmean(omega_sq,axis=1)
# omega_SEM=stats.sem(omega_sq, axis=1, nan_policy='omit')

# mdic = {"omega_time_mean": omega_mean,"omega_time_SEM": omega_SEM}
# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/stage B/"
# savemat(save_path+"omega"+ ".mat", mdic)

# #%% heatmaps of neuron with their omega
# #2d maps (x*z) for each y 

# #remove rows with nan (cell that were not included in the analysis in one of the tasks)
# cell_coor2=(cell_coor[~np.isnan(cell_coor).any(axis=1),:])
# cell_coor2[:,2]=np.round(cell_coor2[:,2]/1000) #convert z axis to mm
# omega_sq_mean2=np.transpose(omega_sq_mean)
# omega_sq_mean2=(omega_sq_mean2[~np.isnan(cell_coor).any(axis=1),:])

# #min,max and range of coordinates across all cells
# min_x=np.amin(cell_coor2[:,0])
# max_x=np.amax(cell_coor2[:,0])
# x_range=np.arange(min_x,max_x+1)

# min_y=np.amin(cell_coor2[:,1])
# max_y=np.amax(cell_coor2[:,1])
# y_range=np.arange(min_y,max_y+1)

# min_z=np.amin(cell_coor2[:,2])
# max_z=np.amax(cell_coor2[:,2])
# z_range=np.arange(min_z,max_z+1)

# task_array=['active','passive','saccade']

# for task_inx,task in enumerate(task_array):
    
#     omega_maps=[]#list of maps for current task (each map is for a different y)
#     for cur_y in y_range:
#         #new map (for each y and each task)
#         cur_map=np.empty([len(x_range),len(np.unique(cell_coor2[:,2]))])
#         cur_map[:]=np.NaN
#         #keep only cells with the current y and omega for the current task
#         cell_coor2_cur_y=cell_coor2[cell_coor2[:,1]==cur_y,:]
#         omega_cur_y=omega_sq_mean2[cell_coor2[:,1]==cur_y,task_inx]
#         for cur_x_inx,cur_x in enumerate(x_range):
#             cell_coor2_cur_xy=cell_coor2_cur_y[cell_coor2_cur_y[:,0]==cur_x,:]
#             omega_cur_xy=omega_cur_y[cell_coor2_cur_y[:,0]==cur_x]
#             for cur_z_inx,cur_z in enumerate(np.unique(cell_coor2[:,2])):
#                 cell_coor2_cur_xyz=cell_coor2_cur_xy[cell_coor2_cur_xy[:,2]==cur_z,:]
#                 omega_cur_xyz=omega_cur_xy[cell_coor2_cur_xy[:,2]==cur_z]
#                 if np.size(omega_cur_xyz)>0:
#                     cur_map[cur_x_inx,cur_z_inx]=np.mean(omega_cur_xyz)
#         #append the map to the list of maps for the current task
#         omega_maps.append(cur_map)
        
#     for y_inx,cur_y in enumerate(y_range):
#         sb.set_theme()
#         ax=sb.heatmap(np.transpose(omega_maps[y_inx]),cmap='Reds', vmin=-0.01, vmax=0.6)
#         sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
#         ax.set_title('task:'+task+ ' y='+str(cur_y))
#         y_ticks_vec=np.arange(0,len(np.unique(cell_coor2[:,2])),10)
        
#         ax.set_yticks(y_ticks_vec) # <--- set the ticks first
#         ax.set_xlabel('x axis (mm)')
#         ax.set_ylabel('z - thomas (mm)')
#         ax.set_yticklabels([np.unique(cell_coor2[:,2])[x] for x in list(y_ticks_vec)])
#         ax.set_xticklabels(np.unique(cell_coor2[:,0]))
#         ax.set_aspect(0.05)
        
#         plt.show()
                

# #%% heatmaps of neuron with their omega upon mri slices
# #2d maps (x*z) for each y 

# #remove rows with nan (cell that were not included in the analysis in one of the tasks)
# cell_coor2=(cell_coor[~np.isnan(cell_coor).any(axis=1),:])
# cell_coor2[:,2]=np.round(cell_coor2[:,2]/1000) #convert z axis to mm
# omega_sq_mean2=np.transpose(omega_sq_mean)
# omega_sq_mean2=(omega_sq_mean2[~np.isnan(cell_coor).any(axis=1),:])

# #min,max and range of coordinates across all cells
# min_x=np.amin(cell_coor2[:,0])
# max_x=np.amax(cell_coor2[:,0])
# x_range=np.arange(min_x,max_x+1)

# min_y=np.amin(cell_coor2[:,1])
# max_y=np.amax(cell_coor2[:,1])
# y_range=np.arange(min_y,max_y+1)

# min_z=np.amin(cell_coor2[:,2])
# max_z=np.amax(cell_coor2[:,2])
# z_range=np.arange(min_z,max_z+1)

# task_array=['active','passive','saccade']

# #paramters for image - depends in the MRI
# #Fiona:
# CHAMBER_EDGES=[490, 590]
# CHAMBER_LENGTH=19
# Z_0Thomas=420 #the 0 thomas in pixel units
# z_mm2pixel=1102/198 #1102 is number of pixel on vertical axis and 198 its length in mm according to mango ruler

# x_coor=np.linspace(CHAMBER_EDGES[0],CHAMBER_EDGES [1],CHAMBER_LENGTH)
# x_coor=[round(x) for x in x_coor]
# x_offset=-9 #take care that left edge of chamber will be -9

# #text parameters
# font = cv2.FONT_HERSHEY_SIMPLEX #font
# fontScale = 1 # fontScale
# thickness_text=1#thickness
# colorTitle=(255, 0, 0,255) #color of title
# title_coor = (CHAMBER_EDGES[0], Z_0Thomas-50) #coordinates of left bottom edge of title

# path_slices="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/slices mri fiona/slices with grid/"
# dir2save="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/slices mri fiona/slices with omega/"

# COL = MplColorHelper('Reds', -0.01, 0.6)
# thickness_square=-1
# for task_inx,task in enumerate(task_array):
    
#     omega_maps=[]#list of maps for current task (each map is for a different y)
#     for cur_y in y_range:
#         #new map (for each y and each task)
#         cur_map=np.empty([len(x_range),len(np.unique(cell_coor2[:,2]))])
#         cur_map[:]=np.NaN
        
#         #load relevant slice
#         image_name='y'+str(round(cur_y))+'_grid.png'
#         img = Image.open(path_slices+image_name)
#         img_np=np.array(img)
        
#         #keep only cells with the current y and omega for the current task
#         cell_coor2_cur_y=cell_coor2[cell_coor2[:,1]==cur_y,:]
#         omega_cur_y=omega_sq_mean2[cell_coor2[:,1]==cur_y,task_inx]
#         for cur_x_inx,cur_x in enumerate(x_range):
#             cell_coor2_cur_xy=cell_coor2_cur_y[cell_coor2_cur_y[:,0]==cur_x,:]
#             omega_cur_xy=omega_cur_y[cell_coor2_cur_y[:,0]==cur_x]
#             for cur_z_inx,cur_z in enumerate(np.unique(cell_coor2[:,2])):
#                 cell_coor2_cur_xyz=cell_coor2_cur_xy[cell_coor2_cur_xy[:,2]==cur_z,:]
#                 omega_cur_xyz=omega_cur_xy[cell_coor2_cur_xy[:,2]==cur_z]
#                 if np.size(omega_cur_xyz)>0:
#                     cur_map[cur_x_inx,cur_z_inx]=np.nanmean(omega_cur_xyz)
#                     cur_z_pixel=Z_0Thomas+round(cur_z*z_mm2pixel)
#                     cur_color=np.array(COL.get_rgb(np.nanmean(omega_cur_xyz)))*255
#                     cv2.rectangle(img_np, pt1=(x_coor[round(cur_x)-x_offset],cur_z_pixel), pt2=(x_coor[round(cur_x+1)-x_offset],math.floor(cur_z_pixel+z_mm2pixel)-1), color=cur_color, thickness=thickness_square)
       
#         #add title to image
#         cv2.putText(img_np, task+'-y:'+str(round(cur_y)), title_coor, font,fontScale, colorTitle, thickness_text, cv2.LINE_AA)
#         img2 = Image.fromarray(img_np)
#         #img2.show()
#         img2.save(dir2save+task+' '+image_name) 
        

#%% test if effect size is different between tasks:
#Option 1
# Perform one-way ANOVA
#f_statistic, p_value = f_oneway(*omega_sq_mean_noNan)

# Print ANOVA results
# print("ANOVA Results:")
# print(f"F-statistic: {f_statistic}")
# print(f"P-value: {p_value}")

# Check if ANOVA result is significant
# if p_value < 0.05:
#     print("ANOVA result is significant. Proceeding with post hoc tests.")
    
#     # Flatten the data into a 1D array for pairwise Tukey's HSD test
#     flat_data = omega_sq_mean_noNan.flatten()
    
#     # Create group labels for Tukey's HSD
#     group_labels = np.repeat(['Group 1', 'Group 2', 'Group 3', 'Group 4'], np.size(omega_sq_mean_noNan[0]))
    
#     # Perform Tukey's HSD post hoc test
#     tukey_results = pairwise_tukeyhsd(flat_data, group_labels, alpha=0.05)
    
#     # Print Tukey's HSD results
#     print("\nTukey's HSD Results:")
#     print(tukey_results)
# else:
#     print("ANOVA result is not significant. No need for post hoc tests.")
    


#option 2 - wilcoxon - dependent nn parametric t-test
# Group labels
#group_labels = ['saccade', 'pursuit', 'suppression', 'cue']
# group_labels = ['saccade', 'pursuit', 'suppression']

# # Index of Group of reference
# group1_index = 1
# # Loop through the other groups and perform t-tests
# for i in range(len(group_labels)):
#     if i != group1_index:
#         group1_data = omega_sq_mean_noNan[group1_index,:]
#         other_group_data = omega_sq_mean_noNan[i,:]
        
#         # Perform independent t-test
#         t_statistic, p_value = scipy.stats.wilcoxon(group1_data, other_group_data)
        
#         # Print t-test results
#         print(f"T-test between Group 1 and {group_labels[i]}:")
#         print(f"T-statistic: {t_statistic}")
#         print(f"P-value: {p_value}")
        
#         # Check if the p-value is significant
#         if p_value < 0.05:
#             print("The difference is statistically significant.")
#         else:
#             print("The difference is not statistically significant.")
        
#         print("\n")
