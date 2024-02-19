# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 15:33:45 2022

@author: Owner
"""

#This script loads rtoated crop MRI coronal slices (outputs from process_slices)
#At each location in the grid the code checks whether there is at least one cell significantly tuned and fills the grid
#The script finnally saves those slices.
import imageio as iio
import os
os.chdir("C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/basic classes")
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import numpy as np
import cv2
from scipy.stats import kruskal
from neuron_class import cell_task,filtCells,load_cell_task,smooth_data
from glob import glob
import pickle
import re
import matplotlib as mpl
from matplotlib import cm
import math
import seaborn as sb
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
#%% Choose a monkey:
cur_monkey='ya'
#cur_monkey='ya'

#%% Find significant cells
cell_task_py_folder="units_task_python_two_monkeys/"
cell_pursuit_list=os.listdir(cell_task_py_folder+'8dir_active_passive_interleaved_100_25') #list of strings
cell_saccade_list=os.listdir(cell_task_py_folder+'8dir_saccade_100_25') #list of strings
  
#Uncomment for yasmin_only
if cur_monkey=='ya':
    cell_pursuit_list=[int(item) for item in cell_pursuit_list if int(item)>8229]
    cell_saccade_list=[int(item) for item in cell_saccade_list if int(item)>8229]

#Uncomment for fiona_only
if cur_monkey=='fi':
    cell_pursuit_list=[int(item) for item in cell_pursuit_list if int(item)<=8229]
    cell_saccade_list=[int(item) for item in cell_saccade_list if int(item)<=8229]

cell_list=np.unique(np.array(cell_pursuit_list+cell_saccade_list)) #all cells recorded in pursuit OR saccade
cell_list=[int(item) for item in cell_pursuit_list if item in cell_saccade_list] #all cells recorded in pursuit AND saccade

saccade_plus_pursuit_inxs=[inx for inx,cell_ID in enumerate(cell_list)if cell_ID in cell_pursuit_list and cell_ID in cell_saccade_list]#incs of cells recorded in pursuit and saccade
pursuit_inxs=[inx for inx,cell_ID in enumerate(cell_list)if cell_ID in cell_pursuit_list ]#incs of cells recorded in pursuit 

task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
trial_type_array=['filterOff','v20a','v20p']
event_array=['motion_onset','motion_onset','motion_onset']
condition_array=['saccade','pursuit active','pursuit passive']

win_begin_array=[0,0,0,0,0]#for PD
win_end_array=[350,350,350,350,350]#for PD
win_begin_PSTH=0
win_end_PSTH=350

SMOOTH_PSTH_EDGE=200

win_begin_baseline=-300 #for the baseline calculation of the cell
win_end_baseline=-100

PSTH_length=win_end_PSTH-win_begin_PSTH
directions=[0,45,90,135,180,225,270,315]

#The coordinate of each cell (x,y,z)
cell_coor=np.empty([len(cell_list),3])
cell_coor[:]=np.NaN

#create an array task*n_tests*cells
#The first test check is there is a main effect event vs baseline. The second test checks if tuning during event is significant
sig_array=np.empty([len(task_array),2,len(cell_list)])
sig_array[:]=np.NaN
for cell_inx,cell_ID in enumerate(cell_list):    
    for condition_inx in np.arange(len(task_array)):
        cur_task=task_array[condition_inx]
        cur_trial_type=trial_type_array[condition_inx]
        cur_event=event_array[condition_inx]
        cur_win_begin=win_begin_array[condition_inx]
        cur_win_end=win_end_array[condition_inx]
 
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
        
        if condition_inx==0:
            cell_coor[cell_inx,:]=cur_cell_task.XYZCoor
            
        dictFilterTrials = {'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0}
       
        # Check whether cell is significant for the event relative to baseline       
        try:
            check_main_effect_cond=cur_cell_task.check_main_effect_motion_vs_baseline(cur_event,dictFilterTrials,crit_value=0.01)
        except:
            check_main_effect_cond=0
        sig_array[condition_inx,0,cell_inx]=check_main_effect_cond

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
                   tuning_effect=test_result[1]<0.01
               except:
                   tuning_effect=0   
        except:
           continue
        sig_array[condition_inx,1,cell_inx]=tuning_effect

#%% 
cell_coor[:,2]=np.round(cell_coor[:,2]/1000) #convert z axis to mm
if cur_monkey=='ya':
    cell_coor[:,0]=-cell_coor[:,0]
sig_inx=1 #0 for main effect and 1 for tuning 

#%% Process slices

#parameters to increase resolution
upscaling_factor=4
original_resolution=1103 #depends on the MRI
new_resolution=original_resolution*upscaling_factor
#Parameter of the image - depends on the MRI
if cur_monkey=='fi':
    path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/slices mri fiona/slices_original"
    dir2save="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/slices mri fiona"
    CHAMBER_EDGES_RATIO=[5.7/13, 7/13] #left edge on the chamber is at 5.5/13% of the figure. Measured on the picture on the screen with a ruler
    Z_TOP=300*upscaling_factor #upper edge of vertical line in the chamber
    Z_BOTTOM=485*upscaling_factor #bottom edge of vertical line in the chamber
    Z_0Thomas=round((5/13)*new_resolution) #supposed edge of the guide (0 of thomas):110*1103/256.
    rotation_angle=28
    crop_left=0.42
    crop_right=0.55
    crop_bottom=0.455
    crop_top=0.37
if cur_monkey=='ya':
    #cd to directory with mri slices
    path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/slices mri yasmin/slices_original"
    dir2save="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/slices mri yasmin"
    CHAMBER_EDGES_RATIO=[6/13, 7/13] #left edge on the chamber is at 5.5/13% of the figure. Measured on the picture on the screen with a ruler
    Z_TOP=300*upscaling_factor #upper edge of vertical line in the chamber
    Z_BOTTOM=530*upscaling_factor #bottom edge of vertical line in the chamber
    Z_0Thomas=round((5.51/13)*new_resolution) #supposed edge of the guide (0 of thomas):110*1103/256.
    rotation_angle=24
    crop_left=0.44
    crop_right=0.56
    crop_bottom=0.5
    crop_top=0.4
os.chdir(path)

CHAMBER_EDGES=[new_resolution*x for x in CHAMBER_EDGES_RATIO] #[119,138]*1103/256. 1103 is the number of pixel and 256 the number of mm in the picture
CHAMBER_LENGTH=19 #in mm
z_mm2pixel=new_resolution/256 #1102 is number of pixel on vertical axis and 198 its length in mm according to mango ruler

#create a list with all the slices in the path directory 
slices_list=os.listdir(path) 
slices_list=[x for x in slices_list if '.png' in x]

#Colors
colorVertLinesRGB_ext=(255, 0, 0) #color of vertical lines in RGBA
colorVertLinesRGB_in=(0,0,0) #color of vertical lines in RGBA
colorCoorRGB=(255,255,255) #color of x coordinates labels
color_Z0_RGB=(0, 255, 0) #color of the thomas 0 line
color_ruler=(255,255,255) #color of the vertical ruler for the z axis

#text parameters
font = cv2.FONT_HERSHEY_SIMPLEX #font
fontScale = 1 # fontScale
thickness_text=4#thickness

min_x=-9
max_x=9
x_range=np.arange(min_x,max_x+1)

min_y=np.amin(cell_coor[:,1])
max_y=np.amax(cell_coor[:,1])
y_range=np.arange(min_y,max_y+1)

x_coor=np.linspace(CHAMBER_EDGES[0],CHAMBER_EDGES [1],CHAMBER_LENGTH)
x_coor=[round(x) for x in x_coor]*upscaling_factor

task_array=['active','passive','saccade']
#divide the chamber in 19 points (19 mm)
x_grid=np.linspace(CHAMBER_EDGES[0],CHAMBER_EDGES [1],CHAMBER_LENGTH)
x_grid=[round(x) for x in x_grid]
x_offset=-9 #take care that left edge of chamber will be -9 (according to grid coordinates)
sig_inx=1
TICK_LENGTH=5*upscaling_factor

#cell_size parameter
radius_parameter=8
thickness_circle=-1

#loop over slices
for task_inx,task in enumerate(task_array):
    for cur_slice in slices_list: 

        image_name=cur_slice[0:-4] #remove .png from name
        # Open image of current slice
        img = Image.open(cur_slice)
        
        #image dimensions abd rescaling
        original_width, original_height = img.size
        new_width = original_width * upscaling_factor
        new_height = original_height * upscaling_factor
        img = img.resize((new_width, new_height), Image.ANTIALIAS)    
        img_np=np.array(img) #convert image to numpy array
        
        #Find the y fitting the current slice:
        match = re.search('y[-,\d]+', image_name)
        cur_y=int(match.group(0)[1:])    
        if cur_y not in y_range: 
            continue

        cell_coor_cur_y=cell_coor[cell_coor[:,1]==cur_y,:] #all the cells recorded in the current slice
        sig_array_y=np.transpose(sig_array[task_inx,sig_inx,cell_coor[:,1]==cur_y])    
        
        #loop across x coordinates in the grid
        for x_inx, x in enumerate(x_grid):
            #vertical lines
            top_point=(round(x),Z_TOP)
            bottom_point=(round(x),Z_BOTTOM)
            if x_inx==0 or x_inx==len(x_grid)-1:
                cv2.line(img_np, top_point,bottom_point, colorVertLinesRGB_ext, thickness=6)
            else:
                cv2.line(img_np, top_point,bottom_point, colorVertLinesRGB_in, thickness=2)

            if x_inx%3==0:
                #add label (number) to each line
                org2 = (x-12, Z_BOTTOM+40)
                cv2.putText(img_np, str(abs(x_inx+x_offset)), org2, font,fontScale, colorCoorRGB, thickness_text, cv2.LINE_AA)
     
            cell_coor_cur_xy=cell_coor_cur_y[cell_coor_cur_y[:,0]==x_range[x_inx],:]
            sig_array_xy=sig_array_y[cell_coor_cur_y[:,0]==x_range[x_inx]]
            for cur_z_inx,cur_z in enumerate(np.unique(cell_coor[:,2])):
                cell_coor_cur_xyz=cell_coor_cur_xy[cell_coor_cur_xy[:,2]==cur_z,:]
                sig_array_xyz=sig_array_xy[cell_coor_cur_xy[:,2]==cur_z]
                if np.size(sig_array_xyz)>0:
                    cur_z_pixel=Z_0Thomas+round(cur_z*z_mm2pixel)
                    if np.max(sig_array_xyz)==1:
                        cur_color=(255,0,0)
                    else:
                        cur_color=(0,0,255)
                    x_center=int((x_coor[round(x_range[x_inx])-x_offset]+x_coor[round(x_range[x_inx]+1)-x_offset])/2)
                    z_center=int((cur_z_pixel+(math.floor(cur_z_pixel+z_mm2pixel)-1))/2)
                    cv2.circle(img_np, center=(x_center,z_center),radius=radius_parameter,color=cur_color,thickness=thickness_circle)   
    
        
        #ruler line
        for cur_depth in np.arange(0,15,3):
            cur_depth_pixel=round(Z_0Thomas+cur_depth*z_mm2pixel)#cur depth in pixel relative to 0 in thomas
            org3=(x_grid[0]-50,cur_depth_pixel+10)
            cv2.putText(img_np, str(cur_depth), org3, font,fontScale, color_ruler, thickness_text, cv2.LINE_AA)
            #ticks
            left_point1=(x_grid[0],cur_depth_pixel)
            right_point1=(x_grid[0]+TICK_LENGTH,cur_depth_pixel)
            cv2.line(img_np, left_point1,right_point1, colorVertLinesRGB_ext, thickness=6)
            left_point2=(x_grid[-1],cur_depth_pixel)
            right_point2=(x_grid[-1]-TICK_LENGTH,cur_depth_pixel)
            cv2.line(img_np, left_point2,right_point2, colorVertLinesRGB_ext, thickness=6)
        #Add thomas 0 line 
        left_point=(x_grid[0],Z_0Thomas)
        right_point=(x_grid[-1],Z_0Thomas)
        cv2.line(img_np, left_point,right_point, color_Z0_RGB, thickness=2)
 
        #Remove transparency
        img = Image.fromarray(img_np[:,:,0:3])#remove transparency (4th dimension in the numpy array) to enable conversion to jpeg
        #rotation
        #img = img.rotate(rotation_angle) 
        # Croping
        width, height = img.size
        left = crop_left*new_width
        top = crop_top*new_height
        right = crop_right*new_width
        bottom = crop_bottom*new_height
        img = img.crop((left, top, right, bottom))
        #Saving
        img.save(dir2save+'/slices with cells/'+image_name+task+'.jpg')

#%% Percentage of significant cell on x-y plane (average z plane)

#min,max and range of coordinates across all cells
min_x=np.amin(cell_coor[:,0])
max_x=np.amax(cell_coor[:,0])
x_range=np.arange(min_x,max_x+1)

min_y=np.amin(cell_coor[:,1])
max_y=np.amax(cell_coor[:,1])
y_range=np.arange(min_y,max_y+1)

sig_cells_xy=np.empty((len(condition_array),np.size(x_range),np.size(y_range)))
sig_cells_xy[:]=np.nan
for x_inx,x in enumerate(x_range):
    x_bool_array=cell_coor[:,0]==x
    x_inxs=[i for i, x in enumerate(x_bool_array) if x]
    for y_inx,y in enumerate(y_range):
        y_bool_array=cell_coor[:,1]==y
        xy_inxs=[i for i, x in enumerate(y_bool_array) if x and (i in x_inxs)]
        sig_cells_xy[:,x_inx,y_inx]=np.mean(sig_array[:,1,xy_inxs],1)
        
for condition_inx,condition in enumerate(condition_array):
    
    # the index of the position of yticks
    xticklabels = np.unique(x_range)
    yticklabels = np.unique(y_range)

    plt.figure(dpi=500)
    g=sb.heatmap(np.transpose(sig_cells_xy[condition_inx,:]), xticklabels=xticklabels, yticklabels=yticklabels,cmap="bwr",vmin=0,vmax=0.7)
    g.set_facecolor('grey')
    plt.title(condition)
    plt.xlabel('X')
    plt.xlabel('Y')
    plt.show()
