# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 15:33:45 2022

@author: Owner
"""

#This script loads MRI coronal slices and draws the grid and the zero thomas coordinates upon the slice. It also crops them and rotate them
#The paramter of the image depends on the MRI.
 
import imageio as iio
import os
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import numpy as np
import cv2

#%% Choose a monkey:
cur_monkey='ya'
#cur_monkey='ya'
#%%

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
    crop_left=0.37
    crop_right=0.63
    crop_bottom=0.58
    crop_top=0.38
if cur_monkey=='ya':
    #cd to directory with mri slices
    path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/slices mri yasmin/slices_original"
    dir2save="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/slices mri yasmin"
    CHAMBER_EDGES_RATIO=[6/13, 7/13] #left edge on the chamber is at 5.5/13% of the figure. Measured on the picture on the screen with a ruler
    Z_TOP=300*upscaling_factor #upper edge of vertical line in the chamber
    Z_BOTTOM=530*upscaling_factor #bottom edge of vertical line in the chamber
    Z_0Thomas=round((5.51/13)*new_resolution) #supposed edge of the guide (0 of thomas):110*1103/256.
    rotation_angle=24
    crop_left=0.41
    crop_right=0.65
    crop_bottom=0.56
    crop_top=0.4
os.chdir(path)

CHAMBER_EDGES=[new_resolution*x for x in CHAMBER_EDGES_RATIO] #[119,138]*1103/256. 1103 is the number of pixel and 256 the number of mm in the picture
CHAMBER_LENGTH=19 #in mm
z_mm2pixel=new_resolution/256 #1102 is number of pixel on vertical axis and 198 its length in mm according to mango ruler

#create a list with all the slices in the path directory 
slices_list=os.listdir(path) 
slices_list=[x for x in slices_list if '.png' in x]

#Colors
colorVertLinesRGB_in=(0, 0, 0) #color of vertical lines in RGBA
colorVertLinesRGB_ext=(255, 0, 0) #color of vertical lines in RGBA

colorCoorRGB=(255,255,255) #color of x coordinates labels
color_Z0_RGB=(0, 255, 0) #color of the thomas 0 line
color_ruler=(255,0,0) #color of the vertical ruler for the z axis

#text parameters
font = cv2.FONT_HERSHEY_SIMPLEX #font
fontScale = 1 # fontScale
thickness_text=3#thickness

#divide the chamber in 19 points (19 mm)
x_grid=np.linspace(CHAMBER_EDGES[0],CHAMBER_EDGES [1],CHAMBER_LENGTH)
x_grid=[round(x) for x in x_grid]
x_offset=-9 #take care that left edge of chamber will be -9 (according to grid coordinates)

TICK_LENGTH=5*upscaling_factor
#loop over slices
for cur_slice in slices_list: 
    image_name=cur_slice[0:-4] #remove .png from name
    # Open image of current slice
    img = Image.open(image_name+'.png')
    original_width, original_height = img.size
    new_width = original_width * upscaling_factor
    new_height = original_height * upscaling_factor
    # Scale the image
    img = img.resize((new_width, new_height), Image.ANTIALIAS)

    img_np=np.array(img) #convert image to numpy array
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
    
    #ruler line
    for cur_depth in np.arange(0,15,3):
        cur_depth_pixel=round(Z_0Thomas+cur_depth*z_mm2pixel)#cur depth in pixel relative to 0 in thomas
        org3=(x_grid[0]-50,cur_depth_pixel+10)
        cv2.putText(img_np, str(cur_depth), org3, font,fontScale, colorCoorRGB, thickness_text, cv2.LINE_AA)
        #ticks
        left_point1=(x_grid[0],cur_depth_pixel)
        right_point1=(x_grid[0]+TICK_LENGTH,cur_depth_pixel)
        cv2.line(img_np, left_point1,right_point1, colorVertLinesRGB_ext, thickness=4)
        left_point2=(x_grid[-1],cur_depth_pixel)
        right_point2=(x_grid[-1]-TICK_LENGTH,cur_depth_pixel)
        cv2.line(img_np, left_point2,right_point2, colorVertLinesRGB_ext, thickness=4)
    #Add thomas 0 line 
    left_point=(x_grid[0],Z_0Thomas)
    right_point=(x_grid[-1],Z_0Thomas)
    cv2.line(img_np, left_point,right_point, color_Z0_RGB, thickness=2) 
    #Remove transparency
    img = Image.fromarray(img_np[:,:,0:3])#remove transparency (4th dimension in the numpy array)
    #rotation
    img = img.rotate(rotation_angle) 
    # Croping
    width, height = img.size
    left = crop_left*new_width
    top = crop_top*new_height
    right = crop_right*new_width
    bottom = crop_bottom*new_height
    img = img.crop((left, top, right, bottom))
    #Saving
    img.save(dir2save+'/slices with grid/'+image_name+ '_grid.jpg')
    
    

