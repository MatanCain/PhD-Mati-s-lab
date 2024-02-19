# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:59:37 2021

@author: Owner
"""
from __future__ import print_function
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from glob import glob
import pickle
import os
import pandas as pd
import numpy as np
import re
import scipy.io
import math
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from mat4py import loadmat
import sys
from scipy.stats import kruskal
from os.path import isfile,join


## OBJECT DEFINIION
##########################################
class cell_task:
    
    def __init__(self,cell_task_info, cell_task_trials):
        self.id = cell_task_info['cell_ID']#ID of the cell
        self.XYZCoor = [cell_task_info['X'],cell_task_info['Y'],cell_task_info['depth_thomas']] #list with x,y,z coor of the cells. z is the depth in units of the thomas MEM
        self.grade=cell_task_info['grade']# cell grade
        self.trials_df = cell_task_trials # a data frame with all data regarding this cell during a specific task
    
        
    def getId(self):
        return self.id

    def getXCoordinate(self):
        return self.XYZCoor
        
    def getTrials_df(self):
        return self.trials_df
    
    def getGrade(self):
        return self.grade

    def getSession(self):
        return re.findall("[a-z]{2}[0-9]{6}", self.trials_df.loc[0,'filename_name'])[0]
    
    
    #this method filters the trials_df according to parameters in dictFilterTrials.
    #To not filter add 'filterOff' as value to the key parameter.
    #dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20a', 'fail':0,'after_fail':0}
    #for trial_name it keeps all trials that contain the pattern in dictFilterTrials['trial_name']
    # n_trials extract n random trials
    #trial_begin_end: a list that extract all trials from rows first to second element or an int and extract only  this row
    #files_begin_end: a list where each element is a number of the maestro files. keep files from first to second element
    #trial_inxs: a list where is element is an index of a trial we want
    def filtTrials(self, dictFilterTrials):
        trials_df = self.trials_df
        
        if ('trial_inxs' in dictFilterTrials) and (dictFilterTrials['trial_inxs'] !='filterOff'):
            trials_df=trials_df.loc[dictFilterTrials['trial_inxs'],:] 
        if  ('files_begin_end' in dictFilterTrials) and (dictFilterTrials['files_begin_end'] !='filterOff'):
            file_begin=dictFilterTrials['files_begin_end'][0]
            file_end=dictFilterTrials['files_begin_end'][1]
            trials_df['file_number']=trials_df['filename_name'].str.split(".",1)   
            trials_df.loc[:, 'file_number']=trials_df.file_number.map(lambda x: int(x[1]))  
            trials_df=trials_df[(trials_df['file_number']>=file_begin) & (trials_df['file_number']<=file_end)]
        if 'fail' in dictFilterTrials and dictFilterTrials['fail'] !='filterOff': 
            trials_df = trials_df[trials_df['fail'] == dictFilterTrials['fail']]
            
        if 'trial_name' in dictFilterTrials and  dictFilterTrials['trial_name'] !='filterOff': 
            trials_df = trials_df[trials_df['trial_name'].str.contains(dictFilterTrials['trial_name'],regex=True)]    
            
    
        if 'after_fail' in dictFilterTrials and  dictFilterTrials['after_fail'] !='filterOff':
            #0 will remove all trials AFTER a failure
            #1 will keep only trials after failure
            fail_inx=trials_df[trials_df['fail']==dictFilterTrials['after_fail'] ].index
            trials_df=trials_df.loc[fail_inx[0:-1]+1,:]  
        
        if 'dir' in dictFilterTrials and dictFilterTrials['dir'] !='filterOff':
            if type (dictFilterTrials['dir'])==list:
                trials_df=trials_df[trials_df['dir'].isin(dictFilterTrials['dir'])]
            else: #if its int or numpy or float...
                trials_df = trials_df[trials_df['dir'] == dictFilterTrials['dir']]
                
                
        if 'screen_rot' in dictFilterTrials and dictFilterTrials['screen_rot'] !='filterOff':
            if type (dictFilterTrials['screen_rot'])==list:
                trials_df=trials_df[trials_df['screen_rotation'].isin(dictFilterTrials['screen_rot'])]
            else: #if its int or numpy or float...
                trials_df = trials_df[trials_df['screen_rotation'] == dictFilterTrials['screen_rot']]                


        if  'saccade_motion' in dictFilterTrials and dictFilterTrials['saccade_motion'] !='filterOff':
            saccade_motion_begin=0 #after motion onset
            saccade_motion_end=250 #after motion onset
            trials_df.loc[:,'saccade_onset'] = trials_df.apply(lambda row: [saccade[0] if type(saccade)==list else row.saccades[0]  for saccade in row.saccades],axis=1)
            trials_df.loc[:,'saccade_motion']= trials_df.apply(lambda row: any([saccade_onset>row.motion_onset+saccade_motion_begin and saccade_onset<row.motion_onset+saccade_motion_end   for saccade_onset in row.saccade_onset]) ,axis=1)
            trials_df.loc[:,'saccade_motion']= trials_df.apply(lambda row: not(row.saccade_motion),axis=1)
            trials_df=trials_df[trials_df['saccade_motion']]
        if  'blink_motion' in dictFilterTrials and dictFilterTrials['blink_motion'] !='filterOff': #delete trials with blinks that begin in the relevant window after motion onset
            blink_window_begin=-200 #after motion onset
            blink_window_end=300 #after motion onset
            trials_df.loc[:,'blink_onset'] = trials_df.apply(lambda row: [blink[0] if type(blink)==list else row.blinks[0]  for blink in row.blinks],axis=1)
            trials_df.loc[:,'blink_motion']= trials_df.apply(lambda row: any([blink_onset>row.motion_onset+blink_window_begin and blink_onset<row.motion_onset+blink_window_end   for blink_onset in row.blink_onset]) ,axis=1)
            trials_df.loc[:,'blink_motion']= trials_df.apply(lambda row: not(row.blink_motion),axis=1)
            trials_df=trials_df[trials_df['blink_motion']]
        if  'n_trials' in dictFilterTrials and dictFilterTrials['n_trials'] !='filterOff':
            if dictFilterTrials['n_trials']< trials_df.shape[0]:
                trials_df=trials_df.sample(n=dictFilterTrials['n_trials'])
                
        if  ('trial_begin_end' in dictFilterTrials) and dictFilterTrials['trial_begin_end'] !='filterOff':
            if type(dictFilterTrials['trial_begin_end'])==list:
                trials_df=trials_df.iloc[dictFilterTrials['trial_begin_end'][0]:dictFilterTrials['trial_begin_end'][1],:]
            if type(dictFilterTrials['trial_begin_end'])==int:
                trials_df=trials_df.iloc[[dictFilterTrials['trial_begin_end']],:]
            if type(dictFilterTrials['trial_begin_end'])==np.ndarray:
                trials_df=trials_df.iloc[dictFilterTrials['trial_begin_end'],:]        

        if  ('block_begin_end' in dictFilterTrials) and dictFilterTrials['block_begin_end'] !='filterOff':
            trials_df=trials_df.loc[dictFilterTrials['block_begin_end'][0]:dictFilterTrials['block_begin_end'][1],:]
            
        if  ('even_odd_rows' in dictFilterTrials) and dictFilterTrials['even_odd_rows'] !='filterOff':
            if dictFilterTrials['even_odd_rows'] =='odd':
                trials_df=trials_df.iloc[0::2]
            elif dictFilterTrials['even_odd_rows'] =='even':
                trials_df=trials_df.iloc[1::2]
        return trials_df
    
    
        
    # return spikes by given timePoint and time window:
    #         window = {"timePoint": timePoint, "timeBefore": -timeBefore,"timeAfter": timeAfter}
    # timePoint is a string and 'timeBefore' and 'timeAfter' are ints
    def spikesInWindow(self, row, kwargs):
        events = (row['spike_times'] > row[kwargs['timePoint']] + kwargs['timeBefore']) & (row['spike_times'] < row[kwargs['timePoint']] + kwargs['timeAfter']) 
        events = row.loc['spike_times'][events] - row[kwargs['timePoint']]
        return events

    #normalize spike times relatively to event and extract only spikes within the window
    def spikesByTime(self, window, dictFilterTrials):
        trials_df = self.filtTrials(dictFilterTrials)
        events=trials_df.apply(self.spikesInWindow, args=(window,), axis=1)
        return events 
    

    def raster(self,window,dictFilterTrials,plot_option=1):
            cur_df=self.trials_df.sort_values(by=['dir'])
            spike_times=self.spikesByTime(window, dictFilterTrials)
            if plot_option:
                plt.eventplot(spike_times)
                plt.ylabel('trial')
                plt.xlabel('time (ms)')
                plt.axvline(x=0, color='r')
                plt.show()
            return  spike_times  

    #The psth method concatenates all the spike times in a np array and then build an histogram
    def PSTH(self,window,dictFilterTrials,smooth_option=1,SMOOTH_KERNEL=20):
        
        # if smooth_option==1: #prevent problems from smoothing problem at edges
        #     window['timeAfter']=window['timeAfter']+100
        #     window['timeBefore']=window['timeBefore']-100
            
        
        BIN_LENGTH=1 # in ms
        
        spike_times=self.spikesByTime(window,dictFilterTrials)
        spike_times=spike_times-window['timeBefore'] #setting 0 as begining of the window
        spike_times_concact=np.empty(shape=(0))
        
        window_length=window['timeAfter']-window['timeBefore']
        n_bins=int(window_length/BIN_LENGTH)
        for i in list(spike_times.index.values): #concactenate all spike_times (maybe there is a better way)
            spike_times_concact=np.concatenate([spike_times.loc[i],spike_times_concact])
        #hist_output=np.histogram(spike_times_concact,bins=n_bins)
        hist_output=np.histogram(spike_times_concact,bins=n_bins,range=(0,window_length))
        PSTH_raw=hist_output[0]
        n_trials=spike_times.index.values.size
        PSTH_raw=PSTH_raw/(BIN_LENGTH*n_trials*0.001) #normalize PSTH, assume time of spikes in ms
        if smooth_option:
             PSTH=smooth_data(PSTH_raw,SMOOTH_KERNEL)
        #     PSTH=PSTH[100:-100]
        else:
            PSTH=PSTH_raw
        return PSTH


#This method calculates the PD of a cell in degrees using the center of mass method
#It furst calculates the exact PD and then finds the closest direction within the 8 discrete direction in dir_array
# Enter 'filterOff' as value of the dir in the dictfitler trials dictionnary 
    def get_PD(self,window,dictFilterTrials):
        try:
            DictFilterTrials={} #to prevent the dict to be  changed out of the function
            for key, value in dictFilterTrials.items():
                DictFilterTrials[key]=dictFilterTrials[key]
            trials_df=self.filtTrials(DictFilterTrials) 
            dir_array=list(np.unique(np.array(trials_df.dir))) #extract all directions present in trials df
            #dir_array=[0,45,90,135,180,225,270,315]
            FR=[]
            x=0 #for center of mass
            y=0 #for center of mass
            for cur_dir_inx,cur_dir in enumerate(dir_array):
                DictFilterTrials['dir'] =cur_dir
                PSTH=self.PSTH(window,DictFilterTrials,0)
                FR.append(np.mean(PSTH))
                x=x+math.cos(math.radians(cur_dir)) *np.mean(PSTH)
                y=y+math.sin(math.radians(cur_dir)) *np.mean(PSTH)    
            Exact_PD=math.degrees(np.arctan2(y,x))%360
            #Find the closest direction to the exact PD within the 8 dir
            absolute_difference_function = lambda dir_array : abs(dir_array - Exact_PD)
            PD = min(dir_array, key=absolute_difference_function)
            trial_df=self.filtTrials(dictFilterTrials)
            screen_rot=trial_df.iloc[0]['screen_rotation']
            PD=(PD+screen_rot)%360
            return PD
        except:
            return False

    def get_exact_PD(self,window,dictFilterTrials): #return the exact PD, not one of the 8 directions
        try:
            DictFilterTrials={} #to prevent the dict to be  changed out of the function
            for key, value in dictFilterTrials.items():
                DictFilterTrials[key]=dictFilterTrials[key]
            trials_df=self.filtTrials(DictFilterTrials) 
            dir_array=list(np.unique(np.array(trials_df.dir))) #extract all directions present in trials df
            #dir_array=[0,45,90,135,180,225,270,315]
            FR=[]
            x=0 #for center of mass
            y=0 #for center of mass
            for cur_dir_inx,cur_dir in enumerate(dir_array):
                DictFilterTrials['dir'] =cur_dir
                PSTH=self.PSTH(window,DictFilterTrials,0)
                FR.append(np.mean(PSTH))
                x=x+math.cos(math.radians(cur_dir)) *np.mean(PSTH)
                y=y+math.sin(math.radians(cur_dir)) *np.mean(PSTH)    
            Exact_PD=math.degrees(np.arctan2(y,x))%360
            #Find the closest direction to the exact PD within the 8 dir
            trial_df=self.filtTrials(dictFilterTrials)
            screen_rot=trial_df.iloc[0]['screen_rotation']
            PD=(Exact_PD+screen_rot)%360
            return PD
        except:
            return False
                
    #this function returns a serie with the mean FR for each trial (after filtering with dictFilterTrials) around a given event    
    def get_mean_FR_event(self,dictFilterTrials,event,window_pre=0,window_post=200):
        
        trial_df=self.filtTrials(dictFilterTrials)
        #convert spike_time serie to df and adds nan to create a rectangular df
        cur_list=trial_df.loc[:,"spike_times"].tolist()
    
        cur_list=[np.append(spike_array,float('nan')) for spike_array in cur_list  ]
        spike_times_df=pd.DataFrame(cur_list) 
            
        #extract event serie from trial_df:
        event_serie=trial_df.loc[:,event].to_numpy()
        
        #define windows around event
        window_len=window_post-window_pre
        
        #Create a boolean array where true means that spike occured within the window
        spike_event_before=spike_times_df.gt(event_serie+window_pre,axis=0)  
        
        spike_event_before=spike_event_before.loc[:len(event_serie)-1,:] #it removes weird rows that are added in the end
        spike_event_before=spike_event_before.to_numpy()
        
        spike_event_after=spike_times_df.lt(event_serie+window_post,axis=0) 
        spike_event_after=spike_event_after.loc[:len(event_serie)-1,:] #it removes weird rows that are added in the end
        spike_event_after=spike_event_after.to_numpy()
        
        spike_event=np.logical_and(spike_event_before,spike_event_after)
            
        #SCalculates number of true events (spikes in the window) trial wise
        n_spikes=np.sum(spike_event,axis=1)
        #convert to FR (Hz)
        FR_event=n_spikes*(1000/window_len) 
        
        FR_event=pd.Series(FR_event)
        return FR_event

    #This function check whether the cell reacts during event (motion/cue) compared to baseline 
    def check_main_effect_motion_vs_baseline(self,event,dictFilterTrials,Window_pre=0,Window_post=350,crit_value=0.05): 
        
        FR_MO=self.get_mean_FR_event(dictFilterTrials,event,window_pre=Window_pre,window_post=Window_post)
        FR_BL=self.get_mean_FR_event(dictFilterTrials,event,window_pre=-300,window_post=-100)
        try:
            [stat,pval]=scipy.stats.wilcoxon(FR_MO,FR_BL,alternative='two-sided')
        except:
            print('no spike in cur cell ')
            pval=1
        sig_bool=pval<crit_value
        return sig_bool


    #This function check whether the cell reacts significantly different to some direction during an event 
    def check_sig_dir(self,dictFilterTrials,event,Window_pre=100,Window_post=500,dir_array=[0,45,90,135,180,225,270,315]): 
        try:
            DictFilterTrials={} #to prevent the dict to be  changed out of the function
            for key, value in dictFilterTrials.items():
                DictFilterTrials[key]=dictFilterTrials[key]
            
            data_array=[]
            for cur_ix,cur_dir in enumerate(dir_array):
               DictFilterTrials['dir'] =cur_dir
               cur_dir_trial_df=self.filtTrials(DictFilterTrials)
               if cur_dir_trial_df.shape[0]>0:
                   FR_cur_dir=self.get_mean_FR_event(DictFilterTrials,event,window_pre=Window_pre,window_post=Window_post)
                   data_array.append(FR_cur_dir)
            try:
                test_result=kruskal(*data_array)
                sig_bool=test_result[1]<0.01
                pval=test_result[1]
            except:
                sig_bool=0
                pval=1
                #print('pb in kruskal in check sig_dir')
            return sig_bool
        except:
            return False
        


##########################################    
#THis function receive the path to cell db_file and a dictionnary with the different filters and return a list of cell IDs that match the dict criteria 
#Input examples:
    #cell_db_file="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_cells_db.xlsx"
    #dictFilterCells  = {'cell_ID':'filterOff','session':'filterOff','grade':7, 'n_trials':'filterOff'}        
def filtCells(cell_db_file,dictFilterCells):
    cell_db_df = pd.read_excel (cell_db_file)
    cell_db_df['n_trials']=cell_db_df['fe_after_stability']-cell_db_df['fb_after_stablility']

    cell_db_filtered=cell_db_df
    if dictFilterCells['cell_ID']!='filterOff':
        cell_list=dictFilterCells['cell_ID']
        cell_db_filtered=cell_db_filtered[cell_db_filtered['cell_ID'].isin(cell_list)]


    if dictFilterCells['session']!='filterOff':
        session_list=dictFilterCells['session']
        cell_db_filtered=cell_db_filtered[cell_db_filtered['session'].isin(session_list)]


    if dictFilterCells['grade']!='filterOff':
        cur_grade=dictFilterCells['grade']
        cell_db_filtered=cell_db_filtered[cell_db_filtered['grade']<cur_grade]

    if dictFilterCells['n_trials']!='filterOff':
        Ntrial_limit=dictFilterCells['n_trials']
        cell_db_filtered=cell_db_filtered[cell_db_filtered['n_trials']>Ntrial_limit]  
    

    return (cell_db_filtered['cell_ID'].values.tolist())


#smooth data with moving average. kernel_size is the width of the uniform filter
def smooth_data(raw_data,kernel_size=20): 
   # kernel = np.ones(kernel_size) / kernel_size
    #smoothed_data = np.convolve(raw_data, kernel, mode='same')
    smoothed_data=scipy.ndimage.gaussian_filter1d(raw_data, kernel_size)
    return smoothed_data
#%% Check the width of a gaussian for a given sigma
# from matplotlib import pyplot as plt
# import scipy.ndimage as ndi
# from matplotlib import pyplot as plt
# from scipy.stats import multivariate_normal
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import warnings
# warnings.filterwarnings("ignore")
# sigma=10 #standard deviation
# mu=0 # mean
# x=np.arange(-30,30.1,0.1)
# gaussContin=gauss=1/(np.sqrt(2*np.pi)*sigma)*np.exp(-1*(x-mu)**2/(2*sigma**2))
# plt.plot(x,gaussContin)
# plt.grid(True)
# plt.title("Continous Gauss Filter")
# plt.show()

#%%
#### Function to load a cell_task
    #inputs:
        #cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
        #task='8dir_active_passive_interleaved_100_25'
        #cell_ID=7385 #int
    #output
        #cur_cell_task (cell_task object)
def load_cell_task(cell_task_py_folder,task,cell_ID):
    filename=cell_task_py_folder+task+'/'+str(cell_ID)
    infile = open(filename,'rb')
    cur_cell_task = pickle.load(infile)
    infile.close()
    return cur_cell_task

def get_cell_list(cell_task_py_folder,task_list):
    cells_list=[]
    for cur_task in task_list:
        cell_tasks_folder=join(cell_task_py_folder,cur_task)
        cells_list=cells_list+os.listdir(cell_tasks_folder)
    return cells_list     
    
    

#########       SAVE DATA IN PYTHON #####

#This script converts all the cell_task in the matlab folder into python cell_task object and save them in the python folder
# cell_task_matlab_folder="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/units_task_two_monkeys_matlab_kinematics/"
# cell_task_py_folder="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/units_task_two_monkeys_python_kinematics/"

# #task_list=os.listdir(cell_task_matlab_folder) #for all task
# task_list=['8dir_saccade_100_25'] #for a specific task
# for cur_task in task_list:
#     cell_task_list=os.listdir(cell_task_matlab_folder+cur_task)
#     cell_task_list=[item[0:-4] for item in cell_task_list] #remove '.mat'
#     for cell in cell_task_list:
#         matlab_file=(cell_task_matlab_folder+cur_task+'/'+cell+'.mat')
#         data=loadmat(matlab_file)
#         data2=scipy.io.loadmat(matlab_file)
#         cell_task_info=data['cell_task']['info']
         
#         #If there is only one trial (it there is more then type will be a list)
#         if type(data['cell_task']['trials']['filename_name'])==str:
#             for key in data['cell_task']['trials']:
#                 data['cell_task']['trials'][key]=[data['cell_task']['trials'][key]]    
               
#         cell_task_trials=pd.DataFrame.from_dict(data['cell_task']['trials'])
#         cell_task_trials['spike_times'] = cell_task_trials['spike_times'].apply(lambda x: np.ceil(x))
#         cur_cell_task=cell_task(cell_task_info, cell_task_trials)
        
#         filename=cell_task_py_folder+cur_task+'/'+cell
#         #create a directory if it does not exist
#         if not os.path.isdir(cell_task_py_folder+cur_task):
#             os.makedirs(cell_task_py_folder+cur_task)
        
#         outfile = open(filename,'wb')
#         pickle.dump(cur_cell_task,outfile)
#         outfile.close()
 
    
#########       SAVE BEHAVIOURAL DATA IN PYTHON #####

#This script converts all the behavioral in the matlab folder into python save them in the python folder
# behaviour_matlab_folder="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/behaviour_matlab/"
# behaviour_py_folder="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/behaviour_python/"


# task_list=os.listdir(behaviour_matlab_folder) #for all task
# #task_list=Tasks=['8dir_saccade'] #for specific task(s)
# for cur_task in task_list:
#     session_list=os.listdir(behaviour_matlab_folder+cur_task)
#     session_list=[item[0:-4] for item in session_list] #remove '.mat'
#     # Continue to next task if directory already exists
#     #if os.path.isdir(behaviour_py_folder+cur_task):
#         # continue
   
#     for cur_session in session_list:
#         matlab_file=(behaviour_matlab_folder+cur_task+'/'+cur_session+'.mat')
#         data=loadmat(matlab_file)
#         #convert dict to df
#         cur_sess=data['session_task']['trials'] 
#         sess_df=pd.DataFrame.from_dict(cur_sess, orient='columns', dtype=None, columns=None)
#         filename=behaviour_py_folder+cur_task+'/'+cur_session
        
#         #create a directory if it does not exist
#         if not os.path.isdir(behaviour_py_folder+cur_task):
#             os.makedirs(behaviour_py_folder+cur_task)

#         outfile = open(filename,'wb')
#         pickle.dump(sess_df,outfile)
#         outfile.close()
 


#########
#load a cell task for tests:
# cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
# task='Motor_learning_CW_100_25_cue'
# cell_ID=7678
# cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)

# dictFilterTrials={'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':'filterOff','saccade_motion':1,'blink_motion':1,}    
# trials_df=cur_cell_task.filtTrials(dictFilterTrials)

# trials_inx=[10,8,7]

# trials_df2=trials_df.iloc[trials_inx]

#######
#%%
#The function calculates PSTH for a list of cells
#This function generates PSTH_array(n_cells*time*directions) and PSTH_array_mean(time*directions)
#The function can also draw the average PSTH across cells
#Examples of inputs:
    #cell_list=[7702,7795]
    #tasks='8dir_active_passive_interleaved_100_25' or ['Motor_learning_CW_100_25_cue','Motor_learning_CCW_100_25_cue','Motor_learning_CW','Motor_learning_CCW']
    #event='cue_onset'  #or 'motion onset'
    
    #trial_type='v20a|v20p'(depends on the task)

    #fail=0 #(0-no fail 1-fails)
    
    #PSTH window around event and filter trials
    #window={"timePoint": event,"timeBefore": win_before,"timeAfter":win_after}
    #dictFilterTrials = {'dir':'filterOff', 'trial_name':trial_type, 'fail':fail, 'after_fail':'filterOff','saccade_motion':'filterOff'}
    
    #PSTH_type='dir_apart' #'dir_apart'-plot directions apart,'meged_dir'-merge all directions, 'PD' - plot only PD  
    
    #PD parameters
    #PD_dict={'PD_event':'cue_onset','PD_win_before':0,'PD_win_after':400,'PD_trial_type':'v20a|v20p','PD_fail':0}
    
    #plot_option=1

def PSTH_across_cells(cell_list,tasks,event,trial_type,window,dictFilterTrials,PSTH_type,PD_dict=[],plot_option=1):
    
    #cd to cell task folder
    path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses"
    os.chdir(path)
    cell_task_py_folder="units_task_python_two_monkeys/"
    
    
    # initialize PSTH array for different kinds of PSTH
    win_before=window['timeBefore']
    win_after=window['timeAfter']
    PSTH_length=win_after-win_before
    if PSTH_type=='merged_dir' or PSTH_type=='PD' or PSTH_type=='null':
        PSTH_array=np.empty([len(cell_list),PSTH_length])
    
    elif PSTH_type=='dir_apart':
        directions=[0, 45, 90, 135, 180, 225, 270, 315]
        PSTH_array=np.empty([len(cell_list),PSTH_length,len(directions)])
    PSTH_array[:]=np.NaN
    
    if type(tasks)==str:#if there is only possible task
        tasks=[tasks]
        
    # loop across cells
    for cell_inx,cell_ID in enumerate(cell_list):
        #load cell task
        for task in tasks:
            try:
                cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
            except:
                continue
        #merged PSTH for all directions
        if PSTH_type=='merged_dir':
            window={"timePoint": event,"timeBefore": win_before,"timeAfter":win_after}
            PSTH_array[cell_inx,:]=cur_cell_task.PSTH(window,dictFilterTrials,smooth_option=1)
            
        #PSTH for each directions apart    
        if PSTH_type=='dir_apart':
            for dir_inx,direction in enumerate(directions):
                window={"timePoint": event,"timeBefore": win_before,"timeAfter":win_after}
                dictFilterTrials['dir']=direction
                cur_PSTH=cur_cell_task.PSTH(window,dictFilterTrials,smooth_option=1)
                PSTH_array[cell_inx,:,dir_inx]=cur_PSTH
            
        # PSTH in PD
        if PSTH_type=='PD' or PSTH_type=='null' :           
            PD_window={"timePoint": PD_dict['PD_event'],"timeBefore": PD_dict['PD_win_before'],"timeAfter":PD_dict['PD_win_after']}
            PD_dictFilterTrials = {'dir':'filterOff', 'trial_name':PD_dict['PD_trial_type'], 'fail':PD_dict['PD_fail'], 'after_fail':'filterOff','saccade_motion':'filterOff'}
            PD_dir=cur_cell_task.get_PD(PD_window,PD_dictFilterTrials)    
            window={"timePoint": event,"timeBefore": win_before,"timeAfter":win_after}
            if PSTH_type=='PD':
                dictFilterTrials['dir']=PD_dir
            elif PSTH_type=='null':
                dictFilterTrials['dir']=(PD_dir+180)%360
            PSTH_array[cell_inx,:]=cur_cell_task.PSTH(window,dictFilterTrials,smooth_option=1)
    
    #average across cells
    PSTH_array_mean=np.nanmean(PSTH_array,axis=0)
    # Draw PSTH
    if plot_option==1:
        x_axis=np.arange(win_before,win_after)
        if PSTH_type=='dir_apart':
            plt.plot(x_axis,PSTH_array_mean)
            plt.legend(directions)
        elif PSTH_type=='PD':
            plt.plot(x_axis,PSTH_array_mean)
        elif PSTH_type=='null':
            plt.plot(x_axis,PSTH_array_mean)
        elif PSTH_type=='merged_dir':
            plt.plot(x_axis,PSTH_array_mean)
            
        plt.axvline(x=0,color='red')
        plt.title('PSTH -' + str(len(cell_list))+' cells'+' trial type:'+trial_type)
        plt.xlabel('Time from '+event+' (ms)')
        plt.ylabel('FR (Hz)')
        plt.show()
        
    return PSTH_array_mean,PSTH_array
