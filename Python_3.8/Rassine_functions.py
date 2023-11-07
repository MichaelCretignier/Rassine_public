#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:34:29 2019

@author: cretignier

"""

from __future__ import print_function

import matplotlib

matplotlib.use('Qt5Agg',force=True)
import glob as glob
import multiprocessing as multicpu
import os
import pickle
import platform
import sys
#from tqdm import tqdm 
import time
from itertools import repeat

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from matplotlib.widgets import Button, RadioButtons, Slider
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import norm

np.warnings.filterwarnings('ignore', category=RuntimeWarning)


py_ver = platform.python_version_tuple()
py_ver = str(int(py_ver[0])+int(py_ver[1])//10)+'.'+py_ver[1]

# =============================================================================
# FUNCTIONS LIBRARY
# =============================================================================

h_planck = 6.626e-34
c_lum = 299.792e6
k_boltz = 1.3806e-23

sys_python = sys.version[0]
protocol_pickle = 'auto'
voice_name = ['Victoria', 'Daniel', None][0] #voice pitch of the auditive feedback

if protocol_pickle == 'auto':
    if sys_python == '3':
        protocol_pick = 3
    else:
        protocol_pick = 2
else:
    protocol_pick = int(protocol_pickle)



if sys_python == '3':
    def my_input(text):
        return input(text)
else:
    def my_input(text):
        return raw_input(text)


# =============================================================================
# FUNCTIONS (alphabetic ordered)
# =============================================================================

def ccf(wave,spec1,spec2,extended=1500):
    """
    CCF for a equidistant grid in log wavelength spec1 = spectrum, spec2 =  binary mask, mask_telluric = binary mask
    
    Parameters
    ----------
    extended : int-type 
    
    Returns
    -------
    
    Return the vrad lag grid as well as the CCF
    
    """
       
    dwave = wave[1]-wave[0]
    spec1 = np.hstack([np.ones(extended),spec1,np.ones(extended)])
    spec2 = np.hstack([np.zeros(extended),spec2,np.zeros(extended)])
    wave = np.hstack([np.arange(-extended*dwave+wave.min(),wave.min(),dwave),wave,np.arange(wave.max()+dwave,(extended+1)*dwave+wave.max(),dwave)])
    shift = np.linspace(0,dwave,10)[:-1]
    shift_save = []
    sum_spec = np.sum(spec2)
    convolution = []
    for j in shift:
        new_spec = interp1d(wave+j,spec2,kind='cubic', bounds_error=False, fill_value='extrapolate')(wave)
        for k in np.arange(-60,61,1):
            new_spec2 = np.hstack([new_spec[-k:],new_spec[:-k]])
            convolution.append(np.sum(new_spec2*spec1)/sum_spec)
            shift_save.append(j+k*dwave)
    shift_save = np.array(shift_save)
    return (c_lum*10**shift_save)-c_lum, np.array(convolution)


def check_none_negative_values(array):
    """
    Remove negative number
    
    """
    neg = np.where(array<=0)[0]
    if len(neg)==0:
        pass
    elif len(neg)==1:
        array[neg] = 0.5*(array[neg+1]+array[neg-1])
    else:
        where = np.where(np.diff(neg)!=1)[0]
        if (len(where)==0)&(np.mean(neg)<len(array)/2):
            array[neg] = array[neg[-1]+1]
        elif (len(where)==0)&(np.mean(neg)>=len(array)/2):
            array[neg] = array[neg[0]-1]            
        else:
            where = np.hstack([where,np.array([0,len(neg)-1])])
            where = where[where.argsort()]
            for j in range(len(where)-1):
                if np.mean(array[neg[where[j]]:neg[where[j+1]]+1])<len(array)/2:
                    array[neg[where[j]]:neg[where[j+1]]+1] = array[neg[where[j+1]]+1]
                else:
                    array[neg[where[j]]:neg[where[j+1]]+1] = array[neg[where[j]]-1]
    return array


def clustering(array, tresh, num):
    """
    Detect and form cluster on an array considering than values are closer than the tresh value. 
    Only cluster containing more elements than num are kept.
    
    """
    difference = np.diff(array)
    cluster = (difference<tresh)
    indice = np.arange(len(cluster))[cluster]
    if sum(cluster):
        j = 0
        border_left = [indice[0]]
        border_right = []
        while j < len(indice)-1:
            if indice[j]==indice[j+1]-1:
                j+=1
            else:
                border_right.append(indice[j])
                border_left.append(indice[j+1])
                j+=1
        border_right.append(indice[-1])        
        border = np.array([border_left,border_right]).T
        border = np.hstack([border,(1+border[:,1]-border[:,0])[:,np.newaxis]])
        
        kept = []
        for j in range(len(border)):
            if border[j,-1]>=num:
                kept.append(array[border[j,0]:border[j,1]+2])
        return np.array(kept,dtype='object') 
    else:
        return [[j] for j in array]


def create_grid(wave_min, dwave, nb_bins):
    """
    Form a  wavelength grid from a minimal wavelength, a dwave and a number of elements.
    
    """
    return np.linspace(wave_min, wave_min+(nb_bins-1)*dwave, nb_bins) # the grid of wavelength of your spectrum (assumed equidistant in lambda)


def doppler_r(lamb,v):
    """
    Relativistic doppler shift of a wavelength by a velocity v in kms
    
    """
    factor = np.sqrt((1+1000*v/c_lum)/(1-1000*v/c_lum))
    lambo = lamb*factor
    lambs = lamb*factor**(-1)
    return lambo, lambs 


def empty_ccd_gap(wave,flux,left=None,right=None, extended=30):
    """
    Ensure a 0 value in the gap between the ccd of HARPS s1d with extended=30 kms extension
    """
    dgrid = np.diff(wave)[0]
    
    if left is not None:
        left = np.array(left).astype('float')
        left = doppler_r(left,-extended)[0] #30 km/s supression
    else:
        left = wave.max()
    
    if right is not None:
        right = np.array(right).astype('float')
        right = doppler_r(right,extended)[0] #30 km/s supression
    else:
        right = wave.min()
    
    flux[(wave>=left-dgrid/2)&(wave<=right+dgrid/2)] = 0
    return flux


def find_nearest(array,value,dist_abs=True):
    """
    Find the closest value in an array
    
    Parameters
    ----------
    
    dist_abs : provide the distance output in absolute
    
    Return
    ------
    
    index of th closest element, value and distance
    
    """
    if type(value)!=np.ndarray:
        value = np.array([value])
    idx = np.argmin((np.abs(array-value[:,np.newaxis])),axis=1)
    distance = abs(array[idx]-value) 
    if dist_abs==False:
        distance = array[idx]-value
    return idx, array[idx], distance


def gaussian(x, cen, amp, offset, wid):
    return amp * np.exp(-0.5*(x-cen)**2 / wid**2)+offset


def conv_time(time):    
    time = np.array(time)
    if (type(time[0])==np.float64)|(type(time[0])==np.int64):
        fmt='mjd'
        if time[0]<2030:
            fmt='decimalyear'
        elif time[0]<50000:
            time+=50000
        if fmt=='mjd':
            t0 = time
            t1 = np.array([Time(i, format=fmt).decimalyear for i in time])
            t2 = np.array([Time(i, format=fmt).isot for i in time])
        else:
            t0 = np.array([Time(i, format=fmt).mjd for i in time])
            t1 = time
            t2 = np.array([Time(i, format=fmt).isot for i in time])            
    elif type(time[0])==np.str_:
        fmt='isot'
        t0 = np.array([Time(i, format=fmt).jd-2400000 for i in time]) 
        t1 = np.array([Time(i, format=fmt).decimalyear for i in time])
        t2 = time  
    return t0,t1,t2

def find_iso_in_filename(filename):
    all_cut = [filename[i:i+23] for i in range(len(filename)-23)]
    t=0
    for string in all_cut:
        try:
            t = conv_time([string])[0][0]
        except ValueError:
            pass
    return t

def grouping(array, tresh, num):
    difference = abs(np.diff(array))
    cluster = (difference<tresh)
    indice = np.arange(len(cluster))[cluster]
    
    j = 0
    border_left = [indice[0]]
    border_right = []
    while j < len(indice)-1:
        if indice[j]==indice[j+1]-1:
            j+=1
        else:
            border_right.append(indice[j])
            border_left.append(indice[j+1])
            j+=1
    border_right.append(indice[-1])        
    border = np.array([border_left,border_right]).T
    border = np.hstack([border,(1+border[:,1]-border[:,0])[:,np.newaxis]])
    
    kept = []
    for j in range(len(border)):
        if border[j,-1]>=num:
            kept.append(array[border[j,0]:border[j,1]+2])
    return np.array(kept,dtype='object'), border 


def local_max(spectre,vicinity):
    vec_base = spectre[vicinity:-vicinity]
    maxima = np.ones(len(vec_base))
    for k in range(1,vicinity):
        maxima *= 0.5*(1+np.sign(vec_base - spectre[vicinity-k:-vicinity-k]))*0.5*(1+np.sign(vec_base - spectre[vicinity+k:-vicinity+k]))
    
    index = np.where(maxima==1)[0]+vicinity
    flux = spectre[index]
    return np.array([index,flux])


def import_files_mcpu_wrapper(args):
   return import_files_mcpu(*args)


def import_files_mcpu(file_liste,kind):
    file_liste = file_liste.tolist()
    #print(file_liste)
    sub = []
    snr = []
    for j in file_liste:
        file = open_pickle(j)
        snr.append(file['parameters']['SNR_5500'])
        sub.append(file['output'][kind])
    return sub, snr 

    
def intersect_all_continuum_sphinx(names, master_spectrum=None, copies_master=0, kind='anchor_index', 
                            nthreads=6, fraction=0.2, threshold = 0.66, tolerance=0.5, add_new = True, feedback=True): 
    """Search for the interction of all the anchors points in a list of filename
    For each anchor point the fraction of the closest distance to a neighbourhood is used. 
    Anchor point under the threshoold are removed (1 = most deleted, 0 = most kept).
    Possible to use multiprocessing with nthreads cpu.
    If you want to fix the anchors points, enter a master spectrum path and the number of copies you want of it. 
    """      
    
    print('Loading of the files, wait ... \n')
    names = np.sort(names)
    sub_dico = 'output'
    
    directory, dustbin = os.path.split(names[0])
    directory = '/'.join(directory.split('/')[0:-1])+'/MASTER/'
    
    previous_files = glob.glob(directory+'Master_tool*.p')
    for file_to_delete in previous_files:
        os.system('rm '+file_to_delete)
    
    save = []
    snr = []
    
    if nthreads >= multicpu.cpu_count():
        print('Your number of cpu (%s) is smaller than the number your entered (%s), enter a smaller value please'%(multicpu.cpu_count(),nthreads))
    else:
        if float(py_ver)<3.7:
            chunks = np.array_split(names,nthreads)
            pool = multicpu.Pool(processes=nthreads)
            product = pool.map(import_files_mcpu_wrapper, zip(chunks, repeat(kind)))
        else:
            product = [import_files_mcpu(names,kind)] #multiprocess not work for some reason, go back to classical loop 

        for j in range(len(product)):
            save = save + product[j][0]
            snr = snr + product[j][1]
    
    snr =  np.array(snr)
            
    if master_spectrum is not None:
        if copies_master==0:
            print('You have to specify the number of copy of the master file you want as argument.')
            copies_master = 2*len(names)
            print('[WARNING] Default value of master copies fixed at %.0f.'%(copies_master))
        file = open_pickle(master_spectrum)
        for j in range(copies_master):
            names = np.hstack([names,master_spectrum])
            save.append(file[sub_dico][kind])     
    
    sum_mask = []
    all_idx = np.hstack(save)
    
    #names = np.array(names)[np.array(snr).argsort()[::-1]]
    save = np.array(save,dtype='object')#[np.array(snr).argsort()[::-1]]
    #snr  = np.array(snr)[np.array(snr).argsort()[::-1]]

    print('Computation of the intersection of the anchors points, wait ... \n')
    for j in range(len(names)):
        #plt.scatter(save[j],j*np.ones(len(save[j])))
        diff = np.min([np.diff(save[j][1:]),np.diff(save[j][0:-1])],axis=0)
        diff = np.array([diff[0]]+list(diff)+[diff[-1]])
        diff = diff*fraction
        diff = diff.astype('int')
        mask = np.zeros(len(open_pickle(names[0])['wave']))
        new = []
        for k in range(len(save[j])):
            new.append(save[j][k] + np.arange(-diff[k],diff[k]))
        new = np.unique(np.hstack(new))
        new = new[(new>0)&(new<len(mask))]
        mask[new.astype('int')] = 1
        sum_mask.append(mask)
    sum_mask = np.array(sum_mask)
    sum_mask_vert = np.sum(sum_mask,axis=0)
        
    strat = np.linspace(int(sum_mask_vert.min()),int(sum_mask_vert.max()),10).astype('int')
    strat = strat[strat!=strat[1]] #supress the first level
    
    for j in range(len(strat)-1)[::-1]:
        sum_mask_vert[(sum_mask_vert>=strat[j])&(sum_mask_vert<strat[j+1])] = strat[j]
    for j in range(len(strat[0:-1]))[::-1]:
        sum_mask_vert[sum_mask_vert==strat[0:-1][j]] = strat[j+1]
        
    #sum_mask_vert -= np.diff(strat)[0]
    sum_mask_vert[sum_mask_vert==np.unique(sum_mask_vert)[0]] = 0
    
    mask_vert2 = sum_mask_vert.copy()
    mask_vert2[0] = 0 ; mask_vert2[-1] = 0
    
    for j in range(len(mask_vert2)-2):
        j+=1
        if (mask_vert2[j]>mask_vert2[j-1])|(mask_vert2[j]>mask_vert2[j+1]): #supression of delta peak (not possible because all the peaks cannot be situated at the exact same wavelength), allow the use of grouping function here after
            mask_vert2[j] = np.max([mask_vert2[j-1],mask_vert2[j+1]])

    val, border = grouping(mask_vert2,1,1)
    border = np.hstack([border,np.array([i[0] for i in val])[:,np.newaxis]])
    
    null = np.where(border[:,-1]==0)[0]

    area = []
    small_length = []
    small_center = []
    center2 = []
    for j in range(len(null)-1):
        area.append(np.sum(border[null[j]+1:null[j+1],2]*border[null[j]+1:null[j+1],3]))
        peak = np.where(border[null[j]+1:null[j+1],3]==(border[null[j]+1:null[j+1],3].max()))[0]
        peak = peak[border[null[j]+1:null[j+1],2][peak].argmax()]
        small_length.append(border[null[j]+1:null[j+1],2][peak])
        small_center.append(border[null[j]+1:null[j+1],0][peak]+small_length[-1]/2)
        center2.append(np.median(all_idx[(all_idx>border[null[j]+1,1])&(all_idx<border[null[j+1],0])]))
    center2 = np.round(center2,0).astype('int')

    left = border[null,1][0:-1]
    right = border[null,0][1:]
    
    center2 = []
    for i,j in zip(left,right):
        center2.append(np.median(all_idx[(all_idx>=i)&(all_idx<=j)]))
    center2 = np.round(center2,0).astype('int')
    center2[center2<0]=0
    
    large_length = (right-left)
    large_center = left + large_length/2
    
    center = np.mean(np.array([[small_center],[large_center]]),axis=0)[0]
    windows = np.mean(np.array([[small_length],[large_length]]),axis=0)[0]
    height = area / windows
        
    center = np.round(center,0).astype('int')
    windows = np.round(windows/2,0).astype('int')
    height = np.round(height,0).astype('int')
    
    center = center2
    
    fig = plt.figure(figsize=(14,7))
    plt.subplots_adjust(left=0.10, bottom=0.25,top=0.95,hspace=0.30)
    plt.subplot(2,1,1)
    file_to_plot = open_pickle(names[snr.argsort()[-1]])
    plt.plot(file_to_plot['wave'],file_to_plot['flux']/file_to_plot[sub_dico]['continuum_linear'],color='k')
    ax = plt.gca()
    plt.ylabel('Flux normalised',fontsize=14)
    plt.title('Selection of the clusters',fontsize=14)
        
    plt.subplot(2,1,2,sharex=ax)
    for i,j in enumerate(names): 
        file_to_read = open_pickle(j)
        plt.scatter(file_to_read[sub_dico]['anchor_wave'],i*np.ones(len(file_to_read[sub_dico]['anchor_wave'])),alpha=0.5) 
    plt.plot(file_to_plot['wave'],sum_mask_vert,color='g')
    plt.axhline(y=(len(names)*1.02),color='r')
    plt.xlabel(r'Wavelength [$\AA$]',fontsize=14)
    plt.ylabel('N° of the spectrum',fontsize=14)
    
    stack_vert3 = np.zeros(len(sum_mask_vert))

    stack_vert = np.zeros(len(sum_mask_vert))
    for j in range(len(save)):
        stack_vert[save[j]] += 1
    
    for j in range(len(center)):
        if center[j] != np.nan:
            stack_vert3[center[j]-windows[j]:center[j]+windows[j]+1] = height[j]
    
    stack_vert3[stack_vert3<(len(names)*threshold)] = 0
    stack_vert3[stack_vert3>(len(names))] = len(names)
    val, border = grouping(stack_vert3,1,1)
    border = np.hstack([border,np.array([i[0] for i in val])[:,np.newaxis]])
    
    for j in range(len(border)):
        if np.sum(stack_vert[int(border[j,0]):int(border[j,1])+1])<(border[j,3]*tolerance):
            stack_vert3[int(border[j,0]):int(border[j,1])+1] = 0
    
    liste = []
    for j in range(len(stack_vert3)-2):
        j+=1
        if (stack_vert3[j]!=stack_vert3[j-1])|(stack_vert3[j]!=stack_vert3[j+1]):
            liste.append(j+1)
    liste = np.array(liste)
    if len(liste)!=0:
        stack_vert3[liste] = 0
    
    val, border = grouping(stack_vert3,1,1)
    border = np.hstack([border,np.array([i[0] for i in val])[:,np.newaxis]])
    
    nb_cluster = np.sum(border[:,-1]!=0)

    gri = file_to_plot['wave']
    l1, = plt.plot(gri, stack_vert3, color='k',lw=2)
    l2, = plt.plot([gri.min(),gri.max()],[len(names)*threshold]*2,color='b')
    plt.axes([0.37,0.57,0.05,0.05])
    plt.axis('off')
    l3 = plt.text(0,0,'Nb of cluster detected : %.0f'%(nb_cluster),fontsize=14)
    axcolor = 'whitesmoke'
    
    axtresh = plt.axes([0.1, 0.12, 0.30, 0.03], facecolor = axcolor)
    stresh = Slider(axtresh, 'Threshold', 0, 1, valinit = threshold, valstep=0.05)

    axtolerance = plt.axes([0.1, 0.05, 0.30, 0.03], facecolor = axcolor)
    stolerance = Slider(axtolerance, 'Tolerance', 0, 1, valinit = tolerance, valstep=0.05)
    
    resetax = plt.axes([0.8, 0.05, 0.1, 0.1])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    class Index():
        nb_clust = nb_cluster
        def update(self,val):
            tresh = stresh.val
            tol = 1 - stolerance.val
            
            stack_vert3 = np.zeros(len(sum_mask_vert))

            stack_vert = np.zeros(len(sum_mask_vert))
            for j in range(len(save)):
                stack_vert[save[j]] += 1

            for j in range(len(center)):
                if center[j] != np.nan:
                    stack_vert3[center[j]-windows[j]:center[j]+windows[j]+1] = height[j]
            
            stack_vert3[stack_vert3<(len(names)*tresh)] = 0
            stack_vert3[stack_vert3>(len(names))] = len(names)
            val, border = grouping(stack_vert3,1,1)
            border = np.hstack([border,np.array([i[0] for i in val])[:,np.newaxis]])
            
            for j in range(len(border)):
                if np.sum(stack_vert[int(border[j,0]):int(border[j,1])+1])<(border[j,3]*tol):
                    stack_vert3[int(border[j,0]):int(border[j,1])+1] = 0
            
            liste = []
            for j in range(len(stack_vert3)-2):
                j+=1
                if (stack_vert3[j]!=stack_vert3[j-1])|(stack_vert3[j]!=stack_vert3[j+1]):
                    liste.append(j+1)
            liste = np.array(liste)
            if len(liste)!=0:
                stack_vert3[liste] = 0
            
            val, border = grouping(stack_vert3,1,1)
            border = np.hstack([border,np.array([i[0] for i in val])[:,np.newaxis]])
            
            nb_cluster = np.sum(border[:,-1]!=0)
            
            l3.set_text('Nb of cluster detected : %.0f'%(nb_cluster))
            l1.set_ydata(stack_vert3)
            l2.set_ydata([len(names)*tresh]*2)
            
            self.nb_clust = nb_cluster
            
            fig.canvas.draw_idle()
    
    callback = Index()
    stresh.on_changed(callback.update)
    stolerance.on_changed(callback.update)
    
    def reset(event):
        stolerance.reset()
        stresh.reset()

    button.on_clicked(reset)
    
    plt.show(block=False)
    
    if feedback:
        make_sound('The sphinx is waiting on you...')
        sphinx('Press ENTER to save the clusters locations (black curve)')
        
    while type(callback.nb_clust)==str:
        print('[WARNING] You have to move the sliders at least once before validating clusters')
        make_sound('Warning')
        sphinx('Press ENTER to save the clusters locations (black curve)')
    
    time.sleep(0.5)
    plt.close()
    time.sleep(0.5)
    
    threshold = stresh.val
    tolerance = stolerance.val
    
    #after plot
    
    val2, border2 = grouping(l1.get_ydata(),1,1)
    border2 = np.hstack([border2,np.array([i[0] for i in val2])[:,np.newaxis]])
    border2 = border2[border2[:,-1] > 0]
    curve = (l1.get_ydata() > 0)
    
    plt.close()   

    output_cluster = {'curve':curve,'border':border2, 'wave':gri,
                      'threshold':'%.2f'%(threshold), 'tolerance':'%.2f'%(tolerance),'fraction':'%.2f'%(fraction),
                      'nb_copies_master':copies_master, 'master_filename':master_spectrum}
    
    save_pickle('',output_cluster)
    
    tool_name = 'Master_tool_%s.p'%(time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()))

    save_pickle(directory+'/'+tool_name, output_cluster)
    
def intersect_all_continuum(names, add_new = True): 
    print('Extraction of the new continua, wait... \n')

    names = np.sort(names)
    sub_dico = 'output'
    
    directory, dustbin = os.path.split(names[0])
    directory = '/'.join(directory.split('/')[0:-1])
    
    master_file = glob.glob(directory+'/MASTER/RASSINE_Master_spectrum*')[0]
    names = np.hstack([master_file,names])
    number_of_files = len(names)
    
    tool_file = glob.glob(directory+'/MASTER/Master_tool*')[0]
    tool_name = tool_file.split('/')[-1]
    tool = pd.read_pickle(tool_file)
    
    fraction = tool['fraction']
    tolerance = tool['tolerance']
    threshold = tool['threshold']
    copies_master = tool['nb_copies_master']
    master_spectrum = tool['master_filename']
    border2 = tool['border']
    cluster_center = (border2[:,1]-border2[:,0])/2+border2[:,0]
    curve = tool['curve']
    
    for i,j in enumerate(names):
        valid = True                
        file = open_pickle(j)
        try: 
            valid = (file['matching_anchors']['parameters']['master_tool'] != tool_name)
        except:
            pass
        
        if valid:
            spectrei = file['flux']
            spectre = file['flux_used']        
            grid = file['wave']  
    
            index = file[sub_dico]['anchor_index']
            wave = file[sub_dico]['anchor_wave']
            flux = file[sub_dico]['anchor_flux']
    
            save = index.copy()
            
            diff = np.min([np.diff(save[1:]),np.diff(save[0:-1])],axis=0)
            diff = np.array([diff[0]]+list(diff)+[diff[-1]])
            diff = diff*np.float(fraction)
            diff = diff.astype('int')
            mask = np.zeros(len(grid))
            new = []
            for k in range(len(save)):
                new.append(save[k] + np.arange(-diff[k],diff[k]))
            new = np.unique(np.hstack(new))
            new = new[(new>0)&(new<len(mask))]
            mask[new.astype('int')] = 1
            
            test = mask*curve
            
            mask_idx = test[save].astype('bool')
            mask_idx[0:file['parameters']['number of cut']] = True
            mask_idx[-file['parameters']['number of cut']:] = True
    
            try:
                flux_denoised = file[sub_dico]['anchor_flux_denoised']
            except:
                flux_denoised = flux
            
            save_before = len(index)
            
            index = index[mask_idx]
            wave = wave[mask_idx]
            flux = flux[mask_idx]
            flux_denoised = flux_denoised[mask_idx]
            
            save_after = len(index)
            
            index2 = (index>=border2[:,0][:,np.newaxis])&(index<=border2[:,1][:,np.newaxis])
            cluster_empty = np.where(np.sum(index2,axis=1)==0)[0]
            cluster_twin = np.where(np.sum(index2,axis=1)>1)[0]
    
            if len(cluster_twin)!=0:
                index3 = np.unique(index*index2[cluster_twin,:])[1:]  
                #centers_with_twin = cluster_center[cluster_twin]
                
                index_kept = index3[match_nearest(cluster_center,index3)[:,1].astype('int')] #only twin closest to the cluster is kept
                index_to_suppress = np.setdiff1d(index3,index_kept)
                mask_idx = ~np.in1d(index,index_to_suppress)
                
                index = index[mask_idx]
                wave = wave[mask_idx]
                flux = flux[mask_idx]
                flux_denoised = flux_denoised[mask_idx]
            
            save_after_twin = len(index)
            
            if add_new:
                index_max, flux_max = local_max(spectre,file['parameters']['vicinity_local_max'])
                
                new_max_index = []
                new_max_flux = []
                new_max_flux_denoised = []
                new_max_wave = []
                
                for k in cluster_empty:
                    kept = (index_max>=border2[k,0])&(index_max<=border2[k,1])
                    if sum(kept)!=0:
                        maxi = flux_max[kept].argmax()
                        new_max_index.append((index_max[kept].astype('int'))[maxi])
                        new_max_flux.append((flux_max[kept])[maxi])
                        new_max_wave.append(grid[(index_max[kept].astype('int'))[maxi]])
                        new_max_flux_denoised.append(np.mean(spectre[(index_max[kept].astype('int'))[maxi]-int(file['parameters']['denoising_dist']):(index_max[kept].astype('int'))[maxi]+int(file['parameters']['denoising_dist'])+1]))
                
                new_max_index = np.array(new_max_index)
                new_max_flux = np.array(new_max_flux)
                new_max_flux_denoised = np.array(new_max_flux_denoised)
                new_max_wave = np.array(new_max_wave)         
                
                index = np.hstack([index,new_max_index])
                wave = np.hstack([wave,new_max_wave])
                flux = np.hstack([flux,new_max_flux])
                flux_denoised = np.hstack([flux_denoised,new_max_flux_denoised])  
                        
            save_after_new = len(index)
            print('\nModification of file (%.0f/%.0f): %s \n# of anchor before : %.0f \n# of anchor after out-of-cluster filtering : %0.f \n# of anchor after twin filtering : %.0f \n# of anchor after adding : %.0f'%(i+1,number_of_files,j,save_before,save_after,save_after_twin,save_after_new))
    
            continuum1, continuum3, continuum1_denoised, continuum3_denoised = make_continuum(wave, flux, flux_denoised, grid, spectrei, continuum_to_produce = [file['parameters']['continuum_interpolated_saved'],file['parameters']['continuum_denoised_saved']])
            
            float_precision = file['parameters']['float_precision']
            if float_precision!='float64':
                flux = flux.astype(float_precision)
                wave = wave.astype(float_precision)
                continuum3 = continuum3.astype(float_precision)
                continuum1 = continuum1.astype(float_precision)
                continuum3_denoised = continuum3_denoised.astype(float_precision)
                continuum1_denoised = continuum1_denoised.astype(float_precision)
                flux_denoised = flux_denoised.astype(float_precision)
            index = index.astype('int')
            
            outputs_interpolation_saved = file['parameters']['continuum_interpolated_saved']
            outputs_denoising_saved = file['parameters']['continuum_denoised_saved']
            
            if (outputs_interpolation_saved=='linear')&(outputs_denoising_saved=='undenoised'):    
                output = {'continuum_linear':continuum1,
                         'anchor_wave':wave,
                         'anchor_flux':flux,
                         'anchor_index':index}
            elif (outputs_interpolation_saved=='cubic')&(outputs_denoising_saved=='undenoised'):    
                output = {'continuum_cubic':continuum3,
                         'anchor_wave':wave,
                         'anchor_flux':flux,
                         'anchor_index':index}
            elif (outputs_interpolation_saved=='linear')&(outputs_denoising_saved=='denoised'):    
                output = {'continuum_linear':continuum1_denoised,
                         'anchor_wave':wave,
                         'anchor_flux':flux_denoised,
                         'anchor_index':index}
            elif (outputs_interpolation_saved=='cubic')&(outputs_denoising_saved=='denoised'):     
                output = {'continuum_cubic':continuum3_denoised,
                         'anchor_wave':wave,
                         'anchor_flux':flux_denoised,
                         'anchor_index':index}
            elif (outputs_interpolation_saved=='all')&(outputs_denoising_saved=='denoised'):     
                output = {'continuum_cubic':continuum3_denoised,
                      'continuum_linear':continuum1_denoised,
                      'anchor_wave':wave,
                      'anchor_flux':flux_denoised,
                      'anchor_index':index}
            elif (outputs_interpolation_saved=='all')&(outputs_denoising_saved=='undenoised'):     
                output = {'continuum_cubic':continuum3,
                      'continuum_linear':continuum1,
                      'anchor_wave':wave,
                      'anchor_flux':flux,
                      'anchor_index':index}
            elif (outputs_interpolation_saved=='linear')&(outputs_denoising_saved=='all'):     
                output = {'continuum_linear':continuum1,
                      'continuum_linear_denoised':continuum1_denoised,
                      'anchor_wave':wave,
                      'anchor_flux':flux,
                      'anchor_flux_denoised':flux_denoised,          
                      'anchor_index':index}  
            elif (outputs_interpolation_saved=='cubic')&(outputs_denoising_saved=='all'):     
                output = {'continuum_cubic':continuum3,
                      'continuum_cubic_denoised':continuum3_denoised,
                      'anchor_wave':wave,
                      'anchor_flux':flux,
                      'anchor_flux_denoised':flux_denoised,          
                      'anchor_index':index}      
            else:
                output = {'continuum_cubic':continuum3,
                      'continuum_linear':continuum1,
                      'continuum_cubic_denoised':continuum3_denoised,
                      'continuum_linear_denoised':continuum1_denoised,
                      'anchor_wave':wave,
                      'anchor_flux':flux,
                      'anchor_flux_denoised':flux_denoised,
                      'anchor_index':index}
            
            
            file['matching_anchors'] = output
            file['matching_anchors']['parameters'] = {'master_tool':tool_name,'master_filename':master_spectrum,'sub_dico_used':sub_dico,'nb_copies_master':copies_master,
                                                      'threshold':threshold,'tolerance':tolerance,'fraction':fraction}
            save_pickle(names[i],file)
    
def matching_diff_continuum_sphinx(names, sub_dico = 'matching_anchors', master=None, savgol_window = 200, zero_point=False):
    snr = []
    for j in names:
        file = open_pickle(j)
        snr.append(file['parameters']['SNR_5500'])
    
    names = np.array(names)[np.array(snr).argsort()[::-1]]
    snr  = np.array(snr)[np.array(snr).argsort()[::-1]]
    
    if master is None:
        master = names[0]

    file_highest = open_pickle(master)
    file_highest['matching_diff'] = file_highest[sub_dico]
    
    save_pickle(master,file_highest)

    dx = np.diff(file_highest['wave'])[0]
    length_clip = int(100/dx) #smoothing on 100 \ang for the tellurics clean
    
    file = open_pickle(names[1])
    
    keys = file_highest[sub_dico].keys()
    if 'continuum_cubic' in keys:
        ref_cont = 'continuum_cubic'
    else:
        ref_cont = 'continuum_linear'
        
    cont2  = file_highest['flux_used']/file_highest[sub_dico][ref_cont]
    cont1 = file['flux_used']/file[sub_dico][ref_cont]
    
    all_continuum = ['continuum_linear','continuum_cubic','continuum_linear_denoised','continuum_cubic_denoised']
    continuum_to_reduce = []
    for i in all_continuum:
        if i in keys:
            continuum_to_reduce.append(i)
            
    diff = cont1-cont2
    med_value = np.nanmedian(diff)
    for k in range(3): #needed to avoid artefact induced by tellurics
        q1, q3, iq = rolling_iq(diff,window=length_clip)
        diff[(diff>q3+1.5*iq)|(diff<q1-1.5*iq)] = med_value
        diff[diff<q1-1.5*iq] = med_value
    
    correction = smooth(diff, savgol_window, shape='savgol') 
    correction = smooth(correction, savgol_window, shape='savgol') 
    
    fig = plt.figure(figsize=(14,7))
    plt.subplots_adjust(left=0.10, bottom=0.25,top=0.95,hspace=0.30)
    plt.title('Selection of the smoothing kernel length',fontsize=14)
    plt.plot(file['wave'], diff, color='b',alpha=0.4,label='flux difference')
    l1, = plt.plot(file['wave'], correction, color='k',label='smoothed flux difference (flux correction)')
    plt.xlabel(r'Wavelength [$\AA$]',fontsize=14)
    plt.ylabel(r'$F - F_{ref}$ [normalised flux units]',fontsize=14)
    plt.legend()
    axcolor = 'whitesmoke'
    axsmoothing = plt.axes([0.2, 0.1, 0.40, 0.03], facecolor = axcolor)
    ssmoothing = Slider(axsmoothing, 'Kernel length', 1, 500, valinit = savgol_window, valstep=1)
    
    resetax = plt.axes([0.8, 0.05, 0.1, 0.1])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        
    class Index():
        def update(self,val):
            smoothing = ssmoothing.val
            correction = smooth(diff, smoothing, shape='savgol') 
            correction = smooth(correction, smoothing, shape='savgol') 
            l1.set_ydata(correction)
            fig.canvas.draw_idle()            
    
    callback = Index()
    ssmoothing.on_changed(callback.update)
    
    def reset(event):
        ssmoothing.reset()
    button.on_clicked(reset)
    
    plt.show(block=False)
    make_sound('The sphinx is waiting on you...')
    sphinx('Press ENTER to save the kernel length for the smoothing')
    savgol_window = ssmoothing.val
    plt.close()        
    time.sleep(1)

    return master, savgol_window

def matching_diff_continuum(names, sub_dico = 'matching_anchors', master=None, savgol_window = 200, zero_point=False): 
    """A savgol fitering can be performed on spectra diff 
    (the spectrum with the highest snr is used as reference) to remove remaining fluctuation.
    If zero point = False, the median level of the spectra difference is not removed and only the fluctuation are."""
    
    snr = []
    for j in names:
        file = open_pickle(j)
        snr.append(file['parameters']['SNR_5500'])
    
    names = np.array(names)[np.array(snr).argsort()[::-1]]
    snr  = np.array(snr)[np.array(snr).argsort()[::-1]]
    
    if master is None:
        master = names[0]

    master_name = master.split('/')[-1]
    file_highest = open_pickle(master)
    dx = np.diff(file_highest['wave'])[0]
    length_clip = int(100/dx) #smoothing on 100 \ang for the tellurics clean
    
    keys = file_highest[sub_dico].keys()
            
    all_continuum = ['continuum_linear','continuum_cubic','continuum_linear_denoised','continuum_cubic_denoised']
    continuum_to_reduce = []
    for i in all_continuum:
        if i in keys:
            continuum_to_reduce.append(i)

    for i,j in enumerate(names):
        valid = True                        
        file = open_pickle(j)
        try: 
            valid = (file['matching_diff']['parameters']['reference_continuum'] != master_name)
        except:
            pass
        
        if valid:
            print('Modification of file (%.0f/%.0f) : %s (SNR : %.0f)'%(i+1,len(names),j,snr[i]))
            spectre = file['flux_used']  
            
            par = {'reference_continuum':master_name,'savgol_window':savgol_window,'recenter':zero_point,'sub_dico_used':sub_dico}
            file['matching_diff'] = {'parameters':par}
    
            for label in continuum_to_reduce:
                cont = file[sub_dico][label]
                cont2 = file_highest['flux_used']/file_highest[sub_dico][label]
                
                cont1 = spectre/cont
                diff = cont1-cont2
                med_value = np.nanmedian(diff)
                for k in range(3): #needed to avoid artefact induced by tellurics
                    q1, q3, iq = rolling_iq(diff,window=length_clip)
                    diff[(diff>q3+1.5*iq)|(diff<q1-1.5*iq)] = med_value
                    diff[diff<q1-1.5*iq] = med_value
                
                correction = smooth(diff, savgol_window, shape='savgol') 
                correction = smooth(correction, savgol_window, shape='savgol') 
                if zero_point:
                    correction = correction - np.nanmedian(correction)   
                cont_corr = cont.copy()/(1-correction.copy())                 
                file['matching_diff'][label] = cont_corr
            
            save_pickle(j,file)


def make_continuum(wave, flux, flux_denoised, grid, spectrei, continuum_to_produce = ['all','all']):
    
    continuum1_denoised = np.zeros(len(grid))
    continuum3_denoised = np.zeros(len(grid))
    continuum1 = np.zeros(len(grid))
    continuum3 = np.zeros(len(grid))
    
    if continuum_to_produce[1]!='undenoised':
        if continuum_to_produce[0]!='cubic':
            Interpol1 = interp1d(wave, flux_denoised, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
            continuum1_denoised = Interpol1(grid)
            continuum1_denoised = troncated(continuum1_denoised,spectrei)
            continuum1_denoised = check_none_negative_values(continuum1_denoised)
        if continuum_to_produce[0]!='linear':
            Interpol3 = interp1d(wave, flux_denoised, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')
            continuum3_denoised = Interpol3(grid)
            continuum3_denoised = troncated(continuum3_denoised,spectrei)
            continuum3_denoised = check_none_negative_values(continuum3_denoised)        

    if continuum_to_produce[1]!='denoised':
        if continuum_to_produce[0]!='cubic':
            Interpol1 = interp1d(wave, flux, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
            continuum1 = Interpol1(grid)
            continuum1 = troncated(continuum1,spectrei)
            continuum1 = check_none_negative_values(continuum1)
        if continuum_to_produce[0]!='linear':
            Interpol3 = interp1d(wave, flux, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')
            continuum3 = Interpol3(grid)
            continuum3 = troncated(continuum3,spectrei)
            continuum3 = check_none_negative_values(continuum3)
    
    return continuum1, continuum3, continuum1_denoised, continuum3_denoised    


def make_sound(sentence):
    if type(voice_name)==str:
        try: 
            os.system('say -v '+voice_name+' "'+sentence+'"')
        except:
            print('\7')
    else:
        print('\7')
        

def match_nearest(array1, array2):
    dmin = np.diff(np.sort(array1)).min()
    dmin2 = np.diff(np.sort(array2)).min()
    array1_r = array1 + 0.001*dmin*np.random.randn(len(array1))
    array2_r = array2 + 0.001*dmin2*np.random.randn(len(array2))
    m = abs(array2_r-array1_r[:,np.newaxis])
    arg1 = np.argmin(m,axis=0)
    arg2 = np.argmin(m,axis=1)
    mask = (np.arange(len(arg1)) == arg2[arg1])
    liste_idx1 = arg1[mask]
    liste_idx2 = arg2[arg1[mask]]
    array1_k = array1[liste_idx1]
    array2_k = array2[liste_idx2]
    return np.hstack([liste_idx1[:,np.newaxis],liste_idx2[:,np.newaxis],
                      array1_k[:,np.newaxis],array2_k[:,np.newaxis],(array1_k-array2_k)[:,np.newaxis]]) 


def open_pickle(filename):    
    if filename.split('.')[-1]=='p':
        a = pd.read_pickle(filename)
        return a
    elif filename.split('.')[-1]=='fits':
        data = fits.getdata(filename)
        header = fits.getheader(filename)
        return data, header


def plot_debug(grid,spectre,wave,flux):
    plt.figure()
    plt.plot(grid,spectre)
    plt.scatter(wave,flux,color='k')
    plt.show(block=False)
    input('blabla')
          

def preprocess_fits(files_to_process, instrument='HARPS', plx_mas=0, final_sound=True):
    """Preprocess  the files depending on the s1d format instrument, HARPS, HARPN, CORALIE or ESPRESSO"""
    files_to_process = np.sort(files_to_process)
    number_of_files = len(files_to_process)
    counter = 0
    init_time = time.time()
    
    if (instrument=='HARPS')|(instrument=='CORALIE')|(instrument=='HARPN'):
        
        directory0, dustbin = os.path.split(files_to_process[0])
        try:
            table = pd.read_csv(directory0+'/DACE_TABLE/Dace_extracted_table.csv')
            print('[INFO] DACE table has been found. Reduction will run with it.')
        except FileNotFoundError:
            print('[INFO] The DACE table is missing. Reduction will run without it.')
        
        for spectrum_name in files_to_process:
            
            counter+=1
            if (counter+1)%100==1:
                after_time = time.time()
                time_it = (after_time - init_time)/100
                init_time = after_time
                if time_it>1:
                    print('[INFO] Number of files preprocessed : --- %.0f/%0.f --- (%.2f s/it, remaining time : %.0f min %s s)'%(counter, number_of_files, time_it, ((number_of_files-counter)*time_it)//60, str(int(((number_of_files-counter)*time_it)%60)).zfill(2)))
                else:
                    print('[INFO] Number of files preprocessed : --- %.0f/%0.f --- (%.2f it/s, remaining time : %.0f min %s s)'%(counter,number_of_files,time_it**-1, ((number_of_files-counter)*time_it)//60, str(int(((number_of_files-counter)*time_it)%60)).zfill(2)))
                    
            directory, name = os.path.split(spectrum_name)
            if not os.path.exists(directory+'/PREPROCESSED/'):
                os.system('mkdir '+directory+'/PREPROCESSED/')
            
            header = fits.getheader(spectrum_name) # load the fits header
            spectre = fits.getdata(spectrum_name).astype('float64') # the flux of your spectrum
            spectre_step = np.round(fits.getheader(spectrum_name)['CDELT1'],8)
            wave_min = np.round(header['CRVAL1'],8) # to round float32
            wave_max = np.round(header['CRVAL1']+(len(spectre)-1)*spectre_step,8) # to round float32
            
            grid = np.round(np.linspace(wave_min,wave_max,len(spectre)),8)

            begin = np.min(np.arange(len(spectre))[spectre>0])
            end = np.max(np.arange(len(spectre))[spectre>0])
            grid = grid[begin:end+1]
            spectre = spectre[begin:end+1]
            wave_min = np.min(grid)
            wave_max = np.max(grid)

            kw = 'ESO'
            if instrument=='HARPN':
                kw='TNG'

            berv = header['HIERARCH '+kw+' DRS BERV']
            lamp = header['HIERARCH '+kw+' DRS CAL TH LAMP OFFSET']
            try:
                pma = header['HIERARCH '+kw+' TEL TARG PMA']*1000
                pmd = header['HIERARCH '+kw+' TEL TARG PMD']*1000
            except:
                pma=0
                pmd=0
            
            if plx_mas:
                distance_m = 1000.0/plx_mas*3.08567758e16
                mu_radps = np.sqrt(pma**2+pmd**2)*2*np.pi/(360.0*1000.0*3600.0*86400.0*365.25)
                acc_sec = distance_m*86400.0*mu_radps**2 # rv secular drift in m/s per days
            else:
                acc_sec = 0
            
            if instrument=='CORALIE':
                if np.mean(spectre)<100000:
                    spectre *= 400780143771.18976 #calibrated with HD8651 2016-12-16 AND 2013-10-24
                
                spectre /= (1.4e10/125**2) #calibrated to match with HARPS SNR
            
            try:
                mjd = table.loc[table['fileroot']==spectrum_name,'mjd'].values[0]                
            except NameError:
                try:
                    mjd = header['MJD-OBS']
                except KeyError:
                    mjd = Time(name.split('/')[-1].split('.')[1]).mjd
            
            jdb = np.array(mjd) + 0.5
            
            out = {'flux':spectre, 'flux_err':0*spectre,
                   'instrument':instrument,'mjd':mjd,'jdb':jdb, 
                   'berv':berv, 'lamp_offset':lamp, 'plx_mas':plx_mas,'acc_sec':acc_sec,
                   'wave_min':wave_min,'wave_max':wave_max,'dwave':spectre_step}
            
            save_pickle(directory+'/PREPROCESSED/'+name[:-5]+'.p',out)
            
    elif instrument=='ESPRESSO':
        for spectrum_name in files_to_process:

            directory, name = os.path.split(spectrum_name)
            if not os.path.exists(directory+'/PREPROCESSED/'):
                os.system('mkdir '+directory+'/PREPROCESSED/')

            header = fits.getheader(spectrum_name) # load the fits header
            spectre = fits.getdata(spectrum_name)['flux'].astype('float64') # the flux of your spectrum
            spectre_error = fits.getdata(spectrum_name)['error'].astype('float64') # the flux of your spectrum
            grid = fits.getdata(spectrum_name)['wavelength_air'].astype('float64') # the grid of wavelength of your spectrum (assumed equidistant in lambda)
            begin = np.min(np.arange(len(spectre))[spectre>0]) # remove border spectrum with 0 value
            end = np.max(np.arange(len(spectre))[spectre>0])   # remove border spectrum with 0 value
            grid = grid[begin:end+1]
            spectre = spectre[begin:end+1]
            spectre_error = spectre_error[begin:end+1]
            wave_min = np.min(grid)
            wave_max = np.max(grid)
            spectre_step = np.mean(np.diff(grid))
            mjd = header['MJD-OBS']
            berv = header['HIERARCH ESO QC BERV']
            lamp = 0 #header['HIERARCH ESO DRS CAL TH LAMP OFFSET'] no yet available
            try:
                pma = header['HIERARCH ESO TEL TARG PMA']*1000
                pmd = header['HIERARCH ESO TEL TARG PMD']*1000
            except:
                pma=0
                pmd=0
            
            if plx_mas:
                distance_m = 1000.0/plx_mas*3.08567758e16
                mu_radps = np.sqrt(pma**2+pmd**2)*2*np.pi/(360.0*1000.0*3600.0*86400.0*365.25)
                acc_sec = distance_m*86400.0*mu_radps**2 # rv secular drift in m/s per days
            else:
                acc_sec = 0
            jdb = np.array(mjd) + 0.5
            
            out = {'wave':grid,'flux':spectre,'flux_err':spectre_error,
                   'instrument':instrument,'mjd':mjd,'jdb':jdb,
                   'berv':berv,'lamp_offset':lamp,'plx_mas':plx_mas,'acc_sec':acc_sec,
                   'wave_min':wave_min,'wave_max':wave_max,'dwave':spectre_step}
            
            save_pickle(directory+'/PREPROCESSED/'+name[:-5]+'.p',out)
    elif instrument=='CSV':
        for spectrum_name in files_to_process:
            
            counter+=1
            if (counter+1)%100==1:
                after_time = time.time()
                time_it = (after_time - init_time)/100
                init_time = after_time
                if time_it>1:
                    print('[INFO] Number of files preprocessed : --- %.0f/%0.f --- (%.2f s/it, remaining time : %.0f min %s s)'%(counter, number_of_files, time_it, ((number_of_files-counter)*time_it)//60, str(int(((number_of_files-counter)*time_it)%60)).zfill(2)))
                else:
                    print('[INFO] Number of files preprocessed : --- %.0f/%0.f --- (%.2f it/s, remaining time : %.0f min %s s)'%(counter,number_of_files,time_it**-1, ((number_of_files-counter)*time_it)//60, str(int(((number_of_files-counter)*time_it)%60)).zfill(2)))
                    
            directory, name = os.path.split(spectrum_name)
            if not os.path.exists(directory+'/PREPROCESSED/'):
                os.system('mkdir '+directory+'/PREPROCESSED/')
            
            tab = pd.read_csv(spectrum_name) # the flux of your spectrum
            spectre = np.array(tab['flux'].astype('float64'))
            if 'flux_std' in tab.keys():
                spectre_error = np.array(tab['flux_std'].astype('float64'))
            else:
                spectre_error = spectre*0
            grid = np.array(tab['wave'].astype('float64')) # the grid of wavelength of your spectrum (assumed equidistant in lambda)
            begin = np.min(np.arange(len(spectre))[spectre>0]) # remove border spectrum with 0 value
            end = np.max(np.arange(len(spectre))[spectre>0])   # remove border spectrum with 0 value
            grid = grid[begin:end+1]
            spectre = spectre[begin:end+1]
            spectre_error = spectre_error[begin:end+1]

            wave_min = np.min(grid)
            wave_max = np.max(grid)      
            spectre_step = np.mean(np.diff(grid))

            jdb = find_iso_in_filename(spectrum_name)
            mjd = (jdb-0.5)*(1-int(jdb==0))

            out = {'wave':grid,'flux':spectre,'flux_err':spectre_error,
                   'instrument':instrument,'mjd':mjd,'jdb':jdb,
                   'berv':0,'lamp_offset':0,'plx_mas':0,'acc_sec':0,
                   'wave_min':wave_min,'wave_max':wave_max,'dwave':spectre_step}
            
            save_pickle(directory+'/PREPROCESSED/'+name[:-4]+'.p',out)

    
    if final_sound:
        make_sound('Preprocessing files has finished')



def preprocess_prematch_stellar_frame(files_to_process, rv=0, dlambda=None):
    """compute needed information to all the spectra of a star in the same stellar frame by specifying the rv in km/s and the dlambda to form the grid in angstrom"""
    
    files_to_process = np.sort(files_to_process)
    
    emergency = 1

    if np.max(abs(rv))>300:
        make_sound('Warning')
        print('\n [WARNING] RV are certainly is m/s instead of km/s ! ')
        rep = sphinx('Are you sure your RV are in km/s ? Purchase with these RV ? (y/n)',rep=['y','n'])    
        if rep=='n':
            emergency=0

    rv_mean = np.median(rv)
    rv -= rv_mean
 
    if type(rv)!=np.array:
        rv = np.hstack([rv])
        if len(rv)==1:
            rv = np.ones(len(files_to_process))*rv
        
    if (len(rv)!=len(files_to_process)):
        make_sound('Warning')
        print('\n [WARNING] RV vector is not the same size than the number of files ! ')
        emergency=0       
    
    if emergency:
        all_length = [] ; wave_min = [] ; wave_max = []
        hole_left = [] ; hole_right = []        
        diff = [] ; berv = [] ; lamp = []
        plx_mas = [] ; acc_sec =[] 
        
        i = -1
        print('Loading the data, wait... \n')  
        nb_total = len(files_to_process)
        for j in files_to_process:
            i+=1
            if not (i%250):
                print(' [INFO] Number of files processed : %s/%.0f (%.1f %%)'%(str(i).zfill(len(str(nb_total))),nb_total,100*i/nb_total))
            f = open_pickle(j)
              
            shift = rv[i]*(len(np.ravel(rv))!=1)
                        
            flux = f['flux']
            try:
                wave = f['wave']
                diff.append(np.unique(np.diff(wave)))
            except KeyError:
                wave = create_grid(f['wave_min'],f['dwave'],len(flux))
                diff.append(f['dwave'])
            wave_min.append(f['wave_min'])      
            wave_max.append(f['wave_max'])      
            all_length.append(len(wave))
            berv.append(f['berv'])
            lamp.append(f['lamp_offset'])
            plx_mas.append(f['plx_mas'])
            acc_sec.append(f['acc_sec'])

            
            null_flux = np.where(flux==0)[0] #criterion to detect gap between ccd
            if len(null_flux):
                mask = grouping(np.diff(null_flux),0.5,0)[-1]
                highest = mask[mask[:,2].argmax()]
                if highest[2]>1000:
                    left = wave[int(null_flux[highest[0]])]
                    right = wave[int(null_flux[highest[1]])]

                    hole_left.append(find_nearest(wave, doppler_r(left,shift)[1])[1])
                    hole_right.append(find_nearest(wave, doppler_r(right,shift)[1])[1])
                
        if len(hole_left)!=0:
            hole_left_k = np.min(hole_left)-0.5 #increase of 0.5 the gap limit by security
            hole_right_k = np.max(hole_right)+0.5 #increase of 0.5 the gap limit by security
            make_sound('Warning')
            print('\n [WARNING] GAP detected in s1d between : %.2f and %.2f ! '%(hole_left_k, hole_right_k))
            #rep = sphinx('Do you confirm these limit for the CCD gap ? (y/n)',rep=['y','n'])    
            #if rep=='n':
            #    hole_left_k = -99.9
            #    hole_right_k = -99.9               
        else:
            hole_left_k = -99.9
            hole_right_k = -99.9
        
        berv = np.array(berv)
        lamp = np.array(lamp)
        plx_mas = np.array(plx_mas)
        acc_sec = np.array(acc_sec)
        
        wave_min = np.round(wave_min,8)#to take into account float32 
        wave_max = np.round(wave_max,8)#to take into account float32 
        wave_min_k = np.array(wave_min).max()
        wave_max_k = np.array(wave_max).min() 
        print('\n [INFO] Spectra borders are found between : %.4f and %.4f'%(wave_min_k, wave_max_k))
       
        if dlambda is None:
            value = np.unique(np.round(np.hstack(diff),8))

            if len(value)==1:
                dlambda = value[0]
                print('\n [INFO] Spectra dwave is : %.4f \n'%(dlambda))
            else:
                make_sound('Warning')
                print('\n [WARNING] The algorithm has not managed to determine the dlambda value of your spectral wavelength grid')
                dlambda = sphinx('Which dlambda value are you selecting for the wavelength grid ?') 
                dlambda = np.round(np.float(dlambda),8)
        else:
            value = np.array([69,69])

        if len(value)==1: #case with static wavelength grid
            static_grid = None
        else:
            static_grid = np.arange(wave_min_k, wave_max_k + dlambda, dlambda)
            
        return wave_min_k, wave_max_k, dlambda, hole_left_k, hole_right_k, static_grid, wave_min, berv, lamp, plx_mas, acc_sec, rv, rv_mean
                
    
def preprocess_match_stellar_frame(files_to_process, args=None, rv=0, dlambda=None, final_sound=True):
    """process all the spectra of a star in the same stellar frame by specifying the rv in km/s and the dlambda to form the grid in angstrom"""
    
    files_to_process = np.sort(files_to_process)
    
    number_of_files = len(files_to_process)
    
    if args is None:
        args = preprocess_prematch_stellar_frame(files_to_process, rv=rv, dlambda=dlambda)
    
    wave_min_k, wave_max_k, dlambda, hole_left_k, hole_right_k, static_grid, wave_min, berv, lamp, plx_mas, acc_sec, rv, rv_mean = args
               
    wave_ref = int(find_nearest(np.arange(wave_min_k,wave_max_k+dlambda,dlambda),5500)[0])
    
    print('Extraction of the new spectra, wait... \n')
                
    init_time = time.time()
    k=-1
    for name in files_to_process:
        k+=1
        if (k+2)%100==1:
            after_time = time.time()
            time_it = (after_time - init_time)/100
            init_time = after_time
            counter = k+1
            if time_it>1:
                print('[INFO] Number of files preprocessed : --- %.0f/%0.f --- (%.2f s/it, remaining time : %.0f min %s s)'%(counter, number_of_files, time_it, ((number_of_files-counter)*time_it)//60, str(int(((number_of_files-counter)*time_it)%60)).zfill(2)))
            else:
                print('[INFO] Number of files preprocessed : --- %.0f/%0.f --- (%.2f it/s, remaining time : %.0f min %s s)'%(counter, number_of_files,time_it**-1, ((number_of_files-counter)*time_it)//60, str(int(((number_of_files-counter)*time_it)%60)).zfill(2)))
                
        f = open_pickle(name)
        flux = f['flux']
        flux_err = f['flux_err']

        if static_grid is None:
            wave = create_grid(wave_min[k], dlambda,len(flux))
            grid = wave.copy()
        else:
            try:
                wave = f['wave']
            except:
                wave = create_grid(f['wave_min'],f['dwave'],len(flux))
            grid = static_grid
                        
        if (rv[k]!=0)|(len(grid)!=len(wave)):
            nflux = interp1d(doppler_r(wave,rv[k])[1], flux, kind='cubic', bounds_error=False, fill_value='extrapolate')(grid)
            nflux_err = interp1d(doppler_r(wave,rv[k])[1], flux_err, kind='linear', bounds_error=False, fill_value='extrapolate')(grid)
        else:
            nflux = flux
            nflux_err = flux_err

        if static_grid is not None:
            wave = static_grid
        
        mask = (wave>=(wave_min_k-dlambda/2.))&(wave<=(wave_max_k+dlambda/2.))
        mask2 = (wave>=(hole_left_k-dlambda/2.))&(wave<=(hole_right_k+dlambda/2.))
        nflux[mask2] = 0
        new_flux = nflux[mask]
        nflux_err[mask2] = 1
        new_flux_err = nflux_err[mask]
        
        continuum_5500 = np.nanpercentile(new_flux[wave_ref-50:wave_ref+50],95)
        SNR = np.sqrt(continuum_5500)
        save_pickle(name,{'flux':new_flux, 'flux_err':new_flux_err,
                       'RV_sys':rv_mean, 'RV_shift':rv[k], 'SNR_5500':SNR, 
                       'berv':berv[k], 'lamp_offset':lamp[k], 'plx_mas':plx_mas[k], 'acc_sec':acc_sec[k],
                       'instrument':f['instrument'], 'mjd':f['mjd'], 'jdb':f['jdb'],
                       'hole_left':hole_left_k, 'hole_right':hole_right_k,
                       'wave_min':wave_min_k,'wave_max':wave_max_k,'dwave':dlambda})
    if final_sound:
        make_sound('Matching stellar spectra has finished')

def preprocess_prestacking(files_to_process, bin_length = 1, dbin = 0):
    """Stack the s1d spectras by bin_length in days. 
    Use dbin to shift the zero point (useful for solar observation).
    Define the zero point of the counter for the product name. 
    """
    
    files_to_process = np.sort(files_to_process)
    directory, dustbin = os.path.split(files_to_process[0])
    directory = '/'.join(directory.split('/')[0:-1])+'/STACKED/'
    
    if not os.path.exists(directory):
        os.system('mkdir '+directory)    
    
    print('Loading the data, wait... \n')    
    jdb = []
    berv = []
    lamp = []
    plx = []
    acc_sec = []
    
    for name in files_to_process:
        jdb.append(pd.read_pickle(name)['jdb'])
        berv.append(pd.read_pickle(name)['berv'])
        lamp.append(pd.read_pickle(name)['lamp_offset'])
        plx.append(pd.read_pickle(name)['plx_mas'])
        acc_sec.append(pd.read_pickle(name)['acc_sec'])

    jdb = np.array(jdb) + dbin
    berv = np.array(berv)
    lamp = np.array(lamp)
    plx = np.array(plx)
    acc_sec = np.array(acc_sec)
    
    if bin_length==0:
        group = np.arange(len(jdb))
        groups = np.arange(len(jdb))
    else:
        groups = (jdb//bin_length).astype('int')
        groups -= groups[0]
        group = np.unique(groups)

    print('Number of bins : %.0f'%(len(group)))
    
    return jdb, berv, lamp, acc_sec, groups
    
def preprocess_stack(files_to_process, bin_length = 1, dbin = 0, make_master=True):
    
    """Stack the s1d spectras by bin_length in days. 
    Use dbin to shift the zero point (useful for solar observation).
    Define the zero point of the counter for the product name. 
    """
    
    files_to_process = np.sort(files_to_process)
    directory, dustbin = os.path.split(files_to_process[0])
    directory = '/'.join(directory.split('/')[0:-1])+'/STACKED/'
    
    jdb, berv, lamp, rv_sec, groups = preprocess_prestacking(files_to_process, bin_length = bin_length, dbin = dbin)
    
    group = np.unique(groups)
    
    num=-1

    all_snr = []
    all_stack = []
    all_berv = []
    for j in group:
        num+=1
        g = np.where(groups==j)[0]
        file_arbitrary = pd.read_pickle(files_to_process[0])
        wave_min =  try_field(file_arbitrary,'wave_min')
        wave_max =  try_field(file_arbitrary,'wave_max')
        dwave =  try_field(file_arbitrary,'dwave')
        grid = create_grid(wave_min, dwave, len(file_arbitrary['flux']))
        RV_sys =  try_field(file_arbitrary,'RV_sys')
        instrument =  try_field(file_arbitrary,'instrument')
        hole_left =  try_field(file_arbitrary,'hole_left')
        hole_right =  try_field(file_arbitrary,'hole_right')
        acc_sec =  try_field(file_arbitrary,'acc_sec')
        stack = 0
        bolo = []     
        rv_shift = []
        name_root_files = []
        for file in files_to_process[g]:
            f = pd.read_pickle(file)
            flux = f['flux']    
            rv_shift.append(try_field(f,'RV_shift'))
            stack += flux
            bolo.append(np.nansum(flux)/len(flux))
            name_root_files.append(file)
        
        bolo = np.array(bolo)
        rv_shift = np.array(rv_shift)
        wave_ref = int(find_nearest(grid,5500)[0])
        continuum_5500 = np.nanpercentile(stack[wave_ref-50:wave_ref+50],95)
        SNR = np.sqrt(continuum_5500)
        all_snr.append(SNR)
        all_stack.append(stack)
        nb_spectra_stacked = len(g)
        jdb_w = np.sum((jdb[g]-dbin)*bolo)/np.sum(bolo)
        date_name = Time(jdb_w-0.5,format='mjd').isot
        berv_w = np.sum(berv[g]*bolo)/np.sum(bolo)
        lamp_w = np.sum(lamp[g]*bolo)/np.sum(bolo)
        all_berv.append(berv_w)
        out = {'flux':stack,
               'jdb':jdb_w, #mjd weighted average by bolo flux
               'mjd':jdb_w - 0.5, #mjd weighted average by bolo flux
               'berv':berv_w,
               'lamp_offset':lamp_w,
               'acc_sec':acc_sec,
               'RV_shift':np.sum(rv_shift*bolo)/np.sum(bolo),
               'RV_sys':RV_sys,'SNR_5500':SNR, 
               'hole_left':hole_left, 'hole_right':hole_right,
               'wave_min':wave_min,'wave_max':wave_max,'dwave':dwave,
               'stacking_length':bin_length,
               'nb_spectra_stacked':nb_spectra_stacked,
               'arcfiles':name_root_files} 
        
        if len(group)!=len(files_to_process):
            save_pickle(directory+'/Stacked_spectrum_bin_'+str(bin_length)+'.'+date_name+'.p',out)
        else:
            save_pickle(directory+'/Prepared_'+files_to_process[num].split('/')[-1],out)
    
    all_snr = np.array(all_snr)

    print('SNR 5500 statistic (Q1/Q2/Q3) : Q1 = %.0f / Q2 = %.0f / Q3 = %.0f'%(np.nanpercentile(all_snr,25),np.nanpercentile(all_snr,50),np.nanpercentile(all_snr,75)))

    master_name=None

    if make_master:
        all_berv = np.array(all_berv)
        stack = np.array(all_stack)
        stack = np.sum(stack,axis=0)
        stack[stack<=0] = 0
        continuum_5500 = np.nanpercentile(stack[wave_ref-50:wave_ref+50],95)
        SNR = np.sqrt(continuum_5500)
        BERV = np.sum(all_berv*all_snr**2)/np.sum(all_snr**2)
        BERV_MIN = np.min(berv)
        BERV_MAX = np.max(berv)
        #plt.figure(figsize=(16,6))
        #plt.plot(grid,stack,color='k')

        master_name = 'Master_spectrum_%s.p'%(time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()))

        save_pickle(directory+'/'+master_name,{'flux':stack, 'master_spectrum':True,
                                                    'RV_sys':RV_sys, 'RV_shift':0, 'SNR_5500':SNR, 
                                                    'lamp_offset':0,'acc_sec':acc_sec,
                                                    'berv':BERV, 'berv_min':BERV_MIN,'berv_max':BERV_MAX,
                                                    'instrument':instrument, 'mjd':0, 'jdb':0,
                                                    'hole_left':hole_left, 'hole_right':hole_right,
                                                    'wave_min':wave_min,'wave_max':wave_max,'dwave':dwave,
                                                    'nb_spectra_stacked':len(files_to_process),'arcfiles':'none'})
        #plt.xlabel(r'Wavelength [$\AA$]',fontsize=14)
        #plt.ylabel('Flux',fontsize=14)
        #plt.show()
        #loop = sphinx('Press Enter to finish the stacking process.')
        #plt.close()

    make_sound('Stacking spectra has finished')
    return master_name

def postprocess_tofits(path_input,path_rassine,anchor_mode,continuum_mode):
    
    files_input = glob.glob(path_input)
    files_Rassine = glob.glob(path_rassine)
    
    files_input.sort()
    files_Rassine.sort()
    
    
    for i in range(len(files_input)):
        
        data_input = fits.open(files_input[i])
        data_Rassine = pd.read_pickle(files_Rassine[i])
        
        
        wave_input = data_input[1].data['wavelength']
        
        wave_Rassine = data_Rassine['wave']
        cont_Rassine = data_Rassine[anchor_mode][continuum_mode]
        
        f = interp1d(wave_Rassine,cont_Rassine,kind='cubic', bounds_error=False, fill_value='extrapolate')
        cont_interp_Rassine = f(wave_input)
        
        
        
        col1 = fits.Column(name = 'stellar_continuum', format = '1D', array = cont_interp_Rassine)
        col2 = fits.Column(name = 'flux_norm', format = '1D', array = data_input[1].data['flux']/cont_interp_Rassine)
        cols = fits.ColDefs([col1, col2])
    
        tbhdu = fits.BinTableHDU.from_columns(data_input[1].data.columns + cols) 
    
        prihdr = fits.Header()
        prihdr = data_input[0].header
        prihdu = fits.PrimaryHDU(header=prihdr)
        
        
        thdulist = fits.HDUList([prihdu, tbhdu])
        thdulist.writeto(files_input[i][:files_input[i].index('.fits')]+'_rassine.fits')

    return
    
def produce_line(grid,spectre,box=5,shape='savgol',vic=7):
    index, line_flux = local_max(-smooth(spectre, box, shape = shape), vic)
    line_flux = -line_flux
    line_index = index.astype('int')
    line_wave = grid[line_index]
    
    index2, line_flux2 = local_max(smooth(spectre, box, shape = shape),vic)
    line_index2 = index2.astype('int')
    line_wave2 = grid[line_index2]
    
    if line_wave[0]<line_wave2[0]:
        line_wave2 = np.insert(line_wave2,0,grid[0])
        line_flux2 = np.insert(line_flux2,0,spectre[0])
        line_index2 = np.insert(line_index2,0,0)
    
    if line_wave[-1]>line_wave2[-1]:
        line_wave2 = np.insert(line_wave2,-1,grid[-1])    
        line_flux2 = np.insert(line_flux2,-1,spectre[-1]) 
        line_index2 = np.insert(line_index2,-1,len(grid)-1)
    
    memory = np.hstack([-1*np.ones(len(line_wave)),np.ones(len(line_wave2))])
    stack_wave = np.hstack([line_wave,line_wave2])
    stack_flux = np.hstack([line_flux,line_flux2])
    stack_index = np.hstack([line_index,line_index2]) 
    
    memory = memory[stack_wave.argsort()]
    stack_flux = stack_flux[stack_wave.argsort()]
    stack_wave = stack_wave[stack_wave.argsort()]
    stack_index = stack_index[stack_index.argsort()]
    
    trash,matrix = grouping(memory,0.01,0)
    
    delete_liste = []
    for j in range(len(matrix)):
        numero = np.arange(matrix[j,0],matrix[j,1]+2)
        fluxes = stack_flux[numero].argsort()
        if trash[j][0] == 1 :
            delete_liste.append(numero[fluxes[0:-1]])
        else:
            delete_liste.append(numero[fluxes[1:]])
    delete_liste = np.hstack(delete_liste)        
    
    memory = np.delete(memory,delete_liste)       
    stack_flux = np.delete(stack_flux,delete_liste)       
    stack_wave = np.delete(stack_wave,delete_liste)       
    stack_index = np.delete(stack_index,delete_liste)       
    
    minima = np.where(memory==-1)[0] ; maxima = np.where(memory==1)[0]
    
    index = stack_index[minima] ; index2 = stack_index[maxima]
    flux = stack_flux[minima] ; flux2 = stack_flux[maxima]
    wave = stack_wave[minima] ; wave2 = stack_wave[maxima]
    
    index = np.hstack([index[:,np.newaxis],index2[0:-1,np.newaxis],index2[1:,np.newaxis]])
    flux = np.hstack([flux[:,np.newaxis],flux2[0:-1,np.newaxis],flux2[1:,np.newaxis]])
    wave = np.hstack([wave[:,np.newaxis],wave2[0:-1,np.newaxis],flux2[1:,np.newaxis]])
    
    return index,wave,flux  
    

def save_pickle(filename,output,header=None): 
    if filename.split('.')[-1]=='p':
        pickle.dump(output,open(filename,'wb'),protocol=protocol_pick)
    if filename.split('.')[-1]=='fits': #for futur work
        pass
    
    
def smooth(y, box_pts, shape='rectangular'): #rectangular kernel for the smoothing
    box2_pts = int(2*box_pts-1)
    if shape=='savgol':
        if box2_pts>=5:
            y_smooth = savgol_filter(y, box2_pts, 3)
        else:
            y_smooth = y
    else:
        if shape=='rectangular':
            box = np.ones(box2_pts)/box2_pts
        if shape == 'gaussian':
            vec = np.arange(-25,26)
            box = norm.pdf(vec,scale=(box2_pts-0.99)/2.35)/np.sum(norm.pdf(vec,scale = (box2_pts-0.99)/2.35))
        y_smooth = np.convolve(y, box, mode='same')
        y_smooth[0:int((len(box)-1)/2)] = y[0:int((len(box)-1)/2)]
        y_smooth[-int((len(box)-1)/2):] = y[-int((len(box)-1)/2):]
    return y_smooth    


def supress_low_snr_spectra(files_to_process, snr_cutoff = 100, supress=True):        
    for j in files_to_process:
        file = pd.read_pickle(j)
        if 'parameters' not in file.keys():
            if file['SNR_5500']<snr_cutoff:
                print('File deleted : %s '%(j))
                if supress:
                    os.system('rm '+j)  
                else:
                    new_name = 'rassine_'+j.split('_')[1]
                    os.system('mv '+j+' '+new_name)
        else:
            if file['parameters']['SNR_5500']<snr_cutoff:
                print('File deleted : %s '%(j))
                if supress:
                    os.system('rm '+j)
                else:
                    new_name = 'rassine_'+j.split('_')[1]
                    os.system('mv '+j+' '+new_name)
    
    
def sphinx(sentence,rep=None,s2=''):
    answer='-99.9'
    print(' ______________ \n\n --- SPHINX --- \n\n TTTTTTTTTTTTTT \n\n Question : '+sentence+'\n\n [Deafening silence ...] \n\n ______________ \n\n --- OEDIPE --- \n\n XXXXXXXXXXXXXX \n ')
    if rep != None:
        while answer not in rep:
            answer = my_input('Answer : '+s2)
    else:
        answer = my_input('Answer : '+s2)
    return answer

def supress_ccd_gap(files_to_process,continuum='linear'):
    """Fill the gap between the ccd of HARPS s1d"""
    for i,j in enumerate(files_to_process):
        print('Modification of file (%.0f/%.0f) : %s'%(i+1,len(files_to_process),j))
        file = pd.read_pickle(j)
        conti = file['matching_anchors']['continuum_'+continuum]
        flux_norm = file['flux']/conti
        cluster = grouping(flux_norm,0.001,0)[-1]
        cluster = cluster[cluster[:,2].argmax(),:]
        left = cluster[0]-10
        right = cluster[1]+10
        flux_norm[int(left):int(right)+1] = 1
        new_flux = smooth(flux_norm, box_pts=6, shape='gaussian')
        flux_norm[left-6:right+7] = new_flux[left-6:right+7]
        file['flux'] = flux_norm*conti
        save_pickle(j,file)


def rm_outliers(array, m=1.5, kind='sigma', direction='sym'):
    if type(array)!=np.ndarray:
        array=np.array(array)
    if kind == 'inter':
        median = np.nanpercentile(array,50)
        Q3 = np.nanpercentile(array, 75)
        Q1 = np.nanpercentile(array, 25)
        IQ = Q3-Q1
        if direction=='sym':
            mask = (array >= Q1-m*IQ)&(array <= Q3+m*IQ)
        if direction=='highest':
            mask = (array <= Q3+m*2*(Q3-median))
        if direction=='lowest':
            mask = (array >= Q1-m*2*(median-Q1))    
    if kind == 'sigma':
        mask = abs(array-np.nanmean(array)) <= m * np.nanstd(array)
    return mask,  array[mask]


def rolling_stat(array,window=1,min_periods=1):
    roll_median = np.ravel(pd.DataFrame(array).rolling(window,min_periods=min_periods,center=True).quantile(0.50))
    roll_Q1 = np.ravel(pd.DataFrame(array).rolling(window,min_periods=min_periods,center=True).quantile(0.25))
    roll_Q3 = np.ravel(pd.DataFrame(array).rolling(window,min_periods=min_periods,center=True).quantile(0.75))
    roll_IQ = roll_Q3 - roll_Q1
    return roll_median,roll_Q1,roll_Q3,roll_IQ


def rolling_iq(array,window=1,min_periods=1):
    roll_Q1 = np.ravel(pd.DataFrame(array).rolling(window,min_periods=min_periods,center=True).quantile(0.25))
    roll_Q3 = np.ravel(pd.DataFrame(array).rolling(window,min_periods=min_periods,center=True).quantile(0.75))
    roll_IQ = roll_Q3 - roll_Q1
    return roll_Q1, roll_Q3, roll_IQ


def try_field(dico,field):
    try:
        a = dico[field]
        return a
    except:
        return None


def troncated(array, spectre, treshold=5):
    maxi = np.percentile(spectre,99.9)
    mini = np.percentile(spectre,0.1)
    tresh = (maxi-mini)/treshold
    array[array<mini-tresh] = mini
    array[array>maxi+tresh] = maxi
    return array
