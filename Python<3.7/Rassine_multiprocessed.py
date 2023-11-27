#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:41:21 2019

@author: cretignier
"""

from __future__ import print_function
import multiprocessing as multicpu
import os
import numpy as np 
import threading
import pandas as pd
import getopt
import sys
import glob as glob
from Rassine_functions import make_sound
from Rassine_functions import preprocess_fits
from Rassine_functions import preprocess_prematch_stellar_frame
from Rassine_functions import preprocess_match_stellar_frame
from Rassine_functions import intersect_all_continuum
from Rassine_functions import matching_diff_continuum


"""COMMMENT OF DEVELOPPERS : WE ADMISE THE USER TO USE THE RASSINE_TRIGGER.PY RATHER THAN TO INTERACT DIRECTLY WITH THE MULTIPROCESSED FILE HERE"""

output_dir = 'unspecified'
only_print_end = True
nthreads = 4 
anchor_file = 'unspecified' 
feedback = False  # allow feedback from rassine DO NOT SWITCH ON this button except if you know what you are doing
files_to_reduce = '/Users/cretignier/Documents/HD8651/data/s1d/STACKED/S' # generic name of the spectra to reduce (found later with a glob call)
plot_end = False
plx_mas = 0
instrument = 'HARPS'
process = 'RASSINE' #either PREPROCESS, MATCHING OR RASSINE


#sys arguments bypass the previous ones
if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'s:i:o:a:l:P:n:e:v:d:k:p:w:')
    for j in optlist:
        if j[0] == '-s': #spectrum file
            files_to_reduce = j[1]
        if j[0] == '-o': #output directory
            output_dir = j[1]
        if j[0] == '-l': #anchor file
            anchor_file = j[1]
        if j[0] == '-P': #only print end
            only_print_end = j[1]
        if j[0] == '-n': #only print end
            nthreads = int(j[1])
        if j[0] == '-e': #only print end
            plot_end = j[1]
        if j[0] == '-i': #instrument
            instrument = j[1]
        if j[0] == '-v': #process to run (PREPROCESS, MATCHING, RASSINE)
            process = j[1]     
        if j[0] == '-p': #only print end
            plx_mas = np.float(j[1])
        if j[0] == '-w': #only print end
            savgol_window = int(np.float(j[1]))
        if j[0] == '-d': 
            dlambda = j[1]
            if dlambda == 'None':
                dlambda = None
            else:
                dlambda = np.float(dlambda)
        if j[0] == '-k': #pickle file containing the RV to remove or float number with systemic velocity
            rv = j[1] 
            if len(rv.split('/'))==1:
                rv = np.float(rv)
            else :
                rv_file = rv
                if rv_file.split('.')[-1]=='csv':
                    rv = pd.read_csv(rv_file)['model']  # RV time-series to remove in kms, (binasry  or known planets) otherwise give the systemic velocity
                elif rv_file.split('.')[-1]=='p':
                    rv = pd.read_pickle(rv_file)['model']
                else:
                    print('Cannot read this file format')

rassine_files_to_preprocess = np.sort(glob.glob(files_to_reduce+'*.fits'))
rassine_files_to_reduce = np.sort(glob.glob(files_to_reduce+'*.p'))

if output_dir=='':
    output_dir = os.path.dirname(rassine_files_to_reduce[0])+'/'


def init(lock):
    global starting
    starting = lock

def run_preprocessing(file_liste):
    starting.acquire() # no other process can get it until it is released
    threading.Timer(0.7, starting.release).start() # release in 6 seconds
    preprocess_fits(file_liste, instrument=instrument, plx_mas=plx_mas, final_sound=False) 

def run_matching_wrapper(args):
    return preprocess_match_stellar_frame(args[0], args = args[1],final_sound=False) 

def run_rassine(file_liste):
    starting.acquire() # no other process can get it until it is released
    threading.Timer(0.7, starting.release).start() # release in 6 seconds
    if anchor_file=='':
        for n in file_liste:
            os.system('python Rassine.py -s '+n+' -o '+output_dir+' -a '+str(feedback)+' -P '+str(only_print_end)+' -e '+str(plot_end))
    else:
        for n in file_liste:
            os.system('python Rassine.py -s '+n+' -o '+output_dir+' -a '+str(feedback)+' -P '+str(only_print_end)+' -l '+anchor_file+' -e '+str(plot_end))            

def run_matching_diff(file_liste):
    starting.acquire() # no other process can get it until it is released
    threading.Timer(0.7, starting.release).start() # release in 6 seconds
    matching_diff_continuum(file_liste, sub_dico = 'matching_anchors', master=anchor_file, savgol_window = savgol_window, zero_point=False) 

def run_matching_diff(file_liste):
    starting.acquire() # no other process can get it until it is released
    threading.Timer(0.7, starting.release).start() # release in 6 seconds
    matching_diff_continuum(file_liste, sub_dico = 'matching_anchors', master=anchor_file, savgol_window = savgol_window, zero_point=False) 

def run_intersect_continuum(file_liste):
    starting.acquire() # no other process can get it until it is released
    threading.Timer(0.7, starting.release).start() # release in 6 seconds
    intersect_all_continuum(file_liste, add_new=True) 


if process=='RASSINE':

    print('Number of files to reduce %.0f'%(len(rassine_files_to_reduce)))

    if nthreads >= multicpu.cpu_count():
        print('Your number of cpu (%s) is smaller than the number your entered (%s), enter a smaller value please'%(multicpu.cpu_count(),nthreads))
    else:
        chunks = [rassine_files_to_reduce[j::nthreads] for j in range(nthreads)]
        pool = multicpu.Pool(processes=nthreads, initializer=init, initargs=[multicpu.Lock()])
        pool.map(run_rassine, chunks)

    make_sound('Racine Multiprocessing has finished')


elif process=='INTERSECT':

    print('Number of files to reduce %.0f'%(len(rassine_files_to_reduce)))

    if nthreads >= multicpu.cpu_count():
        print('Your number of cpu (%s) is smaller than the number your entered (%s), enter a smaller value please'%(multicpu.cpu_count(),nthreads))
    else:
        chunks = [rassine_files_to_reduce[j::nthreads] for j in range(nthreads)]
        pool = multicpu.Pool(processes=nthreads, initializer=init, initargs=[multicpu.Lock()])
        pool.map(run_intersect_continuum, chunks)

    make_sound('Racine Multiprocessing has finished')



elif process=='SAVGOL':

    print('Number of files to reduce %.0f'%(len(rassine_files_to_reduce)))

    if nthreads >= multicpu.cpu_count():
        print('Your number of cpu (%s) is smaller than the number your entered (%s), enter a smaller value please'%(multicpu.cpu_count(),nthreads))
    else:
        chunks = [rassine_files_to_reduce[j::nthreads] for j in range(nthreads)]
        pool = multicpu.Pool(processes=nthreads, initializer=init, initargs=[multicpu.Lock()])
        pool.map(run_matching_diff, chunks)

    make_sound('Racine Multiprocessing has finished')



elif process=='PREPROCESS':

    print('Number of files to preprocess %.0f'%(len(rassine_files_to_preprocess)))

    if nthreads >= multicpu.cpu_count():
        print('Your number of cpu (%s) is smaller than the number your entered (%s), enter a smaller value please'%(multicpu.cpu_count(),nthreads))
    else:
        chunks = np.array_split(rassine_files_to_preprocess, nthreads)
        pool = multicpu.Pool(processes=nthreads,initializer=init, initargs=[multicpu.Lock()])
        pool.map(run_preprocessing, chunks)

    make_sound('Preprocessing in multiprocessed has finished')



elif process=='MATCHING':

    print('Number of files to match %.0f'%(len(rassine_files_to_reduce)))

    if nthreads >= multicpu.cpu_count():
        print('Your number of cpu (%s) is smaller than the number your entered (%s), enter a smaller value please'%(multicpu.cpu_count(),nthreads))
    else:
        args = preprocess_prematch_stellar_frame(rassine_files_to_reduce, dlambda=dlambda, rv=rv)
        
        split = np.array_split(np.arange(len(rassine_files_to_reduce)), nthreads)
        
        chunks = [(rassine_files_to_reduce[split[j]],(args[0],args[1],args[2],args[3],args[4],args[5],args[6][split[j]],args[7][split[j]],args[8][split[j]],args[9][split[j]],args[10][split[j]],args[11][split[j]],args[12])) for j in range(nthreads)]
        
        pool = multicpu.Pool(processes=nthreads)
        pool.map(run_matching_wrapper, chunks)

    make_sound('Matching in multiprocessed has finished')
