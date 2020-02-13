#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:50:04 2019

@author: cretignier
"""

import matplotlib
matplotlib.use('Qt5Agg',force=True)
import glob as glob 
import Rassine_functions as ras      
import numpy as np 
import matplotlib.pylab as plt
import os
import pandas as pd
cwd = os.getcwd()

# =============================================================================
# parameters
# =============================================================================

instrument = 'HARPS'                                                     # instrument (either HARPS, HARPN, CORALIE or ESPRESSO for the moment)
dir_spec_timeseries = cwd+'/spectra_library/CenB/'   # directory containing the s1d spectra timeseries

nthreads_preprocess = 8               # number of threads in parallel for the preprocessing
nthreads_matching = 2                 # number of threads in parallel for the matching (more than 2 is not efficient for some reasons...) 
nthreads_rassine = 4                  # number of threads in parallel for the normalisation (BE CAREFUL RASSINE NEED A LOT OF RAM DEPENDING ON SPECTRUM LENGTH)

rv_timeseries = -22.7      #/Users/cretignier/Documents/Yarara/HD128621/data/s1d/HARPS/DACE_TABLE/Dace_extracted_table.csv'      # RV time-series to remove in kms, (only if binaries with ~kms RV amplitudes) stored in a pickle dictionnary inside 'model' keyword 
                       # otherwise give the systemic velocity
plx_mas = 0            # parallaxe of the star in mas to compute the secular drift  

dlambda = None                     # wavelength grid step of the equidistant grid, only if unevenly wavelength grid on lack of homogeneity in spectra time-series
bin_length_stack = 1               # length of the binning for the stacking in days
dbin = 0                           # dbin to shift the binning (0.5 for solar data)
counter_stack = 0                  # define the first index of stack file names (index = counter + 1) 
use_master_as_reference = False     # use master spectrum as reference for the clustering algorithm (case of low SNR spectra)

# =============================================================================
# buttons
# =============================================================================

preprocessed = 1
match_frame = 1
stacking = 1
rassine_normalisation_master = 1
rassine_normalisation = 1
rassine_intersect_continuum = 1
rassine_diff_continuum = 1

# =============================================================================
# trigger
# =============================================================================

if bin_length_stack != 0:
    fileroot_files_to_rassine = 'Stacked' #usual fileroot of the Stacked spectra produce by RASSINE
else:
    fileroot_files_to_rassine = 'Prepared' #usual fileroot of the Unstacked spectra produce by RASSINE

if preprocessed:
    print('[STEP INFO] Preprocessing...')
    os.system('python Rassine_multiprocessed.py -v PREPROCESS -s '+dir_spec_timeseries+' -n '+str(nthreads_preprocess)+' -i '+instrument+' -p '+str(plx_mas))

if match_frame:
    print('[STEP INFO] Matching frame...')
    os.system('python Rassine_multiprocessed.py -v MATCHING -s '+dir_spec_timeseries+'PREPROCESSED/'+' -n '+str(nthreads_matching)+' -d '+str(dlambda)+' -k '+str(rv_timeseries))

if stacking:
    print('[STEP INFO] Stacking frame...')
    ras.preprocess_stack(glob.glob(dir_spec_timeseries+'PREPROCESSED/*.p'), 
                         bin_length = bin_length_stack, 
                         dbin = dbin,
                         counter = counter_stack,
                         make_master=True)

if rassine_normalisation_master:
    print('[STEP INFO] Normalisation master spectrum')
    os.system('python Rassine.py -s '+dir_spec_timeseries+'STACKED/Master_spectrum.p -a True')

if rassine_normalisation:
    print('[STEP INFO] Normalisation frame...')
    os.system('python Rassine_multiprocessed.py -v RASSINE -s '+dir_spec_timeseries+'STACKED/'+fileroot_files_to_rassine+' -n '+str(nthreads_rassine)+' -l '+dir_spec_timeseries+'STACKED/RASSINE_Master_spectrum.p -P '+str(True)+' -e '+str(False))

if os.path.exists(dir_spec_timeseries+'STACKED/RASSINE_RASSINE_Master_spectrum.p'):
    os.system('rm '+dir_spec_timeseries+'STACKED/RASSINE_RASSINE_Master_spectrum.p')

if use_master_as_reference:
    master_spectrum_name = dir_spec_timeseries+'STACKED/RASSINE_Master_spectrum.p'
else:  
    master_spectrum_name = None

if rassine_intersect_continuum:
    print('[STEP INFO] Normalisation clustering algorithm')
    ras.intersect_all_continuum(glob.glob(dir_spec_timeseries+'STACKED/RASSINE*.p'), master_spectrum=master_spectrum_name, copies_master=0, kind='anchor_index', nthreads=6, fraction=0.2, threshold = 0.66, tolerance=0.5, add_new = True)

if rassine_diff_continuum:
    print('[STEP INFO] Normalisation diff continuum')
    ras.matching_diff_continuum(glob.glob(dir_spec_timeseries+'STACKED/RASSINE*.p'), sub_dico = 'matching_anchors', savgol_window = 200, zero_point=False)

