#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:34:29 2019
19.04.19
@author: michael cretignier & jémémie francfort

# =====================================================================================
# Rolling Alpha Shape for a Spectral Improved Normalisation Estimator (RASSINE)
# =====================================================================================

       ^                  .-=-.          .-==-.
      {}      __        .' O o '.       /   ^  )
     { }    .' O'.     / o .-. O \     /  .--`\
     { }   / .-. o\   /O  /   \  o\   /O /    ^  (RASSSSSSINE)
      \ `-` /   \ O`-'o  /     \  O`-`o /
  jgs  `-.-`     '.____.'       `.____.'

"""

import os 
cwd = os.getcwd()

# =============================================================================
#  ENTRIES
# =============================================================================

spectrum_name = cwd+'/spectra_library/spectrum_cenB.p' # full path of your spectrum pickle/csv file
output_dir  = cwd+'/output/'                             # directory where output files are written

synthetic_spectrum = False   # True if working with a noisy-free synthetic spectra 
anchor_file = ''             # Put a RASSINE output file that will fix the value of the 7 parameters to the same value than in the anchor file

column_wave = 'wave'
column_flux = 'flux'

float_precision = 'float32' # float precision for the output products wavelength grid

#general initial parameters

par_stretching = 'auto_0.5'     # stretch the x and y axes ratio ('auto' available)                            <--- PARAMETER 1
par_vicinity = 7                # half-window to find a local maxima                                           
                                
par_smoothing_box = 6           # half-window of the box used to smooth (1 => no smoothing, 'auto' available)  <--- PARAMETER 2
par_smoothing_kernel = 'savgol' # 'rectangular','gaussian','savgol' if a value is specified in smoothig_kernel
                                # 'erf','hat_exp' if 'auto' is given in smoothing box                     
                   
par_fwhm = 'auto'               # FWHM of the CCF in km/s ('auto' available)                                   <--- PARAMETER 3
CCF_mask = 'master'             # only needed if par_fwhm is in 'auto'
RV_sys = 0                      # RV systemic in kms, only needed if par_fwhm is in 'auto' and CCF different of 'master'
mask_telluric = [[6275,6330],   # a list of left and right borders to eliminate from the mask of the CCF
                 [6470,6577],   # only if CCF = 'master' and par_fwhm = 'auto'
                 [6866,8000]] 

mask_broadline = [[3960,3980],   # a list of left and right borders to eliminate from the anchor point candidates
                 [6560,6562],   
                 [10034,10064]] 

par_R = 'auto'             # minimum radius of the rolling pin in angstrom ('auto' available)                  <--- PARAMETER 4
par_Rmax = 'auto'          # maximum radius of the rolling pin in angstrom ('auto' available)                  <--- PARAMETER 5
par_reg_nu = 'poly_1.0'    # penality-radius law                                                               <--- PARAMETER 6
                           # poly_d (d the degree of the polynome x**d)
                           # or sigmoid_c_s where c is the center and s the steepness

denoising_dist = 5      # half window of the area used to average the number of point around the local max for the continuum
count_cut_lim = 3       # number of border cut in automatic mode (put at least 3 if Automatic mode)
count_out_lim = 1       # number of outliers clipping in automatic mode (put at least 1 if Automatic mode)


interpolation = 'cubic' # define the interpolation for the continuum displayed in the subproducts            
                        # note that at the end a cubic and linear interpolation are saved in 'output' regardless this value

feedback = True        # run the code without graphical feedback and interactions with the sphinx (only wishable if lot of spectra)     
only_print_end = False  # only print in the console the confirmation of RASSINE ending
plot_end = True        # display the final product in the graphic
save_last_plot = False  # save the last graphical output (final output)


outputs_interpolation_saved = 'all' # to only save a specific continuum (output files are lighter), either 'linear','cubic' or 'all'
outputs_denoising_saved = 'undenoised'        # to only save a specific continuum (output files are lighter), either 'denoised','undenoised' or 'all'

light_version = True    # to save only the vital output


config = {'spectrum_name':spectrum_name,
          'synthetic_spectrum':synthetic_spectrum,
          'output_dir':output_dir,
          'anchor_file':anchor_file,
          'column_wave':column_wave,
          'column_flux':column_flux,
          'axes_stretching':par_stretching,
          'vicinity_local_max':par_vicinity,
          'smoothing_box':par_smoothing_box,
          'smoothing_kernel':par_smoothing_kernel,
          'fwhm_ccf':par_fwhm,
          'CCF_mask':CCF_mask,
          'RV_sys':RV_sys,
          'mask_telluric':mask_telluric,
          'mask_broadline':mask_broadline,
          'min_radius':par_R,
          'max_radius':par_Rmax,
          'model_penality_radius':par_reg_nu,
          'denoising_dist':denoising_dist,
          'interpol':interpolation,
          'number_of_cut':count_cut_lim,
          'number_of_cut_outliers':count_out_lim,
          'float_precision':float_precision,
          'feedback':feedback,
          'only_print_end':only_print_end,
          'plot_end':plot_end,
          'save_last_plot':save_last_plot,
          'outputs_interpolation_save':outputs_interpolation_saved,
          'outputs_denoising_save':outputs_denoising_saved,
          'light_file':light_version,
          'speedup':1} 
