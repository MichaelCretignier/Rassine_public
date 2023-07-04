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

from __future__ import print_function

import matplotlib

matplotlib.use('Qt5Agg',force=True)
import getopt
import os
import sys
import time

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import Button, RadioButtons, Slider
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import erf

import Rassine_functions as ras

np.warnings.filterwarnings('ignore', category=RuntimeWarning)

#get_ipython().run_line_magic('matplotlib','qt5')

python_version = sys.version[0]
config = {}

# =============================================================================
#  IMPORT CONFIG FILE
# =============================================================================

if python_version=='3':
    exec(open('Rassine_config.py').read())
elif python_version=='2':
    execfile('Rassine_config.py')


spectrum_name = config['spectrum_name']
output_dir = config['output_dir']
synthetic_spectrum = config['synthetic_spectrum']  
anchor_file = config['anchor_file']            
column_wave = config['column_wave']
column_flux = config['column_flux']
float_precision = config['float_precision'] 

par_stretching = config['axes_stretching']   
par_vicinity = config['vicinity_local_max']                                              
par_smoothing_box = config['smoothing_box']           
par_smoothing_kernel = config['smoothing_kernel']                    
par_fwhm = config['fwhm_ccf']               
CCF_mask = config['CCF_mask']          
RV_sys = config['RV_sys']                    
mask_telluric = config['mask_telluric']
mask_broadline = config['mask_broadline']
par_R = config['min_radius']               
par_Rmax = config['max_radius']        
par_reg_nu = config['model_penality_radius']

denoising_dist = config['denoising_dist']      
count_cut_lim = config['number_of_cut']     
count_out_lim = config['number_of_cut_outliers']   
   
interpol = config['interpol']          
feedback = config['feedback']             
only_print_end = config['only_print_end'] 
plot_end = config['plot_end']     
save_last_plot = config['save_last_plot']  

outputs_interpolation_saved = config['outputs_interpolation_save']
outputs_denoising_saved = config['outputs_denoising_save']
light_version = config['light_file']   
speedup = config['speedup']
            

# =============================================================================
# TAKE THE DATA
# =============================================================================

plt.close('all')

if speedup < 1:
    speedup = 1

#column_wave = 'wave'
#column_flux = 'flux'  

if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'f:s:o:r:R:a:w:l:p:P:e:S:')
    for j in optlist:
        if j[0] == '-f': #flux column key
            column_flux = j[1]
        if j[0] == '-w': #wave column key
            column_wave = j[1]
        if j[0] == '-s': #spectrum file
            spectrum_name = j[1]
            output_dir = os.path.dirname(spectrum_name)+'/'
        if j[0] == '-o': #output directory
            if j[1]!='unspecified':
                output_dir = j[1]
        if j[0] == '-l': #anchor file
            anchor_file = j[1]
        if j[0] == '-r': #Radius minimum
            par_R = j[1]
            par_R = float(par_R)
        if j[0] == '-R': #Radius maximum
            par_Rmax = j[1]
            par_Rmax = float(par_Rmax)
        if j[0] == '-p': #par_stretching
            par_stretching = j[1]
            par_stretching = float(par_stretching)
        if j[0] == '-a': #feedback
            if j[1]!='unspecified':
                feedback = j[1]
        if j[0] == '-P': #only print end
            only_print_end = j[1]
        if j[0] == '-e': #only print end
            plot_end = j[1]
        if j[0] == '-S': #only print end
            save_last_plot = j[1]


    if (only_print_end == 'True')|(only_print_end == 'true')|(only_print_end == '1')|(only_print_end == True):
        only_print_end = True
    else:
        only_print_end = False

    if (plot_end == 'True')|(plot_end == 'true')|(plot_end == '1')|(plot_end == True):
        plot_end = True
    else:
        plot_end = False
    
    if (feedback == 'True')|(feedback == 'true')|(feedback == '1')|(feedback == True):
        feedback = True
    else:
        feedback = False

    if (save_last_plot == 'True')|(save_last_plot == 'true')|(save_last_plot == '1')|(save_last_plot == True):
        save_last_plot = True
    else:
        save_last_plot = False

filename = spectrum_name.split('/')[-1]
cut_extension = len(filename.split('.')[-1]) + 1
new_file = filename[:-cut_extension]

random_number = np.sum([ord(a) for a in filename.split('RASSINE_')[-1]])

#to ignite the variable present after preprocessing

mjd = None 
jdb = None
hole_left = None
hole_right = None
RV_shift = None
acc_sec = None
berv = None
lamp_offset = None
nb_spectra_stacked = None
arcfiles = None

if not os.path.exists(anchor_file):
    anchor_file=''

if anchor_file!='':
    anchor_file = ras.open_pickle(anchor_file)
    par_stretching = anchor_file['parameters']['axes_stretching']
    par_vicinity = anchor_file['parameters']['vicinity_local_max']
    par_smoothing_box = anchor_file['parameters']['smoothing_box']
    par_smoothing_kernel = anchor_file['parameters']['smoothing_kernel']                      
    par_fwhm = anchor_file['parameters']['fwhm_ccf']
    par_R = anchor_file['parameters']['min_radius'] 
    par_Rmax = anchor_file['parameters']['max_radius'] 
    par_reg_nu = anchor_file['parameters']['model_penality_radius']
    count_cut_lim = anchor_file['parameters']['number of cut']

if spectrum_name.split('.')[-1]=='fits': # to load a fits file 
    header = fits.getheader(spectrum_name) # load the fits header
    spectre_step = fits.getheader(spectrum_name)['CDELT1']
    spectrei = fits.getdata(spectrum_name).astype('float64') # the flux of your spectrum
    grid = np.linspace(header['CRVAL1'], header['CRVAL1']+(len(spectrei)-1)*spectre_step, len(spectrei)) # the grid of wavelength of your spectrum (assumed equidistant in lambda)
else: # to load a pickle dictionnary, csv file or txt file
    if spectrum_name.split('.')[-1]=='csv':
        data = pd.read_csv(spectrum_name) # load the pickle dictionnary
        spectrei = np.array(data[column_flux])  # the flux of your spectrum
        grid = np.array(data[column_wave])      # the grid of wavelength of your spectrum
    elif spectrum_name.split('.')[-1]=='p':
        data = ras.open_pickle(spectrum_name) # load the pickle dictionnary
        spectrei = np.array(data[column_flux])  # the flux of your spectrum
        try:
            grid = np.array(data[column_wave])      # the grid of wavelength of your spectrum
        except:
            grid = ras.create_grid(data['wave_min'],data['dwave'],len(data[column_flux]))
        if ras.try_field(data,'RV_sys') is not None:
            RV_sys = ras.try_field(data,'RV_sys')
        RV_shift = ras.try_field(data,'RV_shift')
        mjd = ras.try_field(data,'mjd')
        jdb = ras.try_field(data,'jdb')
        hole_left = ras.try_field(data,'hole_left')
        hole_right = ras.try_field(data,'hole_right')
        berv = ras.try_field(data,'berv')
        lamp_offset = ras.try_field(data,'lamp_offset')
        acc_sec = ras.try_field(data,'acc_sec')
        nb_spectra_stacked = ras.try_field(data,'nb_spectra_stacked')
        arcfiles = ras.try_field(data,'arcfiles')

    elif (spectrum_name.split('.')[-1]=='txt')|(spectrum_name.split('.')[-1]=='rdb'):
        data = np.genfromtxt(spectrum_name)
        if np.shape(data)[1]!=2:
            print('[WARNING] Your txt file does not contain two columns, please take care to format correctly your input')
        if (~(data[0][0]==data[0][0]))&((data[1][0]==data[1][0])):
            print('[WARNING] Your txt file is suspected to contain a header, the first line was removed (please take care to format correctly your txt input)')            
            data = data[1:,:]
        spectrei = data[:,1]  # the flux of your spectrum
        grid = data[:,0]      # the grid of wavelength of your spectrum
        

if output_dir!='':
    if output_dir.split('/')[-1] != '':
        output_dir += '/'
    if not os.path.isdir(output_dir):
        print('The directory does not exist yet, creation of the directory')
        os.system('mkdir '+output_dir)
else:
    output_dir = os.path.dirname(spectrum_name)+'/'


if type(par_stretching)!=str:
    if par_stretching<0:
        print('[WARNING] par_stretching is smaller than 0, please enter a higher value')
        print('[WARNING] par_stretching value fixed at 3')
        par_stretching = 3.0
else:
    if (float(par_stretching.split('_')[1])>1)|(float(par_stretching.split('_')[1])<0):
        print('[WARNING] par_stretching automatic value should be between 0 and 1')
        print('[WARNING] par_stretching value fixed at 0.5')
        par_stretching = 'auto_0.5'        

# =============================================================================
# LOCAL MAXIMA
# =============================================================================

if not only_print_end:
    print('\n [BEGIN] RASSINE is beginning the reduction')

begin = time.time()

if not only_print_end:
    print('\n Computation of the local maxima : LOADING' )

mask_grid = np.arange(len(grid))[(grid-grid)!=0]
mask_spectre = np.arange(len(grid))[(spectrei-spectrei)!=0]

if len(mask_grid)>0:
    print(' Nan values were found, replaced by left and right average...')
    for j in mask_grid:
        grid[j] = (grid[j-1]+grid[j+1])/2

if len(mask_spectre)>0:
    print(' Nan values were found, replaced by left and right average...')
    for j in mask_spectre:
        spectrei[j] = (spectrei[j-1]+spectrei[j+1])/2    

mask_grid = np.arange(len(grid))[(grid-grid)!=0]
mask_spectre = np.arange(len(grid))[(spectrei-spectrei)!=0]        

if np.sum(np.isnan(grid))|np.sum(np.isnan(spectrei)):
    print(' [WARNING] There is too much NaN values, attempting to clean your data')
    spectrei[mask_spectre] = 0

if len(np.unique(np.diff(grid)))>1:
    grid_backup_0 = grid.copy()
    new_grid = np.linspace(grid.min(), grid.max(), len(grid))
    spectrei = interp1d(grid, spectrei, kind='cubic', bounds_error=False, fill_value='extrapolate')(new_grid)
    grid = new_grid.copy()
    
dgrid = grid[1] - grid[0]
dgrid/=5

sorting = grid.argsort() #sort the grid of wavelength
grid = grid[sorting]
dlambda = np.mean(np.diff(grid)) 
spectrei = spectrei[sorting]
spectrei[spectrei<0] = 0
spectrei = ras.empty_ccd_gap(grid,spectrei,left=hole_left,right=hole_right)


minx = grid[0] ; maxx = grid[-1]
miny = np.nanpercentile(spectrei,0.001) ; maxy = np.nanpercentile(spectrei,0.999)

len_x = maxx - minx
len_y = np.max(spectrei) - np.min(spectrei)

wave_ref_snr = 5500
if (wave_ref_snr<np.nanmin(grid))|(wave_ref_snr>np.nanmax(grid)):
    wave_ref_snr = int(np.round(np.nanmean(grid),-2))
idx_wave_ref_snr = int(ras.find_nearest(grid,wave_ref_snr)[0])

continuum_ref_snr = np.nanpercentile(spectrei[idx_wave_ref_snr-50:idx_wave_ref_snr+50],95)
SNR_0 = np.sqrt(continuum_ref_snr)
if np.isnan(SNR_0):
    SNR_0 = -99

if not only_print_end:
    print(' Spectrum SNR at %.0f : %.0f'%(wave_ref_snr,SNR_0))

normalisation = float(len_y)/float(len_x) # stretch the y axis to scale the x and y axis 
spectre = spectrei/normalisation

if synthetic_spectrum:
    spectre += np.random.randn(len(spectre))*1e-5*np.min(np.diff(spectre)[np.diff(spectre)!=0]) #to avoid to same value of flux in synthetic spectra


#Do the rolling sigma clipping on a grid smaller to increase the speed 
np.random.seed(random_number)
subset = np.sort(np.random.choice(np.arange(len(spectre)),size=int(len(spectre)/speedup),replace=False)) # take randomly 1 point over 10 to speed process

for iteration in range(5): #k-sigma clipping 5 times

    maxi_roll_fast = np.ravel(pd.DataFrame(spectre[subset]).rolling(int(100/dgrid/speedup),min_periods=1,center=True).quantile(0.99))
    Q3_fast = np.ravel(pd.DataFrame(spectre[subset]).rolling(int(5/dgrid/speedup),min_periods=1,center=True).quantile(0.75)) #sigma clipping on 5 \AA range 
    Q2_fast = np.ravel(pd.DataFrame(spectre[subset]).rolling(int(5/dgrid/speedup),min_periods=1,center=True).quantile(0.50))  
    Q1_fast = np.ravel(pd.DataFrame(spectre[subset]).rolling(int(5/dgrid/speedup),min_periods=1,center=True).quantile(0.25))
    IQ_fast = 2*(Q3_fast-Q2_fast)  
    sup_fast = Q3_fast+1.5*IQ_fast
    
    if speedup>1:
        maxi_roll_fast = interp1d(subset, maxi_roll_fast, bounds_error=False, fill_value='extrapolate')(np.arange(len(spectre)))
        sup_fast = interp1d(subset, sup_fast, bounds_error=False, fill_value='extrapolate')(np.arange(len(spectre)))
        Q2_fast = interp1d(subset, Q2_fast, bounds_error=False, fill_value='extrapolate')(np.arange(len(spectre)))
    
    if not only_print_end:
        print(' Number of cosmic peaks removed : %.0f'%(np.sum((spectre>sup_fast)&(spectre>maxi_roll_fast))))
    
    mask = (spectre>sup_fast)&(spectre>maxi_roll_fast)
    for j in range(int(par_vicinity/2)):
        mask = mask|np.roll(mask,-j)|np.roll(mask,j) #supress the peak + the vicinity range
    if sum(mask)==0:
        break
    spectre[mask] = Q2_fast[mask]

conversion_fwhm_sig = (10*minx/(2.35*3e5)) #5sigma width in the blue

if par_fwhm == 'auto':

    mask = np.zeros(len(spectre))
    continuum_right = np.ravel(pd.DataFrame(spectre).rolling(int(30/dgrid)).quantile(1)) #by default rolling maxima in a 30 angstrom window
    continuum_left = np.ravel(pd.DataFrame(spectre[::-1]).rolling(int(30/dgrid)).quantile(1))[::-1]
    continuum_right[np.isnan(continuum_right)] = continuum_right[~np.isnan(continuum_right)][0] 
    continuum_left[np.isnan(continuum_left)] = continuum_left[~np.isnan(continuum_left)][-1]
    both = np.array([continuum_right,continuum_left])
    continuum = np.min(both,axis=0)
    
    continuum = ras.smooth(continuum, int(15/dgrid), shape='rectangular') #smoothing of the envelop 15 anstrom to provide more accurate weight
    
    log_grid = np.linspace(np.log10(grid).min(),np.log10(grid).max(),len(grid))
    log_spectrum = interp1d(np.log10(grid), spectre/continuum, kind='cubic', bounds_error=False, fill_value='extrapolate')(log_grid)
    
    
    if CCF_mask != 'master':
        mask_harps = np.genfromtxt(CCF_mask+'.txt')
        line_center = ras.doppler_r(0.5*(mask_harps[:,0]+mask_harps[:,1]),RV_sys)[0]
        distance = np.abs(grid - line_center[:,np.newaxis])   
        index = np.argmin(distance,axis=1)
        mask = np.zeros(len(spectre))
        mask[index] = mask_harps[:,2]
        log_mask = interp1d(np.log10(grid),mask, kind='linear', bounds_error=False, fill_value='extrapolate')(log_grid)        
    else:
        index, wave, flux = ras.produce_line(grid,spectre/continuum)
        keep = (0.5*(flux[:,1]+flux[:,2])-flux[:,0])>0.2
        flux = flux[keep]
        wave = wave[keep]
        index = index[keep]
        mask = np.zeros(len(spectre))
        mask[index[:,0]] = 0.5*(flux[:,1]+flux[:,2])-flux[:,0]
        log_mask = interp1d(np.log10(grid), mask, kind='linear', bounds_error=False, fill_value='extrapolate')(log_grid)        
        if len(mask_telluric)>0:
            for j in range(len(mask_telluric)):
                tellurics = (log_grid>np.log10(mask_telluric[j][0]))&(log_grid<np.log10(mask_telluric[j][1]))
                log_mask[tellurics] = 0
        
    vrad, ccf = ras.ccf(log_grid,log_spectrum,log_mask,extended=500)
    ccf = ccf[vrad.argsort()]
    vrad = vrad[vrad.argsort()]
    popt, pcov = curve_fit(ras.gaussian, vrad/1000, ccf, p0 = [0, -0.5, 0.9, 3])
    errors_fit = np.sqrt(np.diag(pcov))
    if not only_print_end:
        print(' [AUTO] FWHM computed from the CCF is about : %.2f [km/s]'%(popt[-1]*2.35))
    if errors_fit[-1]/popt[-1]>0.2:
        print(' [WARNING] Error on the FWHM of the CCF > 20% ! Check the CCF and/or enter you own mask')
        plt.figure(figsize=(10,6))
        plt.plot(vrad/1000,ccf,label='CCF')
        plt.plot(vrad/1000,ras.gaussian(vrad/1000,popt[0],popt[1],popt[2],popt[3]),label='gaussian fit')
        plt.legend()
        plt.title('Debug graphic : CCF and fit to determine the FWHM\n Check that the fit has correctly converged')
        plt.xlabel('Vrad [km/s]')
        plt.ylabel('CCF')
        
    par_fwhm = popt[-1]*2.35

if par_smoothing_kernel=='rectangular':
    active_b = 0
elif par_smoothing_kernel=='gaussian':
    active_b = 1
else:
    active_b = 2
    
if (feedback)&(par_smoothing_box != 'auto'):
    spectre_backup = spectre.copy()
    fig = plt.figure(figsize=(14,7))
    plt.subplots_adjust(left=0.07,bottom=0.25,right=0.96,hspace=0,top=0.95)
    plt.title('Selection of the smoothing kernel length',fontsize=14)
    plt.plot(grid, spectre, color='b',alpha=0.4,label='input spectrum')
    plt.xlabel(r'Wavelength [$\AA$]',fontsize=14)
    plt.ylabel('Flux [arb. unit]',fontsize=14)
    l1, = plt.plot(grid, ras.smooth(spectre,int(par_smoothing_box),shape = par_smoothing_kernel), color='k',label='smoothed spectrum')
    plt.legend()
    axcolor = 'whitesmoke'
    axsmoothing = plt.axes([0.14, 0.1, 0.40, 0.02], facecolor = axcolor)
    ssmoothing = Slider(axsmoothing, 'Kernel length', 1, 10, valinit = int(par_smoothing_box), valstep=1)
    
    resetax = plt.axes([0.8, 0.05, 0.1, 0.1])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    rax = plt.axes([0.65, 0.05, 0.10, 0.10], facecolor=axcolor)
    radio = RadioButtons(rax, ('rectangular', 'gaussian','savgol'), active = active_b)
    
    class Index():
        shape = par_smoothing_kernel
        def update(self,val):
            smoothing = ssmoothing.val
            l1.set_ydata(ras.smooth(spectre,int(smoothing),shape=self.shape))
            fig.canvas.draw_idle()            
                
        def change_kernel(self,label):
            self.shape = label
    
    callback = Index()
    ssmoothing.on_changed(callback.update)
    radio.on_clicked(callback.change_kernel)
    radio.on_clicked(callback.update)
    
    def reset(event):
        ssmoothing.reset()
    button.on_clicked(reset)
    
    plt.show(block=False)
    answer = ras.sphinx(' Press ENTER to save the kernel length for the smoothing')
    spectre = ras.smooth(spectre,int(ssmoothing.val),shape=callback.shape)
    if not only_print_end:
        print(' Smoothed kernel length saved at : %s with %s shape'%(ssmoothing.val,callback.shape))
    plt.close()
    smoothing_shape = callback.shape
    smoothing_length = ssmoothing.val
    
    median = np.median(abs(spectre_backup-spectre))
    IQ = np.percentile(abs(spectre_backup-spectre),75) - median
    mask_out = np.where(abs(spectre_backup-spectre)>(median+20*IQ))[0]
    mask_out = np.unique(mask_out+np.arange(-smoothing_length,smoothing_length+1,1)[:,np.newaxis])
    mask_out = mask_out[(mask_out>=0)&(mask_out<len(grid))]
    spectre[mask_out.astype('int')] = spectre_backup[mask_out.astype('int')] #supress the smoothing of peak to sharp which create sinc-like wiggle
else:
    if par_smoothing_box == 'auto': 
        if par_smoothing_kernel not in ['erf','hat_exp']:
            if not only_print_end:
                print(' [WARNING] Your smoothing kernel is not correctly specified, pleaser enter either : erf or hat_exp')
                print(' The kernel is fixed by default to erf kernel')
            par_smoothing_kernel = 'erf' 
        grid_vrad = (grid-minx)/grid * ras.c_lum/1000 #grille en vitesse radiale (unités km/s)
        grid_vrad_equi = np.linspace(grid_vrad.min(),grid_vrad.max(),len(grid)) #new grid equidistant
        dv = np.diff(grid_vrad_equi)[0]  ##delta velocity
        spectrum_vrad = interp1d(grid_vrad, spectre, kind='cubic', bounds_error=False, fill_value='extrapolate')(grid_vrad_equi)
        
        sp = np.fft.fft(spectrum_vrad) 
        freq = np.fft.fftfreq(grid_vrad_equi.shape[-1])/dv #List of frequencies
        sig1 = par_fwhm/2.35 #fwhm-sigma conversion
        
        if par_smoothing_kernel == 'erf':
            alpha1 = np.exp(np.polyval(np.array([ 0.00210819, -0.04581559, 0.49444111, -1.78135102]), np.log(SNR_0))) #using the calibration curve calibration
            alpha2 = np.polyval(np.array([-0.04532947, -0.42650657,  0.59564026]), SNR_0)
        elif par_smoothing_kernel == 'hat_exp':
            alpha1 = np.exp(np.polyval(np.array([ 0.01155214, -0.20085361, 1.34901688, -3.63863408]), np.log(SNR_0))) #using the calibration curve calibration
            alpha2 = np.polyval(np.array([-0.06031564, -0.45155956, 0.67704286]), SNR_0)
        
        fourier_center = alpha1/sig1
        fourier_delta = alpha2/sig1
        cond = abs(freq) < fourier_center

        if par_smoothing_kernel == 'erf':        
            fourier_filter = 0.5*(erf((fourier_center-abs(freq))/fourier_delta)+1) #erf function
            smoothing_shape = 'erf'
        elif par_smoothing_kernel == 'hat_exp':
            fourier_filter = cond + (1-cond) * np.exp(-(abs(freq)- fourier_center)/fourier_delta) #Top hat with an exp  
            smoothing_shape = 'hat_exp'
        
        fourier_filter = fourier_filter/fourier_filter.max()
        
        spectrei_ifft = np.fft.ifft(fourier_filter*(sp.real+1j*sp.imag))
        #spectrei_ifft *= spectre.max()/spectrei_ifft.max()
        spectrei_ifft = np.abs(spectrei_ifft)
        spectre_back = interp1d(grid_vrad_equi, spectrei_ifft, kind='cubic', bounds_error=False, fill_value='extrapolate')(grid_vrad)
        median = np.median(abs(spectre_back-spectre))
        IQ = np.percentile(abs(spectre_back-spectre),75) - median
        mask_out_fourier = np.where(abs(spectre_back-spectre)>(median+20*IQ))[0]
        #plt.plot(grid_vrad_equi,abs(spectrei_ifft-spectrum_vrad))
        #plt.axhline(y=median+20*IQ)
        length_oversmooth = int(1/fourier_center/dv)
        mask_fourier = np.unique(mask_out_fourier+np.arange(-length_oversmooth,length_oversmooth+1,1)[:,np.newaxis])
        mask_fourier = mask_fourier[(mask_fourier>=0)&(mask_fourier<len(grid))]
        spectre_back[mask_fourier] = spectre[mask_fourier] #supress the smoothing of peak to sharp which create sinc-like wiggle
        spectre_back[0:length_oversmooth+1] = spectre[0:length_oversmooth+1] #suppression of the border which are at high frequencies
        spectre_back[-length_oversmooth:] = spectre[-length_oversmooth:]
        spectre = spectre_back.copy()
        smoothing_length = par_smoothing_box
    else:
        spectre_backup = spectre.copy()
        spectre = ras.smooth(spectre,int(par_smoothing_box),shape = par_smoothing_kernel)
        smoothing_shape = par_smoothing_kernel
        smoothing_length = par_smoothing_box
        median = np.median(abs(spectre_backup-spectre))
        IQ = np.percentile(abs(spectre_backup-spectre),75) - median
        mask_out = np.where(abs(spectre_backup-spectre)>(median+20*IQ))[0]
        mask_out = np.unique(mask_out+np.arange(-smoothing_length,smoothing_length+1,1)[:,np.newaxis])
        mask_out = mask_out[(mask_out>=0)&(mask_out<len(grid))]
        spectre[mask_out.astype('int')] = spectre_backup[mask_out.astype('int')] #supress the smoothing of peak to sharp which create sinc-like wiggle

par_fwhm = par_fwhm*conversion_fwhm_sig #conversion of the fwhm to angstrom lengthscale in the bluest part  

spectre = ras.empty_ccd_gap(grid,spectre,left=hole_left,right=hole_right)

index, flux = ras.local_max(spectre,par_vicinity)
index = index.astype('int')
wave = grid[index]

if flux[0] < spectre[0]:
    wave = np.insert(wave,0,grid[0])
    flux = np.insert(flux,0,spectre[0])
    index = np.insert(index,0,0)

if flux[-1] < spectre[-1]:
    wave = np.hstack([wave,grid[-1]])
    flux = np.hstack([flux,spectre[-1]])
    index = np.hstack([index,len(spectre)-1])
   
#supression of cosmic peak
median = np.ravel(pd.DataFrame(flux).rolling(10,center=True).quantile(0.50))
IQ = np.ravel(pd.DataFrame(flux).rolling(10,center=True).quantile(0.75)) - median
#plt.plot(wave,np.ravel(pd.DataFrame(flux).rolling(10,center=True).quantile(0.50))+10*IQ,color='k')
#plt.scatter(wave,flux)
IQ[np.isnan(IQ)] = spectre.max()
median[np.isnan(median)] = spectre.max()
mask = (flux > median + 20 * IQ)
#plt.show()
if not only_print_end:
    print(' Number of cosmic peaks removed : %.0f'%(np.sum(mask)))
wave = wave[~mask]
flux = flux[~mask]
index = index[~mask]

#print(' Rough estimation of the typical width of the lines : median=%.3f mean=%.3f'%(np.median(np.diff(wave))/conversion_fwhm_sig,np.mean(np.diff(wave))/conversion_fwhm_sig))

computed_parameters = 0.390/51.3*np.median(abs(np.diff(flux)))/np.median(np.diff(wave)) #old calibration

calib_low = np.polyval([-0.08769286, 5.90699857],par_fwhm/conversion_fwhm_sig)
calib_high = np.polyval([-0.38532535,20.17699949],par_fwhm/conversion_fwhm_sig)

if not only_print_end:
    print(' Suggestion of a streching parameter to try : %.0f +/- %.0f'%(calib_low + (calib_high-calib_low)*0.5,(calib_high-calib_low)*0.25))

out_of_calibration = False
if par_fwhm/conversion_fwhm_sig>30:
    out_of_calibration = True
    print(' [WARNING] Star out of the FWHM calibration range')

if type(par_stretching) == str:
    if not out_of_calibration:
        par_stretching = calib_low + (calib_high-calib_low) * float(par_stretching.split('_')[1])
        #par_stretching = 20*computed_parameters #old calibration
        if not only_print_end:
            print(' [AUTO] par_stretching fixed : %.2f'%(par_stretching))
    else:
        print(' [AUTO] par_stretching out of the calibration range, value fixed at 7')
        par_stretching = 7

spectre = spectre/par_stretching
flux = flux/par_stretching
normalisation = normalisation*par_stretching

locmaxx = wave.copy()
locmaxy = flux.copy()
locmaxz = index.copy()

if not only_print_end:
    print(' Computation of the local maxima : DONE' )

loc_max_time = time.time()

if not only_print_end:
    print(' Time of the step : %.2f'%(loc_max_time-begin))

waves = wave - wave[:,np.newaxis]
distance = np.sign(waves)*np.sqrt((waves)**2+(flux - flux[:,np.newaxis])**2)
distance[distance<0] = 0

numero = np.arange(len(distance)).astype('int')

# =============================================================================
#  PENALITY
# =============================================================================

if not only_print_end:
    print('\n Computation of the penality map : LOADING' )

# general parameters for the algorithm 
# (no need to modify the values except if you are visually unsatisfied of the penality plot)
# iteration increase the upper zone of the penality top

np.random.seed(random_number+1)
subset = np.sort(np.random.choice(np.arange(len(spectre)),size=int(len(spectre)/speedup),replace=False)) # take randomly 1 point over 10 to speed process

windows = 10. #10 typical line width scale (small window for the first continuum)
big_windows = 100.  #100 typical line width scale (large window for the second continuum)
iteration = 5
reg = par_reg_nu
par_model = reg.split('_')[0]
Penalty = False

if par_R=='auto':
    par_R = np.round(10*par_fwhm,1)
    if not only_print_end:
        print(' [AUTO] R fixed : %.1f'%(par_R))
    if par_R > 5:
        if not only_print_end:
            print(' [WARNING] R larger than 5, R fixed at 5')
        par_R = 5

if out_of_calibration:
    windows = 2. #2 typical line width scale (small window for the first continuum)
    big_windows = 20.  #20typical line width scale (large window for the second continuum)

law_chromatic = wave/minx

radius = par_R * np.ones(len(wave)) * law_chromatic
if (par_Rmax!=par_R)|(par_Rmax=='auto'):
    Penalty = True
    dx = par_fwhm/np.median(np.diff(grid))
    
    continuum_small_win = np.ravel(pd.DataFrame(spectre[subset]).rolling(int(windows*dx/speedup),center=True).quantile(1)) #rolling maximum with small windows
    continuum_right = np.ravel(pd.DataFrame(spectre[subset]).rolling(int(big_windows*dx/speedup)).quantile(1))
    continuum_left = np.ravel(pd.DataFrame(spectre[subset][::-1]).rolling(int(big_windows*dx/speedup)).quantile(1))[::-1]
    continuum_right[np.isnan(continuum_right)] = continuum_right[~np.isnan(continuum_right)][0] 
    continuum_left[np.isnan(continuum_left)] = continuum_left[~np.isnan(continuum_left)][-1]
    both = np.array([continuum_right,continuum_left])
    continuum_small_win[np.isnan(continuum_small_win)&(2*grid[subset]<(maxx+minx))] = continuum_small_win[~np.isnan(continuum_small_win)][0] 
    continuum_small_win[np.isnan(continuum_small_win)&(2*grid[subset]>(maxx+minx))] = continuum_small_win[~np.isnan(continuum_small_win)][-1] 
    continuum_large_win = np.min(both,axis=0) #when taking a large window, the rolling maximum depends on the direction make both direction and take the minimum
    
    median_large = np.ravel(pd.DataFrame(continuum_large_win).rolling(int(10*big_windows*dx),min_periods=1,center=True).quantile(0.5))
    Q3_large = np.ravel(pd.DataFrame(continuum_large_win).rolling(int(10*big_windows*dx),min_periods=1,center=True).quantile(0.75))
    q3_large = np.ravel(pd.DataFrame(continuum_large_win).rolling(int(big_windows*dx),min_periods=1,center=True).quantile(0.75))
    Q1_large = np.ravel(pd.DataFrame(continuum_large_win).rolling(int(10*big_windows*dx),min_periods=1,center=True).quantile(0.25))
    q1_large = np.ravel(pd.DataFrame(continuum_large_win).rolling(int(big_windows*dx),min_periods=1,center=True).quantile(0.25))
    IQ1_large = Q3_large - Q1_large
    IQ2_large = q3_large - q1_large
    sup_large = np.min([Q3_large+1.5*IQ1_large,q3_large+1.5*IQ2_large],axis=0)

    if speedup > 1:
        sup_large = interp1d(subset, sup_large, bounds_error=False, fill_value='extrapolate')(np.arange(len(spectre)))
        continuum_large_win = interp1d(subset, continuum_large_win, bounds_error=False, fill_value='extrapolate')(np.arange(len(spectre)))
        median_large = interp1d(subset, median_large, bounds_error=False, fill_value='extrapolate')(np.arange(len(spectre)))    

    mask = (continuum_large_win > sup_large)
    continuum_large_win[mask] = median_large[mask] 

    median_small = np.ravel(pd.DataFrame(continuum_small_win).rolling(int(10*big_windows*dx/speedup),min_periods=1,center=True).quantile(0.5))
    Q3_small = np.ravel(pd.DataFrame(continuum_small_win).rolling(int(10*big_windows*dx/speedup),min_periods=1,center=True).quantile(0.75))
    q3_small = np.ravel(pd.DataFrame(continuum_small_win).rolling(int(big_windows*dx/speedup),min_periods=1,center=True).quantile(0.75))
    Q1_small = np.ravel(pd.DataFrame(continuum_small_win).rolling(int(10*big_windows*dx/speedup),min_periods=1,center=True).quantile(0.25))
    q1_small = np.ravel(pd.DataFrame(continuum_small_win).rolling(int(big_windows*dx/speedup),min_periods=1,center=True).quantile(0.25))
    IQ1_small = Q3_small - Q1_small
    IQ2_small = q3_small - q1_small
    sup_small = np.min([Q3_small+1.5*IQ1_small,q3_small+1.5*IQ2_small],axis=0)
    
    if speedup>1:
        sup_small = interp1d(subset, sup_small, bounds_error=False, fill_value='extrapolate')(np.arange(len(spectre)))
        continuum_small_win = interp1d(subset, continuum_small_win, bounds_error=False, fill_value='extrapolate')(np.arange(len(spectre)))
        median_small = interp1d(subset, median_small, bounds_error=False, fill_value='extrapolate')(np.arange(len(spectre)))
    
    mask = (continuum_small_win > sup_small)
    continuum_small_win[mask] = median_small[mask]
        
    loc_out = ras.local_max(continuum_large_win,2)[0]
    for k in loc_out.astype('int'):
        continuum_large_win[k] = np.min([continuum_large_win[k-1],continuum_large_win[k+1]])
    
    loc_out = ras.local_max(continuum_small_win,2)[0]
    for k in loc_out.astype('int'):
        continuum_small_win[k] = np.min([continuum_small_win[k-1],continuum_small_win[k+1]])    
    
    continuum_large_win = np.where(continuum_large_win==0, 1.0, continuum_large_win) #replace null values
    penalite0 = (continuum_large_win - continuum_small_win)/continuum_large_win
    penalite0[penalite0<0] = 0
    penalite = penalite0.copy()
        
    for j in range(iteration): #make the continuum less smooth (step-like function) to improve the speed later
        continuum_right = np.ravel(pd.DataFrame(penalite).rolling(int(windows*dx)).quantile(1))
        continuum_left = np.ravel(pd.DataFrame(penalite[::-1]).rolling(int(windows*dx)).quantile(1))[::-1]
        continuum_right[np.isnan(continuum_right)] = continuum_right[~np.isnan(continuum_right)][0] #define for the left border all nan value to the first non nan value
        continuum_left[np.isnan(continuum_left)] = continuum_left[~np.isnan(continuum_left)][-1] #define for the right border all nan value to the first non nan value
        both = np.array([continuum_right,continuum_left])
        penalite = np.max(both,axis=0)    
    
    penalite_step = penalite.copy()
    mini = penalite_step.min()
    penalite_step = penalite_step - mini
    maxi = penalite_step.max()
    penalite_step = penalite_step/maxi
    
    penalite_graph = penalite_step[index] ; #take the penalite value at the local maxima position
        
    threshold = 0.75
    loop = True
    if par_Rmax=='auto':
        while (loop)&(threshold>0.2):
            difference = (continuum_large_win<continuum_small_win).astype('int')  
            cluster_broad_line = ras.grouping(difference,0.5,0)[-1]
            if cluster_broad_line[0][0]==0:#rm border left
                cluster_broad_line = cluster_broad_line[1:]
            if cluster_broad_line[-1][1]==len(grid)-2:#rm border right
                cluster_broad_line = cluster_broad_line[0:-1]
            
            penality_cluster = np.zeros(len(cluster_broad_line[:,2]))
            for j in range(len(cluster_broad_line[:,2])):
                penality_cluster[j] = np.max(penalite0[cluster_broad_line[j,0]:cluster_broad_line[j,1]+1])
            cluster_length = np.hstack([cluster_broad_line,penality_cluster[:,np.newaxis]])
            cluster_length = cluster_length[cluster_length[:,3]>threshold,:] #only keep cluster with high enough penality
            if len(cluster_length)==0:
                threshold -=0.05
                continue
            cluster_length = np.hstack([cluster_length,np.zeros(len(cluster_length))[:,np.newaxis]])
            for j in range(len(cluster_length)):
                cluster_length[j,4] = np.nanpercentile(abs(np.diff(spectre[int(cluster_length[j,0]):int(cluster_length[j,1])])),10)
            cluster_length = cluster_length[cluster_length[:,4]!=0,:]
            if len(cluster_length)==0:
                threshold -=0.05
                continue
            else:
                loop = False
        if threshold>0.2:
            band_center = np.mean(grid[cluster_length[:,0:2].astype('int')],axis=1)   
            cluster_length = np.hstack([cluster_length,band_center[:,np.newaxis]])
            largest_cluster = np.argmax(cluster_length[:,2]/cluster_length[:,5]) #largest radius in vrad unit
            largest_radius = cluster_length[largest_cluster,2]*dgrid #largest radius in vrad unit
    
            par_Rmax = 2*np.round(largest_radius*minx/cluster_length[largest_cluster,5]/cluster_length[largest_cluster,3],0)
        else:
            par_Rmax=par_R
        if not (only_print_end)|(threshold<0.2):
            print(' [AUTO] Rmax found around %.0f AA and fixed : %.0f'%(cluster_length[largest_cluster,5],par_Rmax))
        if par_Rmax > 150:
            if not only_print_end:
                print(' [WARNING] Rmax larger than 150, Rmax fixed at 150')
            par_Rmax = 150
      
    par_R = np.round(par_R,1)
    par_Rmax = np.round(par_Rmax,1)
    
    if feedback:
        t = np.linspace(0,1,100)
        radius = grid/minx * ( par_R + (par_Rmax - par_R) * penalite_step ** (float(reg.split('_')[-1])))
        law = par_R + (par_Rmax - par_R) * t ** float(reg.split('_')[-1])
        if reg.split('_')[0] == 'poly':
            alpha1 = 1 ; alpha2 = 0 ; actif = 0 ; ini = 0.5
        elif reg.split('_')[0] == 'sigmoid':
            alpha1 = 0 ; alpha2 = 1 ; actif = 1  ; ini = float(reg.split('_')[-2])         
        radius2 = grid/minx * (par_R + (par_Rmax - par_R) * (1+np.exp(-10*float(reg.split('_')[-1]) * (penalite_step - ini))) ** -1)
        law2 = par_R + (par_Rmax - par_R) * (1+np.exp(-10*float(reg.split('_')[-1]) * (t-ini))) ** -1
        
        fig = plt.figure(figsize=(12,6))
        plt.subplot(3,2,1)
        plt.plot(grid, spectre,zorder=1,color='k',alpha=0.5)
        plt.plot(grid, continuum_small_win,color='r',zorder=3,lw=2,label=r'$S_1$ continuum')
        plt.plot(grid, continuum_large_win,color='k',zorder=4,lw=3,label=r'$S_2$ continuum')
        plt.scatter(wave,flux,color='blue',s=1,zorder=2,label='local maxima')
        plt.xlabel(r'Wavelength [$\AA$]',fontsize=13) ; plt.ylabel('Flux [arb. unit]',fontsize=13) ; ax = plt.gca()  
        plt.tick_params(direction='in',top=True,right=True)
        plt.legend(loc=2)
        
        plt.subplot(3,2,3,sharex=ax)
        plt.plot(grid, penalite0,color='k',alpha=0.5, label='penalty computed')
        plt.plot(grid, penalite_step, color='k', lw=2, label='penalty extracted')
        plt.xlabel('Wave',fontsize=13)
        plt.ylabel('Penalty',fontsize=13)
        plt.tick_params(direction='in',top=True,right=True)
        plt.legend()
        
        plt.subplot(3,2,5,sharex=ax)
        l1, = plt.plot(grid, radius, color='k',lw=2,alpha=alpha1)
        l3, = plt.plot(grid, radius2, color='k',lw=2,alpha=alpha2)
        plt.plot(grid,grid/minx * par_R,color='r',ls=':',label='chromatic law')
        plt.legend()
        ax = plt.gca()
        plt.xlabel(r'Wavelength [$\AA$]',fontsize=13)
        plt.ylabel(r'Radius [$\AA$]',fontsize=13) 
        plt.tick_params(direction='in',top=True,right=True)

        axcolor = 'whitesmoke'
        axexponent = plt.axes([0.55, 0.25, 0.35, 0.03], facecolor = axcolor)
        sexponent = Slider(axexponent, 'Nu', 0.1, 5.0, valinit=float(reg.split('_')[-1]), valstep=0.05)
        axexponent2 = plt.axes([0.55, 0.20, 0.35, 0.03], facecolor = axcolor)
        sexponent2 = Slider(axexponent2, 'Mu', 0, 1, valinit=ini, valstep=0.05)
        axrmin = plt.axes([0.55, 0.3, 0.35, 0.03], facecolor = axcolor)
        srmin = Slider(axrmin, 'R', 1.0, 10.0, valinit=par_R, valstep=0.1)
        axrmax = plt.axes([0.55, 0.35, 0.35, 0.03], facecolor = axcolor)
        srmax = Slider(axrmax, 'Rmax', par_R, 150, valinit=par_Rmax, valstep=1)
        plt.subplot(2,2,2)
        plt.title('Selection of the penalty-radius law',fontsize=14)
        l2, = plt.plot(t,law,color='k',alpha=alpha1)
        l4, = plt.plot(t,law2,color='k',alpha=alpha2)
        plt.xlabel('Penalty',fontsize=13)
        plt.ylabel(r'Radius [$\AA$]',fontsize=13) 
        ax2 = plt.gca()
        
        rax = plt.axes([0.55, 0.05, 0.15, 0.10], facecolor=axcolor)
        radio = RadioButtons(rax, ('poly', 'sigmoid'), active=actif)
        
        class Index():
            model = reg.split('_')[0]
            def update(self,val):
                expo = sexponent.val
                expo2 = sexponent2.val
                par_R = srmin.val
                par_Rmax = srmax.val
                if self.model == 'poly':
                    ax.set_ylim(par_R-(par_Rmax-par_R)*0.1-1,par_Rmax+(par_Rmax-par_R)*0.1+1)
                    ax2.set_ylim(par_R-(par_Rmax-par_R)*0.1-1,par_Rmax+(par_Rmax-par_R)*0.1+1)
                    radius = grid/minx * (par_R + (par_Rmax-par_R) * penalite_step ** (expo))
                    l1.set_ydata(radius)
                    l2.set_ydata(par_R + (par_Rmax-par_R) * t ** expo)
                else:
                    ax.set_ylim(par_R - (par_Rmax-par_R) * 0.1-1,  par_Rmax + (par_Rmax-par_R) * 0.1+1)
                    ax2.set_ylim(par_R - (par_Rmax-par_R) * 0.1-1, par_Rmax + (par_Rmax-par_R) * 0.1+1)
                    radius2 = grid/minx * (par_R + (par_Rmax-par_R) * (1+np.exp(-10*expo*(penalite_step-expo2))) ** -1)
                    l3.set_ydata(radius2)
                    l4.set_ydata(par_R + (par_Rmax-par_R) * (1+np.exp(-10*expo*(t-expo2))) ** -1)
                fig.canvas.draw_idle()
            
            def change_model(self,label):
                self.model = label
                if self.model=='poly':
                    l1.set_alpha(1) ; l2.set_alpha(1)
                    l3.set_alpha(0) ; l4.set_alpha(0)
                else:
                    l1.set_alpha(0) ; l2.set_alpha(0)
                    l3.set_alpha(1) ; l4.set_alpha(1)
                fig.canvas.draw_idle()
        
        callback = Index()
        radio.on_clicked(callback.change_model)
        radio.on_clicked(callback.update)
        sexponent.on_changed(callback.update)
        sexponent2.on_changed(callback.update)
        srmin.on_changed(callback.update)
        srmax.on_changed(callback.update)
        
        resetax = plt.axes([0.8, 0.05, 0.1, 0.1])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        
        def reset(event):
            sexponent.reset()
            sexponent2.reset()
            srmin.reset()
            srmax.reset()
        button.on_clicked(reset)
        plt.subplots_adjust(hspace=0,top=0.95,left=0.06,right=0.96)
        plt.show(block=False)
        answer = ras.sphinx('Press ENTER to save penality law')
        par_R = srmin.val
        par_Rmax = srmax.val
        expo = np.round(sexponent.val,2)
        expo2 = sexponent2.val
        if callback.model =='poly':
            radius = law_chromatic * (par_R + (par_Rmax-par_R) * penalite_graph ** (expo))
            par_model = 'poly_%.2f'%(expo)
            if not only_print_end:
                print(' You selected the poly law (exponent %.2f) with R : %.2f and Rmax %.2f'%(expo,par_R,par_Rmax))
        else: 
            radius = law_chromatic * (par_R + (par_Rmax-par_R) * (1+np.exp(-10*expo*(penalite_graph-expo2))) ** -1)
            par_model = 'sigmoid_%.2f_%.2f'%(expo,expo2)
            if not only_print_end:
                print(' You selected the sigmoid law (sigma %.2f, center %.2f) with R : %.2f and Rmax %.2f'%(expo,expo2,par_R,par_Rmax))
        plt.close()
    else:
        if reg.split('_')[0] == 'poly':
           expo = float(reg.split('_')[-1])
           radius = law_chromatic * (par_R + (par_Rmax-par_R) * penalite_graph ** (expo))
           par_model = reg
        elif reg.split('_')[0] == 'sigmoid':
           center = float(reg.split('_')[-2])
           width = float(reg.split('_')[-1])
           radius = law_chromatic * (par_R + (par_Rmax-par_R) * (1+np.exp(-10*width*(penalite_graph-center))) ** -1)
           par_model = reg
        else:
            if not only_print_end:
                print(' the law should be either poly_d or sigmoid_c_s')       
 
if not only_print_end:
    print(' Computation of the penality map : DONE' )
    
loc_penality_time = time.time()


if not only_print_end:
    print(' Time of the step : %.2f'%(loc_penality_time-loc_max_time))

# =============================================================================
#  ROLLING PIN
# =============================================================================

if not only_print_end:
    print('\n Rolling pin is rolling : LOADING' )

mask = (distance>0)&(distance<2.*par_R)

loop = 'y'
count_iter = 0
k_factor = []

while loop == 'y':
    mask = np.zeros(1)
    radius[0] = radius[0]/1.5
    keep = [0]
    j = 0
    R_old = par_R
    
    while (len(wave)-j>3):
        par_R = float(radius[j]) #take the radius from the penality law
        mask = (distance[j,:]>0)&(distance[j,:]<2.*par_R) #recompute the points closer than the diameter if Radius changed with the penality      
        while np.sum(mask)==0:
            par_R *=1.5
            mask = (distance[j,:]>0)&(distance[j,:]<2.*par_R) #recompute the points closer than the diameter if Radius changed with the penality      
        p1 = np.array([wave[j],flux[j]]).T #vector of all the local maxima 
        p2 = np.array([wave[mask],flux[mask]]).T #vector of all the maxima in the diameter zone
        delta = p2 - p1 # delta x delta y
        c = np.sqrt(delta[:,0]**2+delta[:,1]**2) # euclidian distance 
        h = np.sqrt(par_R**2-0.25*c**2) 
        cx = p1[0] + 0.5*delta[:,0] - h/c*delta[:,1] #x coordinate of the circles center
        cy = p1[1] + 0.5*delta[:,1] + h/c*delta[:,0] #y coordinates of the circles center
                
        cond1 = (cy-p1[1])>=0
        thetas = cond1*(-1*np.arccos((cx - p1[0])/par_R)+np.pi) + (1-1*cond1)*(-1*np.arcsin((cy - p1[1])/par_R) + np.pi)
        j2 = thetas.argmin()
        j = numero[mask][j2] #take the numero of the local maxima falling in the diameter zone
        keep.append(j)
    flux = flux[keep] #we only keep the local maxima with the rolling pin condition
    wave = wave[keep]
    index = index[keep]
            
    if feedback:
        plt.figure(figsize=(16,8))
        plt.plot(grid,spectre)
        plt.xlabel(r'Wavelength [$\AA$]',fontsize=14)
        plt.ylabel('Flux [arb. unit]',fontsize=14)
        plt.tick_params(top=True)
        Interpol = interp1d(wave[1:-1], flux[1:-1], kind=interpol, bounds_error=False, fill_value='extrapolate')
        continuum = Interpol(grid)
        plt.plot(grid, continuum,label='intermediate continuum')
        plt.scatter(wave,flux,color='k',label='anchor points')
        plt.legend()
        plt.title('Visualisation of intermediate continuum',fontsize=14)
        plt.ylim(spectre.min(), spectre.max())
        #ax = plt.axes([0.20,0.5,0.3,0.3])
        #ax.patch.set_alpha(0.4)
        #ax.spines['bottom'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        #ax.xaxis.set_ticks_position('top')
        #ax.set_ybound(-R_old, R_old)
        #ax.set_xbound(-R_old, R_old)
        #ax.plot(np.linspace(-R_old, R_old, 100), np.sqrt(R_old**2-(np.linspace(-R_old, R_old, 100))**2), color='k')
        #ax.plot(np.linspace(-R_old, R_old, 100), -np.sqrt(R_old**2-(np.linspace(-R_old, R_old, 100))**2), color='k')
        plt.show(block = False)
        loop = ras.sphinx('Do you want to perform a new loop with R = KxR (y/n)? Enter the K that you want (if (y) is given the default value is K=1.5)',rep=['y','n']+['%.0f'%(j) for j in np.arange(1,10)]+['%.1f'%(j) for j in np.arange(1,10,0.1)])
        plt.close()
        if loop=='n':
            break
        if (loop != 'y')&(loop != 'n'):
            par_R = float(loop)*R_old
            radius = float(loop)*radius
            k_factor.append(float(loop))
            loop = 'y'
        elif loop == 'y':
            par_R = 1.5*R_old
            radius = 1.5*radius
        waves = wave - wave[:,np.newaxis]
        distance = np.sign(waves)*np.sqrt((waves)**2+(flux - flux[:,np.newaxis])**2)
        distance[distance<0]=0
        numero = np.arange(len(distance)).astype('int')
        count_iter+=1
    else:
        loop = 'n'

if not only_print_end:
    print(' Rolling pin is rolling : DONE' )

loc_rolling_time = time.time()

if not only_print_end:
    print(' Time of the step : %.2f'%(loc_rolling_time-loc_penality_time))

# =============================================================================
# EDGE CUTTING
# =============================================================================

if not only_print_end:
    print('\n Edge cutting : LOADING' )

count_cut = 0
if feedback:
    fig = plt.figure(figsize=(12,6))
    plt.subplots_adjust(left=0.10, bottom=0.25,top=0.9,hspace=0.35)
    plt.subplot(2,1,1)
    plt.title('Edges cutting step',fontsize=14)
    plt.plot(grid, spectre,label='spectrum')
    l1, = plt.plot(wave, flux, ls='',color='k',marker='o', label='anchor points')
    plt.xlabel(r'Wavelength [$\AA$]',fontsize=14)
    plt.ylabel('Flux [arb. unit]',fontsize=14)
    
    Interpol = interp1d(wave, flux, kind=interpol, bounds_error=False, fill_value='extrapolate')
    continuum = Interpol(grid)
    continuum = ras.troncated(continuum,spectre)
    l2, = plt.plot(grid, continuum, label='intermediate continuum')
    plt.xlim(None, minx+len_x/10.)
    plt.ylim(None,spectre[0:int(len(grid)/10.)].max())
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(grid, spectre)
    l3, = plt.plot(wave, flux, ls='',color='k',marker='o')
    plt.xlabel(r'Wavelength [$\AA$]',fontsize=14)
    plt.ylabel('Flux [arb. unit]',fontsize=14)
    l4, = plt.plot(grid, continuum)
    plt.xlim(minx+9*len_x/10.,None)
    plt.ylim(None,spectre[-int(len(grid)/10.):].max())
    plt.show(block = False)
        
    class Index(object):
        ind = 0
        flux_backup = [flux.copy()]
        def update(self,event):
            wave = l1.get_xdata()
            flux = self.flux_backup[-1].copy()
            flux[0:self.ind+1]=flux[self.ind+1] ; flux[-1-self.ind:] = flux[-2-self.ind]
            self.ind +=1
            self.flux_backup.append(flux)
            continuum = interp1d(wave, flux, kind = interpol, bounds_error = False, fill_value = 'extrapolate')(grid)            
            continuum = ras.troncated(continuum,spectre)
            l1.set_ydata(flux) 
            l2.set_ydata(continuum)
            l3.set_ydata(flux)
            l4.set_ydata(continuum)
            plt.draw()
            fig.canvas.draw_idle()
        
        def backup(self,event):            
            if len(self.flux_backup)>1:
                wave = l1.get_xdata()
                self.flux_backup = self.flux_backup[:-1]                
                
                flux = self.flux_backup[-1]
                continuum = interp1d(wave, flux, kind = interpol, bounds_error = False, fill_value = 'extrapolate')(grid)
                continuum = ras.troncated(continuum,spectre)
                l1.set_ydata(flux) 
                l2.set_ydata(continuum)
                l3.set_ydata(flux)
                l4.set_ydata(continuum)
                
                if self.ind != 0:
                    self.ind-=1
                plt.draw()
                fig.canvas.draw_idle()
    
    callback = Index()
    axnext = plt.axes([0.2, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Cut')
    bnext.on_clicked(callback.update)
    axprev = plt.axes([0.75, 0.05, 0.1, 0.075])
    anext = Button(axprev, 'Cancel')
    anext.on_clicked(callback.backup)
    loop = ras.sphinx('Press Enter to move on the last step')
    wave = l1.get_xdata()
    flux = l1.get_ydata()
    count_cut = callback.ind
    plt.close()
else:
    j = 0
    while count_cut<count_cut_lim:
        flux[0:j+1]=flux[j+1] ; flux[-1-j:] = flux[-2-j]
        j += 1
        count_cut += 1

if not only_print_end:
    print(' Edge cutting : DONE' )

loc_cutting_time = time.time()

if not only_print_end:
    print(' Time of the step : %.2f'%(loc_cutting_time-loc_rolling_time))

# =============================================================================
# CAII MASKING
# =============================================================================

mask_caii = ((wave>3929)&(wave<3937))|((wave>3964)&(wave<3972))  

wave = wave[~mask_caii]
flux = flux[~mask_caii]
index = index[~mask_caii]

# =============================================================================
# OUTLIERS REMOVING
# =============================================================================

if not only_print_end:
    print('\n Outliers removing : LOADING' )

count_out = 0
win_grap = 10#int(30*typ_line_width/dlambda)
lines_showed = 5

if feedback:
    fig = plt.figure(figsize=(12,6))
    diff_deri = abs(np.diff(np.diff(flux)/np.diff(wave)))
    sort_val = np.argsort(diff_deri)+1
    sort_val = np.insert(sort_val,0,[0,len(flux)-1])[::-1]
    mask_out = np.zeros(len(flux)).astype('bool')
    mask_out[sort_val[0:lines_showed]] = True
    center = index[sort_val[0]]
    
    plt.subplots_adjust(left = 0.10, bottom = 0.25, top = 0.95,hspace=0.35)
    plt.subplot(2,1,1)
    plt.plot(grid, spectre)
    plt.title('Anomalous maxima suppression step',fontsize=14)
    plt.xlabel(r'Wavelength [$\AA$]',fontsize=14)
    plt.ylabel('Flux [arb. unit]',fontsize=14)
    
    l1, = plt.plot(wave, flux, ls='',marker='o',color='k',label='anchor points')
    l2,l3 = plt.plot(wave[mask_out], flux[mask_out], 'ro', 
                  wave[sort_val[0]], flux[sort_val[0]], 'go', 
                  ls='',zorder=99,label='doubtful anchor points')       

    lvline = plt.axvline(x = wave[sort_val[0]], color='k')
    
    Interpol = interp1d(wave, flux, kind = interpol, bounds_error = False, fill_value = 'extrapolate')
    continuum = Interpol(grid)
    continuum = ras.troncated(continuum,spectre)
    l4, = plt.plot(grid, continuum, alpha = 1)
    plt.legend()

    Interpol = interp1d(wave[~mask_out], flux[~mask_out], kind = interpol, bounds_error = False, fill_value = 'extrapolate')
    continuum = Interpol(grid)
    continuum = ras.troncated(continuum,spectre)
    l5, = plt.plot(grid, continuum, color='k',ls=':',alpha = 1)            
    
    plt.subplot(2,1,2)
    plt.xlabel(r'Wavelength [$\AA$]',fontsize=14)
    plt.ylabel('Flux [arb. unit]',fontsize=14)
        
    win_grap1 = np.min([sort_val[0],win_grap]) 
    win_grap2 = np.min([len(wave)-sort_val[0],win_grap])-1
    
    plt.plot(grid, spectre)
    plt.xlim(wave[sort_val[0]-win_grap1],wave[sort_val[0]+win_grap2])
    y1, y2 = np.min(spectre[center-win_grap1:center+win_grap2]),np.max(spectre[center-win_grap1:center+win_grap2])
    y1, y2 = np.min(flux[sort_val[0]-win_grap1:sort_val[0]+win_grap2]),np.max(flux[sort_val[0]-win_grap1:sort_val[0]+win_grap2])

    plt.ylim(y1-(y2-y1)*0.2, y2+(y2-y1)*0.2)
    ax = plt.gca()
    
    l6, = plt.plot(wave, flux, ls='',marker='o',color='k')
    
    mask_out2 = np.zeros(len(flux)).astype('bool')
    mask_out2[sort_val[0]] = True
    l7, = plt.plot(wave[mask_out2], flux[mask_out2], ls='', marker='o', color='g')    

    Interpol = interp1d(wave, flux, kind = interpol, bounds_error = False, fill_value = 'extrapolate')
    continuum = Interpol(grid)
    continuum = ras.troncated(continuum,spectre)
    l8, = plt.plot(grid, continuum, alpha = 1,label='current continuum')
    Interpol = interp1d(wave[~mask_out2], flux[~mask_out2], kind = interpol, bounds_error = False, fill_value = 'extrapolate')
    continuum = Interpol(grid)
    continuum = ras.troncated(continuum,spectre)
    l9, = plt.plot(grid, continuum, color='k',ls=':',alpha = 1,label='updated continuum')            
    plt.legend()
    plt.show(block=False)
    
    class Index(object):
        ind = 0
        wave_backup = [wave]
        flux_backup = [flux]
        index_backup = [index]
        sort_backup = [sort_val]
        out_backup = []
        
        def rm_all(self,event):
            mask_out = np.zeros(len(self.flux_backup[-1])).astype('bool')
            mask_out[self.sort_backup[-1][0:lines_showed]] = True
            self.out_backup.append(mask_out)
            self.ind = 0
            
        def rm_one(self,event):
            mask_out = np.zeros(len(self.flux_backup[-1])).astype('bool')
            mask_out[self.sort_backup[-1][self.ind]] = True
            self.out_backup.append(mask_out)
            self.ind = 0
            
        def update(self,event):
            
            wave = self.wave_backup[-1][~self.out_backup[-1]]
            flux = self.flux_backup[-1][~self.out_backup[-1]]
            index = self.index_backup[-1][~self.out_backup[-1]]
            
            self.wave_backup.append(wave)
            self.flux_backup.append(flux)
            self.index_backup.append(index)
            
            diff_deri = abs(np.diff(np.diff(flux)/np.diff(wave)))
            sort_val = np.argsort(diff_deri)+1
            sort_val = np.insert(sort_val,0,[0,len(flux)-1])[::-1]
            center = index[sort_val[0]]
            mask_out = np.zeros(len(flux)).astype('bool')
            mask_out[sort_val[0:lines_showed]] = True
            self.sort_backup.append(sort_val)
            
            Interpol = interp1d(wave, flux, kind = interpol, bounds_error = False, fill_value = 'extrapolate')
            continuum = Interpol(grid)
            continuum = ras.troncated(continuum,spectre)
            
            l1.set_xdata(wave) ; l1.set_ydata(flux)
            l6.set_xdata(wave) ; l6.set_ydata(flux)
            
            l3.set_xdata(grid[center]) ; l3.set_ydata(spectre[center])
            lvline.set_xdata(x=grid[center])

            l7.set_xdata(grid[center]) ; l7.set_ydata(spectre[center])    
            
            win_grap1 = np.min([sort_val[0],win_grap]) 
            win_grap2 = np.min([len(wave)-sort_val[0],win_grap])-1
                    
            ax.set_xlim(wave[sort_val[0]-win_grap1],wave[sort_val[0]+win_grap2])
            y1, y2 = np.min(flux[sort_val[0]-win_grap1:sort_val[0]+win_grap2]),np.max(flux[sort_val[0]-win_grap1:sort_val[0]+win_grap2])

            ax.set_ylim(y1-(y2-y1)*0.2, y2+(y2-y1)*0.2)

            l2.set_xdata(wave[mask_out]) ; l2.set_ydata(flux[mask_out])
            l4.set_ydata(continuum)
            l8.set_ydata(continuum)
            
            Interpol = interp1d(wave[~mask_out], flux[~mask_out], kind = interpol, bounds_error = False, fill_value = 'extrapolate')
            continuum = Interpol(grid)
            l5.set_ydata(continuum)
            
            mask_out2 = np.zeros(len(flux)).astype('bool')
            mask_out2[sort_val[self.ind]] = True
            Interpol = interp1d(wave[~mask_out2], flux[~mask_out2], kind = interpol, bounds_error = False, fill_value = 'extrapolate')
            continuum = Interpol(grid)
            continuum = ras.troncated(continuum,spectre)
            l9.set_ydata(continuum) 
            
            self.ind = 0 
            plt.draw()
            fig.canvas.draw_idle()
        
        def backup(self,event):            
            if len(self.wave_backup)>1:
                self.wave_backup = self.wave_backup[:-1]
                self.flux_backup = self.flux_backup[:-1]
                self.index_backup = self.index_backup[:-1]
                self.sort_backup = self.sort_backup[:-1]
                sort_val = self.sort_backup[-1]
                
                mask_out = np.zeros(len(self.flux_backup[-1])).astype('bool')
                mask_out[self.sort_backup[-1][0:lines_showed]] = True
                
                Interpol = interp1d(self.wave_backup[-1], self.flux_backup[-1], kind = interpol, bounds_error = False, fill_value = 'extrapolate')
                continuum = Interpol(grid)
                continuum = ras.troncated(continuum,spectre)
                l4.set_ydata(continuum)
                l8.set_ydata(continuum)
                            
                Interpol = interp1d(self.wave_backup[-1][~self.out_backup[-1]], self.flux_backup[-1][~self.out_backup[-1]], kind = interpol, bounds_error = False, fill_value = 'extrapolate')
                continuum = Interpol(grid)
                continuum = ras.troncated(continuum,spectre)
                l5.set_ydata(continuum)
                
                l1.set_xdata(self.wave_backup[-1]) 
                l1.set_ydata(self.flux_backup[-1])
                l2.set_xdata(self.wave_backup[-1][mask_out])
                l2.set_ydata(self.flux_backup[-1][mask_out])
                
                center = self.index_backup[-1][self.sort_backup[-1][0]]
                
                lvline.set_xdata(x=grid[center])
                l3.set_xdata(grid[center]) ; l3.set_ydata(spectre[center]) 
                l7.set_xdata(grid[center]) ; l7.set_ydata(spectre[center])  
                
                win_grap1 = np.min([sort_val[0],win_grap]) 
                win_grap2 = np.min([len(wave)-sort_val[0],win_grap])-1
                ax.set_xlim(wave[sort_val[0]-win_grap1],wave[sort_val[0]+win_grap2])
                y1, y2 = np.min(flux[sort_val[0]-win_grap1:sort_val[0]+win_grap2]),np.max(flux[sort_val[0]-win_grap1:sort_val[0]+win_grap2])

                ax.set_ylim(y1-(y2-y1)*0.2, y2+(y2-y1)*0.2)
                
                self.ind = 0
                self.out_backup = self.out_backup[:-1]
                plt.draw()
                fig.canvas.draw_idle()
                
        def next_line(self,event):
            if self.ind<lines_showed-1:
                self.ind += 1
                index =  self.index_backup[-1]
                wave =  self.wave_backup[-1]
                sort_val = self.sort_backup[-1]
                flux = self.flux_backup[-1]
                mask_out2 = np.zeros(len(flux)).astype('bool')
                mask_out2[sort_val[self.ind]] = True
                Interpol = interp1d(wave[~mask_out2], flux[~mask_out2], kind = interpol, bounds_error = False, fill_value = 'extrapolate')
                continuum = Interpol(grid)
                continuum = ras.troncated(continuum,spectre)
                l9.set_ydata(continuum)    
                
                center = index[sort_val[self.ind]]
                lvline.set_xdata(x=grid[center])
                ax.set_xlim(wave[sort_val[self.ind]-win_grap],wave[sort_val[self.ind]+win_grap+1])
                y1, y2 = np.min(flux[sort_val[self.ind]-win_grap:sort_val[self.ind]+win_grap+1]),np.max(flux[sort_val[self.ind]-win_grap:sort_val[self.ind]+win_grap+1])
                ax.set_ylim(y1-(y2-y1)*0.2, y2+(y2-y1)*0.2)
                l7.set_xdata(grid[center]) ; l7.set_ydata(spectre[center])
                l3.set_xdata(grid[center]) ; l3.set_ydata(spectre[center])
                plt.draw()
                fig.canvas.draw_idle()
            
        def prev_line(self,event):
            if self.ind>0:
                self.ind -= 1
                index =  self.index_backup[-1]
                sort_val = self.sort_backup[-1]
                flux = self.flux_backup[-1]
                wave =  self.wave_backup[-1]
                mask_out2 = np.zeros(len(flux)).astype('bool')
                mask_out2[sort_val[self.ind]] = True
                Interpol = interp1d(wave[~mask_out2], flux[~mask_out2], kind = interpol, bounds_error = False, fill_value = 'extrapolate')
                continuum = Interpol(grid)
                continuum = ras.troncated(continuum,spectre)
                l9.set_ydata(continuum)   
                
                center = index[sort_val[self.ind]]
                lvline.set_xdata(x=grid[center])
                ax.set_xlim(wave[sort_val[self.ind]-win_grap],wave[sort_val[self.ind]+win_grap+1])
                y1, y2 = np.min(flux[sort_val[self.ind]-win_grap:sort_val[self.ind]+win_grap+1]),np.max(flux[sort_val[self.ind]-win_grap:sort_val[self.ind]+win_grap+1])
                ax.set_ylim(y1-(y2-y1)*0.1, y2+(y2-y1)*0.1)
                l7.set_xdata(grid[center]) ; l7.set_ydata(spectre[center]) 
                l3.set_xdata(grid[center]) ; l3.set_ydata(spectre[center]) 
                plt.draw()
                fig.canvas.draw_idle()
    
    callback = Index()
    axdelete = plt.axes([0.1, 0.05, 0.15, 0.075])
    bdelete = Button(axdelete, 'Delete all')
    bdelete.on_clicked(callback.rm_all)
    bdelete.on_clicked(callback.update)

    axnext = plt.axes([0.475, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next_line)

    axprev = plt.axes([0.35, 0.05, 0.1, 0.075])
    bprev = Button(axprev, 'Prev')
    bprev.on_clicked(callback.prev_line)
    
    axdel = plt.axes([0.60, 0.05, 0.1, 0.075])
    bdel = Button(axdel, 'Delete')
    bdel.on_clicked(callback.rm_one)
    bdel.on_clicked(callback.update)    
    
    axcancel = plt.axes([0.8, 0.05, 0.1, 0.075])
    bcancel = Button(axcancel, 'Cancel')
    bcancel.on_clicked(callback.backup)
    loop = ras.sphinx('Press Enter to move on the last step')
    wave = l1.get_xdata()
    flux = l1.get_ydata()
    index = callback.index_backup[-1]
    plt.close()
else:
    while count_out<count_out_lim:
        diff_deri = abs(np.diff(np.diff(flux)/np.diff(wave)))
        mask_out = diff_deri  > (np.percentile(diff_deri,99.5))
        mask_out = np.array([False]+mask_out.tolist()+[False])
        
        wave = wave[~mask_out]
        flux = flux[~mask_out]
        index = index[~mask_out]
        count_out += 1
        

if not only_print_end:
    print(' Outliers removing : DONE' )

loc_outliers_time = time.time()

if not only_print_end:
    print(' Time of the step : %.2f'%(loc_outliers_time-loc_cutting_time))

# =============================================================================
# EQUIDISTANT GRID FORMATION
# =============================================================================

wave_backup = wave.copy()
flux_backup = flux.copy()
index_backup = index.copy()
criterion = 1

for j in range(5):
    diff_x = np.log10(np.min(np.vstack([abs(np.diff(wave)[1:]),abs(np.diff(wave)[0:-1])]),axis=0))
    diff_diff_x = np.log10(abs(np.diff(np.diff(wave)))+1e-5)
    
    diff_x = np.hstack([0,diff_x,0])
    diff_diff_x = np.hstack([0,diff_diff_x,0])        
    if criterion==1:
        parameter = diff_x - diff_diff_x
    elif criterion==2:
        parameter = diff_x
    IQ = 2*(np.nanpercentile(parameter,50) - np.nanpercentile(parameter,25))
    mask_out = (parameter < (np.nanpercentile(parameter,50)-1.5*IQ))
    if not sum(mask_out):
        criterion+=1
        if criterion==2:
            continue
        elif criterion==3:
            break
    mask_out_idx = np.arange(len(parameter))[mask_out]
    if len(mask_out_idx)>1:
        cluster_idx = ras.clustering(mask_out_idx,3,1) 
        unique =  np.setdiff1d(mask_out_idx,np.hstack(cluster_idx))
        cluster_idx = list(cluster_idx)
        for j in unique:
            cluster_idx += [np.array([j])] 
        
        mask_out_idx = [] 
        for j in cluster_idx:
            j = np.array(j)
            which = np.argmin(flux[j.astype('int')])
            mask_out_idx.append(j[which])
    mask_out_idx = np.array(mask_out_idx)
    mask_out_idx = list(mask_out_idx[(mask_out_idx>3)&(mask_out_idx<(len(wave)-3))])
    mask_out_idx2 = []
    for j in mask_out_idx:
        sub_wave = wave[j-2:j+3]
        sub_diff_diff = []
        for k in [1,2,3]:
            sub_diff_diff.append(np.max(abs(np.diff(np.diff(np.delete(sub_wave,k))))))
        mask_out_idx2.append(j-1+np.argmin(sub_diff_diff))
    
    mask_final = np.ones(len(wave)).astype('bool')
    mask_final[mask_out_idx2] = False 
    
    wave = wave[mask_final]
    flux = flux[mask_final]
    index = index[mask_final]

if len(wave)!=len(wave_backup):
#        plt.plot(grid, spectre, zorder=0)
#        for j in np.setdiff1d(wave_backup,wave):
#            plt.axvline(x=j,color='k',ls='-')
#        Interpol = interp1d(wave, flux, kind = interpol, bounds_error = False, fill_value = 'extrapolate')
#        continuum = Interpol(grid)
#        continuum = troncated(continuum)
#        plt.plot(grid,continuum,ls=':',label='new continuum')
#        Interpol = interp1d(wave_backup, flux_backup, kind = interpol, bounds_error = False, fill_value = 'extrapolate')
#        continuum = Interpol(grid)
#        continuum = troncated(continuum)
#        plt.plot(grid,continuum,label = 'old continuum')
#        plt.legend()
#        plt.show()
#        answer = sphinx('Do you accept the following grid rearangement ? (y/n)',rep=['y','n'])
    answer='y'
    if answer == 'n':
        wave = wave_backup.copy()
        flux = flux_backup.copy()

if not only_print_end:
    print(' Number of points removed to build a more equidistant grid : %s'%(len(wave_backup)-len(wave)))


# =============================================================================
# MANUAL ADDING/SUPRESSING 
# =============================================================================

if feedback:
    fig, ax = plt.subplots(figsize=(15,6))
    plt.subplots_adjust(left=0.07,right=0.96,hspace=0,top=0.95)
    plt.plot(grid, spectre, alpha=0.6,label='spectrum')
    plt.scatter(locmaxx,locmaxy,color='b',s=5,zorder=3,label='local maxima')
    #plt.scatter(wave, flux, color = 'k', label='anchor_points (%s)'%(int(len(wave))))
    Interpol3 = interp1d(wave, flux, kind = interpol, bounds_error = False, fill_value = 'extrapolate')
    continuum3 = Interpol3(grid)
    continuum3 = ras.troncated(continuum3, spectre)
    plt.title('Manual maxima suppression/adjunction',fontsize=14)
    l1, = plt.plot(grid, continuum3,zorder=5, label='continuum')
    l2, = plt.plot(wave,flux,'ko',label='anchor points',zorder=4)
    plt.xlabel(r'Wavelength [$\AA$]',fontsize=14)
    plt.ylabel('Flux [arb. unit]',fontsize=14)
    plt.legend()
    plt.show(block=False)

    class Index():
    
        inix = locmaxx ; iniy = locmaxy ; iniz = locmaxz
        vecx = wave ; vecy = flux ; vecz = index
                    
        def update_data(self,newx,newy):
            
            dist1 = (newx-self.inix)**2+(newy-self.iniy)**2
            dist2 = (newx-self.vecx)**2+(newy-self.vecy)**2
            
            dist_min1 = [np.sort(dist1)[0], np.argsort(dist1)[0], np.sort(dist1)[1], np.argsort(dist1)[1]]
            dist_min2 = [np.min(dist2), np.argmin(dist2)]
            
            if dist_min1[2]<dist_min2[0]:
                self.vecx = np.append(self.vecx, self.inix[dist_min1[1]])
                self.vecy = np.append(self.vecy,  self.iniy[dist_min1[1]])
                self.vecz = np.append(self.vecz, self.iniz[dist_min1[1]])
            else:
                where = ras.find_nearest(self.vecx,self.vecx[dist_min2[1]])[0]
                self.vecx = np.delete(self.vecx,where)
                self.vecy = np.delete(self.vecy,where)
                self.vecz = np.delete(self.vecz,where)
                    
            self.vecy = self.vecy[np.argsort(self.vecx)]
            self.vecz = self.vecz[np.argsort(self.vecx)]
            self.vecx = np.sort(self.vecx)

            Interpol3 = interp1d(self.vecx, self.vecy, kind = interpol, bounds_error = False, fill_value = 'extrapolate')
            continuum3 = Interpol3(grid)
            continuum3 = ras.troncated(continuum3,spectre)
        
            l1.set_ydata(continuum3)
            l2.set_xdata(self.vecx)
            l2.set_ydata(self.vecy)
            plt.gcf().canvas.draw_idle()
    
    t = Index()
    
    def onclick(event):
        newx = event.xdata
        newy = event.ydata
        if event.dblclick:
            t.update_data(newx,newy)  
    
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    
    loop = ras.sphinx('Double click on : 1) blue points to add them 2) black point to remove them. Press Enter to finish the process.')
    plt.close()
    wave, flux, index = t.vecx, t.vecy, t.vecz

# =============================================================================
# WINDOWS BROADLINE MANUAL REJECTION
# =============================================================================

if len(mask_broadline):
    for w in mask_broadline:
        mask_broad = (wave>=w[0])&(wave<=w[1])
        wave = wave[~mask_broad]
        flux = flux[~mask_broad]
        index = index[~mask_broad]

# =============================================================================
# PHYSICAL MODEL FITTING (to develop)
# =============================================================================

flux_denoised = flux.copy()
for i,j in enumerate(index):
    if (i<count_cut)|((len(index)-i)<count_cut):
        pass
    else:
        new = np.mean(spectre[j-denoising_dist:j+denoising_dist+1])
        if abs(new - flux[i])/flux[i]<0.10:
            flux_denoised[i] = new

flux = flux*normalisation
flux_denoised = flux_denoised*normalisation

# =============================================================================
# FINAL PLOT 
# =============================================================================

end = time.time()

if not only_print_end:
    print('\n [END] RASSINE has finished to compute your continuum in %.2f seconds \n'%(end-begin))

jump_point = 1 # make lighter figure for article
if (feedback)|(plot_end)|(save_last_plot):
    fig = plt.figure(figsize=(16,6))
    plt.subplot(2,1,1)
    plt.plot(grid[::jump_point], spectrei[::jump_point], label = 'spectrum (SNR=%.0f)'%(int(SNR_0)),color='g')
    plt.plot(grid[::jump_point], spectre[::jump_point]*normalisation, label = 'spectrum reduced',color='b',alpha=0.3)
    plt.scatter(wave, flux, color = 'k', label='anchor points (%s)'%(int(len(wave))),zorder=100)

if interpol=='cubic':
    continuum1, continuum3, continuum1_denoised, continuum3_denoised = ras.make_continuum(wave, flux, flux_denoised, grid, spectrei, continuum_to_produce = [interpol, 'undenoised'])
    conti = continuum3
elif interpol=='linear':
    continuum1, continuum3, continuum1_denoised, continuum3_denoised = ras.make_continuum(wave, flux, flux_denoised, grid, spectrei, continuum_to_produce = [interpol, 'undenoised'])
    conti = continuum1

continuum1, continuum3, continuum1_denoised, continuum3_denoised = ras.make_continuum(wave, flux, flux_denoised, grid, spectrei, continuum_to_produce = [outputs_interpolation_saved, outputs_denoising_saved])

if (feedback)|(plot_end)|(save_last_plot):
    plt.plot(grid[::jump_point], conti[::jump_point], label='continuum',zorder=101,color='r')
    plt.xlabel('Wavelength',fontsize=14)
    plt.ylabel('Flux',fontsize=14)
    plt.legend()
    plt.title('Final products of RASSINE',fontsize=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    plt.tick_params(direction='in',top=True,which='both')
    plt.subplot(2,1,2,sharex=ax)
    plt.plot(grid[::jump_point], spectrei[::jump_point]/conti[::jump_point], color='k')
    plt.axhline(y=1,color='r',zorder=102)
    plt.xlabel(r'Wavelength [$\AA$]',fontsize=14)
    plt.ylabel('Flux normalised',fontsize=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    plt.tick_params(direction='in',top=True,which='both')
    plt.subplots_adjust(left=0.07,right=0.96,hspace=0,top=0.95)
    if save_last_plot:
        plt.savefig(output_dir+new_file+'_output.png')
    if (feedback):
        plt.show(block=False)
        loop = ras.sphinx('Press Enter to finish the execution and save the final product')
        plt.close()
    else:
        plt.close()

# =============================================================================
# SAVE OF THE PARAMETERS
# =============================================================================

if (hole_left is not None)&(hole_left!=-99.9):
    hole_left = ras.find_nearest(grid,ras.doppler_r(hole_left,-30)[0])[1][0]
if (hole_right is not None)&(hole_right!=-99.9):
    hole_right = ras.find_nearest(grid,ras.doppler_r(hole_right,30)[0])[1][0]

parameters = {'number_iteration':count_iter,
              'K_factors':k_factor,
              'axes_stretching':np.round(par_stretching,1),
              'vicinity_local_max':par_vicinity,
              'smoothing_box':smoothing_length,
              'smoothing_kernel':smoothing_shape,
              'fwhm_ccf':np.round(par_fwhm/conversion_fwhm_sig,2),
              'CCF_mask':CCF_mask,
              'RV_sys':RV_sys,
              'min_radius':np.round(R_old,1),
              'max_radius':np.round(par_Rmax,1),
              'model_penality_radius':par_model,
              'denoising_dist':denoising_dist,
              'number of cut':count_cut,
              'float_precision':float_precision,
              'windows_penality':windows,
              'large_window_penality':big_windows,
              'number_points':len(grid),
              'number_anchors':len(wave),
              'SNR_5500':int(SNR_0),
              'mjd':mjd,
              'jdb':jdb,
              'wave_min':minx,
              'wave_max':maxx,
              'dwave':dgrid,
              'hole_left':hole_left,
              'hole_right':hole_right,
              'RV_shift':RV_shift,
              'berv':berv,
              'lamp_offset':lamp_offset,
              'acc_sec':acc_sec,
              'light_file':light_version,
              'speedup':speedup,
              'continuum_interpolated_saved':outputs_interpolation_saved,
              'continuum_denoised_saved':outputs_denoising_saved,
              'nb_spectra_stacked':nb_spectra_stacked,
              'arcfiles':arcfiles} 

name_parameters = ['number_iteration','K_factors','par_stretching','par_vicinity',
                   'par_smoothing_box','par_smoothing_kernel','par_fwhm','CCF_mask',
                   'RV_sys','par_R','par_Rmax','par_reg_nu','denoising_dist','count_cut_lim',
                   'float_precision','windows_penality','large_window_penality','number of points',
                   'number of anchors', 'SNR_5500','mjd','jdb','wave_min','wave_max','dwave','hole_left','hole_right','RV_shift',
                   'berv','lamp_offset','acc_sec','light_file','speedup','continuum_interpolated_saved','continuum_denoised_saved','nb_spectra_stacked','arcfiles']


if not only_print_end:
    print('\n------TABLE------- \n')
    for i,j in zip(name_parameters,parameters.keys()):
        print(i+' : '+str(parameters[j]))
    print('\n----------------- \n')


if not Penalty:
    penalite_step = None
    penalite0 = None

#conversion in fmt format
    
if float_precision!='float64':
    grid = grid.astype(float_precision)
    spectrei = spectrei.astype(float_precision)
    flux = flux.astype(float_precision)
    wave = wave.astype(float_precision)
    flux_used = (spectre*normalisation).astype(float_precision)
    continuum3 = continuum3.astype(float_precision)
    continuum1 = continuum1.astype(float_precision)
    continuum3_denoised = continuum3_denoised.astype(float_precision)
    continuum1_denoised = continuum1_denoised.astype(float_precision)
    flux_denoised = flux_denoised.astype(float_precision)
index = index.astype('int')

# =============================================================================
# SAVE THE OUTPUT 
# =============================================================================
            
if (outputs_interpolation_saved=='linear')&(outputs_denoising_saved=='undenoised'):    
    basic = {'continuum_linear':continuum1,
             'anchor_wave':wave,
             'anchor_flux':flux,
             'anchor_index':index}
elif (outputs_interpolation_saved=='cubic')&(outputs_denoising_saved=='undenoised'):    
    basic = {'continuum_cubic':continuum3,
             'anchor_wave':wave,
             'anchor_flux':flux,
             'anchor_index':index}
elif (outputs_interpolation_saved=='linear')&(outputs_denoising_saved=='denoised'):    
    basic = {'continuum_linear':continuum1_denoised,
             'anchor_wave':wave,
             'anchor_flux':flux_denoised,
             'anchor_index':index}
elif (outputs_interpolation_saved=='cubic')&(outputs_denoising_saved=='denoised'):     
    basic = {'continuum_cubic':continuum3_denoised,
             'anchor_wave':wave,
             'anchor_flux':flux_denoised,
             'anchor_index':index}
elif (outputs_interpolation_saved=='all')&(outputs_denoising_saved=='denoised'):     
    basic = {'continuum_cubic':continuum3_denoised,
          'continuum_linear':continuum1_denoised,
          'anchor_wave':wave,
          'anchor_flux':flux_denoised,
          'anchor_index':index}
elif (outputs_interpolation_saved=='all')&(outputs_denoising_saved=='undenoised'):     
    basic = {'continuum_cubic':continuum3,
          'continuum_linear':continuum1,
          'anchor_wave':wave,
          'anchor_flux':flux,
          'anchor_index':index}
elif (outputs_interpolation_saved=='linear')&(outputs_denoising_saved=='all'):     
    basic = {'continuum_linear':continuum1,
          'continuum_linear_denoised':continuum1_denoised,
          'anchor_wave':wave,
          'anchor_flux':flux,
          'anchor_flux_denoised':flux_denoised,          
          'anchor_index':index}  
elif (outputs_interpolation_saved=='cubic')&(outputs_denoising_saved=='all'):     
    basic = {'continuum_cubic':continuum3,
          'continuum_cubic_denoised':continuum3_denoised,
          'anchor_wave':wave,
          'anchor_flux':flux,
          'anchor_flux_denoised':flux_denoised,          
          'anchor_index':index}      
else:
    basic = {'continuum_cubic':continuum3,
          'continuum_linear':continuum1,
          'continuum_cubic_denoised':continuum3_denoised,
          'continuum_linear_denoised':continuum1_denoised,
          'anchor_wave':wave,
          'anchor_flux':flux,
          'anchor_flux_denoised':flux_denoised,
          'anchor_index':index}


if light_version:
    output = {'wave':grid,'flux':spectrei, 'flux_used':flux_used, 'output':basic, 'parameters':parameters}    
else:
    output = {'wave':grid, 'flux':spectrei, 'flux_used':flux_used, 'output':basic,
          'penality_map':penalite_step, 'penality':penalite0, 'parameters':parameters}
    

output['parameters']['filename'] = 'RASSINE_'+new_file+'.p'

ras.save_pickle(output_dir+'RASSINE_'+new_file+'.p',output)
print('Ouput file saved under : %s (SNR at 5500 : %.0f)'%(output_dir+'RASSINE_'+new_file+'.p',output['parameters']['SNR_5500']))

if False:
    ras.make_sound('Congratulations ! Racine has finished your spectra normalisation. Time to make science.')
