————————
IMPORTANT INFORMATIONS :
————————

Please cite Cretignier et al., 2020 paper (see the paper in the directory for a better understanding of the code).
ADS Link : https://ui.adsabs.harvard.edu/abs/2019arXiv191205192C/abstract

RASSINE is a free access (https://github.com/MichaelCretignier/Rassine_public) Python3 code, compatible with Python2, which is a simple tool to normalise spectra with the less "fine-tuning buttons" possible.



By default, the input file has to be either a pickle or csv dictionary with at minimum two columns containing the key words 'wave' and 'flux'.
RASSINE can also read fits file by preprocessing the file with the  « preprocess_fits » function.
If you are a beginner user, let all the parameters to their default automatic value until you are sufficiently familiar with them (automatic mode often provide good enough result without any fine-tuning).

The code work almost exclusively with pickle files. By default, the protocol version is automatically determined based on the python version (sys.version[0]).
If you want to change that change the : << protocol_pickle = ‘auto’ >> code line in the Rassine_functions.py file.

You can change the parameters value in the Rassine_config.py file. Running this python file will produce a config pickle dictionary that will be read by RASSINE.
The user do not need to compile it by himself/herself with Python since it will be automatically done by the main code Rassine.py

RASSINE can be run in multi-processed using the Rassine_multiprocessed.py code.
Finally, RASSINE contains a python code Rassine_trigger.py which is automatically launching all the different steps if you are working with a spectra time-series.

————————
BASIC RULES :
————————

If :
1) Individual spectra of different stars : Adapt the Rassine_config.py parameters —> launch Rassine.py in Ipython shell
2) Spectra time-series : Adapt the Rassine_trigger.py parameters —> launch Rassine_trigger.py in Ipython shell

The trigger will make different steps that you can control with the boolean buttons in the Rassine_trigger.py file :

1) Preprocessing
2) Matching spectra (produce a common wavelength grid)
3) Stacking spectra
4) Launch RASSINE in feedback mode on the master spectrum created in 3)
5) Launch RASSINE without feedback in multiprocessed on the stacked spectra created in 3) (the master output file created in 4) is used as anchor file)
6) Launch the intersect_all_continuum on spectra created in 5)
7) Launch the matching_all_continuum on spectra created in 6)

————————
MORE INFORMATIONS :
————————

You can also run the main Rassine.py code in sys mode, in this case the options are :

-i : the spectrum full path name
-o : the output directory
-f : the flux keyword of your dictionary
-w : the wave keyword of your dictionary
-l : the optional anchor file
-r : the minimum radius (parameter 5)
-R : the maximum radius (parameter 6)
-p : the par_stretching parameter (parameter 1)
-a : activate or suppress graphical feedback
-P : to only print when RASSINE finished the normalisation

Sys options are dominant in comparison to the values written in the Rassine_config.py file.
You can provide a RASSINE output file to normalise another spectra from the same star using the 'anchor_file' parameter.
The anchor file will acts as as a new config file for the parameters. 
Anchor file is dominant compared to sys mode.

In summary :

Anchor file > Sys mode > Config file

RASSINE contains 6 parameters but only 3 are really relevant and don't need to be fine-tuned (Parameter 1,  Parameter 3,  Parameter 6)
The code can be run in interactive mode (feedback = True in config file). If so, it will display the subproducts at each step, before the Sphinx ask your feedback in the terminal.
By interacting with the Sphinx you can control the process until the wanted product is reached.

Most parameters can be replaced by ‘auto’, if so always keep an eye on the feedback AUTO INFO if you are using the automatic mode.
Keep in mind that no guaranty is given about the final product.

————————
REMARKS
————————

Remark :
    1) you don't need to change any parameters in the code except the ones in the Rassine_config.py or Rassine_trigger.py file
    2) depending on the radius, the rolling pin can go THROUGH the spectrum if it is the case, either increase the radius parameter par_R or par_stretching parameter (Parameter 1)
    3) the K factors to increase the radius when interacting with the Sphinx has to be between 1 and 10
    4) you can enter 'auto' for some parameters and RASSINE will try to find good value for the parameters (no guaranty to work)
    5) Rmax has to be smaller than 150 angstrom

The output are given by the dictionary 'output','matching_anchors' or 'matching_diff' depending on your reduction progress.
The parameters are saved in the dictionary 'parameters'.
