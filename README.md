# P300-feature extracction-classification

% P300 feature extraction and classification code
%%% Procedure to handle the code %%%

%1. Download data and put in the folder ...\P300_featureextraction_classification
%  link for the database: https://nbml.ir/FA/pages/NBML-Free-Databases data: iBCIC2021 
% data base description according to: A novel hybrid BCI speller based on RSVP and SSVEP paradigm

% 2. Set @LogitBoost in the Matlab path 
% 3. Run: Main_P300_featureextraction_classification.m file in the folder ...\P300_featureextraction_classification



%%% concepts %%%
% P300 evokes in your EEG signal around 300 ms after you watch 
% a picture, primarily defined in your mind (target picture).
% If the target picture consists of characters, it can be used in a BCI speller system designed for writing with your mind.
% How? The code detects P300 and the corresponding set of characters is typed on the screen.

% First, the P300 pattern is highlighted in the EEG signal. 
%To do this, raw signals are denoised using transient suppression-based convex optimization and thresholding-based high amplitude artifact suppression.

% Second, the feature file for each subject (time-electrode matrix of denoised EEG) is saved as an output ".mat" file (features_subi.mat). 
% It will be uploaded as input feature (time-electrode-amplitude colormap)
%in the CNN classifier code (written in python).
 
% Third, the logit boost classifier is applied.
