# An efficient CNN based P300 Classification in an RSVP Experiment

## Transforming mind to text through reading and analyzing brain signals in an experiment in which the subject focuses on one of the 9 predefined 3-character images in each 10 seconds trial. 


**P300 feature extraction and classification code**

- Procedure to handle the code

1. Download data and put in the folder... \P300_featureextraction_classification <br> Link for the database: https://nbml.ir/FA/pages/NBML-Free-Databases data: iBCIC2021 <br> Database description according to: A novel hybrid BCI speller based on RSVP and SSVEP paradigm

2. Set @LogitBoost in the Matlab path

3. Run: Main_P300_featureextraction_classification.m file 
or you can run Test_pretrainedP300Classifier.m to see the results with the pretrained classifier (l_subi.mat) for each subject 

**Output**

The code calculates the classification accuracy and writes the trial, the target stimulus label and the estimated labels for each trial.

**Concepts**

- P300 is evoked as a positive peak on your EEG signal around 300 ms after you watch 
a picture, primarily defined in your mind (target picture).
- If the target picture consists of characters, it can be used in a BCI speller system designed for writing with your mind.


How? The code detects the presence of P300 on the EEG in each epoch of stimulus image, and the corresponding set of characters is typed on the screen at the end of a trial.

- First, the P300 pattern is highlighted in the EEG signal. 
- To do this, raw signals are denoised using transient suppression-based convex optimization and thresholding-based high amplitude artifact suppression.

- Second, the feature file for each subject (time-electrode matrix of denoised EEG during each epoch of stimulus ) is saved as an output ".mat" file (features_subi.mat). 
- It will be uploaded as input feature (time-electrode-amplitude colormap) in the CNN classifier code (written in python).
- Third, the logit boost classifier (gradient boosting; an ensemble of decision trees) is applied.
