# P300-feature extracction-classification

P300 evokes in your EEG signal when you watch a picture, primarily defined in your mind (target picture). If the target picture consists of characters, it can be used in a BCI speller system designed for writing with your mind.
How? The code detects P300 over time for each trial, and the corresponding set of characters is typed on the screen.
   

*P300 feature extraction and classification code*
* procedure to handle the code*

link for the database:
https://nbml.ir/FA/pages/NBML-Free-Databases data: iBCIC2021 First Step

Set @LogitBoost in the Matlab path

Run: Main_P300_featureextraction_classification.m file in the folder ...\P300_featureextraction_classification

First, the P300 pattern is highlighted in the EEG signal.
To do this, raw signals are denoised using transient suppression-based convex optimization and thresholding-based high amplitude artifact suppression.

Second, the feature file for each subject (time-electrode matrix of denoised EEG) is saved as an output ".mat" file (features_subi.mat).
It will be uploaded as input in the CNN classifier Python code.
The time-electrode-amplitude map for each stimulus image, set of characters, is plotted.
The target stimulus is visually detected among 9 colo-maps.


Third, the logit boost classifier is applied for automatic detection of the target stimulus.
