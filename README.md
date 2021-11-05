# P300-feature extracction-classification

P300 evokes in your EEG signal when you watch a picture that has been predefined in your mind (target picture). If the target picture consists of characters, it can be used in a BCI speller system designed for writing with our mind.
How? The code detects P300 and the corresponding set of characters is called the target character.
   
*P300 feature extraction and classification code*
* procedure to handle the code*

link for the database:
https://nbml.ir/FA/pages/NBML-Free-Databases data: EEG_32Channel_VisualOddball.rar

Set @LogitBoost in the Matlab path
Run: Main_P300_featureextraction_classification.m file in the folder ...\P300_featureextraction_classification

First, the P300 pattern is highlighted in the EEG signal.
To do this, raw signals are denoised using transient suppression-based convex optimization and thresholding-based high amplitude artifact suppression.

Second, the feature file for each subject (time-electrode matrix of denoised EEG) is saved as an output ".mat" file (features_subi.mat).
It will be uploaded as input in the CNN classifier code.

Third, the logit boost classifier is applied.
