# Multimodal-dataset-for-Personality-Prediction
This file introduces the related source code of the project "Multimodal Dataset for Personality Prediction".

## 1. General Information
This folder contains the source code (Matlab and Python scripts) used for data preprocessing, feature extraction and personality trait prediction. Details for each script is as follows:
1. PreprocessEEG.m: Preprocess steps for raw EEG data, including channel selection (selected 18 channels from 24 channels) , re-reference operation, data filtering, and ICA etc.
2. Pip_ICARecon.m: Script for reconstruction of EEG data from ICA components obtained by last step.
3. Format_Data.m: The function of this script is to seperate the preprocessed  EEG data and original GSR and PPG data into individual trials. 
4. Format_video.py: This python script aims to divide the whole face video of a subject into sub video clips corresponding to each trial.
5. AlignAllData.py: This script align all signals (i.e., EEG, GSR, PPG and face video) of each trial with the end of the trial as reference point.
6. fea4eeg.py: This python script extracts differential entropy (DE) features from 5 frequency bands (i.e., \delta (1-3Hz), \theta (4-7Hz), \alpha (8-13Hz), \beta (14-30Hz) and \gamma (31-50Hz)) of each channel (totally 18 channels).
7. perifeaext.py: The function of this script is to extract both time domain and frequency domain features from GSR and PPG signals. Note that preprocessing steps (band pass filtering) are implemented for both GSR and PPG before feature extraction.
8. fea4video.py: This script implements features from video clips using pretrained VGG model, which is defined in models.py.
9. PrepareFeatures4Subj.m: This matlab script puts EEG features, GSR features, PPG features and VGG features together.
10. PrepareData4CLS_SubDep.m: The PrepareData4CLS_SubDep function helps to prepare the training set and test set for positive, negative and mixed emotion classification. The signal data of each trial is divided into two parts according to 4:1, and the first and second parts of all trials of a subject form the training and test set respectively. 
11. CLS.py: We implement personality prediction in this script. We test combinations of features from different modalities and four classifiers (MLP, kNN, SVM and RF). The experiment is carried out in a 5-fold cross validation across all subjects.
 

## Usage
1. The Matlab version is R2019b.
2. We use EEGLab Matlab toolbox for eeg signal processing, and the corresponding version is v2021.1.
3. Python 3.8 is used to run all python scripts, and to process face videos, opencv-python 4.5 is needed.
4. The scripts should to be run in the order in which they were introduced in section 'General Information'.
5. Note that when attempting to run the program, you need to modify the data path in the program according to the actual location of the data.
