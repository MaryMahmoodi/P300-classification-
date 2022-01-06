% Test a pretrained P300 classifier
% you will see the real and estimated characters for each trial
clear
clc
%%%%%%%%%%%%%%%%%%%%%%%%
%% input parameters
%%%%%%%%%%%%%%%%%%%%%%%%
SubjectNumber=1;

load index_similarstimulusCharacters % order of repetition for each of the 9 stimulus images
index_images=index_similarstimulusCharacters;

trainingLogitBoost=0;% for P300 classification
iter=20; % if low training accuracy, iter=50
if ~trainingLogitBoost
    l =LogitBoost(iter, 0.05, 1);%LogitBoost number of iterations=50,
    load (['l_sub',num2str(SubjectNumber),'.mat']);
end

repetition=5; % 5 or 3 if training=0, selecet two groups of test data with 5 or 3 repetitions
training=1; % to use training (1) or test data (0)

data.TrainLabels=[];

landau=70; %  transient suppression with Total variation denoising (TVD)
% % % if no TVD, landau=0;
Rereference=0;% Rereference all electrodes to the average signal of occipital electrodes
f_delta=[0.5 4]; f_theta=[4 7]; f_alpha=[8 12];
f_sigma=[14 16] ;f_beta=[29 31]; f_gama=[44 47];
f_ssvep=15;%Hz
maxfreq=47;
minfreq=0.5;
freqrange=[minfreq,maxfreq];
showplots=1;%show figures
myogenicrejection=1;
thresh_EMG=20;
blinkingrejection=0;% detection and replacement approach  if blinking_rejection=1;thresh_EMG=80;

%%%%%%%%%%%%%%%%%%%%%%%%
%% initial %%
%%%%%%%%%%%%%%%%%%%%%%%%
if exist(['Subject',num2str(SubjectNumber),'_Data.mat'])
    data = load (['Subject',num2str(SubjectNumber),'_Data.mat']);
    % data.TrainLabels=data.trainLabels;save (['Subject',num2str(SubjectNumber),'_Data.mat'], data);
    fs=data.Fs;
    
    
    fprintf ('initializing parameters...')
    [signal, time]=initialize_signalattributes(data, training, repetition,f_ssvep , freqrange, Rereference );
    
    %% fix damaged electrodes and Rereferencing
    fprintf (' fix_damagedelectrodes...')
    signal  = fix_damagedelectrodes( signal,Rereference );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%  Preprocessing %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1.baseline correction
    % 2. Total variation denoising
    % 3.BP filter in the freqrange
    % 4. Myogenic artefact rejection
    % 5. Blinking rejection
    % 6. 4Hz lowpass filter
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % d1= fdesign.bandpass('N,Fst1,Fp1,Fp2,Fst2,C',50,1,1.1,30,maxfreq+0.5,fs);%50,0.01,0.16,45,50.5,fs);%36
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     Hd1=design(d1,'equiripple');
    
    tic,
    %%% initialize BP filter parameters %%%
    [filterparams]=initialize_filterparams (fs,freqrange,f_delta,f_theta,f_alpha,f_sigma,f_beta,f_gama);
    
    [ signal ] = Preprocessing_RSVPexperiment( signal,landau,myogenicrejection,thresh_EMG,blinkingrejection,filterparams);
    
    fprintf( ' Preprocessing: ')
    toc,
%%%%%%%%%%%%%%%%%%%%%%%
%% P300-RSVP_feature extraction %%
%%%%%%%%%%%%%%%%%%%%%%%
    
    [signal, f ]=P300RSVP_featureextraction ( signal, data,training,repetition );
    f. Characters=signal.Characters;
    f.num_trial=signal.num_trial;
    f.num_labels=signal.num_labels;
    save ( ['features_sub',num2str(SubjectNumber)','.mat'],  'f')
else
    load (['features_sub',num2str(SubjectNumber)','.mat']);
end
%% normalize feature data for classification %%%
x=f.P300features;
y=f.P300labels;
x = reshape(x,size(x,1)*size(x,2),size(x,3));

for num_epoch=1:size(x,2)
    x(:,num_epoch)=x(:,num_epoch)./max(x(:,num_epoch));
end
%% classify
% signal.estCharacters(tr,:)
% signal.mainCharacters(tr,:)
% Accuracy_p300;
[Accuracy_p300,signal]=calculate_accuracy_estCharacters(x,y, f, l, training,index_images);
