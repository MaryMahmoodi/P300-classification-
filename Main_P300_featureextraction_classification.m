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
% How? The code detects the presence of P300 on the EEG in the duration of each stimulus image, and the corresponding set of characters is typed on the screen at the end of a trial.

% First, the P300 pattern is highlighted in the EEG signal. 
%To do this, raw signals are denoised using transient suppression-based convex optimization and thresholding-based high amplitude artifact suppression.

% Second, the feature file for each subject (time-electrode matrix of denoised EEG) is saved as an output ".mat" file (features_subi.mat). 
% It will be uploaded as input feature (time-electrode-amplitude colormap)
%in the CNN classifier code (written in python).
 
% Third, the logit boost classifier is applied.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% input parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
load index_similarstimulusCharacters
index_images=index_similarstimulusCharacters;

SubjectNumber=1;
training=1; % to use training (1) or test data (0)
repetition=5; % 5 or 3 if training=0, selecet two groups of test data with 5 or 3 repetitions

trainingLogitBoost=0;% for P300 classification

landau=70; %  transient suppression with Total variation denoising (TVD)
% % % if no TVD, landau=0;
Rereference=0;% Rereference all electrodes to the average signal of occipital electrodes



iter=20; % if low training accuracy, iter=50
if ~trainingLogitBoost
    l =LogitBoost(iter, 0.05, 1);%LogitBoost number of iterations=50,
    load (['l_sub',num2str(SubjectNumber),'.mat']);
end

data.TrainLabels=[];
data = load (['Subject',num2str(SubjectNumber),'_Data.mat']);
% data.TrainLabels=data.trainLabels;save (['Subject',num2str(SubjectNumber),'_Data.mat'], data);



fs=data.Fs;


% for power-feature extraction
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initial %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

% the Parks-McClellan method is used via the �remez� function of MATLAB
rp = 0.01; % Passband ripple
rs = 26; % Stopband ripple
f = freqrange; % Cutoff frequencies
a = [1 0]; % Desired amplitudes
% Compute deviations
dev = [(10^(rp/20)-1)/(10^(rp/20)+1) 10^(-rs/20)];
[n,fo,ao,w] = remezord(f,a,dev,fs);
B = remez(n,fo,ao,w);
A=1;
% freqz(B,A);
% Cz1=filter(B,A,Cz);

[B1, A1]=butter(6,f_delta/(fs/2),'bandpass');
[B2, A2]=butter(6,f_theta/(fs/2),'bandpass');
[B3, A3]=butter(6,f_alpha/(fs/2),'bandpass');
[B4, A4]=butter(6,f_sigma/(fs/2),'bandpass');
[B5, A5]=butter(6,f_beta/(fs/2),'bandpass');
[B6, A6]=butter(6,f_gama/(fs/2),'bandpass');

[ signal ] = Preprocessing_RSVPexperiment( signal,landau,myogenicrejection,thresh_EMG,blinkingrejection,A, B, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5, A6, B6 );

fprintf( ' Preprocessing: ')
toc,
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% P300-RSVP_feature extraction %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[signal, f ]=P300RSVP_featureextraction ( signal, data,training,repetition );

save ( ['features_sub',num2str(SubjectNumber)','.mat'],  'f')

% num_trial=num_epoch/signal.num_stimulus;
%num_epochintrial=num_epoch/num_trial;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% P300 classification with logistic regression (logit boost)
%%% Logistric regressor %%%
% A Boosting Approach to P300 Detection with Application to Brain-Computer Interfaces 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% normalize feature data for classification %%%
x=f.P300features;
y=f.P300labels;
x = reshape(x,size(x,1)*size(x,2),size(x,3));

for num_epoch=1:size(x,2)
    x(:,num_epoch)=x(:,num_epoch)./max(x(:,num_epoch));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_channels=size(x,1);

%%% prepare index sets for cross-validation %%%
n_permutations = 5;
n_epochs = size(x,2);%450
testsetsize = round(n_epochs / 10);
[trainsets, testsets] = crossValidation(1:n_epochs, testsetsize, ...
    n_permutations);
correct = [];figure(1);
tic,
for i = 1:n_permutations
    train_x = x(:, trainsets(i,:));
    if training
        train_y = y(:, trainsets(i,:));%y is logical output for x
        test_y  = y(:, testsets(i,:));
    end
    test_x  = x(:, testsets(i,:));
    % train classifier and apply to test data
    if trainingLogitBoost
        l =LogitBoost(iter, 0.05, 1);%LogitBoost number of iterations=iter,
        l = train(l, train_x, train_y, n_channels);
        save (['l_sub',num2str(SubjectNumber), '.mat'],'l');
    else
        load (['l_sub',num2str(SubjectNumber),'.mat']);%num2str(SubjectNumber)
    end
    p = classify(l, test_x);
    % evaluate classification accuracy
    i0 = find(p <= 0.5);
    i1 = find(p > 0.5);
    est_y = zeros(size(p));
    est_y(i0) = 0;
    est_y(i1) = 1;
    if training
        for j = 1:size(est_y,1)
            n_correct(j) = length(find(est_y(j,:) == test_y));% each electrode is concidered for all testing epochs to find which one has p300
        end
        p_correct = n_correct / size(est_y,2);% it's an array (not a number) containing number of corrects
        correct = [correct ; p_correct];
        % plot number of steps vs. classification accuracy
        if (i>1)
            hold on; plot(mean(correct));
            xlabel('number of boosting iterations');
            ylabel('classification accuracy');
            drawnow;
        end
    end
end
hold on
fprintf( ' P300 classifier: ')
toc,



if training
    counttrue=0; %number of true answers with at least 85% accuracy
    for i2=1:size(testsets,2) %number of test_input epochs or signals(each signal has21*45samples)
       
        p2=round(classify(l,x(:,testsets(i,i2)))); % answer to a specific data
        y2=length(find(p2==y(testsets(i,i2)) ));
      
        if y2>=.85*length(p2)% iter is number of iterationsor answers of logitboost for each input
            counttrue=counttrue+1;
        end
        
    end
    disp('number of true answers with atleast 85% accuracy=')
    counttrue
    Accuracy_test=counttrue/size(testsets,2)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calculate classifier detection accuracy and estimated characters for all trials
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% signal.estCharacters(tr,:)
% signal.mainCharacters(tr,:)
% Accuracy_p300;
[Accuracy_p300,signal]=calculate_accuracy_estCharacters(x,y, signal, l, training,index_images);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% P300 detection by morphology (time-electrode amplitude map, time_peak and ramp)
% show group average of signals for each of 9 stimulus images in each trial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% show group average of signals for each of 9 stimulus images in each trial%%%
elec=10:30;
% % % %   elec=num_SSVEPchannels;
spect=1;% to show all channels as an image if 0, signals are shown in each epoch length of 0.9 s
showplots=1;
trial_number=1;
training;
% load P300wave
show_timeelectrodemap(trial_number, signal, index_similarstimulusCharacters, training, elec, showplots, spect )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% show the signal preprocessing steps %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trial_number=4;
show_signalPreprocessings (trial_number, signal,data);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% save the output: P300 features used in our convolutional neural network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f=[];
f.P300features=signal.P300features;
f.P300labels=signal.P300labels;
f.fs=signal.fs;
f.downsamplingfactor=10;
f.P300freqrange=[0.5 4];%Hz

save ( ['features_sub',num2str(SubjectNumber)','.mat'],  'f')