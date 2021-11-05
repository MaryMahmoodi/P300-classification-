% P300 feature extraction and classification code
%%% Procedure to handle the code %%%

%1. Download data and put in the folder ...\P300_featureextraction_classification
%  link for the database: https://nbml.ir/FA/pages/NBML-Free-Databases data: EEG_32Channel_VisualOddball.rar
% data base description according to: A novel hybrid BCI speller based on RSVP and SSVEP paradigm

% 2. Set @LogitBoost in the Matlab path 
% 3. Run: Main_P300_featureextraction_classification.m file in the folder ...\P300_featureextraction_classification



%%% concepts %%%
% P300 evokes in your EEG signal when you watch 
% a picture, primarily defined in your mind (target picture).
% If the target picture consists of characters, it can be used in a BCI speller system designed for writing with our mind.
% How? The code detects P300 and the corresponding set of characters is called the target character.

% First, the P300 pattern is highlighted in the EEG signal. 
%To do this, raw signals are denoised using transient suppression-based convex optimization and thresholding-based high amplitude artifact suppression.

% Second, the feature file for each subject (time-electrode matrix of denoised EEG) is saved as an output ".mat" file (features_subi.mat). 
% It will be uploaded as input feature (time-electrode-amplitude colormap)
%in the CNN classifier code (written in python).
 
% Third, the logit boost classifier is applied.

%%
% input parameters
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

%% initial %%
fprintf ('initializing parameters...')
[signal, time]=initialize_signalattributes(data, training, repetition,f_ssvep , freqrange, Rereference );

%% fix damaged electrodes and Rereferencing
fprintf (' fix_damagedelectrodes...')
signal  = fix_damagedelectrodes( signal,Rereference );
%%  Preprocessing %%% 
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

% the Parks-McClellan method is used via the ‘remez’ function of MATLAB
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

%% P300-RSVP_feature extraction
[signal, f ]=P300RSVP_featureextraction ( signal, data,training,repetition );

save ( ['features_sub',num2str(SubjectNumber)','.mat'],  'f')

% num_trial=num_epoch/signal.num_stimulus;
%num_epochintrial=num_epoch/num_trial;

%% P300 classification with logistic regression (logit boost)
% %%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%% Logistric regressor %%%%%%
% A Boosting Approach to P300 Detection with Application to Brain-Computer Interfaces % %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

%%% normalize feature data for classification %%%
x=f.P300features;
y=f.P300labels;
x = reshape(x,size(x,1)*size(x,2),size(x,3));

for num_epoch=1:size(x,2)
    x(:,num_epoch)=x(:,num_epoch)./max(x(:,num_epoch));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%
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


%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%
%%**********calculate classifier detection accuracy for all epochs*******************
%%%%%%%%%%%%%%%%%%%%%%%%
counttrue3=0; est_y=[];
for num_epoch=1:size(x,2)
    p3=round(classify(l,x(:,num_epoch)) );
    if training
        y3=length(find(p3==y(num_epoch)));
    else
        y3=length(find(p3==1));
    end
    if y3>=.6*length(p3)% iter is number of iterations or answers of logitboost for each input
        counttrue3=counttrue3+1;
        if training
            est_y(1,num_epoch)=y(num_epoch);
        else
            est_y(1,num_epoch)=1;
        end
    else
        if training
            est_y(1,num_epoch)=abs(1-y(num_epoch));
        else
            est_y(1,num_epoch)=0;
        end
    end
end


% find character
counter2=1;
step2=size(index_images,1);est_label=[];label=[];
allCharacters=signal.Characters(1:size(index_images,1),:);
for tr=1: size(signal.data,3)
    if training
        label=y(1,counter2:counter2+step2-1);
    end
    est_label=est_y(1,counter2:counter2+step2-1);
    counter2=counter2+step2;
    num2=find(est_label==1);
    if ~isempty(num2); num2=num2(1);else num2=1; end
    signal.estnum_labels2(1,tr)=num2;
    signal.estCharacters(tr,:)=allCharacters(num2,:);
end

if training
    num_true=0;
    for tr=1:size(signal.estCharacters,1)
        signal.mainCharacters(tr,:)=allCharacters(signal.num_labels(1,tr),:);
        if strcmp(signal.mainCharacters(tr,:),signal.estCharacters(tr,:))
            num_true=num_true+1;
        end
    end
    Accuracy_p300 = num_true/tr;
end
%% P300 detection by morphology (time_peak and ramp)
% show group average of signals for each of 9 stimulus images in each trial%%%

signal.data_forP300classifier;%size:( num_channels, length_epoch, signal.num_stimulus*size(signal.data,3))
% load P300wave
num1=signal.num_stimulus/size(index_similarstimulusCharacters,1);
index_images=index_similarstimulusCharacters (:,1:num1);
stimulusCharacters=signal.Characters(1:size(index_similarstimulusCharacters,1),:);
fs=signal.fs;
counter=1;
step=signal.num_stimulus;
elec=1:size(signal.data,1)-1;

x=zeros(size(signal.data_forP300classifier,1), size(signal.data_forP300classifier,2),  size(index_images,1)*size(signal.data,3) );
counter2=1;
step2=size(index_images,1);
y=[]; %P300-label--> 0 or 1
% length(find(y)==1)

for tr=1: size(signal.data,3)
    if ~isempty(signal.num_labels); num_target= signal.num_labels(1,tr); else num_target=size(index_images,1)+1; end
    A=signal.data_forP300classifier(elec,:,counter:counter+step-1);
    %     data_forSSVEPclassifier  data_forP300classifier
    counter=counter+step;
    
    for ch=1:size(A,1) %synchronized averaging around similar stimulus images at each electrode
        y1=zeros(1,size(index_images,1));
        for im=1:size(index_images,1)
            number=index_images(im,:);
            A2=A(ch,:,number); A2=reshape(A2, size(A2,2),size(A2,3)); A2=A2';
            B(ch,1:size(A2,2),im)=mean(A2,1);
            if training
                if num_target==im; label(1,im)=1; else label(1,im)=0;end
            end
        end
    end
    if training
        y=[y label];
    end
    x (:,:,counter2:counter2+step2-1)= B;
    counter2=counter2+step2;
    
end

% x=x(:,:,:);
% y=y;
time=[1:size(x,2)]/fs;


% P300 detection by morphology (time_peak and ramp)
% show group average of signals for each of 9 stimulus images in each trial%%%


allCharacters=signal.Characters(1:size(index_images,1),:);
est_y=[];
duration=[1:round(0.9*fs)];%0.75
time=duration/fs;
x2=[];
if training
    labels_direction=signal.triallabel_direction;
    labels_direction=labels_direction-1;
end
elec=10:30;
% % % %   elec=num_SSVEPchannels;
spect=1;% to show all channels as an image
showplots=1;
for tr=1
    %:size(signal.data,3)  %bad spects: 3 5 7 17 19 21 24 27 29 38 40
    sig=[];spec=[];
    labels=zeros(1,size(index_images,1));
    ramps=zeros(1,size(index_images,1));
    AUC=zeros(1,size(index_images,1));
    Score=zeros(1,size(index_images,1));
    
    counter2=(tr-1)*size(index_images,1)+1;
    step2=size(index_images,1);
    if showplots
        figure(2); clf; hold on;
    end
    for im=1:size(index_images,1)
        W=msetcca(x(elec,duration,counter2),1);
        [wx1,wy1,r1]=cca(x(elec,duration,counter2),W'* x(elec,duration,counter2) );
        num1=find(r1==max(r1)); num1=num1(1);
        x2(:,duration,counter2)=abs(wx1)'*x(elec,duration,counter2);
        
        sig=mean(x(:,duration,counter2),1);
        
        spec=x(elec,duration,counter2); %x2(:,duration,counter2) or x(elec,duration,counter2)
        T1=[1:size(spec,2)]/fs; T1=T1';
        F=[1:size(spec,1)]; F=F';
        P=spec;
        
        max1=find(sig==max(sig)); max1=max1(end);
        min1=find(sig(1:max1)==min(sig(1:max1))); min1=min1(1);
        if isempty (min1); min1=1;end
        ramp1=sig(max1)-sig(min1);
        
        if (max1/fs)>=0.28 && (max1/fs)<=0.59  % (max1/fs)>=0.28 && (max1/fs)<=0.58 each P300 peaks in this duration after target stimulus onset
            ramps(1,im)=ramp1;
            AUC(1,im)=sum(sig(1,1:max1));
            
        else
            ramps(1,im)=0;
            AUC(1,im)=sum(sig(1,1:max1));
        end
        
        if showplots
            subplot(size(index_images,1)/3,3,im) ;
            
            if spect
                a=imagesc(T1,F,P); colormap(jet); colorbar off;set(gca,'YDir','normal');
                
                %  a=imshow( P(:,1:6:end)/max(max(P(:,1:6:end))),[] );colormap(jet); colorbar off;
                if training
                    %  ylabel (['electrodes','dir:' , num2str(labels_direction(tr)) ])
                    ylabel (['electrodes' ])
                    
                end
            else
                plot(time, x(:,duration,counter2),'b');
                hold on;  plot(time, sig,'m'); hold on;
                
            end
            if training
                if y(counter2)==1;
                    title(['target ' , 'peak: ', num2str(max1/fs)  ,' ramp:', num2str(ramp1),signal.Characters(im,:)]);
                    title(['target character' ,signal.Characters(im,:) ]);
                    
                else
                    %             title([ 'peak: ', num2str(max1/fs) ,' ramp:', num2str(ramp1),signal.Characters(im,:) ]);
                    title([ signal.Characters(im,:) ]);
                    
                end
                hold on;
            else
                %             title([ 'peak: ', num2str(max1/fs) ,' ramp:', num2str(ramp1), signal.Characters(im,:) ]);
                title([ signal.Characters(im,:) ]);
                
            end
        end
        counter2=counter2+1;
        
    end
    
    num= find (ramps==max(ramps)); num=num(1);
    % if length(num)>1
    % num= find (AUC==max(AUC)); num=num(1);
    % end
    if training
        labels(1,num)=1;
        est_y=[est_y labels];
        signal.mainCharacters(tr,:)=allCharacters(signal.num_labels(1,tr),:);
    end
    
    % signal.estCharacters(tr,:)=char(allCharacters(num,:));
    
    
end
hold on; xlabel('time (s)')




%% show results
showplots=1;

figure(5); clf;
tr=4;
fs=signal.fs;
time=[1:size(signal.data,2)]./fs;
orgsig=data.TrainData([signal.num_Cz] ,:,tr );
sig_denoised=signal.data([signal.num_Cz] ,:,tr );
mrk=signal.markersignal(:,tr);
sig_LPF=signal.dataLPF([signal.num_Cz] ,:,tr );
plot(time,orgsig,'b');hold on; plot(time,5*mrk,'r');text (1,7,'Raw signal and markers of target stimulus (P300 duration)')
hold on; plot(time, sig_denoised-20,'b'); text (1, -20,'denoised signal');
hold on; plot(time, sig_LPF-40,'b'); ylim([-50 40]); text(1,-40, 'lowpass filtered signal');
xlabel ('time (s)'); ylabel('Amplitude (\muV)');

sig=[];
num_tr= size(signal.data,3);%num_trial/10;
if num_tr>10; num_tr=round(num_tr/4);end
spect=0;

fs=signal.fs;

time=[1:size(signal.data,2)]./fs;

%% output: P300 features used in our convolutional neural network
f=[];
f.P300features=signal.P300features;
f.P300labels=signal.P300labels;
f.fs=signal.fs;
f.downsamplingfactor=10;
f.P300freqrange=[0.5 4];%Hz

save ( ['features_sub',num2str(SubjectNumber)','.mat'],  'f')



