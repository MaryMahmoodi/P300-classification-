clear
load index_similarstimulusCharacters
load headmodel;
load ('scalpmesh');
load elec_realigned_67;

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

signal=[];

if training
signal.data=data.TrainData;

else

if repetition==5
signal.data=data.TestData_five_repetition;
end

if repetition==3
signal.data=data.TestData_three_repetition;
end


end



%%% initial input
% % % % % % % % % % % % % % % % % % % wavelet_level=6; % 0.5- 7 Hz delta-theta wave
% for power feature extraction
f_delta=[0.5 4]; f_theta=[4 7]; f_alpha=[8 12]; 
f_sigma=[14 16] ;f_beta=[29 31]; f_gama=[44 47]; 
%f_sigma=[14 16] ;f_beta=[29 31]; f_gama=[44 47]; 


signal.num_p300epoch=4; % each epoch length=signal.sample_stimulus

f_ssvep=15;%Hz
signal.f_ssvep=f_ssvep;
maxfreq=47;
minfreq=0.5;
freqrange=[minfreq,maxfreq];
signal.freqrange=freqrange;

showplots=1;%show figures


myogenicrejection=1;
thresh_EMG=20; % 20 %45;%80

blinkingrejection=0;% detection and replacement approach  if blinking_rejection=1;thresh_EMG=80;





signal.Rereference=Rereference;


%% %%%%%%%%%%%
fs=data.Fs;
[m,n]=size(signal.data(end,:,1));
sample0=length(find( signal.data(end,:,1)==zeros(m,n) ) );
sample01=length(find( signal.data(end,1: round(10*fs),1)==zeros(m,round(10*fs)) ) );
signal.sample01=sample01;
signal.sample0=sample0;
max_amp=max(signal.data(end,:,1));
num_stimulus=max_amp;% number of stimulus images ( with three characters)
signal.num_stimulus=num_stimulus;
time_stimulus=(length(signal.data(end,:,1))-sample0)/(num_stimulus*fs); % seconds (s)
sample_stimulus=round((length(signal.data(end,:,1))-sample0)/(num_stimulus)); % duration of showing each stimulus image
signal.sample_stimulus=sample_stimulus;   


signal.fs=fs;

% % % signal.data=data.TrainData;
signal.Characters=data.Characters;
Channels=data.Channels;
% signal.data(1:end-1,:,:);

signal.Channels=data.Channels;
signal.num_Cz=find(strcmp(lower(signal.Channels), lower('Cz')));
signal.num_Oz=find(strcmp(lower(signal.Channels), lower('Oz')));

signal.freqrange=freqrange;
num_trial=size(signal.data,3);


time_ssvepstimulus=1/f_ssvep; %s
sample_ssvepstimulus=fix(time_stimulus/time_ssvepstimulus); %s

if ~isempty (data.TrainLabels)

signal.labels=data.TrainLabels;
[ signal]=findlabels(signal,num_stimulus);

for i=1:size(signal.num_labels,1)
    for j=1:size(signal.num_labels,2)
        num=signal.num_labels(i,j);
signal.startsample_labels(i,j)=sample01+(num-1)*sample_stimulus;
    end
end

end

time=[1:size(signal.data,2)]./fs;
signal.time=time;


%% preprocess ; Rereferencing, BP filter; myogenic artefact rejection, blinking rejection
tic,
% Replace parietooccipital channels with the average 
% of their neigboring electrodes
signal.num_O1=find(strcmp(lower(signal.Channels), lower('O1')));
signal.num_O2=find(strcmp(lower(signal.Channels), lower('O2')));
signal.num_PO3=find(strcmp(lower(signal.Channels), lower('PO3')));
signal.num_PO4=find(strcmp(lower(signal.Channels), lower('PO4')));
signal.num_PO7=find(strcmp(lower(signal.Channels), lower('PO7')));
signal.num_PO8=find(strcmp(lower(signal.Channels), lower('PO8')));
signal.num_Pz=find(strcmp(lower(signal.Channels), lower('Pz')));
signal.num_POz=find(strcmp(lower(signal.Channels), lower('POz')));
signal.num_Oz=find(strcmp(lower(signal.Channels), lower('Oz')));
signal.num_P3=find(strcmp(lower(signal.Channels), lower('CP1')));
signal.num_P4=find(strcmp(lower(signal.Channels), lower('CP2')));
signal.num_P7=find(strcmp(lower(signal.Channels), lower('CP5')));
signal.num_P8=find(strcmp(lower(signal.Channels), lower('CP6')));

num_SSVEPchannels=[signal.num_O1 signal.num_PO3 signal.num_PO7 signal.num_P7 signal.num_P3 signal.num_Pz   signal.num_P4  signal.num_P8   signal.num_PO4 signal.num_PO8 signal.num_O2 signal.num_Oz];
if length(num_SSVEPchannels)>9; num_SSVEPchannels=[signal.num_O1 signal.num_PO3 signal.num_PO7 signal.num_P3 signal.num_Pz   signal.num_P4  signal.num_P8   signal.num_PO4 signal.num_PO8 signal.num_O2 signal.num_Oz];
end
num_parelec=[signal.num_Oz signal.num_O1 signal.num_O2 signal.num_PO3 signal.num_PO4 signal.num_PO7 signal.num_PO8];
avgsignal=zeros(1,size(signal.data,2),size(signal.data,3));

% average signal of parieto-occipital electrodes
for tr=1:size(signal.data,3)
    for i=1:length(num_parelec)
    avgsignal(1,:,tr)=avgsignal(1,:,tr)+signal.data(num_parelec(i),:,tr);
    end
    avgsignal(1,:,tr)=avgsignal(1,:,tr)/length(num_parelec);
end

% Replace signal of damaged occipital electrodes by
% average signal of their neigbor electrodes
for tr=1:size(signal.data,3)
    for i=1:size(signal.data,1)
        if max(signal.data (i,:,tr))<1 || std (signal.data (i,:,tr))<1 %damaged electrode
        signal.data (i,:,tr)=avgsignal(1,:,tr);
        else
            % Rereference all other electrodes to the average signal of occipital electrodes  
            if Rereference
                    signal.data (i,:,tr)=signal.data (i,:,tr)-avgsignal(1,:,tr);
            end
        end
        
    end
end


%BP parameters
% d1= fdesign.bandpass('N,Fst1,Fp1,Fp2,Fst2,C',50,1,1.1,30,maxfreq+0.5,fs);%50,0.01,0.16,45,50.5,fs);%36
%     Hd1=design(d1,'equiripple');

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
    % the Parks-McClellan method is used via the ‘remez’ function of MATLAB
rp = 0.01; % Passband ripple
rs = 26; % Stopband ripple
a = [1 0]; % Desired amplitudes
% Compute deviations
dev = [(10^(rp/20)-1)/(10^(rp/20)+1) 10^(-rs/20)];
A=1;

% [n,fo,ao,w] = remezord(f_delta,a,dev,fs); B1 = remez(n,fo,ao,w);
% 
% [n,fo,ao,w] = remezord(f_theta,a,dev,fs);B2 = remez(n,fo,ao,w);
% 
% [n,fo,ao,w] = remezord(f_alpha,a,dev,fs);B3= remez(n,fo,ao,w);
% 
% [n,fo,ao,w] = remezord(f_sigma,a,dev,fs);B4 = remez(n,fo,ao,w);
% 
% [n,fo,ao,w] = remezord(f_beta,a,dev,fs);B5 = remez(n,fo,ao,w);

[B1, A1]=butter(6,f_delta/(fs/2),'bandpass');
[B2, A2]=butter(6,f_theta/(fs/2),'bandpass');
[B3, A3]=butter(6,f_alpha/(fs/2),'bandpass');
[B4, A4]=butter(6,f_sigma/(fs/2),'bandpass');
[B5, A5]=butter(6,f_beta/(fs/2),'bandpass');
[B6, A6]=butter(6,f_gama/(fs/2),'bandpass');


for tr=1:num_trial    
for i=1:size(signal.data,1)-1
    
sig=signal.data(i,:,tr );

%%%baseline correction%%%
sig=sig-mean(sig)*ones(size(sig));

if ~ exist('tvd.mexw64')
    mex tvd.c
end

sig = tvd(sig,length(sig),landau);

% sig=filter(Hd1,sig);
sig=filtfilt(B,A,sig);

%%%myogenic artefact rejection%%%
if myogenicrejection
step=2;%seconds
sig=myogenic_rejection(sig, fs, step,thresh_EMG);
th1=3.3*(1/length(sig))*sum(abs(sig));
sig=myogenic_rejection(sig,fs,0.25,th1);
end

sig2=sig;

if blinkingrejection
    spikes_index=[];spikes=[];
    [ sig, spikes,spikes_index ]= blinking_rejection( sig,fs );
    [ sig2, spikes,spikes_index ]= blinking_rejection( sig2,fs );

end



% lowpass filter
% figure(1);plot(time, sig); 
 [a2,b2]=butter(3,4/(fs/2),'low');
A5=filtfilt(a2,b2,sig);


% hold on; plot(time,A5,'k');
signal.dataDWT(i,:,tr )=A5;
signal.data(i,:,tr )=sig2;
% % % signal.dataDWT(i,:,tr )=sig2;











[delta,theta,alpha,sigma,beta,gama]=filter_freqbands (sig2,fs, B1, B2, B3, B4,B5,B6,A1,A2,A3,A4,A5,A6 );
% sig2 is not filtered by TVD.
% sig is filtered by TVD.

signal.delta(i,:,tr )=delta;
signal.theta(i,:,tr )=theta;
signal.alpha(i,:,tr )=alpha;
signal.sigma(i,:,tr )=sigma;
signal.beta(i,:,tr )=beta;
signal.gama(i,:,tr)=gama;

% [T1, F1, P1]=mainSpectrogram(sig,maxfreq,1,freqrange);
% signal.time=T1;  
% signal.frequency=F1;
% signal.spectrogram{i,tr}=P1;

end
end



fprintf( ' Preprocessing: ')
toc,

%% P300-SSVEP feature extraction for classifiers

    data_forP300classifier=[];
    data_forSSVEPclassifier=[];
    signal.epochlabel_P300=zeros(signal.num_stimulus,size(signal.data,3));
    signal.triallabel_direction=[];
    signal.num_stimulus;
    tr=size(signal.data,3) ;
    signal.detectedmarkersignal=zeros(size(signal.data,2),tr);
    signal.markersignal=zeros(size(signal.data,2),tr);
    signal.labels=signal.labels;
    epochnumber=0;    
    minus=2;
    extra=2;
%                 [a2,b2]=butter(6,[4, f_ssvep+extra]/(fs/2),'bandpass'); % for visual inspection of SSVEP whole trial around stimulus frequency
                  
%                 [a3,b3]=butter(6,[14, f_ssvep+2]/(fs/2),'bandpass'); % for SSVEP visual inspection and calculations in each image (epoch) of each trial
% [a3,b3]=butter(6,[f_ssvep-1, f_ssvep+1]/(fs/2),'bandpass');

Channels=signal.Channels;
%update electrodes according to headmesh electrodes labels
counter=1;num2=[];
for k=1:length(elec_realigned.label) 
    num1=find(strcmp(lower(Channels),lower(elec_realigned.label(k))));
    if ~isempty (num1)
        num2(counter)=num1;
        Channels2{counter,1}=Channels{num1};
        chanpos2(counter,:)=elec_realigned.chanpos(k,:);
        counter=counter+1;
    end
end
signal.data2=signal.data(num2,:,:);
% find zero channels and interpolate with neighboring electrodes
for tr=1:size(signal.data2 ,3)
for i=1:size(signal.data2  ,1)
%   figure(2);subplot(7,4,i);hold on;
sig=signal.data2  (i,:,tr);
if max(sig)<1 || std(sig)<1
%   plot(sig);title((f.Channels2(i)))
[ mindistance1,index1,distances_ascend, indexes_ascend] = min_distance(chanpos2, chanpos2(i,:) );
signal.data2  (i,:,tr)=sum(signal.data2  (indexes_ascend(1:4),:,tr))./4;
end
end
end

signal.data_forSSVEPclassifier_2=[];epochnumber=1;
for tr=1:size(signal.data2,3) % trial
        starttrial=signal.sample01+(1-1)*signal.sample_stimulus;
        endtrial=starttrial+signal.num_stimulus*signal.sample_stimulus-1;
        signal.data_forSSVEPclassifier1_2(1:size(signal.data2,1),:,tr)=filter (1,1,signal.data2(1:size(signal.data2,1),starttrial:endtrial,tr));% whole trial  filter (a2,b2,signal.data(1:size(signal.data,1)-1,starttrial:endtrial,tr));% whole trial
for num=1:signal.num_stimulus
            start1=signal.sample01+(num-1)*signal.sample_stimulus;
            end1=start1+ signal.num_p300epoch*signal.sample_stimulus-1;
            epochnumber=epochnumber+1;
    signal.data_forSSVEPclassifier_2(1:size(signal.data2,1),:,epochnumber)=filter (1,1,signal.data2(1:size(signal.data2,1),start1:end1,tr));

end
end



    for tr=1:size(signal.data,3) % trial
        starttrial=signal.sample01+(1-1)*signal.sample_stimulus;
        endtrial=starttrial+signal.num_stimulus*signal.sample_stimulus-1;

        data_forSSVEPclassifier1(1:size(signal.data,1)-1,:,tr)=filter (1,1,signal.data(1:size(signal.data,1)-1,starttrial:endtrial,tr));% whole trial  filter (a2,b2,signal.data(1:size(signal.data,1)-1,starttrial:endtrial,tr));% whole trial
           
           
        for ch=1:size(signal.data,1)-1
            powerfeaturedata(ch,:,tr)=[norm(signal.delta(ch,starttrial:endtrial,tr),'fro')  norm(signal.theta(ch,starttrial:endtrial,tr),'fro')   norm(signal.alpha(ch,starttrial:endtrial,tr),'fro')  norm(signal.sigma(ch,starttrial:endtrial,tr),'fro')  norm(signal.beta(ch,starttrial:endtrial,tr),'fro')    norm(signal.gama(ch,starttrial:endtrial,tr),'fro')];
            powerfeaturedata(ch,:,tr)=powerfeaturedata(ch,:,tr)/norm(powerfeaturedata(ch,:,tr));
            end
        for num=1:signal.num_stimulus
            start1=signal.sample01+(num-1)*signal.sample_stimulus;
            end1=start1+ signal.num_p300epoch*signal.sample_stimulus-1;
            epochnumber=epochnumber+1;
            data_forP300classifier(1:size(signal.data,1)-1,:,epochnumber) = signal.dataDWT(1:size(signal.data,1)-1,start1:end1,tr);
            %%%%%%%%%%%%%%%%          
                data_forSSVEPclassifier(1:size(signal.data,1)-1,:,epochnumber)=filter (1,1,signal.data(1:size(signal.data,1)-1,start1:end1,tr));
                start11=start1; end11=end1;

                
            
            if ~isempty (data.TrainLabels)
                k=[]; k=find (num==signal.num_labels(:,tr));
                if ~isempty (k)
                    
%                     signal.epochlabel_P300(num,tr)=1;
                    % as we consider 1s epoch length (4*signal.sample_stimulus-1)
%                     instead of the length of stimulus (signal.sample_stimulus (0.234s)) we
%                    stick label 1 to more than one small epoch
if signal.num_p300epoch==3; step1=2;end
if signal.num_p300epoch==2; step1=1;end
if signal.num_p300epoch==4; step1=3;end

if num<=signal.num_p300epoch-step1
% if num>=signal.num_stimulus
    signal.epochlabel_P300(1:num,tr)=1;
% else
%         signal.epochlabel_P300(1:num+1,tr)=1;
% end
else
    
%     if num>=signal.num_stimulus
        signal.epochlabel_P300(num-(signal.num_p300epoch-step1):num,tr)=1;
        % else
        %         signal.epochlabel_P300(num-3:num+1,tr)=1;
%     end
    
end
                    signal.markersignal(start1:end1,tr)=1; % binary p300 marker 1 from the start of target stimulus               
                    a=signal.direction_labels(1,tr);
                    
                    if strcmp(lower(a),lower('l')); a=1;end
                    if strcmp(lower(a),lower('u')); a=2;end
                    if strcmp(lower(a),lower('r')); a=3;end
                    signal.triallabel_direction(1,tr)=a;% character direction 1 2 3 ( corresponding left, up, right)
                    
                                end
                
            end
        end
    end
    
signal.data_forP300classifier=data_forP300classifier;
signal.powerfeaturedata=powerfeaturedata;
signal.epochlabel_P300=reshape(signal.epochlabel_P300, 1, size(signal.epochlabel_P300,1)*size(signal.epochlabel_P300,2));
signal.data_forSSVEPclassifier=data_forSSVEPclassifier;

signal.data_forSSVEPclassifier1=data_forSSVEPclassifier1;
signal.triallabel_direction;
% % % % % % % % % % This part has been calculated in last loop for each epoch 
% % % % % % % % % % power and SEF feature extraction for each stimulus 3-character image duration (sample_stimulus)
% % % % % % % % % % [signal]=power_SEF_featureextraction (signal,fs);

% save signal_sub2 signal








signal.data_forP300classifier;%size:( num_channels, length_epoch, signal.num_stimulus*size(signal.data,3))
load index_similarstimulusCharacters

index_images=index_similarstimulusCharacters;
stimulusCharacters=signal.Characters(1:9,:);
fs=signal.fs;
counter=1;
step=signal.num_stimulus;
elec=1:32;

x=zeros(size(signal.data_forP300classifier,1), size(signal.data_forP300classifier,2),  size(index_images,1)*size(signal.data,3) );
counter2=1;
step2=size(index_images,1);
y=[]; %P300-label--> 0 or 1
% length(find(y)==1)
B=[];
for tr=1: size(signal.data,3)
    if ~isempty(signal.num_labels); num_target= signal.num_labels(1,tr); else num_target=size(index_images,1)+1; end
    A=signal.data_forP300classifier(elec,:,counter:counter+step-1);
    counter=counter+step;
    
    for ch=1:size(A,1) %synchronized averaging around similar stimulus images at each electrode
        y1=zeros(1,size(index_images,1));
        for im=1:size(index_images,1)
            number=index_images(im,:);number=number(1:repetition);
            A2=A(ch,:,number); A2=reshape(A2, size(A2,2),size(A2,3)); A2=A2';
             B(ch,:,im)=mean(A2,1);
            if num_target==im; label(1,im)=1; else label(1,im)=0;end
        end
    end
    if training
    y=[y label];
    end
    x (:,:,counter2:counter2+step2-1)= B;
    counter2=counter2+step2;
    
    
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %%%%%% canonical correlation analysis (CCA) %%%%%%%%%%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % for tr=1:size(signal.data,3)  %bad spects: 3 5 7 17 19 21 24 27 29 38 40
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % counter2=(tr-1)*size(index_images,1)+1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % step2=size(index_images,1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % for im=1:size(index_images,1)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % W=msetcca(x(:,duration,counter2),1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % [wx1,wy1,r1]=cca(x(:,duration,counter2),W'* x(:,duration,counter2) );
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % num1=find(r1==max(r1)); num1=num1(1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % x(:,duration,counter2)=abs(wx1)'*x(:,duration,counter2);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=x(10:30,1:10:end,:); % 10 is downsampling order
%%%%% rankfeature and dimensionality reduction of P300 features%%%%% 
  if size(x,2)>45; x=x(:,1:45,:); end % 0.7s downsampled with decimation factor 10
% y=y;


if training
signal.P300features=x;
signal.P300labels=y;
end

f=[];
f.P300features=signal.P300features;
f.P300labels=signal.P300labels;
f.fs=signal.fs;
f.downsamplingfactor=10;
f.P300freqrange=[0.5 4];%Hz

save ( ['features_sub',num2str(SubjectNumber)','.mat'],  'f')

%normalize feature data for classification.
% consider all channels for P300 classifier
% length(find(signal.epochlabel_P300==1))
% reshape x into feature vectors
x = reshape(x,size(x,1)*size(x,2),size(x,3));

for num_epoch=1:size(x,2)
x(:,num_epoch)=x(:,num_epoch)./max(x(:,num_epoch));
end

% num_trial=num_epoch/signal.num_stimulus;
%num_epochintrial=num_epoch/num_trial;

%% P300 classification with logit boost 
% %%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%% decision tree %%%%%%
% A Boosting Approach to P300 Detection with Application to Brain-Computer Interfaces % %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

n_channels=size(x,1);%/size(signal.data_forP300classifier,2);
%prepare index sets for cross-validation
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
        %%%n_correct=n_correct1(j)+n_correct;
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
counttrue=0; %number of true answers with atleast 85% accuracy  
for i2=1:size(testsets,2) %number of test_input epochs or signals(each signal has23*128samples or features)
% disp('enter i2='); testsets(i,i2);
% pause(0.5);
  p2=round(classify(l,x(:,testsets(i,i2)))); % answer to a specific data
  y2=length(find(p2==y(testsets(i,i2)) ));
  % number of true answers of classifier for each data for iter iteration of classification
%   pause(0.5);
  if y2>=.85*length(p2)% iter is number of iterationsor answers of logitboost for each input
      counttrue=counttrue+1;
  end

end
disp('number of true answers with atleast 85% accuracy=')
counttrue
Accuracy_test=counttrue/size(testsets,2)
end
%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances

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
% if training; Accuracy=counttrue3/num_epoch; end
% figure;plot(y,'b'); hold on;plot(est_y,'r--')




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
Accuracy_p300 = num_true/tr
 Accuracy_p300;
end
%% P300 detection by morphology (time_peak and ramp) 
% show group average of signals for each of 9 stimulus images in each trial%%%

signal.data_forP300classifier;%size:( num_channels, length_epoch, signal.num_stimulus*size(signal.data,3))
load index_similarstimulusCharacters
load P300wave
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
sig_LPF=signal.dataDWT([signal.num_Cz] ,:,tr );
plot(time,orgsig,'b');hold on; plot(time,5*mrk,'r');text (1,5,'Raw signal')
hold on; plot(time, sig_denoised-20,'b'); text (1, -20,'denoised signal');
hold on; plot(time, sig_LPF-40,'b'); ylim([-50 40]); text(1,-40, 'lowpass filtered signal');
xlabel ('time (s)'); ylabel('Amplitude (\muV)');

sig=[];
num_tr= size(signal.data,3);%num_trial/10;
if num_tr>10; num_tr=round(num_tr/4);end
spect=0;

fs=signal.fs;

time=[1:size(signal.data,2)]./fs;

%% output: P300/SSVEP features
f=[];
f.data=signal.data;
f.P300features=signal.P300features;
f.P300labels=signal.P300labels;
f.fs=signal.fs;
f.downsamplingfactor=10;
f.P300freqrange=[0.5 4];%Hz
f.num_stimulus=signal.num_stimulus;
f.Characters=signal.Characters;% the order of characters shown to subject in each trial
f.mainCharacters=signal.Characters(signal.num_labels(1,:),:);% each trial
f.stimulusCharacters=stimulusCharacters;%all stimulus characters
f.Channels=signal.Channels;
f.labels=signal.labels;
f.Channels2=Channels2;
f.chanpos2=chanpos2;
% f.data_forSSVEPclassifier=signal.data_forSSVEPclassifier;
% f.data_forSSVEPclassifier1=signal.data_forSSVEPclassifier1;
f.data_forSSVEPclassifier_2=signal.data_forSSVEPclassifier_2;
f.data_forSSVEPclassifier1_2=signal.data_forSSVEPclassifier1_2;
f.triallabel_direction=signal.triallabel_direction-1;%0:left 1:Up 2:right 
f.f_ssvep= signal.f_ssvep  ;
f.freqrange=signal.freqrange  ;
% adjust channel numbers (labels) according to the headmesh electrodes labels
Channels=f.Channels;

%update electrodes according to headmesh electrodes labels
counter=1;num2=[];
for k=1:length(elec_realigned.label) 
    num1=find(strcmp(lower(f.Channels),lower(elec_realigned.label(k))));
    if ~isempty (num1)
% % %         f.power2(counter,:)=f.power1(num1,:);
        f.Channels2{counter,1}=f.Channels{num1};
        f.chanpos2(counter,:)=elec_realigned.chanpos(k,:);
        %save numbers
        num2(counter)=num1;
        counter=counter+1;
    end
end

% % % % % % % % % for k=1:size(f.data_forSSVEPclassifier,3)
% % % % % % % % % f.data_forSSVEPclassifier_2(:,:,k)=f.data_forSSVEPclassifier(num2,:,k);
% % % % % % % % % end
% % % % % % % % % for k=1:size(f.data_forSSVEPclassifier1,3)
% % % % % % % % % f.data_forSSVEPclassifier1_2(:,:,k)=f.data_forSSVEPclassifier1(num2,:,k);
% % % % % % % % % end

% filter data around flickering frequency
%BP parameters
d1= fdesign.bandpass('N,Fst1,Fp1,Fp2,Fst2,C',50,14,14.1,16,16+0.5,f.fs);%50,0.01,0.16,45,50.5,fs);%36
Hd1=design(d1,'equiripple');

tr=1;%trial
for tr=1:size(f.data_forSSVEPclassifier1_2 ,3)
for i=1:size(f.data_forSSVEPclassifier1_2  ,1)
%   figure(2);subplot(7,4,i);hold on;
sig=f.data_forSSVEPclassifier1_2  (i,:,tr);
sig=filter(Hd1,sig);
f.data_forSSVEPclassifier1_2  (i,:,tr)=sig;
%   plot(sig);title((f.Channels2(i)))
end
end

for tr=1:size(f.data_forSSVEPclassifier1_2 ,3)
for i=1:size(f.data_forSSVEPclassifier1_2  ,1)
%    figure(2);subplot(7,4,i);hold on;
sig=f.data_forSSVEPclassifier1_2  (i,:,tr);
if max(sig)<1 || std(sig)<1% find electrodes without amplitudes
    sprintf(['found one...','electrode:',num2str(i),'tr:',num2str(tr)])
f.data_forSSVEPclassifier1_2  (i,:,tr)=sig;
%    plot(sig);title((f.Channels2(i)))
[ mindistance1,index1,distances_ascend, indexes_ascend] = min_distance(f.chanpos2, f.chanpos2(i,:) );
f.data_forSSVEPclassifier1_2  (i,:,tr)=sum(f.data_forSSVEPclassifier1_2  (indexes_ascend(1:4),:,tr))./4;
end
end
end


for k=1:size(f.data_forSSVEPclassifier_2 ,3)
for i=1:size(f.data_forSSVEPclassifier_2  ,1)
%    figure(2);subplot(7,4,i);hold on;
sig=f.data_forSSVEPclassifier_2  (i,:,k);
sig=filter(Hd1,sig);
f.data_forSSVEPclassifier_2  (i,:,k)=sig;
%    plot(sig);title((f.Channels2(i)))
end
end




% Calculate laplacian and tangential electric field matrices
% these matrices will be multiplied by EEG matrices to increase spatial
% resolution by increasing the signal to noise ratio (SNR)
f.elec_SSVEP.chanpos=f.chanpos2;
f.elec_SSVEP.label=f.Channels2;
[Et,Ep,L,S] = scalpef( f.elec_SSVEP.chanpos , 3 , 0.1);

% for k=1:size(f.data_forSSVEPclassifier_2,3)
% %apply laplacian or TEF
% f.data_forSSVEPclassifier_2(:,:,k)=L*f.data_forSSVEPclassifier_2(:,:,k);
% end

% for k=1:size(f.data_forSSVEPclassifier1_2,3)
% %apply laplacian or TEF
% f.data_forSSVEPclassifier1_3(:,:,k)=L*f.data_forSSVEPclassifier1_2(:,:,k);
% end
% % % % % calculate power-in a trial
% for k=1:size(f.data_forSSVEPclassifier1_3,3)
% for i=1:size(f.data_forSSVEPclassifier1_3(:,:,k),1)
% f.power2_lap(i,k)=power_SE(f.data_forSSVEPclassifier1_3(i,:,k),f.fs,[f.f_ssvep-1 f.f_ssvep+1]);
% end
% f.power2_lap(:,k)=f.power2(:,k)/max(f.power2(:,k));
% end
% % % % % calculate power-in a trial
for k=1:size(f.data_forSSVEPclassifier1_2,3)
for i=1:size(f.data_forSSVEPclassifier1_2(:,:,k),1)
f.power2(i,k)=power_SE(f.data_forSSVEPclassifier1_2(i,:,k),f.fs,[f.f_ssvep-1 f.f_ssvep+1]);
end
f.power2(:,k)=f.power2(:,k)/max(f.power2(:,k));
end


% %%% calculate power around flickering frequency- 
% % average around stimulus in a trial

% for k=1:size(f.data_forSSVEPclassifier_2,3)
% for i=1:size(f.data_forSSVEPclassifier_2(:,:,k),1)
% f.power2_epoch(i,k)=power_SE(f.data_forSSVEPclassifier_2(i,:,k),f.fs,[f.f_ssvep-1 f.f_ssvep+1]);
% end
% end
% % calculate average power of all epochs in a trial
% counter=1;%trial
% for k=1:f.num_stimulus:size(f.power2_epoch,2)-f.num_stimulus+1
%     a=(sum(f.power2_epoch(:,k:k+f.num_stimulus-1)'))/f.num_stimulus;
% f.power2_avgepoch(:,counter)=a';
% f.power2_avgepoch(:,counter)=f.power2_avgepoch(:,counter)/max(f.power2_avgepoch(:,counter));
% counter=counter+1;
% end



%see power features
% num1=find(f.triallabel_direction==0);
% num2=find(f.triallabel_direction==1);
% num3=find(f.triallabel_direction==2);
% figure;
% plot(f.power2(:,num1),'r');
% hold on;plot(f.power2(:,num2),'g');
% hold on;plot(f.power2(:,num3),'b');




%     counter=1; %using  headmodel for laplacian_Geo, code to reconstruct headmodel has been sudeenly removed
% for k=1:length(M.Electrode.Label)
% %      num1=find(strcmp(lower(f.Channels2),lower(M.Electrode.Label{k,1})));
%     [ mindistance1,index1,distances_ascend, indexes_ascend] = min_distance(f.chanpos2,M.Electrode.Coordinate(k,:)  );
% f.chanpos3(counter,:)=M.Electrode.Coordinate(k,:);
%      
%     f.Channels3{counter,1}=M.Electrode.Label{k,1};
%     f.power3(counter,:)=f.power2(index1,:);
%     counter=counter+1;
%     
% end





method= 'cubic' ;          % cubic % RBF % linear % avg

Channels=f.Channels2;
Values=f.power2(:,1);
[scalpmesh,scalpmesh2, elec, elec_realigned2]=show_3Dbrainmap2( method, Channels, Values,scalpmesh,elec_realigned,0);% or elec or elec_realigned
% view(3);
% step=20;k1=16.3;
% view (-37.5+(k1-1)*step,30);%rotate to the occipital view and save as an RGB m*n*3 
% f.triallabel_direction(1)

num_occipital=[ 2 3 5 6 7 8 9 17 18 19 26  28]';
if ~isempty (num_occipital)
Channels={};
for i=1:length(num_occipital)
Channels{i,1}=f.Channels2{num_occipital(i),1};%f.Channels2
end
num=num_occipital;
else
    Channels={};

  Channels= f.Channels2; 
num=[1:length(Channels)]';
end

for k=1:size(f.power2,2)
k    
Values=f.power2(num,k);%f.power2 has better map compared with power2_avgepoch  
% % % Values=L*Values;%Et*Values %L*Values %it makes all maps the same
% Apply lapacian on power data
% D.Data=f.power3(:,k)';% M*Ne each row contains the values for all electrodes
% D.ExcludeChannel  =[];%column vector numbers of noisy channels
% varargout = ssltool(D,M);
% Values=varargout.Sph';%varargout.Sph  %varargout.Geo

[scalpmesh,scalpmesh2, elec, elec_realigned2]=show_3Dbrainmap2( method, Channels, Values,scalpmesh,elec_realigned,0);% or elec or elec_realigned
figure(2);
hold on; subplot(5,10,k);
ft_plot_mesh(scalpmesh,'vertexcolor',scalpmesh.power ,'edgecolor','none','facealpha',0.9,'edgealpha',0.9);%title('BrainMap');%'edgecolor','none or [0 0 0]'
% hold on; ft_plot_sens(f.elec_SSVEP,'label','on','facecolor','r','fontsize',14);%title (method)%'facecolor',electrodecolor
cM=colorMap([0:0.01:1]');
colormap(cM); 
hold on;view(3);
step=20;k1=16.3;
view (-37.5+(k1-1)*step,30);%rotate to the occipital view and save as an RGB m*n*3 
title(['direction:  ',num2str(f.triallabel_direction  (k))])
 image=getframe (); 
 im=image.cdata; % 3d RGB image with uint8 values between 0 and 255 
 im=imresize(im,[256,256]);
 f.imRGB(:,:,:,k)=im;
% % % %figure;imagesc(f.imRGB(:,:,:,k))
% % % imwrite (im  , 'brainmap.jpg');
end

save ( ['features_sub',num2str(SubjectNumber)','.mat'],  'f')
% load ( ['features_sub',num2str(SubjectNumber)','.mat']);
%% %%%%%%%%%%%%%%%%%
% % % rp = 0.01; % Passband ripple
% % % rs = 26; % Stopband ripple
% % % freq = [f.f_ssvep-1 , f.f_ssvep+1]; % Cutoff frequencies
% % % a = [1 0]; % Desired amplitudes
% % % % Compute deviations
% % % dev = [(10^(rp/20)-1)/(10^(rp/20)+1) 10^(-rs/20)];
% % % [n,fo,ao,w] = remezord(freq,a,dev,fs);
% % % B = remez(n,fo,ao,w);
% % % A=1;

% % % [B,A]=butter(6,[f.f_ssvep-1 , f.f_ssvep+1]./(f.fs/2),'bandpass');
 %  sig=filter(A,B,f.data_forSSVEPclassifier1_2  (i,:,tr));

%%%BP parameters %%%
% d1= fdesign.bandpass('N,Fst1,Fp1,Fp2,Fst2,C',50,14,14.1,16,16+0.5,f.fs);%50,0.01,0.16,45,50.5,fs);%36
% Hd1=design(d1,'equiripple');
% 
 tr=1;%trial
% for tr=1:size(f.data_forSSVEPclassifier1_2 ,3)
for i=1:size(f.data_forSSVEPclassifier1_2  ,1)
  figure(2);subplot(7,4,i);hold on;
sig=f.data_forSSVEPclassifier1_2  (i,:,tr);
% % % % % sig=filter(Hd1,sig);
  plot(sig);title((f.Channels2(i)))
end
% end
% 

% for k=1:size(f.data_forSSVEPclassifier_2 ,3)
% for i=1:size(f.data_forSSVEPclassifier_2  ,1)
%    figure(2);subplot(7,4,i);hold on;
% sig=f.data_forSSVEPclassifier_2  (i,:,k);
% % % % % % % sig=filter(Hd1,sig);
%    plot(sig);title((f.Channels2(i)))
% end
% end
