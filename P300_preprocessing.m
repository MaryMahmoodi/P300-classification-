% we have 26 english characters and one ".".
% So we have 27 characters to be
% classified in the SSVEP+P300 experiment 
% during EEG recording in each trial each subject seeks for 
% a predefined target character on each of the 45 stimulus images  
% each stimulus image contains 3 characters
% In each trial, We have 9 3-character images 
% which are repeated 5 times, i.e., 45  stimulus images  

% SSVEP power spectrum on occipital channels
% has some peaks around stimulus frequency

%  when the subject gazes at a predefined target character on the stimulus image,
% the SSVEP sinusoidal peaks are modulated 

% the direction of eye movement toward target character makes changes
% to the topography of SSVEP spectrum
% we have 3 directions (left, up, right) 
% corresponding 3 characters on each stimulus image
tic,

clear
clf
close all
SubjectNumber=2;
data.TrainLabels=[];
data = load (['Subject',num2str(SubjectNumber),'_Data.mat']);
signal=[];

%% initial input
wavelet_level=6; % 0.5- 7 Hz delta-theta wave
signal.num_p300epoch=4; % each epoch length=signal.sample_stimulus

f_ssvep=15;%Hz
maxfreq=46;
minfreq=0.5;
freqrange=[minfreq,maxfreq];

showplots=1;%show figures

landau= 70; %  transient suppression with Total variation denoising (TVD)

myogenicrejection=1;
thresh_EMG=45;%45;%80

blinkingrejection=0;% detection and replacemnt approach  if blinking_rejection=1;thresh_EMG=80;

Rereference=0;% Rereference all electrodes to the average signal of occipital electrodes

useDWT=1; %using DB10 similar to p300 wave to increase SNR for better classification 

calc_TEO=0;
%     calc_TEO=1; % if you want to calculate Teager energy coeffients to multiply by signal, this increases SNR and spectrogram resolution

calc_PSD=0; % if you want to calculate frequency-fft power (PSD) curves

%% %%%%%%%%%%%%%%%%%%%%%%%%%
fs=data.Fs;
fftlength=16*fs; % for PSD analysis

signal.Rereference=Rereference;
signal.fs=fs;
signal.data=data.TrainData;
signal.Characters=data.Characters;

signal.Channels=data.Channels;
signal.num_Cz=find(strcmp(lower(signal.Channels), lower('Cz')));
signal.num_Oz=find(strcmp(lower(signal.Channels), lower('Oz')));

signal.freqrange=freqrange;
num_trial=size(signal.data,3);
[m,n]=size(signal.data(end,:,1));
max_amp=max(signal.data(end,:,1));
num_stimulus=max_amp;% number of stimulus images ( with three characters)
signal.num_stimulus=num_stimulus;

sample0=length(find( data.TrainData(end,:,1)==zeros(m,n) ) );
sample01=length(find( data.TrainData(end,1: round(10*fs),1)==zeros(m,round(10*fs)) ) );
signal.sample01=sample01;
signal.sample0=sample0;
time_stimulus=(length(data.TrainData(end,:,1))-sample0)/(num_stimulus*fs); % seconds (s)
sample_stimulus=round((length(data.TrainData(end,:,1))-sample0)/(num_stimulus)); % duration of showing each stimulus image
signal.sample_stimulus=sample_stimulus;
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

f_delta=[0.5 4]; f_theta=[4 7]; f_alpha=[8 12]; f_sigma=[14 16] ;f_beta=[15 45]; 



%% preprocess ; Rereferencing, BP filter; myogenic artefact rejection, blinking rejection

% Replace parietooccipital channels with the average 
% of their neigboring electrodes
signal.num_O1=find(strcmp(lower(signal.Channels), lower('O1')));
signal.num_O2=find(strcmp(lower(signal.Channels), lower('O2')));
signal.num_PO3=find(strcmp(lower(signal.Channels), lower('PO3')));
signal.num_PO4=find(strcmp(lower(signal.Channels), lower('PO4')));
signal.num_PO7=find(strcmp(lower(signal.Channels), lower('PO7')));
signal.num_PO8=find(strcmp(lower(signal.Channels), lower('PO8')));
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
% Rereference all other electrodes to the average signal of occipital electrodes  
for tr=1:size(signal.data,3)
    for i=1:size(signal.data,1)
        if max(signal.data (i,:,tr))<1 %damaged electrode
        signal.data (i,:,tr)=avgsignal(1,:,tr);
        else
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

[B1, A1]=butter(3,f_delta/(fs/2),'bandpass');
[B2, A2]=butter(3,f_theta/(fs/2),'bandpass');
[B3, A3]=butter(3,f_alpha/(fs/2),'bandpass');
[B4, A4]=butter(3,f_sigma/(fs/2),'bandpass');
[B5, A5]=butter(3,f_beta/(fs/2),'bandpass');


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
step=2;
sig=myogenic_rejection(sig, fs, step,20);

if blinkingrejection
    spikes_index=[];spikes=[];
    [ sig, spikes,spikes_index ]= blinking_rejection( sig,fs );
end

% % entropy
% fftlength=4*fs; step=round (0.25*fs);
% for k=1:step: (length(sig)-step)
%     [SE]=SpectralEntropy(sig(1,k:k+step-1),fs,fftlength);
%     if isnan(SE); SE=0;end     
%     SE1(1,k:k+step-1)=SE;
% end
%     SE=SpectralEntropy(sig(1,k:length(sig)),fs,fftlength);
% SE1(1,k:length(sig))=SE;
% signal.entropy(i,:,tr )=SE1;



if useDWT
% N=length(sig);
% ww=DWT(N);%building wavelet matrix
% signal.data(i,:,tr )=sig*ww';

% wavelet decomposition at level 7 (delta range, fs=500 or 510 )
% [C1,L] = wavedec(sig, wavelet_level, 'db10'); 
% %[h,g]= wfilters('db10','d'); 
% % wavelet decomposition vector C and the bookkeeping vector L
% cD5 = detcoef(C1,L,wavelet_level);% detail at level7
% cA5 = appcoef(C1,L,'db10',wavelet_level);%approximation coefficient at level 7
% D5 = wrcoef('d',C1,L,'db10',wavelet_level);% Reconstruct detail at level 7->theta(4-7Hz)
% A5 = wrcoef('a',C1,L,'db10',wavelet_level);% Reconstruct approximation at level 7->delta(0-4Hz)
sig=sig-mean(sig);
% figure(1);plot(time, sig); 
[a2,b2]=butter(3,4/(fs/2),'low');
A5=filtfilt(a2,b2,sig);

A5=A5-mean(A5);

% hold on; plot(time,A5,'k');

%%%myogenic artifact rejection%%%
if  myogenicrejection
step=0.25; thresh_EMG=10;
A5=myogenic_rejection(A5, fs, step,thresh_EMG);
end

signal.dataDWT(i,:,tr )=A5;
signal.data(i,:,tr )=sig;

if calc_TEO
%teager energy operator coefficients to increase SNR
%teager energy operator coefficients to increase SNR
TEO= T(signal.dataDWT(i,:,tr )); TEO(1)=0;TEO=TEO/max(TEO); %TEO=abs(5*log10(TEO));
signal.dataTEO(i,:,tr )=TEO;%signal.dataDWT(i,:,tr ).*TEO;
% signal.dataDWT(i,:,tr )=signal.dataDWT(i,:,tr ).*TEO;
% hold on; plot(time,signal.dataDWT(i,:,tr ),'r');
end
else
    sig=sig-mean(sig);
    signal.data(i,:,tr )=sig;
    signal.dataDWT(i,:,tr )=sig;
if calc_TEO
%teager energy operator coefficients to increase SNR
TEO= T(signal.dataDWT(i,:,tr )); TEO(1)=0;TEO=10*TEO/max(TEO); %TEO=abs(5*log10(TEO));
signal.dataTEO(i,:,tr )=TEO;%signal.dataDWT(i,:,tr ).*TEO;
% hold on; plot(time,signal.dataDWT(i,:,tr ),'r');
end

end





if calc_PSD
% PSD
[SE,PSD, freq]=SpectralEntropy(sig,fs,fftlength);
signal.PSD(i,:,tr )=PSD;
signal.freq=freq;
end

[delta,theta,alpha,sigma,beta]=filter_freqbands (sig,fs, B1, B2, B3, B4,B5,A1,A2,A3,A4,A5 );


signal.delta(i,:,tr )=delta;
signal.theta(i,:,tr )=theta;
signal.alpha(i,:,tr )=alpha;
signal.sigma(i,:,tr )=sigma;
signal.beta(i,:,tr )=beta;


% [T1, F1, P1]=mainSpectrogram(sig,maxfreq,1,freqrange);
% signal.time=T1;  
% signal.frequency=F1;
% signal.spectrogram{i,tr}=P1;

end
end


if calc_PSD
    sig=[];
    if ~isempty (signal.startsample_labels)
        
        %  PSD for Cz and Oz channel averaged over target stimulus durations
        for tr=1:num_trial    % i.e., size(signal.data  ,3)
            
            sig (1,:)=signal.data(signal.num_Cz  ,:,tr );%/ max(signal.data(signal.num_Cz ,:,tr ));
            sig (2,:)=signal.data(signal.num_Oz,:,tr );%/ max(signal.data( signal.num_Oz ,:,tr ));
            
            %  markers=zeros(1,size(sig,2));
            
            for i=1:size(signal.startsample_labels,1);
                start1=signal.startsample_labels(i,tr);
                end1=signal.startsample_labels(i,tr)+sample_stimulus;
                %      markers( start1 : end1)=1;
                [SE,PSD, freq]=SpectralEntropy(sig (1,start1:end1),fs,fftlength);
                signal.freq=freq;
                
                if i==1; PSD1=zeros(size(PSD));PSD2=zeros(size(PSD));end
                PSD1=PSD1+PSD;
                [SE,PSD, freq]=SpectralEntropy(sig (2,start1:end1),fs,fftlength);
                PSD2=PSD2+PSD;
                
            end
            
            %     signal.markersignal(:,tr)=markers';
            PSD1=PSD1/i;%  PSD for Cz channel averaged over around target stimulus durations
            signal.PSD_Cz(:,tr)=PSD1';
            
            PSD2=PSD2/i; % PSD for Oz channel averaged over around target stimulus durations
            signal.PSD_Oz(:,tr)=PSD2';
            
        end
        % find zero PSDs and fill them with past trial
        for tr=1:num_trial
            if signal.PSD_Cz(:,tr)==zeros(size(signal.PSD_Cz(:,tr))); signal.PSD_Cz(:,tr)=signal.PSD_Cz(:,tr-1);end
            if signal.PSD_Oz(:,tr)==zeros(size(signal.PSD_Oz(:,tr))); signal.PSD_Oz(:,tr)=signal.PSD_Oz(:,tr-1);end
            
        end
        
    end
end
toc,

%% prepare data, classes and features for classifiers

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
    for tr=1:size(signal.data,3) % trial
        starttrial=signal.sample01+(1-1)*signal.sample_stimulus;
            endtrial=starttrial+signal.num_stimulus*signal.sample_stimulus-1;
        data_forSSVEPclassifier(1:size(signal.data,1)-1,:,tr)=signal.dataDWT(1:size(signal.data,1)-1,starttrial:endtrial,tr);
           
        for ch=1:size(signal.data,1)-1
            powerfeaturedata(ch,:,tr)=[norm(signal.delta(ch,starttrial:endtrial,tr),'fro')  norm(signal.theta(ch,starttrial:endtrial,tr),'fro')   norm(signal.alpha(ch,starttrial:endtrial,tr),'fro')  norm(signal.sigma(ch,starttrial:endtrial,tr),'fro')  norm(signal.beta(ch,starttrial:endtrial,tr),'fro')];
            powerfeaturedata(ch,:,tr)=powerfeaturedata(ch,:,tr)/norm(powerfeaturedata(ch,:,tr));
            end
        for num=1:signal.num_stimulus
            start1=signal.sample01+(num-1)*signal.sample_stimulus;
            end1=start1+ signal.num_p300epoch*signal.sample_stimulus-1;
            epochnumber=epochnumber+1;
            data_forP300classifier(1:size(signal.data,1)-1,:,epochnumber) = signal.dataDWT(1:size(signal.data,1)-1,start1:end1,tr);
if calc_TEO
            dataTEO_forP300classifier(1:size(signal.data,1)-1,:,epochnumber) = signal.dataTEO(1:size(signal.data,1)-1,start1:end1,tr);
end
            
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
    if calc_TEO
signal.dataTEO_forP300classifier=dataTEO_forP300classifier;    
    end
signal.data_forP300classifier=data_forP300classifier;
signal.powerfeaturedata=powerfeaturedata;
signal.epochlabel_P300=reshape(signal.epochlabel_P300, 1, size(signal.epochlabel_P300,1)*size(signal.epochlabel_P300,2));
signal.data_forSSVEPclassifier=data_forSSVEPclassifier;
signal.triallabel_direction;
% This part has been calculated in last loop for each epoch 
% power and SEF feature extraction for each stimulus 3-character image duration (sample_stimulus)
% [signal]=power_SEF_featureextraction (signal,fs);

% save signal_sub2 signal


%% show results 
% load signal_sub2
fs=signal.fs;

time=[1:size(signal.data,2)]./fs;

if showplots
    
num_tr= 10;%num_trial/10;
counter=1;
if calc_PSD
step=4;
else
step=3;  
end

figure(3);clf;

for tr=1:num_tr    % i.e., size(signal.data  ,3)
 
 sig=signal.dataDWT([signal.num_Cz signal.num_Oz] ,:,tr );
 
 markers=zeros(1,size(sig,2));
 if ~isempty (signal.startsample_labels)
    for i=1:size(signal.startsample_labels,1);
    start1=signal.startsample_labels(i,tr);
    end1=signal.startsample_labels(i,tr)+(signal.num_p300epoch)*signal.sample_stimulus;
     markers( start1 : end1)=1;
    end
% markers=signal.markersignal(:,tr)';
 end

% signal (denoised)
 hold on;
subplot(num_tr,step,counter); 
% plot(time, sig(1,:), 'b'); hold on; plot(time, sig(2,:), 'k')
% plot(time, signal.dataDWT(signal.num_Cz  ,:,tr ), 'b'); hold on; plot(time, signal.dataDWT(signal.num_Oz  ,:,tr )-10, 'k');
elec=1:32;

th1=mean(mean(signal.dataDWT(elec,:,tr ),1))+0.5*std(mean(signal.dataDWT(elec,:,tr ),1));
th1=0.5*(1/length(mean(signal.dataDWT(elec,:,tr ),1))*sum(abs(mean(signal.dataDWT(elec,:,tr ),1))));

plot(time, mean(signal.dataDWT(elec,:,tr ),1), 'b'); 
  hold on; plot(time, th1*ones(size(signal.dataDWT(: ,:,tr ))), 'k');
if calc_TEO
 th1=mean(mean(signal.dataTEO(elec ,:,tr ),1))+0.5*std(mean(signal.dataTEO(elec,:,tr ),1));
th1=0.5*(1/length(mean(signal.dataTEO(elec,:,tr ),1))*sum(abs(mean(signal.dataTEO(elec,:,tr ),1))));

 hold on; plot(time, 5*mean(signal.dataTEO(elec,:,tr ),1)-2, 'g'); 
hold on; plot(time, 5*th1*ones(size(signal.dataTEO(elec ,:,tr )))-2, 'k');
end
hold on; plot(time,5*markers,'r');
xlim([1, time(end)])
ylim([-5,5]);

if tr==1
xlabel('time(s)'); ylabel('Amplitude (\muV)');
% title(['denoised signal,',signal.Channels{signal.num_Cz},', ' ,signal.Channels{signal.num_Oz}])
 title('denoised signal and signal of energy changes')
%  legend(signal.Channels{signal.num_Cz},signal.Channels{signal.num_Oz}, 'markers')
 end
 
% power spectrogram map for Cz  ( or power or spectral entropy map)
hold on; subplot(num_tr,step,counter+1); 
% plot(time, signal.SEFd([signal.num_Cz ] ,:,tr ), 'b'); ylim(freqrange);
% plot(time, signal.powerdelta([signal.num_Cz ] ,:,tr )/max(signal.powerdelta([signal.num_Cz ] ,:,tr )), 'b'); 
% hold on;
% plot(time, signal.powerdelta([signal.num_Oz ] ,:,tr )/max(signal.powerdelta([signal.num_Oz ] ,:,tr )), 'k'); 
% 
% hold on; plot(time, markers,'r');
% xlim([1, time(end)]);  


[T1, F1, P1]=mainSpectrogram(sig(1,:),maxfreq,1,freqrange,fs,1);
xlim([T1(1) T1(end)]);  
ylim(freqrange);

 if tr==1
xlabel('time(s)'); %ylabel('frequency (Hz)');
  title([signal.Channels{signal.num_Cz}])
%  title('power-delta');
 end

% power spectrogram map for Oz ( or power or spectral entropy map)
hold on; subplot(num_tr,step,counter+2); 
% hold on; plot(time, signal.SEFd([signal.num_Oz ] ,:,tr ), 'b');ylim(freqrange);
% plot(time, signal.poweralpha([signal.num_Cz ] ,:,tr )/max(signal.poweralpha([signal.num_Cz ] ,:,tr )), 'b'); 
% hold on;
% plot(time, signal.poweralpha([signal.num_Oz ] ,:,tr )/max(signal.poweralpha([signal.num_Oz ] ,:,tr )), 'k'); 
% 
% hold on; plot(time,markers,'r');
% xlim([1, time(end)]);   



[T1, F1, P1]=mainSpectrogram(sig(2,:),maxfreq,1,freqrange,fs,1);
ylim(freqrange); 
xlim([T1(1) T1(end)]);  

if tr==1
xlabel('time(s)'); %ylabel('frequency (Hz)');
 title([ signal.Channels{signal.num_Oz}])
% title('power-alpha');
end

if calc_PSD
%PSD
hold on; subplot(num_tr,step,counter+3); 
% plot(signal.freq, signal.PSD(signal.num_Cz  ,:,tr)/norm(signal.PSD(signal.num_Cz  ,:,tr)),'b'); 
% hold on; plot(signal.freq, signal.PSD(signal.num_Oz  ,:,tr)/norm(signal.PSD(signal.num_Oz  ,:,tr)) ,'k');

plot(signal.freq, signal.PSD_Cz(:,tr)/max(signal.PSD_Cz(:,tr)),'b');
hold on; plot(signal.freq, signal.PSD_Oz(:,tr)/max(signal.PSD_Oz(:,tr)),'k');

% % % AIC_Cz=sum(signal.PSD_Cz(:,tr)/max(signal.PSD_Cz(:,tr)));
% % % AIC_Oz=sum(signal.PSD_Oz(:,tr)/max(signal.PSD_Oz(:,tr)));

xlim(freqrange);
text (5, 0.5,  [ (signal.labels(tr))  ', direction:'  (signal.direction_labels (1, tr))  ]  );
if tr==1
xlabel('frequency (Hz)'); ylabel('power'); 
title(['PSD, ', signal.Channels{signal.num_Cz}, ', ',signal.Channels{signal.num_Oz} ]);
legend(signal.Channels{signal.num_Cz},signal.Channels{signal.num_Oz})
end

end
% hold on; subplot(num_tr,step,counter+4);
% plot(time, signal.entropy(signal.num_Cz  ,:,tr )/max(signal.entropy(signal.num_Cz  ,:,tr )),'b');
% hold on; plot(time, signal.entropy(signal.num_Oz ,:,tr )/max(signal.entropy(signal.num_Oz  ,:,tr )),'k');
% hold on; plot(time,markers,'r');
% xlim([time(1) time(end)]);
% 
% if tr==1
% xlabel('time(s)'); title('entropy'); 
% title(['entrpoy, ', signal.Channels{signal.num_Cz}, ', ',signal.Channels{signal.num_Oz} ]);
% legend(signal.Channels{signal.num_Cz},signal.Channels{signal.num_Oz}, 'markers')
% end


counter=counter+step;
end


%%% AUC _PSD analysis for Cz and Oz around target stimulus
if calc_PSD
    
freqmin=signal.freqrange(1);
num_freqmin=find(signal.freq>=freqmin);num_freqmin=num_freqmin(1);
freqmax=signal.freqrange(2);
num_freqmax=find(signal.freq<=freqmax);num_freqmax=num_freqmax(end);

for tr=1:size(signal.data,3)
signal. PSD_Cz (:,tr)=signal. PSD_Cz(:,tr) /max(signal. PSD_Cz(:,tr) );
signal. PSD_Oz(:,tr) =signal. PSD_Oz (:,tr) /max(signal. PSD_Oz (:,tr));

signal. AIC_Cz (1,tr)=sum(signal.PSD_Cz(num_freqmin:num_freqmax,tr) );
signal. AIC_Oz (1,tr)=sum(signal.PSD_Oz(num_freqmin:num_freqmax,tr));
end

figure(1);clf;
plot(signal. AIC_Cz ,'b'); 
hold on; plot(signal. AIC_Oz ,'k');
legend('Cz','Oz'); title('AUC from PSD curve around stimulus targets in each trial  ')
xlabel('trials'); ylabel ('AUC')

end
 
end

%%% P300 template %%%
% one good P300 template for subject2 is 4th target stimulus (4th rectangle) 
% in the 5th trial 
% it's corresponding start time is:
TargetStimulus=4; trial=2;
ss=signal.startsample_labels(TargetStimulus,trial);
% the template ends in: 
st=signal.startsample_labels(TargetStimulus,trial)+signal.num_p300epoch*signal.sample_stimulus-1
signal.P300_template=[]; signal.P300_template=signal.dataDWT(:,ss:st,5);
time=[1:signal.num_p300epoch*signal.sample_stimulus]./fs;
figure(2);clf;plot(time, signal.P300_template(:,:)); title('P300 wave');
xlabel('time(s)'); ylabel('Amplitude (\muV)')
 P300wave=signal.P300_template(17,:); 
% save P300wave P300wave
% figure;plot(time,P300wave )



% for i=1:9
%     num=find(signal.num_labels(1,:)==i); num=num(1);
%     index_similarstimulusCharacters(i,:)=signal.num_labels(:,num)';
% end
% save index_similarstimulusCharacters   index_similarstimulusCharacters
% save signal_sub2 signal;   save P300wave   P300wave;




%% logistic regressor classifier
cd('D:\code_electronicdevice\Hackaton\NBML_iBCIC2021_p300')
% save signal_sub2 signal ; save P300wave P300wave; save index_similarstimulusCharacters  index_similarstimulusCharacters
%  load signal_sub2 ;  load P300wave;
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

for tr=1: size(signal.data,3)
    if ~isempty(signal.num_labels); num_target= signal.num_labels(1,tr); else num_target=size(index_images,1)+1; end
    A=signal.data_forP300classifier(elec,:,counter:counter+step-1);
    counter=counter+step;
    
    for ch=1:size(A,1) %synchronized averaging around similar stimulus images at each electrode
        y1=zeros(1,size(index_images,1));
        for im=1:size(index_images,1)
            number=index_images(im,:);
            A2=A(ch,:,number); A2=reshape(A2, size(A2,2),size(A2,3)); A2=A2';
            B(ch,:,im)=mean(A2,1);
            if num_target==im; label(1,im)=1; else label(1,im)=0;end
        end
    end
    y=[y label];
    x (:,:,counter2:counter2+step2-1)= B;
    counter2=counter2+step2;
    
    
end

x=x(15:25,:,:);
y=y;


% if 0
    
iter=20;
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

%%%%%% Logistric regressor%%%%%%
addpath ('D:\code_electronicdevice\Hackaton\NBML_iBCIC2021_p300\boostingp300')
cd('D:\code_electronicdevice\Hackaton\NBML_iBCIC2021_p300\boostingp300\@LogitBoost')

n_channels=size(x,1)-1;
%prepare index sets for cross-validation
n_permutations = 2;
n_epochs = size(x,2);%942
testsetsize = round(n_epochs / 10);
[trainsets, testsets] = crossValidation(1:n_epochs, testsetsize, ...
n_permutations);
correct = [];figure(1);
for i = 1:n_permutations
    train_x = x(:, trainsets(i,:));
    train_y = y(:, trainsets(i,:));%y is logical output for x
    test_x  = x(:, testsets(i,:));
    test_y  = y(:, testsets(i,:));    
     % train classifier and apply to test data
if trainingLogitBoost
     l =LogitBoost(iter, 0.05, 1);%LogitBoost number of iterations=50,
    l = train(l, train_x, train_y, n_channels);
else
     load (['l_sub',num2str(SubjectNumber),'.mat']);
 
end
    p = classify(l, test_x); 
% evaluate classification accuracy 
    i0 = find(p <= 0.5);
    i1 = find(p > 0.5);
    est_y = zeros(size(p));
    est_y(i0) = 0;
    est_y(i1) = 1;
    for j = 1:size(est_y,1)
        n_correct(j) = length(find(est_y(j,:) == test_y));% each electrode is concidered for all testing epochs to find which one has p300
        %n_correct=n_correct1(j)+n_correct;
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
hold on


%%%%%%%%%%%%%%%%%%%%%%%%  

counttrue=0;%number of true answers with atleast 85% accuracy  
for i2=1:size(testsets,2) %number of test_input epochs or signals(each signal has23*128samples or features)
% disp('enter i2='); testsets(i,i2);
% pause(0.5);
  p2=round(classify(l,x(:,testsets(i,i2))));%answer to a specific data
  y2=length(find(p2==y(testsets(i,i2)) ));
  % number of true answers of classifier for each data for 50 iteration of classification
%   pause(0.5);
  if y2>=.85*iter% 50 is number of iterationsor answers of logitboost for each input
      counttrue=counttrue+1;
  end
end
disp('number of true answers with atleast 85% accuracy=')
counttrue
Accuracy_test=counttrue/size(testsets,2)

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances

%**********calculate classifier detection accuracy for all epochs*******************
counttrue3=0;
for num_epoch=1:size(x,2)
p3=round(classify(l,x(:,num_epoch)) );
y3=length(find(p3==y(num_epoch)));
 if y3>=.85*iter% 50 is number of iterationsor answers of logitboost for each input
      counttrue3=counttrue3+1;
  end
end
Accuracy=counttrue3/num_epoch
cd('D:\code_electronicdevice\Hackaton\NBML_iBCIC2021_p300')

save l_sub2 l

% end




%% Deep learning prepare train and test data  by cross validation
f1=signal.powerfeaturedata (18:end,:,:);%size:( num_channels, num_features, size(signal.data,3))
signal.data_forSSVEPclassifier;
labels=signal.triallabel_direction;
labels=labels-1;
% consider parieto occipital channel (18:32) for SSVEP
datafeature=reshape(f1, size(f1,1)*size(f1,2),size(f1,3));
for i=1:size(datafeature,1)
    for j=1:size(datafeature,2)
    if isnan(datafeature(i,j)); datafeature(i,j)=rand(1);end 
    end
end

n_permutations = 2;
n_epochs = size(datafeature,2);
testsetsize = round(n_epochs/ 10);
[trainsets, testsets] = crossValidation(1:n_epochs, testsetsize,...
n_permutations); 
res={};
more off;

%%% Define DBN classifier
l=size(datafeature,1);
dbn=DBN('classifier');
% RBM1
%%data.valueType=ValueType.probability;gaussian
rbmParams=RbmParameters(size(datafeature,1),ValueType.gaussian);%30
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.learningRate = 0.001;
rbmParams.moment = 0.5;
rbmParams.batchSize =size(datafeature,1);
%rbmParams.batchSize =4;
rbmParams.penalty = 0.0002;
rbmParams.performanceMethod='reconstruction';
rbmParams.maxEpoch=100;%200
dbn.addRBM(rbmParams);
% RBM2
rbmParams=RbmParameters(5,ValueType.binary);%1000 binary
%rbmParams.rbmType =rbmType.discriminative;
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.performanceMethod='reconstruction';
rbmParams.learningRate = 0.001;
rbmParams.moment = 0.5;
rbmParams.batchSize = size(datafeature,1);
%rbmParams.batchSize =4;
rbmParams.penalty = 0.0002;
rbmParams.maxEpoch=100;%200
dbn.addRBM(rbmParams);
%RBM
% RBM2
% rbmParams=RbmParameters(50,ValueType.binary);%1000 binary
% %rbmParams.rbmType =rbmType.discriminative;
% rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
% rbmParams.performanceMethod='reconstruction';
% rbmParams.learningRate = 0.001;
% rbmParams.moment = 0.5;
% rbmParams.batchSize = size(datafeature,1);
% %rbmParams.batchSize =4;
% rbmParams.penalty = 0.0002;
% rbmParams.maxEpoch=200;%200
% dbn.addRBM(rbmParams);
% RBM2
rbmParams=RbmParameters(4,ValueType.binary);%1000 binary
%rbmParams.rbmType =rbmType.discriminative;
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.performanceMethod='pretrain';
rbmParams.learningRate = 0.001;
rbmParams.moment = 0.5;
rbmParams.batchSize = size(datafeature,1);
%rbmParams.batchSize =4;
rbmParams.penalty = 0.0002;
rbmParams.maxEpoch=100;%200
dbn.addRBM(rbmParams);
% RBM3

rbmParams=RbmParameters(max(labels)+1,ValueType.gaussian);%2000 binary
%yyrbmParams.rbmType =rbmType.discriminative;
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.rbmType=RbmType.discriminative;
rbmParams.learningRate = 0.001;
rbmParams.moment = 0.5;
rbmParams.batchSize = size(datafeature,1);%2
%rbmParams.batchSize =4;
rbmParams.penalty = 0.0002;
rbmParams.performanceMethod='classification';
rbmParams.maxEpoch=100;%400
%rbmParams.numberOfVisibleSoftmax=3;
dbn.addRBM(rbmParams);




%
% select=1;%change between features 
% for select=1:1 

for j=1:size(datafeature,1)
data1.dataMean(1,j)=sum(datafeature(j,:))/size(datafeature,2);
data1.dataStd(1,j) =std(datafeature(j,:));
end
for j=1:size(datafeature,1)
data1.dataMin(1,j)=min(datafeature(j,:));
data1.dataMax(1,j) =max(datafeature(j,:));
end


%data1.dataMin=[];    
%data1.dataMax=[];
data1.normilizeMethod='meanvar'; 
data1.valueType=ValueType.probability;
data1.trainData=[datafeature(:,trainsets)]';% datafeature(:,trainsets) datafeature(:,trainsets) datafeature(:,trainsets) datafeature(:,trainsets) datafeature(:,trainsets)]';
data1.trainLabels=[labels(1,trainsets)]';% y(1,trainsets) y(1,trainsets) y(1,trainsets) y(1,trainsets) y(1,trainsets)]';
data1.validationData=[datafeature(:,testsets)]';% datafeature(:,testsets) datafeature(:,testsets) datafeature(:,testsets) datafeature(:,testsets) datafeature(:,testsets)]';
data1.validationLabels=[labels(1,testsets)]';% y(1,testsets) y(1,testsets) y(1,testsets) y(1,testsets) y(1,testsets)]';
data1.testData=[datafeature(:,testsets)]';% datafeature(:,testsets) datafeature(:,testsets) datafeature(:,testsets) datafeature(:,testsets) datafeature(:,testsets)]';
data1.testLabels=[labels(1,testsets)]';% y(1,testsets) y(1,testsets) y(1,testsets) y(1,testsets) y(1,testsets)]';
 
% data1.shuffle();
%
clear data
load data_makeme
data.valueType=data1.valueType;
data.trainData=data1.trainData;
data.trainLabels=data1.trainLabels;
data.validationData=data1.validationData;
data.validationLabels=data1.validationLabels;
data.testData=data1.testData;
data.testLabels=data1.testLabels;
data.dataMean=data1.dataMean;
data.dataStd=data1.dataStd;
data.dataMin=data1.dataMin;
data.dataMax=data1.dataMax;









% train classifier
ticID=tic;
dbn.train(data);
toc(ticID)
dbn.backpropagation(data);



% load   dbn_sub2

OUTPUT=dbn.getOutput(data.testData,'bySampling');
error=sum(OUTPUT~=data.testLabels)/length(OUTPUT);
sensitivity=abs(sum(OUTPUT~=data.testLabels)-length(OUTPUT))/length(OUTPUT)*100;
% show=1;
% if show
% figure;plot(data.testLabels,'k-')
% hold on;plot(OUTPUT,'k^','markerfacecolor',[1 0 0])
% legend('data.testLabels','classifier output')
% end


data.trainData;
data.trainLabels;
OUTPUT=dbn.getOutput(data.trainData,'bySampling');
error=sum(OUTPUT~=data.trainLabels)/length(OUTPUT);
sensitivity=abs(sum(OUTPUT~=data.trainLabels)-length(OUTPUT))/length(OUTPUT)*100;



datafeature';
labels';
OUTPUT=dbn.getOutput(datafeature','bySampling');
error=sum(OUTPUT~=labels')/length(OUTPUT);
sensitivity=abs(sum(OUTPUT~=labels')-length(OUTPUT))/length(OUTPUT)*100

 save dbn_sub2 dbn
% load dbn
%%%%%%%%%%%%%%
OUTPUT=dbn.getOutput(data.testData,'bySampling');
error=sum(OUTPUT~=data.testLabels)/length(OUTPUT);
sensitivity=abs(sum(OUTPUT~=data.testLabels)-length(OUTPUT))/length(OUTPUT)*100;


for select1=0:0%or 0:0 or any label
    count=0;
for i=1:size(OUTPUT,1)
    if OUTPUT(i,1)==select1
        
   if     OUTPUT(i,1)== data.testLabels(i,1)
        count=count+1;
   end     
 end
end
end
count2=length(find(data.testLabels==select1));
    sensitivity_left = (count/count2)*100; %for either positive or negative(1 or 0 correspond to select1)
    
    
    for select2=1:1
    count=0;
for i=1:size(OUTPUT,1)
    if OUTPUT(i,1)==select2
        
   if     OUTPUT(i,1)== data.testLabels(i,1)
        count=count+1;
   end     
 end
end
end
count2=length(find(data.testLabels==select2));
    sensitivity_up = (count/count2)*100;
    
for select2=2:2
    count=0;
for i=1:size(OUTPUT,1)
    if OUTPUT(i,1)==select2
        
   if     OUTPUT(i,1)== data.testLabels(i,1)
        count=count+1;
   end     
 end
end
end
count2=length(find(data.testLabels==select2));
sensitivity_right = (count/count2)*100;  
  

 
errors = [error]
sensitivity = [sensitivity]
allsensitivity=[sensitivity_left sensitivity_up sensitivity_right ];
% How many data we have for each class?
num_0=length(find(labels==0));%left
num_1=length(find(labels==1));%up
num_2=length(find(labels==2));%right













%%  P300 detection by scoring each stimulus image: 
% % % each stimulus image is repeated 5 times in each trial
% % % we have 9 stimulus image (each has 3 characters)
%  load signal_sub2 ;  load P300wave;
load index_similarstimulusCharacters
index_images=index_similarstimulusCharacters;
stimulusCharacters=signal.Characters(1:9,:);
fs=signal.fs;
counter=1;
step=signal.num_stimulus;
elec=1:32;
for tr=1: size(signal.data,3)
    cnt=1;
    for epoch=counter:counter+step-1 % signals of epochs in each trial are synchronized averaged
        sig_epoch(1,:,cnt)=mean(signal.data_forP300classifier(elec,:,epoch),1);
        cnt=cnt+1;
    end
    counter=counter+step;
    for im=1:size(index_images,1)
        s2=zeros(size(s1,3),size(s1,2));
        s1=sig_epoch(1,:,index_images(im,:));
        for i=1:size(s1,3)
            s2(i,:)=s1(1,:,i);
        end
        sig_image(1,:,im)=mean(s2,1);
    end
    
    for im=1:size(index_images,1)
% % % %         sig_image(1,:,im)=sig_image(1,:,im)/ norm(sig_image(1,:,im)) ;
        a=corrcoef(sig_image(1,:,im),P300wave);
        allcoef(1,im)=a(1,2);
        allmaxamp(1,im)=max(sig_image(1,:,im));
        max1=find(sig_image(1,:,im)==max(sig_image(1,:,im)));
        alltimemax(1,im)=max1- (2*signal.sample_stimulus);
        min1=find(sig_image(1,:,im)==min(sig_image(1,:,im)));
        
         if max1>min1
            allmaxramp(1,im)=sig_image(1,max1,im)-sig_image(1,min1,im);
         else
            allmaxramp(1,im)=sig_image(1,min1,im)-sig_image(1,max1,im);
         end
        
    end
number=find(allmaxramp==max(allmaxramp)); %allcoef; allmaxramp; allmaxamp;
number=number(1);
%     max1=find(sig_image(1,:,number)==max(sig_image(1,:,number)));
%     if max1>=2*signal.sample_stimulus; 
%         number=number+1;
%         if number>size(index_images,1); number=number-1;end
%     end

    % figure;plot(allcoef) ; allcoef; allmaxramp; allmaxamp;
    % figure;plot(sig_image(1,:,number))
    signal.estCharacters(tr,:)=stimulusCharacters(number,:);
    
end

for i=1:size(signal.estCharacters,1)
    out(1,i)=strcmp(signal.mainCharacters(i,:),signal.estCharacters(i,:));
end
count_true=length(find(out==1));
Accuracy=length(find(out==1))/size(signal.estCharacters,1);
[count_true Accuracy*100]

%% P300 detection by morphological operations on preprocessed signal
%  load signal_sub2 ;  load P300wave; load index_similarstimulusCharacters
if 0
fs=signal.fs;

signal.estepochlabel_P300=zeros(size(signal.epochlabel_P300));
counter=1;
step=signal.num_stimulus;
elec=1:32;
for tr=1:size(signal.data,3)
    th=0.5*(1/length(mean(signal.dataTEO(elec,:,tr ),1))*sum(abs(mean(signal.dataTEO(elec,:,tr ),1))));
    th2=0.5*(1/length(mean(signal.dataDWT(elec,:,tr ),1))*sum(abs(mean(signal.dataDWT(elec,:,tr ),1))));
    
    th3=max(mean(signal.dataDWT (elec,:,tr),1));
    
    for epoch=counter:counter+step-1
        sig=mean(signal.data_forP300classifier(elec,:,epoch),1);
        sigTEO=mean(signal.dataTEO_forP300classifier(elec,:,epoch),1);
        
        
        %     sig=conv(sig,P300wave);
        %     sig=sig(1, 1:length(P300wave));
        sig=sig;
        a=zeros(2,2);
        a=corrcoef(sig,P300wave);
        % a=(sum((sig))/length(sig));
        
        signal.corrcoef_epoch(1,epoch)=a(1,2);
        
        min1=find(sig==min(sig));
        max1=find(sig==max(sig));
        max0=find(sigTEO==max(sigTEO));
        duration= (max1-min1)/signal.fs;
        ramp=sig(max1)-sig(min1);
        ramp2= sig(min1)-sig(1);
        
        if      max(sigTEO)>=th %&& max(sig)>=th2 %
            
            if    max1>min1 % max1>min1 && sig(max1)>=sig(max0)
                %              if (0.1<=duration)  && (duration<=0.3) %&& (1.5<=ramp) && (ramp<=25) % s, uV
                signal.estepochlabel_P300(1,epoch)=1;
            else
                signal.estepochlabel_P300(1,epoch)=0;
                
            end
            %         end
        end
    end
    
    counter=counter+step;
end

tr=1;
figure(4);clf; plot(signal.estepochlabel_P300((tr-1)*signal.num_stimulus+1:(tr)*signal.num_stimulus))
hold on;
plot(1/2*signal.epochlabel_P300((tr-1)*signal.num_stimulus+1:(tr)*signal.num_stimulus),'g')
corcoef=signal.corrcoef_epoch;
hold on; plot(corcoef((tr-1)*signal.num_stimulus+1:(tr)*signal.num_stimulus), 'r')



m=find(signal.estepochlabel_P300==signal.epochlabel_P300);
num_True=length(find(signal.estepochlabel_P300==signal.epochlabel_P300));
num_Positives=length(find(signal.estepochlabel_P300(m)==1));
Accuracy=num_True/length(signal.epochlabel_P300);
TPR=num_Positives/length(find(signal.epochlabel_P300==1));
[Accuracy TPR]
% num_tr=num_epoch/signal.num_stimulus;
% num_stimulus=num_epoch/num_tr;
counter=1; signal.num_stimulus;
step=signal.num_stimulus;
for tr=1:size(signal.data,3)
    numbers=signal.estepochlabel_P300(1,counter:counter+step-1)';
    m=find(numbers==1);
    signal.estnum_labels{1,tr}=m;
    signal.corrcoeff{1,tr}=signal.corrcoef_epoch(1,counter:counter+step-1);
    counter=counter+signal.num_stimulus;
end

%%%%%%%
% tr=3; num_stimulus=signal.estnum_labels{1,tr}  ;
% % signal.direction_labels(1,num_stimulus)
% signal.num_labels(:,tr)
% char=signal.Characters(num_stimulus,:)
% corrcoeffs =(signal.corrcoeff{1,tr}(num_stimulus)')
% strcmp(signal.Characters(35,:), signal.Characters(36,:))


counter=1;
step=signal.num_stimulus;
signal.estnum_labels2=[];

for tr=1:size(signal.data,3)
    % corrcoef1=signal.corrcoef_epoch(1,counter:counter+step-1);
    % corrcoef2=zeros(size(corrcoef1));
    num_stimulus=signal.estnum_labels{1, tr};
    % corrcoef2(num_stimulus)=corrcoef1(num_stimulus);
    % num=find(corrcoef2==max(corrcoef2));
    % num=[];
    % signal.estnum_labels2 (1,tr)=num(1);
    allCharacters=['HRT'; 'IFS'; 'CN.' ; 'BQV'; 'OJZ'; 'MUY'; 'XKP'; 'ALW'; 'EGD'] ;
    Characters=signal.Characters(num_stimulus,:);
    for i=1:length(allCharacters)
        counter2(1,i)=0;
        for j=1:length(Characters)
            if strcmp(allCharacters(i,:),Characters(j,:)) ;counter2(1,i)=counter2(1,i)+1;end
        end
    end
    num2=find(counter2==max(counter2));
    num2=num2(1);
    
    signal.estnum_labels2(1,tr)=num2;
    signal.estCharacters(tr,:)=allCharacters(num2,:);
    signal.mainCharacters(tr,:)=signal.Characters(signal.num_labels(1,tr),:);
    
    if ~isempty(signal.labels)
        num3=find(signal.num_labels(1,:)==num2);
        num4=signal.num_labels(:,num3(1));
    end
    a=zeros(1,step);
    a(num4)=1;
    signal.estepochlabel_P300(1,counter:counter+step-1)=a ;
    
    counter=counter+step;
end

figure(5);plot(signal.estepochlabel_P300(1:1000),'g'); hold on;
plot(signal.epochlabel_P300(1:1000));

for i=1:size(signal.estCharacters,1)
    out(1,i)=strcmp(signal.mainCharacters(i,:),signal.estCharacters(i,:));
end
Accuracy=length(find(out==1))/size(signal.estCharacters,1)
end

%% correlation coefficient method
if 0
% save signal_sub2 signal
%  load signal_sub2
load P300wave;
% score each stimulus image by averaging from
% correlation coefficients (xcorr) of signals with a P300 template 
signal.data_forP300classifier;
y=signal.epochlabel_P300;

signal.corrcoef_epoch=[];
for epoch=1:size(signal.data_forP300classifier,3)
for ch=1:size(signal.data_forP300classifier,1)
sig=signal.data_forP300classifier(ch,:,epoch);
a=corrcoef(sig,P300wave);
signal.corrcoef_epoch(ch,epoch)=a(1,2);
end
end
signal.corrcoef_epoch=mean(signal.corrcoef_epoch(15:20,:),1);%average over electrodes
% num_tr=num_epoch/signal.num_stimulus;
% num_stimulus=num_epoch/num_tr;
counter=1; signal.num_stimulus;
for tr=1:size(signal.data,3)
    score_trial=signal.corrcoef_epoch(1,counter:counter+signal.num_stimulus-1)';
    signal.score_trial(:,tr)=score_trial;
    counter=counter+signal.num_stimulus;
end
tr=1;
figure(4);stem(signal.score_trial(:,tr));

% num=41;
% time_stimulusintrial=(signal.sample01+(num-1)*signal.sample_stimulus)/signal.fs

end


