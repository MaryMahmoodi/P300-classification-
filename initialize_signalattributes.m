function [signal, time]=initialize_signalattributes(data, training, repetition,f_ssvep , freqrange, Rereference )
% initialize signal attributes for the RSVP experiment
% according to the paper: A novel hybrid BCI speller based on RSVP and SSVEP paradigm
signal=[]; % data, training, repetition,f_ssvep , freqrange, Rereference  
signal.f_ssvep=f_ssvep;

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
fs=data.Fs;
signal.fs=fs;
signal.fftlength=16*fs; % for PSD analysis

signal.num_p300epoch=4; % each epoch length=signal.sample_stimulus

signal.freqrange=freqrange;
signal.Rereference=Rereference;



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

signal.Characters=data.Characters;

signal.Channels=data.Channels;
signal.num_Cz=find(strcmp(lower(signal.Channels), lower('Cz')));
signal.num_Oz=find(strcmp(lower(signal.Channels), lower('Oz')));

signal.freqrange=freqrange;
num_trial=size(signal.data,3);
signal.num_trial=num_trial;

time_ssvepstimulus=1/f_ssvep; %seconds
signal.time_ssvepstimulus=time_ssvepstimulus;

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



end

