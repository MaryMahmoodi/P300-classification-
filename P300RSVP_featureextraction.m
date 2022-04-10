function [ signal, f ] = P300RSVP_featureextraction( signal, data,training,repetition )
f_ssvep=signal.f_ssvep;
fs=signal.fs;
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

[a3,b3]=butter(6,[4, f_ssvep+2]/(fs/2),'bandpass');

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
        data_forP300classifier(1:size(signal.data,1)-1,:,epochnumber) = signal.dataLPF(1:size(signal.data,1)-1,start1:end1,tr);
        
        data_forSSVEPclassifier(1:size(signal.data,1)-1,:,epochnumber)=filter (a3,b3,signal.data(1:size(signal.data,1)-1,start1:end1,tr));
        start11=start1; end11=end1;
        
        
        if ~isempty (data.TrainLabels)
            k=[]; k=find (num==signal.num_labels(:,tr));
            if ~isempty (k)
                
                
                if signal.num_p300epoch==3; step1=2;end
                if signal.num_p300epoch==2; step1=1;end
                if signal.num_p300epoch==4; step1=3;end
                
                if num<=signal.num_p300epoch-step1
                    signal.epochlabel_P300(1:num,tr)=1;
                    
                else
                    
                    signal.epochlabel_P300(num-(signal.num_p300epoch-step1):num,tr)=1;
                    
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
y=[];
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

x=x(10:30,1:10:end,:); % 10 is downsampling order
%%%%% dimensionality reduction of P300 features%%%%%
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


end

