function  show_timeelectrodemap(trial_number, signal, index_similarstimulusCharacters, training, elec1, showplots, spect )
% show EEG time-electrode-amplitude map in the duration of each stimulus image
% (epoch) in each predefined trial

%%  input
% trial_number
% signal.Characters
% index_images
% signal.fs;
% signal.num_stimulus
% signal.data
% signal.data_forP300classifier;% size:( num_channels, length_epoch, signal.num_stimulus*size(signal.data,3))
% index_similarstimulusCharacters % order of repetition for each of the 9
% stimulus images
% image
% showplots; 1 to show; 
% spect; 1 to show time-electrode color map, 0 to show signals
%% main
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


allCharacters=signal.Characters(1:size(index_images,1),:);
est_y=[];
duration=[1:round(0.9*fs)];
time=duration/fs;
x2=[];
if training
    labels_direction=signal.triallabel_direction;
    labels_direction=labels_direction-1;
end
elec1;
for tr=trial_number
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
        W=msetcca(x(elec1,duration,counter2),1);
        [wx1,wy1,r1]=cca(x(elec1,duration,counter2),W'* x(elec1,duration,counter2) );
        num1=find(r1==max(r1)); num1=num1(1);
        x2(:,duration,counter2)=abs(wx1)'*x(elec1,duration,counter2);
        
        sig=mean(x(:,duration,counter2),1);
        
        spec=x(elec1,duration,counter2); %x2(:,duration,counter2) or x(elec,duration,counter2)
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


end

