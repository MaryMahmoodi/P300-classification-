function [ signal, spikes,spikes_index ] = blinking_rejection( signal,fs )
% spike blinking detection by thresholding and rejection by interpolation 
if size(signal,1)>1; signal=signal';end

% spike blinking rejection
threshold=3;%3.5*(1/length(signal)*sum(abs(signal)));
coef=0.5;

spikes=zeros(size(signal));

    for i=1:length(signal)
            if signal(i)>threshold
                spikes(i)=signal(i);
            elseif (signal(i)>-threshold) && (signal(i)<threshold)
                                spikes(i)=0;
            elseif  signal(i)<-threshold
                                spikes(i)=signal(i);
            end

    end

org_spikes=spikes;
binary =abs(spikes)>0;
% find consequtive detected SPIKE blinking samples
E = binary(2:end)-binary(1:end-1);
sise = size(binary);

begins = find(E==1)+1;

if binary(1) == 1
    if sise(1) > 1
        begins = [1; begins];
    elseif sise(2) > 1
        begins = [1 begins];
    else
        error('The input signal is not one dimensional')
    end
elseif numel(begins) == 0 && binary(1) == 0
    begins = NaN;
end

ends = find(E==-1);
if binary(end) == 1
    if sise(1) > 1
        ends = [ends; length(binary)];
    elseif sise(2) > 1
        ends = [ends length(binary)];
    else
        error('The input signal is not one dimensional')
    end
elseif numel(ends) == 0 && binary(end) == 0
    ends = NaN;
end

counter=1; spikes_index=[];
if ~isnan(begins)
for i=1:length(begins)
    if ends(i)-begins(i)>1*fs
        spikes(begins(i):ends(i))=zeros(size(  spikes(begins(i):ends(i))));
    else
        if max(abs(spikes(begins(i):ends(i))))>=threshold%45
        spikes_index(1,counter)=round( (begins(i)+ends(i))/2);
  counter=counter+1;
        else
                    spikes(begins(i):ends(i))=zeros(size(  spikes(begins(i):ends(i))));

        end

    end
  
end
end
% % % coef=0.1;
if ~ isempty(spikes_index)
for i=1:length(spikes_index)
if  i<length(spikes_index)  &&  spikes_index(i+1)-spikes_index(i)<=round(coef*fs)
    spikes_index(i+1)=spikes_index(i);
end
if           spikes_index(i)- round(coef*fs)>0
ss=round(spikes_index(i)- round(coef*fs));
      if ss<0; ss= spikes_index(i)- round(coef*fs)+ss+1;end
                  st=round(spikes_index(i)+ round(coef*fs));
                  if st>length(signal(1,:)); st=length(signal(1,:));end
                 
                  a1=ss-length(signal(1,ss:st))+1;
                  if a1<0; 
                      st=st+a1; 
                                    a1=ss-length(signal(1,ss:st))+1+1;
                                                 signal(1,ss:st)=[0 signal(1,a1:ss)]+(signal(1,st)-signal(1,ss))*ones(1,length(signal(1,ss:st)));%interp1q(a',signal(i-fs/2+1:i)',b') ;

                  else
                      if a1==0; a1=1; ss=ss+1; st=st+1;end
                                signal(1,ss:st)=signal(1,a1:ss)+(signal(1,st)-signal(1,ss))*ones(1,length(signal(1,ss:st)));%interp1q(a',signal(i-fs/2+1:i)',b') ;

                  end

                  
end
      end
end


fprintf ('eye blink removed...\n')

% %%%%myogenic artifact rejection%%%%%%%
% for i=1:2*fs:length(signal)-2*fs
%     if i+2*fs-1<length(signal)-2*fs
% 
%     if max(abs(signal(1,i:i+2*fs-1)))>=300%% median(Cz2(1,i:i+fs-1))>35
%         signal(1,i:i+2*fs-1)=0;
%          
%     end
%               
%             end
% end



end

