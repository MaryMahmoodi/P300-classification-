function sig=myogenic_rejection(sig, fs, step,thresh_EMG)
% step(in seconds) is duration of myogenic detection and rejection by random
%numbers
% fs: sampling frequency
% thresh_EMG: threshold (uV) for myogenic artefact detection 
%%%%myogenic artefact rejection%%%%%%%
step=round(step*fs);

for j=1:step:length(sig)-step
        if j+step-1<length(sig)-step
        
        if max(abs(sig(1,j:j+step-1)))>=thresh_EMG
            sig(1,j:j+step-1)=rand(size(sig(1,j:j+step-1)));
            
        end
        
    end
end



if j<length(sig)
    if max(abs(sig(1,j:end)))>=thresh_EMG
        sig(1,round(j-0*fs):end)=rand(size( sig(1,round(j-0*fs):end)));
        
    end
end


end