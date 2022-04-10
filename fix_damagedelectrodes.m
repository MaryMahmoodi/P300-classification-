function [ signal ] = fix_damagedelectrodes( signal,Rereference )
% find damaged signals and Rereference
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
        if max(signal.data (i,:,tr))<1 %damaged electrode
            signal.data (i,:,tr)=avgsignal(1,:,tr);
        else
            % Rereference all other electrodes to the average signal of occipital electrodes
            if Rereference
                signal.data (i,:,tr)=signal.data (i,:,tr)-avgsignal(1,:,tr);
            end
        end
        
    end
end


end

