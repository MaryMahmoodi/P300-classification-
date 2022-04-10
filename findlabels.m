function [ signal]=findlabels(signal,num_stimulus)
% input: signal.labels,signal.Characters
% output: signal.num_labels=num_labels;

for i=1:length(signal.labels) %trials labels
    counter=1;
for  j=1:num_stimulus % images
a=[signal.labels(i) signal.labels(i) signal.labels(i)];
binary=a==signal.Characters(j,:);
find1=find(binary==1);
num=length(find(binary==1));
if num>=1; 
    num_labels (counter,i)=j; 
if find1==1;
    direction_labels(counter,i)=['l'];
elseif find1==2;
        direction_labels(counter,i)=['u'];
else
    direction_labels(counter,i)=['r'];
end
counter=counter+1;
end
end
end
signal.num_labels=num_labels;
signal.direction_labels=direction_labels;
end