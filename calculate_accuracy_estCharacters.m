function [ Accuracy_p300,signal ] = calculate_accuracy_estCharacters( x,y, signal, l, training, index_images )

%%% inputs %%%
% x (l, num_epochs): each column is a feature vector for an epoch (occurance of an stimulus
% image)
% y: vector of  0, 1 labels
% signal.Characters;
% signal.num_trial
%  signal.num_labels % order of happening (repetition) of each "target" stimulus image in each trial
% l: logitboost classifier object 
% training: 0, if you use your pretrained logitboost classifier object  (l) 
% index_images: each row is the constant order of repetiotions of each stimulus image

%%%% output %%%
%  signal.estnum_labels2
% signal.estCharacters % for each trial
 

allCharacters=signal.Characters(1:size(index_images,1),:);
for tr=1:signal.num_trial
signal.mainCharacters(tr,:)= (allCharacters(signal.num_labels(1,tr),:));
end

counttrue3=0; est_y=[];
for num_epoch=1:size(x,2)
    p3=round(classify(l,x(:,num_epoch)) );
    if training
        y3=length(find(p3==y(num_epoch)));
    else
        y3=length(find(p3==1));
    end
    if y3>=.6*length(p3) % iter is number of iterations or answers of logitboost for each input
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
for tr=1: signal.num_trial
    if training
        label=y(1,counter2:counter2+step2-1);
    end
    est_label=est_y(1,counter2:counter2+step2-1);
    counter2=counter2+step2;
    num2=find(est_label==1);
    if ~isempty(num2); num2=num2(1);else num2=1; end
    signal.estnum_labels2(1,tr)=num2;
    signal.estCharacters(tr,:)=allCharacters(num2,:);
sprintf(['trial=',num2str(tr), ', main character: ', signal.mainCharacters(tr,:),  ' ,Estimated character: ', signal.estCharacters(tr,:)])
pause (1);
end

if training
    num_true=0;
    for tr=1:size(signal.estCharacters,1)
        signal.mainCharacters(tr,:)=allCharacters(signal.num_labels(1,tr),:);
        if strcmp(signal.mainCharacters(tr,:),signal.estCharacters(tr,:))
            num_true=num_true+1;
        end
    end
    Accuracy_p300 = num_true/tr ;
    sprintf ([' Detection Accuracy = ', num2str(Accuracy_p300*100), ' %%'])
end



end

