function [ signal ] = Preprocessing_RSVPexperiment( signal,landau,myogenicrejection,thresh_EMG,blinkingrejection,filterparams )
fs=signal.fs;

for tr=1:signal.num_trial
    for i=1:size(signal.data,1)-1
        
        sig=signal.data(i,:,tr );
        
        %%%baseline correction%%%
        sig=sig-mean(sig)*ones(size(sig));
        
        if ~ exist('tvd.mexw64')
            mex tvd.c
        end
        
        sig = tvd(sig,length(sig),landau);
        
        % sig=filter(Hd1,sig);
        sig=filtfilt(filterparams.B,filterparams.A,sig);
        
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
        signal.dataLPF(i,:,tr )=A5;
        signal.data(i,:,tr )=sig2;
        
        [delta,theta,alpha,sigma,beta,gama]=filter_freqbands (sig2,fs, filterparams.B1,filterparams.B2,filterparams.B3,filterparams.B4,filterparams.B5,filterparams.B6,filterparams.A1,filterparams.A2,filterparams.A3,filterparams.A4,filterparams.A5,filterparams.A6 );
        % sig2 is not filtered by TVD.
        % sig is filtered by TVD.
        
        signal.delta(i,:,tr )=delta;
        signal.theta(i,:,tr )=theta;
        signal.alpha(i,:,tr )=alpha;
        signal.sigma(i,:,tr )=sigma;
        signal.beta(i,:,tr )=beta;
        signal.gama(i,:,tr)=gama;
        
        
    end
end



end

