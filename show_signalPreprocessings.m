function  show_signalPreprocessings( tr, signal, data )
figure(5); clf;
% tr=4;
fs=signal.fs;
time=[1:size(signal.data,2)]./fs;
orgsig=data.TrainData([signal.num_Cz] ,:,tr );
sig_denoised=signal.data([signal.num_Cz] ,:,tr );
mrk=signal.markersignal(:,tr);
sig_LPF=signal.dataLPF([signal.num_Cz] ,:,tr );
plot(time,orgsig,'b');hold on; plot(time,5*mrk,'r');text (1,7,'Raw signal and markers of target stimulus (P300 duration)')
hold on; plot(time, sig_denoised-20,'b'); text (1, -20,'denoised signal');
hold on; plot(time, sig_LPF-40,'b'); ylim([-50 40]); text(1,-40, 'lowpass filtered signal');
xlabel ('time (s)'); ylabel('Amplitude (\muV)');


end

