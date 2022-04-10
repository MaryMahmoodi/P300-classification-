function [delta,theta,alpha,sigma,beta,gama]=filter_freqbands (sig,fs, B1, B2, B3, B4,B5,B6,A1,A2,A3,A4,A5,A6 )
% input:
% sig is a row vector signal 
% fs:sampling frequency
% matrix of filter coefficients [B1; B2; B3; B4]

%BP parameters
% d1= fdesign.bandpass('N,Fst1,Fp1,Fp2,Fst2,C',50,1,1.1,30,maxfreq+0.5,fs);%50,0.01,0.16,45,50.5,fs);%36
%     Hd1=design(d1,'equiripple');
if nargin<3

f_delta=[0.5 4]; f_theta=[4 7]; f_alpha=[8 12]; 
f_sigma=[14 16] ;f_beta=[29 31]; f_gama=[44 46]; 
% we used these frequency bands for direction classification in our P300 speller 
    
    % the Parks-McClellan method is used via the ‘remez’ function of MATLAB
% rp = 0.01; % Passband ripple
% rs = 26; % Stopband ripple
% a = [1 0]; % Desired amplitudes
% % Compute deviations
% dev = [(10^(rp/20)-1)/(10^(rp/20)+1) 10^(-rs/20)];
% A=1;
% 
% [n,fo,ao,w] = remezord(f_delta,a,dev,fs); B1 = remez(n,fo,ao,w);
% 
% [n,fo,ao,w] = remezord(f_theta,a,dev,fs);B2 = remez(n,fo,ao,w);
% 
% [n,fo,ao,w] = remezord(f_alpha,a,dev,fs);B3= remez(n,fo,ao,w);
% 
% [n,fo,ao,w] = remezord(f_sigma,a,dev,fs);B4 = remez(n,fo,ao,w);
% 
% [n,fo,ao,w] = remezord(f_beta,a,dev,fs);B5 = remez(n,fo,ao,w);
[B1, A1]=butter(3,f_delta/(fs/2),'bandpass');
[B2, A2]=butter(3,f_theta/(fs/2),'bandpass');
[B3, A3]=butter(3,f_alpha/(fs/2),'bandpass');
[B4, A4]=butter(3,f_sigma/(fs/2),'bandpass');
[B5, A5]=butter(3,f_beta/(fs/2),'bandpass');
[B6, A6]=butter(3,f_gama/(fs/2),'bandpass');

end
% freqz(B1,A);
delta=filter(B1,A1,sig); 
theta=filter(B2,A2,sig);
alpha=filter(B3,A3,sig);
sigma=filter(B4,A4,sig);
beta=filter(B5,A5,sig);
gama=filter(B6,A6,sig);
end