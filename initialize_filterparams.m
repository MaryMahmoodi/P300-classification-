function [ filterparams ] = initialize_filterparams(fs,freqrange,f_delta,f_theta,f_alpha,f_sigma,f_beta,f_gama)
filterparams=[];
% the Parks-McClellan method is used via the ‘remez’ function of MATLAB
rp = 0.01; % Passband ripple
rs = 26; % Stopband ripple
f = freqrange; % Cutoff frequencies
a = [1 0]; % Desired amplitudes
% Compute deviations
dev = [(10^(rp/20)-1)/(10^(rp/20)+1) 10^(-rs/20)];
[n,fo,ao,w] = remezord(f,a,dev,fs);
B = remez(n,fo,ao,w);
A=1;
filterparams.A=A;
filterparams.B=B;
% freqz(B,A);
% Cz1=filter(B,A,Cz);

[B1, A1]=butter(6,f_delta/(fs/2),'bandpass');
[B2, A2]=butter(6,f_theta/(fs/2),'bandpass');
[B3, A3]=butter(6,f_alpha/(fs/2),'bandpass');
[B4, A4]=butter(6,f_sigma/(fs/2),'bandpass');
[B5, A5]=butter(6,f_beta/(fs/2),'bandpass');
[B6, A6]=butter(6,f_gama/(fs/2),'bandpass');

filterparams.A1=A1; filterparams.B1=B1;
filterparams.A2=A2; filterparams.B2=B2;
filterparams.A3=A3; filterparams.B3=B3;
filterparams.A4=A4; filterparams.B4=B4;
filterparams.A5=A5; filterparams.B5=B5;
filterparams.A6=A6; filterparams.B6=B6;




end

