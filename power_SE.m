function [power,SE1]=power_SE(signal,fs,freqrange)
% calculate power and spectral entropy
fftlength=fs;
freq=0:fs/fftlength:fs/2;
xdft=fft(signal,fftlength);
xdft=abs(xdft);
xdft=xdft.^2;
xdft1=(real(xdft));
normalized_xdft1=xdft1.*(1/sum(xdft1)) ;
SE1 = (-1/(length(freq)))*sum( (normalized_xdft1) .*log10(normalized_xdft1) ) ;

a=freqrange(1);
b=freqrange(2);

num1=find(freq>=a);
num2=find(freq<=b);
power=sum(xdft(num1(1):num2(end)));




end