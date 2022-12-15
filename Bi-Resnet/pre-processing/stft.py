# -*- coding: utf-8 -*-
import scipy.io.wavfile as wav
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display as display
from scipy.signal import butter, lfilter  
            
def Normalization(x):
    x = x.astype(float)
    max_x = max(x)
    min_x = min(x)
    for i in range(len(x)):
        
        x[i] = float(x[i]-min_x)/(max_x-min_x)
           
    return x

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
 
 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def save_pic(wav_dir,save_dir):
    txt_name = ''
    txt_dir ='../data/ICBHI/'
    for file in os.listdir(wav_dir):
        num = file[-5]
        if file[:22]!=txt_name[:-4]:
            txt_name = file[:22]+'.txt'
            array = np.loadtxt(txt_dir+txt_name)
            label = array[:,2:4]
            
        sig,fs= librosa.load(wav_dir+'/'+file)
        sig = Normalization(sig)
        if fs>4000:
            sig = butter_bandpass_filter(sig, 1, 4000, fs, order=3)
        stft = librosa.stft(sig, n_fft=int(0.02*fs), hop_length=int(0.01*fs), window='hann')
        if fs>4000:
            display.specshow(librosa.amplitude_to_db(stft[0:int(len(stft)/2),:],ref=np.max),y_axis='log',x_axis='time')
        else:
            display.specshow(librosa.amplitude_to_db(stft,ref=np.max),y_axis='log',x_axis='time')

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)

        crackles = label[int(num),0]
        wheezes = label[int(num),1]
        if crackles==0 and wheezes==0:
            plt.savefig(save_dir+'zero/'+file[:24]+'png', cmap='Greys_r')
        elif crackles==1 and wheezes==0:
            plt.savefig(save_dir+'one/'+file[:24]+'png', cmap='Greys_r')
        elif crackles==0 and wheezes==1:
            plt.savefig(save_dir+'two/'+file[:24]+'png', cmap='Greys_r')
        else:
            plt.savefig(save_dir+'three/'+file[:24]+'png', cmap='Greys_r')
        plt.clf()
        
        
if __name__ == '__main__':
   save_pic('../data/train','../analysis/stft/train/')
   save_pic('../data/test','../analysis/stft/test/')
   
