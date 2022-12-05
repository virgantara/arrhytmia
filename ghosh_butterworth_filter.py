import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.stats import entropy
import glob
import os
import librosa, librosa.display
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.signal import hilbert
from scipy.fftpack import fft, fftfreq
from cmath import phase

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def ekstrak():
    path = 'dataset/Yaseen'
    dirs = ['AS','MR','MS','N']

    for dir in dirs:


        dir_input = 'dataset/Yaseen/' + dir
        dir_output = 'dataset/Ghosh/' + dir
        dir_output_filtered = 'dataset/GhoshFiltered/'+dir

        nomor_file = 1
        for fn in glob.glob(os.path.join(dir_input, "*.wav")):
            print(fn)
            myaudio = AudioSegment.from_file(fn, "wav")
            durasi = int(myaudio.duration_seconds * 1000)
            chunk_length_ms = int(durasi / 3)  # pydub calculates in millisec
            print("Durasi",durasi)
            print("Chunk Length: ",chunk_length_ms)
            chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of one sec

            # Export all of the individual chunks as wav files

            for i, chunk in enumerate(chunks):
                chunk_name = dir_output+"/chunk_"+str(nomor_file)+"_"+str(i)+".wav"
                print("exporting", chunk_name)
                chunk.export(chunk_name, format="wav")

            nomor_file = nomor_file + 1

        nomor_file = 1
        for fn in glob.glob(os.path.join(dir_output, "*.wav")):
            print(fn)

            # open the audio file and extract some information
            spf = wave.open(fn, 'r')
            (nChannels, sampWidth, sampleRate, nFrames, compType, compName) = spf.getparams()

            # extract audio from wav file
            input_signal = spf.readframes(-1)
            input_signal = np.fromstring(input_signal, 'int16')
            spf.close()

            fs = 8000.0
            lowcut = 25.0
            highcut = 900.0

            # apply filter
            # output_signal = signal.filtfilt(b, a, input_signal)

            # for order in [3]:
            order = 5
            output_signal = butter_bandpass_filter(input_signal, lowcut, highcut, fs, order=order)

            # ceate output file
            filename = dir + "_" + str(nomor_file) + "_" + str(order) + ".wav"
            wav_out = wave.open(dir_output_filtered + "/" + filename, "w")
            wav_out.setparams((nChannels, sampWidth, sampleRate, nFrames, compType, compName))

            print("Writing frames")
            # write to output file
            wav_out.writeframes(output_signal.tobytes())
            wav_out.close()
            nomor_file = nomor_file + 1
        # # plot the signals
        # t = numpy.linspace(0, nFrames/sampWidth, nFrames, endpoint = False)
        # plt.plot(t, input_signal, label='Input')
        # plt.plot(t, output_signal, label='Output')
        # plt.show()


def gaussian_chirplet(t, alpha1=1., alpha2=0., beta=2., fc=1., phi=0., tau=0.):
    """Gaussian chirplet function given input parameters
    Keyword arguments:
    t -- time vector
    alpha1 -- bandwidth factor    alpha2 -- chirp-rate
    beta -- amplitude
    fc -- center frequency
    phi -- phase
    tau -- time of arrival
    Return values:
    y -- chirplet
    Reference: Yufeng Lu, Ramazan Demirli, Guilherme Cardosa, and Jafar Saniie,
    "A Successive Parameter estimation Algorithm for Chirplet Signal
    Decomposition," IEEE Trans. UFFC, vol 53, no 11, November 2006.
    """
    y = beta * np.exp(-alpha1 * (t - tau) ** 2 + 1j * 2 * np.pi * fc * (t - tau)
                      + 1j * phi + 1j * alpha2 * (t - tau) ** 2)

    return y

y, sr = librosa.load('dataset/Ghosh/N/chunk_1_0.wav')

# stft = librosa.stft(y=y, n_fft=int(0.02 * sr), hop_length=int(0.01 * sr))
# spectogram = np.abs(stft)
# energy_entropy = entropy(spectogram, base=2)
# energy_entropy = energy_entropy / np.linalg.norm(energy_entropy)
# print(energy_entropy.shape)

r_c = gaussian_chirplet(y)
# r_c = r_c[:200]
# y = y[:200]
# print(r_c.real)

plt.figure(figsize=(20,10))
# plt.plot(y,r_c.real)
plt.xlabel('Time',fontsize=20)
plt.ylabel('Amplitude',fontsize=20)
plt.show()
