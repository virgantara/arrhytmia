import librosa
import glob
import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
import pywt
from kymatio.numpy import Scattering1D
import scipy.io.wavfile

def cutoff_frekuensi(cut_off, sr):
    cutOff = cut_off  # Cutoff frequency
    nyq = 0.5 * sr
    N = 6  # Filter order
    fc = cutOff / nyq  # Cutoff frequency normal
    b, a = signal.butter(N, fc)

    return b, a


def filter_signal(data, cut_off, sr):
    b, a = cutoff_frekuensi(cut_off=cut_off, sr=sr)
    filtered_signal = signal.filtfilt(b, a, data)

    return filtered_signal

def mfcc_extract(filename):
    try:

        y, sr = librosa.load(filename, sr=44100)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=int(0.02 * sr), hop_length=int(0.01 * sr))
        mfcc_scaled = np.mean(mfcc,axis=1)

        return mfcc_scaled
    except:
        return



def yadav_feature_extract(filename):
    try:

        y, sr = librosa.load(filename, sr=44100)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid, axis=1)
        spectral_centroid_std = np.std(spectral_centroid, axis=1)

        centroid_ft = np.hstack((spectral_centroid_mean, spectral_centroid_std))

        zcrs = librosa.feature.zero_crossing_rate(y)
        zcrs_mean = np.mean(zcrs,axis=1)
        zcrs_std = np.mean(zcrs, axis=1)
        zcrs_ft = np.hstack((zcrs_mean, zcrs_std))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        spectral_rolloff_mean = np.mean(spectral_rolloff, axis=1)
        spectral_rolloff_std = np.std(spectral_rolloff, axis=1)
        roll_off_ft = np.hstack((spectral_rolloff_mean, spectral_rolloff_std))

        stft = librosa.stft(y=y, n_fft=int(0.02 * sr), hop_length=int(0.01 * sr))
        spectogram = np.abs(stft)
        energy_entropy = entropy(spectogram, base=2)
        energy_entropy = energy_entropy / np.linalg.norm(energy_entropy)
        energy_entropy_mean = np.mean(energy_entropy)
        energy_entropy_std = np.std(energy_entropy)
        energy_ft = np.hstack((energy_entropy_mean, energy_entropy_std))

        hop_length = int(librosa.time_to_samples(1. / 200, sr=sr))
        odf_default = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        spectral_flux_mean = np.mean(odf_default)
        spectral_flux_std = np.std(odf_default)
        spectral_flux_ft = np.hstack((spectral_flux_mean, spectral_flux_std))

        stacked_ft = np.hstack((centroid_ft, zcrs_ft))
        stacked_ft = np.hstack((stacked_ft, roll_off_ft))
        stacked_ft = np.hstack((stacked_ft, energy_ft))
        stacked_ft = np.hstack((stacked_ft, spectral_flux_ft))
        print(stacked_ft.shape)
        return stacked_ft
    except:
        return


def compute_wavelet_features(X):
    # Define a few parameters
    wavelet = 'morl'  # wavelet type: morlet
    sr = 8000  # sampling frequency: 8KHz
    widths = np.arange(1, 256)  # scales for morlet wavelet
    dt = 1 / sr  # timestep difference

    frequencies = pywt.scale2frequency(wavelet, widths) / dt  # Get frequencies corresponding to scales

    # Create a filter to select frequencies between 80Hz and 5KHz
    upper = ([x for x in range(len(widths)) if frequencies[x] > 1000])[-1]
    lower = ([x for x in range(len(widths)) if frequencies[x] < 80])[0]
    widths = widths[upper:lower]  # Select scales in this frequency range

    # Compute continuous wavelet transform of the audio numpy array
    wavelet_coeffs, freqs = pywt.cwt(X, widths, wavelet=wavelet, sampling_period=dt)
    # print(wavelet_coeffs.shape)
    # sys.exit(1)

    # Split the coefficients into frames of length 800
    start = 0
    end = wavelet_coeffs.shape[1]
    frames = []
    frame_size = 400
    count = 0

    while start + frame_size <= end - 1:
        f = wavelet_coeffs[:, start:start + frame_size]

        # Total samples in a frame will not be a multiple of 800 everytime. If the last frame length is less than 800, we can skip it.
        assert f.shape[1] == frame_size  # assert frame lengths are equal to the frame_size parameter

        frames.append(f)
        start += frame_size

    # Convert frames to numpy array
    frames = np.array(frames)
    frames = frames.reshape((len(frames), wavelet_coeffs.shape[0], frame_size))

    return frames

def yaseen_feature_extract(filename):
    try:

        y, sr = librosa.load(filename, sr=44100)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=19, n_fft=int(0.02 * sr), hop_length=int(0.01 * sr))
        mfcc_mean = np.mean(mfcc, axis=1)

        coeffs = pywt.wavedec(data=y, wavelet='db1', level=3)
        cA2, cA1, cD2, cD1 = coeffs
        cA2_mean = np.mean(cA2)
        cA2_std = np.std(cA2)
        cA2_max = np.max(cA2)

        cA1_mean = np.mean(cA1)
        cA1_std = np.std(cA1)
        cA1_max = np.max(cA1)

        cD2_mean = np.mean(cD2)
        cD2_std = np.std(cD2)
        cD2_max = np.max(cD2)

        cD1_mean = np.mean(cD1)
        cD1_std = np.std(cD1)
        cD1_max = np.max(cD1)

        dwt_ft = np.hstack((
            cA2_mean, cA2_std, cA2_max,cA1_mean, cA1_std, cA1_max,
            cD2_mean, cD2_std, cD2_max, cD1_mean, cD1_std, cD1_max
        ))

        stacked_ft = np.hstack((mfcc_mean, dwt_ft))
        # print(stacked_ft.shape)
        return stacked_ft
    except Exception as e:
        print(e)
        return

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

def ghosh_feature_extract(filename):
    try:

        y, sr = librosa.load(filename, sr=44100)

        stft = librosa.stft(y=y, n_fft=int(0.02 * sr), hop_length=int(0.01 * sr))
        spectogram = np.abs(stft)
        energy_entropy = entropy(spectogram, base=2)
        energy_entropy = energy_entropy / np.linalg.norm(energy_entropy)
        entrop = np.array(energy_entropy[:100])
        print("Entropy",entrop.shape)
        r_c = gaussian_chirplet(y)
        rc_real = r_c.real
        rc_real = np.array(rc_real[:200])
        print("RC", rc_real.shape)
        stacked_ft = np.hstack((rc_real, entrop))
        print("Stacked",stacked_ft.shape)
        return stacked_ft
    except Exception as e:
        print(e)
        return

def ensemble_feature_extract(filename):
    try:

        y, sr = librosa.load(filename, sr=44100)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=int(0.02 * sr), hop_length=int(0.01 * sr))
        mfcc_scaled = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_varians = np.var(mfcc,axis=1)

        ensembled_feature = np.append(mfcc_scaled, mfcc_std)
        ensembled_feature = np.append(ensembled_feature, mfcc_varians)
        zcrs = librosa.feature.zero_crossing_rate(y)
        zcrs = np.mean(zcrs,axis=1)
        ensembled_feature = np.append(ensembled_feature, zcrs)

        chroma_stft = librosa.feature.chroma_stft(y=y,sr=sr)
        chroma_stft = np.mean(chroma_stft, axis=1)
        ensembled_feature = np.append(ensembled_feature, chroma_stft)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid = np.mean(spectral_centroid, axis=1)
        ensembled_feature = np.append(ensembled_feature, spectral_centroid)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth = np.mean(spectral_bandwidth, axis=1)
        ensembled_feature = np.append(ensembled_feature, spectral_bandwidth)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff = np.mean(spectral_rolloff, axis=1)
        ensembled_feature = np.append(ensembled_feature, spectral_rolloff)

        return ensembled_feature
    except:
        return

def stft_extract(filename):
    try:
        y, sr = librosa.load(filename, sr=44100)
        stft = librosa.stft(y=y, n_fft=int(0.02 * sr), hop_length=int(0.01 * sr))

        spectogram = np.abs(stft)
        return spectogram
    except:
        return


def mei_feature_extract(filename):
    try:

        sr, x = scipy.io.wavfile.read(filename)
        x = x / np.max(np.abs(x))

        T = x.shape[-1]
        # print(2**13)
        J = 8
        Q = 12
        scattering = Scattering1D(J, T, Q)
        Sx = scattering(x)
        meta = scattering.meta()
        order0 = np.where(meta['order'] == 0)
        order1 = np.where(meta['order'] == 1)
        order2 = np.where(meta['order'] == 2)
        print("0-ord shape", Sx[order0].shape)
        print("1-ord shape", Sx[order1].shape)
        print("2-ord shape", Sx[order2].shape)
        feat = np.vstack((Sx[order0], Sx[order1], Sx[order2]))
        feat = feat[:, 0:10]
        feat = np.mean(feat, axis=0)
        print(feat.shape)

        return feat
    except Exception as e:
        print(e)
        return

def parse_audio_files(parent_dir, sub_dirs, limit):
    labels = []
    features = []
    for label, sub_dir in enumerate((sub_dirs)):
        i = 0
        for fn in glob.glob(os.path.join(parent_dir,sub_dir,"*.wav")):
            # if i >= limit:
            #     break
            features.append(mfcc_extract(fn))
            labels.append(label)
            i+=1
    return features, labels

def write_our_features(parent_dir, label):

    features = []
    # features.append('Class')

    for fn in glob.glob(os.path.join(parent_dir,"*.wav")):
        print(fn)
        # ft = mfcc_extract(fn)
        ft = ensemble_feature_extract(fn)

        features.append(ft)

    file = open('features_ensemble_mean_std'+label+'.csv','w+')
    #
    np.savetxt(file, features,delimiter=',')

def write_yadav_features(parent_dir, label, dataset_name = 'Yaseen'):

    features = []
    # features.append('Class')

    for fn in glob.glob(os.path.join(parent_dir,"*.wav")):
        print(fn)
        # ft = mfcc_extract(fn)
        ft = yadav_feature_extract(fn)
        # print(ft.shape)
        features.append(ft)

    # file = open('yadav_features_'+label+'.csv','w+')
    features = np.array(features, dtype='float32')
    df = pd.DataFrame(features)
    df.to_csv('yadav_features_'+dataset_name+'_'+label+'.csv', index=False,header=False)
    # np.savetxt(file, features,delimiter=',')

def write_yaseen_features(parent_dir, label):

    features = []
    # features.append('Class')

    for fn in glob.glob(os.path.join(parent_dir,"*.wav")):
        print(fn)
        # ft = mfcc_extract(fn)
        ft = yaseen_feature_extract(fn)
        features.append(ft)

    # file = open('yadav_features_'+label+'.csv','w+')
    features = np.array(features, dtype='float32')
    df = pd.DataFrame(features)
    df.to_csv('yaseen_features_'+label+'.csv', index=False,header=False)
    # np.savetxt(file, features,delimiter=',')


def write_mei_features(parent_dir, label):

    features = []
    # features.append('Class')

    for fn in glob.glob(os.path.join(parent_dir,"*.wav")):
        print(fn)
        # ft = mfcc_extract(fn)
        ft = mei_feature_extract(fn)
        features.append(ft)

    # file = open('yadav_features_'+label+'.csv','w+')
    features = np.array(features, dtype='float32')
    df = pd.DataFrame(features)
    df.to_csv('mei_features_'+label+'.csv', index=False,header=False)

def write_ghosh_features(parent_dir, label, dataset_name = 'Yaseen'):

    features = []
    # features.append('Class')

    for fn in glob.glob(os.path.join(parent_dir,"*.wav")):
        print(fn)
        # ft = mfcc_extract(fn)
        ft = ghosh_feature_extract(fn)
        # print(ft.shape)
        features.append(ft)

    # file = open('yadav_features_'+label+'.csv','w+')
    # features = np.array(features)
    df = pd.DataFrame(features)
    df.to_csv('ghosh_features_'+dataset_name+'_'+label+'.csv', index=False,header=False)
#
# test = ['atr'+str(i+1) for i in range(13)]
# # test.append(test)
# test.append(['z'+str(i+1) for i in range(13)])
# print(test)