import librosa
import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sklearn

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

class jrock:
    def __init__(self, path1, path2):
        self.path1 = path1
        self.path2 = path2

        audio_file_1, self.sr_1 = librosa.load(f'{path1}')
        self.audio_file1, self.index_1 = librosa.effects.trim(audio_file_1)
        audio_file_2, self.sr_2 = librosa.load(f'{path2}')
        self.audio_file2, self.index_2 = librosa.effects.trim(audio_file_2)

        spectral_centroids_f1 = librosa.feature.spectral_centroid(y=self.audio_file1, sr=self.sr_1)[0]
        frames_f1 = range(len(spectral_centroids_f1))
        self.t1 = librosa.frames_to_time(frames_f1)
        spectral_centroids_f2 = librosa.feature.spectral_centroid(y=self.audio_file2, sr=self.sr_2)[0]
        frames_f2 = range(len(spectral_centroids_f2))
        self.t2 = librosa.frames_to_time(frames_f2)

    def mfcc(self):
        # Compute MFCCs for each audio file
        mfcc_1 = librosa.feature.mfcc(y=self.audio_file1, sr=self.sr_1)
        mfcc_2 = librosa.feature.mfcc(y=self.audio_file2, sr=self.sr_2)

        # Compute the absolute difference between the MFCCs
        shape1 = np.shape(mfcc_1)[1]
        shape2 = np.shape(mfcc_2)[1]
        shape = shape1 if shape1 < shape2 else shape2

        mfcc_diff = np.abs(mfcc_1[:, :shape] - mfcc_2[:, :shape])

        fig = plt.figure()

        # Plot the difference between the MFCCs
        plt.imshow(mfcc_diff, cmap='hot', interpolation='nearest')
        plt.xlabel('Frames')
        plt.ylabel('MFCC Coefficients')
        plt.title('Difference between MFCCs of the two audio files')
        plt.colorbar()
        
        return fig
    
    def melspec(self):
        # Compute the Mel spectrogram
        mel_spec1 = librosa.feature.melspectrogram(y=self.audio_file1, sr=self.sr_1)
        mel_spec2 = librosa.feature.melspectrogram(y=self.audio_file2, sr=self.sr_2)

        fig = plt.subplots(2, 1)

        plt.subplot(211)
        librosa.display.specshow(librosa.power_to_db(mel_spec1, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        plt.title(f'Mel spectrogram of {self.path1[-12:]}')
        plt.tight_layout()

        plt.subplot(212)
        librosa.display.specshow(librosa.power_to_db(mel_spec2, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        plt.title(f'Mel spectrogram of {self.path2[-10:]}')
        plt.tight_layout()

        return fig
        
    def centroid(self):
        # Compute the spectral centroid for each audio file
        spectral_centroid_1 = librosa.feature.spectral_centroid(y=self.audio_file1, sr=self.sr_1)[0]
        spectral_centroid_2 = librosa.feature.spectral_centroid(y=self.audio_file2, sr=self.sr_2)[0]
        min_length = min(len(spectral_centroid_1), len(spectral_centroid_2))
        shortened_centroid_1 = spectral_centroid_1[:min_length]
        shortened_centroid_2 = spectral_centroid_2[:min_length]

        # Create a new figure and plot the spectral centroid for audio file 1
        fig = plt.subplots(figsize=(14, 6))
        gs = gridspec.GridSpec(3, 4, height_ratios=[3, 3, 4])
        gs.update(wspace=1)
        gs.update(hspace=1)

        ax1 = plt.subplot(gs[0, 1:3])
        img1 = librosa.display.waveshow(self.audio_file1, sr=self.sr_1, color='orange', label=f'{self.path1[-12:]}')
        img2 = librosa.display.waveshow(self.audio_file2, sr=self.sr_2, alpha=0.75, color='g', label=f'{self.path2[-10:]}')
        ax1.set_title(f'Audio Files')
        ax1.set_ylabel('Amplitude')
        ax1.set_xlabel('Time (seconds)')

        ax2 = plt.subplot(gs[1, :2])
        img3 = librosa.display.waveshow(y=self.audio_file1, sr=self.sr_1, alpha=0.4, ax=ax2)
        ax2.plot(self.t1, normalize(spectral_centroid_1), color='orange', label=f'{self.path1[-12:]}')
        ax2.set_title(f'Spectral Centroids')
        ax2.set_ylabel('Amplitude')
        ax2.set_xlabel('Time (seconds)')
        ax2.legend()

        # Plot the spectral centroid for audio file 2 in the same figure
        ax3 = plt.subplot(gs[1, 2:])
        img4 = librosa.display.waveshow(y=self.audio_file2, sr=self.sr_2, alpha=0.4, ax=ax3)
        ax3.plot(self.t2, normalize(spectral_centroid_2), color='g', label=f'{self.path2[-10:]}')
        ax3.set_title(f'Spectral Centroids')
        ax3.set_ylabel('')
        ax3.set_xlabel('Time (seconds)')
        ax3.legend()

        ax4 = plt.subplot(gs[2, 1:3])
        ax4.plot(shortened_centroid_1, color='orange', label=f'{self.path1[-12:]}')
        ax4.plot(shortened_centroid_2, color='g', label=f'{self.path2[-10:]}')
        ax4.set_title(f'Spectral Centroids Comparison')
        ax4.set_ylabel('Hz')
        ax4.set_xlabel('Time (seconds)')
        ax4.legend()

        return fig
    
    def hpss_helper(self, audio_file):
        stft = librosa.stft(audio_file)
        magnitude, phase = librosa.magphase(stft)
        harmonic, percussive = librosa.decompose.hpss(magnitude)

        return magnitude, harmonic, percussive
    
    def hpss(self):

        magnitude, harmonic, percussive = self.hpss_helper(self.audio_file1)
        magnitude1, harmonic1, percussive1 = self.hpss_helper(self.audio_file2)

        fig, ax = plt.subplots(3,2)

        plt.subplot(321)
        librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Original {self.path1[-12:]}')

        plt.subplot(323)
        librosa.display.specshow(librosa.amplitude_to_db(harmonic, ref=np.max), y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Harmonic {self.path1[-12:]}')

        plt.subplot(325)
        librosa.display.specshow(librosa.amplitude_to_db(percussive, ref=np.max), y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Percussive {self.path1[-12:]}')

        plt.subplot(322)
        librosa.display.specshow(librosa.amplitude_to_db(magnitude1, ref=np.max), y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Original {self.path2[-10:]}')

        plt.subplot(324)
        librosa.display.specshow(librosa.amplitude_to_db(harmonic1, ref=np.max), y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Harmonic {self.path2[-10:]}')

        plt.subplot(326)
        librosa.display.specshow(librosa.amplitude_to_db(percussive1, ref=np.max), y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Percussive {self.path2[-10:]}')

        return fig
    
    def rolloff_helper(self, audio, sr):
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.99)
        rolloff_min = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.01)
        S, phase = librosa.magphase(librosa.stft(audio))

        return rolloff, rolloff_min, S

    def rolloff(self):
        fig, ax = plt.subplots(2, 1)

        rolloff, rolloff_min, S = self.rolloff_helper(self.audio_file1, self.sr_1)
        rolloff1, rolloff_min1, S1 = self.rolloff_helper(self.audio_file2, self.sr_2)

        plt.subplot(211)
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
        plt.plot(librosa.times_like(rolloff), rolloff[0], label='Roll-off frequency (0.99)')
        plt.plot(librosa.times_like(rolloff), rolloff_min[0], color='w', label='Roll-off frequency (0.01)')
        plt.legend(loc='lower right')
        ax1 = plt.subplot(211)
        ax1.set_title(f'log Power spectrogram {self.path1[-12:]}')

        plt.subplot(212)
        librosa.display.specshow(librosa.amplitude_to_db(S1, ref=np.max), y_axis='log', x_axis='time')
        plt.plot(librosa.times_like(rolloff1), rolloff1[0], label='Roll-off frequency (0.99)')
        plt.plot(librosa.times_like(rolloff1), rolloff_min1[0], color='w', label='Roll-off frequency (0.01)')
        plt.legend(loc='lower right')
        ax2 = plt.subplot(212)
        ax2.set_title(f'log Power spectrogram {self.path2[-10:]}')

        return fig
    
    def contrast_helper(self, y, sr):
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

        return S, contrast

    def contrast(self):
        fig, ax = plt.subplots(2,2)

        S, contrast = self.contrast_helper(self.audio_file1, self.sr_1)
        S1, contrast1 = self.contrast_helper(self.audio_file2, self.sr_2)

        img1 = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax[0, 0])
        fig.colorbar(img1, ax=[ax[0, 0]], format='%+2.0f dB')
        ax[0, 0].set(title=f'Power spectrogram of {self.path1[-12:]}')
        ax[0, 0].label_outer()
        img2 = librosa.display.specshow(contrast, x_axis='time', ax=ax[1, 0])
        fig.colorbar(img2, ax=[ax[1, 0]])
        ax[1, 0].set(ylabel='Frequency bands', title=f'Spectral contrast of {self.path1[-12:]}')

        img3 = librosa.display.specshow(librosa.amplitude_to_db(S1, ref=np.max), y_axis='log', x_axis='time', ax=ax[0, 1])
        fig.colorbar(img3, ax=[ax[0, 1]], format='%+2.0f dB')
        ax[0, 1].set(title=f'Power spectrogram of {self.path2[-10:]}')
        ax[0, 1].label_outer()
        img4 = librosa.display.specshow(contrast1, x_axis='time', ax=ax[1, 1])
        fig.colorbar(img4, ax=[ax[1, 1]])
        ax[1, 1].set(ylabel='Frequency bands', title=f'Spectral contrast of {self.path2[-10:]}')

        return fig
    
    def tonnetz_helper(self, y, sr):
        hamonic_y = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=hamonic_y, sr=sr)

        return hamonic_y, tonnetz
    
    def tonnetz(self):

        harmonic_y, tonnetz = self.tonnetz_helper(self.audio_file1, self.sr_1)
        harmonic_y1, tonnetz1 = self.tonnetz_helper(self.audio_file2, self.sr_2)

        fig, ax = plt.subplots(2, 2)

        img1 = librosa.display.specshow(tonnetz, y_axis='tonnetz', x_axis='time', ax=ax[0, 0])
        ax[0, 0].set(title=f'Tonal Centroids (Tonnetz) of {self.path1[-12:]}')
        ax[0, 0].label_outer()
        img2 = librosa.display.specshow(librosa.feature.chroma_cqt(y=harmonic_y, sr=self.sr_1), y_axis='chroma', x_axis='time', ax=ax[1, 0])
        ax[1, 0].set(title=f'Chroma of {self.path1[-12:]}')
        fig.colorbar(img1, ax=[ax[0, 0]])
        fig.colorbar(img2, ax=[ax[1, 0]])

        img3 = librosa.display.specshow(tonnetz1, y_axis='tonnetz', x_axis='time', ax=ax[0, 1])
        ax[0, 1].set(title=f'Tonal Centroids (Tonnetz) of {self.path2[-10:]}')
        ax[0, 1].label_outer()
        img4 = librosa.display.specshow(librosa.feature.chroma_cqt(y=harmonic_y1, sr=self.sr_2), y_axis='chroma', x_axis='time', ax=ax[1, 1])
        ax[1, 1].set(title=f'Chroma of {self.path2[-10:]}')
        fig.colorbar(img3, ax=[ax[0, 1]])
        fig.colorbar(img4, ax=[ax[1, 1]])

        return fig
    
    def fft_func(self):
    # Open the first WAV file and extract audio data
        with wave.open(self.path1, 'rb') as wav_file:
            # Extract parameters from WAV file
            sample_rate = wav_file.getframerate()
            num_samples = wav_file.getnframes()
            duration = num_samples / float(sample_rate)

            # Read in audio data and convert to array
            audio_data1 = np.frombuffer(wav_file.readframes(num_samples), dtype=np.int16)

        # Open the second WAV file and extract audio data
        with wave.open(self.path2, 'rb') as wav_file:
            # Extract parameters from WAV file
            sample_rate = wav_file.getframerate()
            num_samples = wav_file.getnframes()
            duration = num_samples / float(sample_rate)

            # Read in audio data and convert to array
            audio_data2 = np.frombuffer(wav_file.readframes(num_samples), dtype=np.int16)

        # Perform DFT on audio data
        dft_data1 = np.fft.fft(audio_data1)
        dft_data2 = np.fft.fft(audio_data2)

        # Calculate frequencies and magnitudes
        frequencies1 = np.fft.fftfreq(len(dft_data1)) * sample_rate
        magnitudes1 = np.abs(dft_data1)
        frequencies2 = np.fft.fftfreq(len(dft_data2)) * sample_rate
        magnitudes2 = np.abs(dft_data2)

        # Plot the spectra
        fig, ax = plt.subplots()
        ax.plot(frequencies1[:int(len(frequencies1) / 2)], magnitudes1[:int(len(magnitudes1) / 2)], label='Медиатор')
        ax.plot(frequencies2[:int(len(frequencies2) / 2)], magnitudes2[:int(len(magnitudes2) / 2)], alpha=0.5,
                label='Пальцы')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title('DFT Spectrum Comparison')
        ax.legend()
        
        return fig

    def chromagram_func(self):
        chroma1 = librosa.feature.chroma_stft(y=self.audio_file1, sr=self.sr_1)
        chroma2 = librosa.feature.chroma_stft(y=self.audio_file2, sr=self.sr_2)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(16, 5))

        # Plot chromagram for audio file 1
        img1 = librosa.display.specshow(chroma1, y_axis='chroma', x_axis='time', ax=ax1)
        ax1.set(title='Chromagram fingerstyle')

        # Plot chromagram for audio file 2
        img2 = librosa.display.specshow(chroma2, y_axis='chroma', x_axis='time', ax=ax2)
        ax2.set(title='Chromagram pick')

        plt.subplots_adjust(hspace=0.5, right=1)

        # Add colorbar to both subplots
        fig.colorbar(ax1.collections[0], ax=[ax1, ax2]) # , format='%+2.0f dB' - шкалирование странное

        # Set labels
        plt.xlabel('Time (s)')

        # Adjust layout and display the plot
        return fig

    '''Zero-Crossing Rate'''
    def zcr_func(self):
        n0 = 6000
        n1 = 7000
        zrc_total_f1 = librosa.feature.zero_crossing_rate(self.audio_file1 + 0.0001, pad=False)# проверить еще раз смысл pad=False
        zrc_total_f2 = librosa.feature.zero_crossing_rate(self.audio_file2 + 0.0001, pad=False)
        zrc_interval_f1 = librosa.zero_crossings(self.audio_file1[n0:n1] + 0.0001, pad=False)
        zrc_interval_f2 = librosa.zero_crossings(self.audio_file2[n0:n1] + 0.0001, pad=False)

        print(self.audio_file1.shape)

        fig = plt.subplots(figsize=(14, 5))
        gs = gridspec.GridSpec(4, 4)
        # gs.update(wspace=0.5)

        ax1 = plt.subplot(gs[0, :2])
        img1 = librosa.display.waveshow(y=self.audio_file1, sr=self.sr_1, ax=ax1, )
        ax1.set_title(f'Waveplot fingerstyle (zero-crossing rate = {zrc_total_f1.sum()})')

        ax2 = plt.subplot(gs[0, 2:])
        img2 = librosa.display.waveshow(y=self.audio_file2, sr=self.sr_2, ax=ax2, color='green')
        ax2.set_title(f'Waveplot pick (zero-crossing rate = {zrc_total_f2.sum()})')

        major_ticks = np.arange(n0, n1, 100)

        ax3 = plt.subplot(gs[1, :2])
        ax3.plot(self.audio_file1[n0:n1])
        ax3.set_xticks([i for i in range(0, 1000, 100)], major_ticks)
        ax3.grid()
        ax3.set_title(f'Waveplot fingerstyle zoomed (zero-crossing rate = {zrc_interval_f1.sum()})')
        # ax3.set_xlabel('Time') # в чём измеряется время?
        ax3.set_ylabel('Amplitude')

        ax4 = plt.subplot(gs[1, 2:])
        ax4.plot(self.audio_file2[n0:n1], color='green')
        ax4.grid()
        ax4.set_title(f'Waveplot pick zoomed (zero-crossing rate = {zrc_interval_f2.sum()})')
        # ax4.set_xlabel('Time')
        ax4.set_ylabel('Amplitude')

        ax5 = plt.subplot(gs[2, :2])
        ax5.plot(zrc_total_f1[0])
        ax5.plot(zrc_total_f1[0])
        ax5.grid()
        ax5.set_title('Zero-cross rate fingerstyle')

        ax6 = plt.subplot(gs[2, 2:], sharey=ax5)
        ax6.plot(zrc_total_f2[0])
        ax6.plot(zrc_total_f2[0], color='green')
        ax6.grid()
        ax6.set_title('Zero-cross rate pick')

        ax7 = plt.subplot(gs[3, 1:3])
        ax7.plot(zrc_total_f1[0])
        ax7.plot(zrc_total_f2[0], color='green')
        ax7.legend() # add legend
        ax7.set_title('Zero-cross rate comparison')

        plt.tight_layout()
        
        return fig

    def spect_bendwidth_func(self, _p: int = 2):
        spectral_bandwidth_f1 = librosa.feature.spectral_bandwidth(y=self.audio_file1 + 0.0001, sr=self.sr_1, p=_p)[0]
        spectral_bandwidth_f2 = librosa.feature.spectral_bandwidth(y=self.audio_file2 + 0.0001, sr=self.sr_2, p=_p)[0]

        fig = plt.subplots(figsize=(14, 5))
        gs = gridspec.GridSpec(2, 4)
        gs.update(wspace=2)

        ax1 = plt.subplot(gs[0, :2])
        img1 = librosa.display.waveshow(y=self.audio_file1, sr=self.sr_1, alpha=0.4)
        ax1.plot(self.t1, normalize(spectral_bandwidth_f1))

        ax2 = plt.subplot(gs[0, 2:])
        img2 = librosa.display.waveshow(y=self.audio_file2, sr=self.sr_2, alpha=0.4)
        ax2.plot(self.t2, normalize(spectral_bandwidth_f2))

        ax3 = plt.subplot(gs[1, 1:3])
        ax3.plot(self.t1, normalize(spectral_bandwidth_f1))
        ax3.plot(self.t2, normalize(spectral_bandwidth_f2))

        return fig
    

