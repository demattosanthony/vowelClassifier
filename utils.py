import matplotlib.pyplot as plt
import torch
import torchaudio
import librosa
from IPython.display import Audio, display

def print_metadata(metadata, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    print(" - sample_rate:", metadata.sample_rate)
    print(" - num_channels:", metadata.num_channels)
    print(" - num_frames:", metadata.num_frames)
    print(" - bits_per_sample:", metadata.bits_per_sample)
    print(" - encoding:", metadata.encoding)
    print()

def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
    if num_channels > 1:
        axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
        axes[c].set_xlim(xlim)
    if ylim:
        axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)
    
def plot_waveforms(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    figure, axes = plt.subplots(nrows=1, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))

    for x in range(5):
        wave = waveform[x][0].numpy()
        num_channels, num_frames = wave.shape
        time_axis = torch.arange(0, num_frames) / sample_rate
#         if num_channels == 1:
#             axe = [axes[x]]
    
        for c in range(num_channels):
            axes[x].set_title(waveform[x][1])
            axes[x].plot(time_axis, waveform[x][0][c], linewidth=1)
            axes[x].grid(True)
        if num_channels > 1:
            axes[x].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[x].set_xlim(xlim)
        if ylim:
            axes[x].set_ylim(ylim)
        figure.suptitle(title)

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for y in range(5):
        data = list(fft.values())[i]
        Y, freq = data[0], data[1]
        axes[y].set_title(list(fft.keys())[i])
        axes[y].plot(freq, Y)
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i += 1

def plot_spectograms(spectograms, fig):
    rows = 1
    columns = 5
    for i in range(5):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(spectograms[i][0].squeeze().numpy(), cmap='hot')
    
def plot_signals(signals):
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for y in range(5):
        axes[y].set_title(list(signals.keys())[i])
        axes[y].plot(list(signals.values())[i])
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i += 1

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
        axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
        axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")
        
def transform(waveform, sr, fixed_sample_rate):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=fixed_sample_rate)
    audio_mono = torch.mean(resample_transform(waveform), dim=0, keepdim=True).to(device)

    mel_spectogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=fixed_sample_rate, n_mels=128).to(device)
    melspectogram_db_transform = torchaudio.transforms.AmplitudeToDB()
    melspectogram = mel_spectogram_transform(audio_mono)
    melspectogram_db = melspectogram_db_transform(melspectogram)

    fixed_length = 3 * (fixed_sample_rate//200)
    if melspectogram_db.shape[2] < fixed_length:
        melspectogram_db = torch.nn.functional.pad(
          melspectogram_db, (0, fixed_length - melspectogram_db.shape[2]))
    else:
        melspectogram_db = melspectogram_db[:, :, :fixed_length]

    melspectogram_db = melspectogram_db.unsqueeze(0)
    
    return melspectogram_db
