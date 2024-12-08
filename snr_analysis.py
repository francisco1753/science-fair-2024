import numpy as np
import librosa
import matplotlib.pyplot as plt

# function to calculate SNR between original and compressed files
def calculate_snr(original, compressed):
    signal_power = np.mean(np.square(original))
    noise_power = np.mean(np.square(original - compressed))
    if noise_power == 0:
        return np.inf  # infinite SNR for perfect reconstruction
    return 10 * np.log10(signal_power / noise_power)

# list of files, compression algorithms, file sizes (in MB), durations (in seconds), and bit rates (in kbps)
files = [
    {"original": "elliot-og.wav", "compressed": "elliot-mp3.mp3", "algorithm": "ELLIOT-MP3", "size_mb": 2.53, "duration_sec": 2 * 60 + 36, "bitrate_kbps": 128},
    {"original": "flow-og.wav", "compressed": "flow-aac.aac", "algorithm": "FLOW-AAC", "size_mb": 3.14, "duration_sec": 3 * 60 + 25, "bitrate_kbps": 160},
    {"original": "sinatra-og.wav", "compressed": "sinatra-flac.flac", "algorithm": "SINATRA-FLAC", "size_mb": 16.00, "duration_sec": 3 * 60 + 14, "bitrate_kbps": 1000},
    {"original": "waltz-og.wav", "compressed": "waltz-ogg.ogg", "algorithm": "WALTZ-OGG", "size_mb": 2.60, "duration_sec": 3 * 60 + 50, "bitrate_kbps": 192},
]

# initialize lists for storing SNR values, labels, file sizes, durations, and bit rates
compressed_snr_values = []
wav_snr_values = []
algorithm_labels = []
file_sizes = []
durations = []
bit_rates = []

# loop through files and calculate SNR for each file pair
for i, file in enumerate(files, start=1):
    original_signal, fs1 = librosa.load(file["original"], sr=None)
    compressed_signal, fs2 = librosa.load(file["compressed"], sr=None)

    # ensure signals are the same length (truncate to minimum length)
    min_length = min(len(original_signal), len(compressed_signal))
    original_signal = original_signal[:min_length]
    compressed_signal = compressed_signal[:min_length]

    # calculate SNR for compressed files
    compressed_snr = calculate_snr(original_signal, compressed_signal)
    wav_snr = calculate_snr(original_signal, original_signal)  # SNR for WAV itself (perfect reconstruction)

    # store results
    compressed_snr_values.append(compressed_snr)
    wav_snr_values.append(wav_snr)
    algorithm_labels.append(file["algorithm"])
    file_sizes.append(file["size_mb"])
    durations.append(file["duration_sec"] / 60)  # convert seconds to minutes
    bit_rates.append(file["bitrate_kbps"])

    # print results
    print(f"Compressed SNR for {file['algorithm']} ({file['compressed']}): {compressed_snr:.2f} dB")
    print(f"WAV SNR for {file['original']}: {wav_snr:.2f} dB\n")

# replace `inf` with a placeholder value for visualization (e.g., 100 dB)
compressed_visual_snr_values = [100 if v == np.inf else v for v in compressed_snr_values]
wav_visual_snr_values = [100 if v == np.inf else v for v in wav_snr_values]

# plot SNR for WAV files
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(wav_visual_snr_values)), wav_visual_snr_values, color='green', alpha=0.7, label="WAV SNR")
plt.xticks(range(len(files)), [file['original'] for file in files], rotation=45, ha="right")
plt.xlabel("Original WAV Files", fontsize=14)
plt.ylabel("SNR (dB)", fontsize=14)
plt.title("Signal-to-Noise Ratio (SNR) of Original WAV Files", fontsize=16)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# add labels for WAV bars
for i, bar in enumerate(bars):
    value = wav_snr_values[i]
    if value == np.inf:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, "∞", ha="center", va="bottom", fontsize=12, color="red")
    else:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{value:.2f}", ha="center", va="bottom", fontsize=12)

plt.tight_layout()


# plot combined SNR and file size comparison
fig, ax1 = plt.subplots(figsize=(10, 6))

# plot SNR (primary y-axis)
bars = ax1.bar(range(len(compressed_visual_snr_values)), compressed_visual_snr_values, color='skyblue', alpha=0.7, label="SNR (dB)")
ax1.set_xlabel("Compression Algorithm", fontsize=14)
ax1.set_ylabel("SNR (dB)", fontsize=14, color='skyblue')
ax1.set_xticks(range(len(algorithm_labels)))
ax1.set_xticklabels(algorithm_labels, rotation=45, ha="right")
ax1.set_title("Comparison of SNR and File Sizes for Compression Algorithms", fontsize=16)
ax1.grid(axis="y", linestyle="--", alpha=0.7)

# add labels for SNR bars
for i, (value, bar) in enumerate(zip(compressed_snr_values, bars)):
    if value == np.inf:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, "∞", ha="center", va="bottom", fontsize=12, color="red")
    else:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{value:.2f}", ha="center", va="bottom", fontsize=12)

# plot file sizes (secondary y-axis)
ax2 = ax1.twinx()
file_size_bars = ax2.plot(range(len(file_sizes)), file_sizes, color='orange', marker='o', markersize=8, linestyle='-', label="File Size (MB)")
ax2.set_ylabel("File Size (MB)", fontsize=14, color='orange')

# add legend
fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.9), bbox_transform=ax1.transAxes)

plt.tight_layout()


# plot file durations
plt.figure(figsize=(10, 6))
plt.bar(algorithm_labels, durations, color='purple', alpha=0.7)
plt.xlabel("Compression Algorithm", fontsize=14)
plt.ylabel("Duration (minutes)", fontsize=14)
plt.title("File Durations of Compressed Files", fontsize=16)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()


# plot bit rates
plt.figure(figsize=(10, 6))
plt.bar(algorithm_labels, bit_rates, color='orange', alpha=0.7)
plt.xlabel("Compression Algorithm", fontsize=14)
plt.ylabel("Bit Rate (kbps)", fontsize=14)
plt.title("Bit Rates of Compressed Files", fontsize=16)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
