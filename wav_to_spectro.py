#generating images from the .wav files from the segments

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display, Image
import argparse

def convert_to_image(files_dir,output_dir):
    filename= "/data1/priyanshus/Displace2024_baseline/speaker_diarization/track2_cluster/segs_to_wav/segments/B010-0000147-0001209-00000000-1.500.wav"
    y, sr = librosa.load(filename,sr=16000)
    final,_=librosa.effects.trim(y)

    n_fft=1024
    hop_length=256
    n_mels=128
    S = librosa.feature.melspectrogram(y=final, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 6));
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
    plt.colorbar(format='%+2.0f dB');
    plt.savefig('output_256.png')

def parse_args():
    parser = argparse.ArgumentParser(description="Process audio segments and create sub-segments.")
    parser.add_argument('--files_dir', required=True, help='Directory containing wav files files')
    parser.add_argument('--output_dir', required=False, help='Directory to store output images')
    return parser.parse_args()

def main():
    print("debug")
    args=parse_args()
    #files_dir is segs_to_wav

    convert_to_image(args.files_dir,args.output_dir)

if __name__ == '__main__':
    main()
