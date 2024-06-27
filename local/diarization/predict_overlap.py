from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines import OverlappedSpeechDetection as OverlappedSpeechDetectionPipeline
import argparse
import numpy as np
import os
from pdb import set_trace as bp
from pathlib import Path

parser = argparse.ArgumentParser(description='overlap prediction')
parser.add_argument(
        'data_dir', type=Path, help='Subset data directory')
parser.add_argument(
        'output_dir', type=Path, help='Output directory')
parser.add_argument(
        '--pyannote_pretrained_model', type=str, default="../vad_benchmarking/VAD_model/pytorch_model.bin"
)
args = parser.parse_args()


def generate_overlap_labels(num_frames,wavfile,overlap_filename,step=100):
        # reduce False Alarm
        best_params =  {"onset": 0.8104268538848918, 
                        "offset": 0.4806866463041527, 
                        "min_duration_on": 0.05537587440407595, 
                        "min_duration_off": 0.09791355693027545} # changed after optimizing

        # pipeline = Pipeline.from_pretrained("pyannote/overlapped-speech-detection",
        #                                     use_auth_token="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")
        
        model = Model.from_pretrained(args.pyannote_pretrained_model)
        # pipeline = OverlappedSpeechDetectionPipeline(use_auth_token="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").instantiate(best_params)
        pipeline = OverlappedSpeechDetectionPipeline(segmentation=model).instantiate(best_params)
        output = pipeline(wavfile)

        overlap_idx = np.zeros(num_frames,dtype=int)
        # if os.stat(overlap_labels).st_size!=0:
        for speech in output.get_timeline().support():
            # overlap_frames = open(overlap_labels).readlines()
            # overlap_frames = np.genfromtxt(overlap_labels,dtype=int)[:,1:3]
            # for line in overlap_frames:
            #   st = int(line.rsplit()[1])-1
            #   ed = min(int(line.rsplit()[2]),num_frames)
            st = int(speech.start*step) # convert to frames
            ed = min(int(speech.end*step)+1,num_frames)
            
            overlap_idx[st:ed] = 1
        # overlap_dir = f'{output_dir}/per_file_overlap'
        overlap_dir = os.path.dirname(overlap_filename)
        if not os.path.exists(overlap_dir):
            # create directory
            os.mkdir(overlap_dir)
        
        np.savetxt(overlap_filename, overlap_idx)

def load_wavfiles_path(fpath):
    """Load mapping from URIs to wavefiles path from ``fpath``.

    The file is expected to be in the format of a Kaldi ``wav.scp`` file;
    that is, two space-delimited columns:

    - URI
    - wavefiles_path
    """
    
    wavfiles_path = {}
    with open(fpath, 'r') as f:
        for line in f:
            uri = line.strip().split()[0]
            wavpath = line.strip().split()[2]
            wavfiles_path[uri] = wavpath
    return wavfiles_path

def load_frame_counts(fpath):
    """Load mapping from URIs to frame counts from ``fpath``.

    The file is expected to be in the format of a Kaldi ``utt2num_frames`` file;
    that is, two space-delimited columns:

    - URI
    - frame count
    """
    frame_counts = {}
    with open(fpath, 'r') as f:
        for line in f:
            uri, n_frames = line.strip().split()
            n_frames = int(n_frames)
            frame_counts[uri] = n_frames
    return frame_counts

if __name__ == "__main__":
    frame_counts = load_frame_counts(
            Path(args.data_dir, "utt2num_frames"))
    recording_ids = sorted(frame_counts.keys())
    # wav.scp 
    wavfiles_path = load_wavfiles_path(
        Path(args.data_dir, "wav.scp"))

    for recording_id in recording_ids:
        overlap_filename = Path(args.output_dir, "per_file_overlap", recording_id + '.txt')
        generate_overlap_labels(frame_counts[recording_id],wavfiles_path[recording_id],overlap_filename)

