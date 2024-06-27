#!/usr/bin/env python


#to convert the segments after SAD to wav format to transform to images.
import os
import subprocess
import argparse

def window_audio_segments(segments_dir, output_dir, window_secs=1.5, period_secs=0.75):
    sample_rate = 16000  
    
    window_samples = int(window_secs * sample_rate)
    period_samples = int(period_secs * sample_rate)

    print("Performing windowing...")
    
    for root, _, files in os.walk(segments_dir):
        for file in files:
            print(file)
            if (file=="segments"):
                segments_file = os.path.join(root, file)
                # Extract the audio ID from the filename
                audio_id = os.path.splitext(file)[0]
                # Create the output subdirectory based on the audio ID
                output_subdir = os.path.join(output_dir, audio_id)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                
                with open(segments_file, 'r') as f:
                    for line in f:
                        segment_id, audio_id, start_time, end_time = line.strip().split()
                        start_time = float(start_time)
                        end_time = float(end_time)

                        audio_file = "/data1/priyanshus/Displace2024_baseline/speaker_diarization/DISPLACE2023_dev/data/wav/"+audio_id+".wav"
                        print(audio_file)
                        segment_duration = end_time - start_time
                        subseg_id_counter = 0
                        
                        if segment_duration <= window_secs:
                            output_file = os.path.join(output_subdir, f"{segment_id}-{subseg_id_counter:08d}-{segment_duration:.3f}.wav")
                            command = [
                                'ffmpeg', '-i', audio_file, '-ss', str(start_time), '-to', str(end_time), 
                                '-c', 'copy', output_file
                            ]
                            # print(command)
                            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        else:
                            subseg_start = start_time
                            while subseg_start < end_time:
                                subseg_end = min(subseg_start + window_secs, end_time)
                                subseg_duration = subseg_end - subseg_start
                                output_file = os.path.join(output_subdir, f"{segment_id}-{subseg_id_counter:08d}-{subseg_duration:.3f}.wav")
                                command = [
                                    'ffmpeg', '-i', audio_file, '-ss', str(subseg_start), '-to', str(subseg_end), 
                                    '-c', 'copy', output_file
                                ]
                                # print(command)
                                subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                
                                subseg_start += period_secs
                                subseg_id_counter += 1



def parse_args():
    parser = argparse.ArgumentParser(description="Process audio segments and create sub-segments.")
    parser.add_argument('--segments_dir', required=True, help='Directory containing .segments files')
    parser.add_argument('--output_dir', required=True, help='Directory to store output sub-segments')
    parser.add_argument('--window_secs', type=float, default=1.5, help='Window length in seconds')
    parser.add_argument('--period_secs', type=float, default=0.75, help='Period length in seconds')
    return parser.parse_args()

def main():
    print("debug")
    args=parse_args()
    window_audio_segments(args.segments_dir, args.output_dir, args.window_secs, args.period_secs)

if __name__ == '__main__':
    main()
