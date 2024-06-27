#!/bin/bash

# Copyright     2017-2018  Daniel Povey
#               2017-2018  David Snyder
#               2017-2018  Matthew Maciejewski
# Apache 2.0.

# This script is a modified version of diarization/extract_ivectors.sh
# that extracts x-vectors instead of i-vectors for speaker diarization.
#
# The script assumes that the x-vector DNN has already been trained, and
# a data directory that contains a segments file and features for the
# x-vector DNN exists.  The segments file was most likely created by a
# speech activity detection system that identified the speech segments in
# the recordings.  This script performs a subsegmentation, that further
# splits the speech segments into very short overlapping subsegments (e.g.,
# 1.5 seconds, with a 0.75 overlap).  Finally, x-vectors are extracted
# for each of the subsegments.  After this, you will most likely use
# diarization/nnet3/xvector/score_plda.sh to compute the similarity
# between all pairs of x-vectors in a recording.

# Begin configuration section.
nj=30
cmd="run.pl"
chunk_size=-1 # The chunk size over which the embedding is extracted.
              # If left unspecified, it uses the max_chunk_size in the nnet
              # directory.
stage=0
window=1.5
period=0.75
pca_dim=
min_segment=0.5
hard_min=false
apply_cmn=true # If true, apply sliding window cepstral mean normalization
#use_gpu=false
use_gpu=false
architecture=ECAPA_speechbrain_voxceleb_pretrained

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 <nnet-dir> <data> <xvector-dir>"
  echo " e.g.: $0 exp/xvector_nnet data/train exp/xvectors"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --window <window|1.5>                            # Sliding window length in seconds"
  echo "  --period <period|0.75>                           # Period of sliding windows in seconds"
  echo "  --pca-dim <n|-1>                                 # If provided, the whitening transform also"
  echo "                                                   # performs dimension reduction"
  echo "  --min-segment <min|0.5>                          # Minimum segment length in seconds per xvector"
  echo "  --hard-min <bool|false>                          # Removes segments less than min-segment if true."
  echo "                                                   # Useful for extracting training xvectors."
  echo "  --chunk-size <n|-1>                              # If provided, extracts embeddings with specified"
  echo "                                                   # chunk size, and averages to produce final embedding"
  echo "  --apply-cmn <true,false|true>                    # If true, apply sliding window cepstral mean"
  echo "                                                   # normalization to features"
  echo "  --nj <n|10>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  exit 1;
fi

srcdir=$1
data=$2
dir=$3

echo 'from extract-xvector.sh, use_gpu is : '$use_gpu
#for f in $srcdir/final.raw $srcdir/min_chunk_size $srcdir/max_chunk_size $data/feats.scp ; do
#  [ ! -f $f ] && echo "No such file $f" && exit 1;
#done

min_chunk_size=35
max_chunk_size=1000

nnet=random

if [ $chunk_size -le 0 ]; then
  chunk_size=$max_chunk_size
fi

if [ $max_chunk_size -lt $chunk_size ]; then
  echo "$0: specified chunk size of $chunk_size is larger than the maximum chunk size, $max_chunk_size" && exit 1;
fi

sub_data=$dir/subsegments_data
mkdir -p $sub_data

# Set up sliding-window subsegments
if [ $stage -le 0 ]; then
  if $hard_min; then
    awk -v min=$min_segment '{if($4-$3 >= min){print $0}}' $data/segments \
        > $dir/pruned_segments
    segments=$dir/pruned_segments
  else
    segments=$data/segments
  fi
  utils/data/get_uniform_subsegments.py \
      --max-segment-duration=$window \
      --overlap-duration=$(echo "$window-$period" | bc) \
      --max-remaining-duration=$min_segment \
      --constant-duration=True \
      $segments > $dir/subsegments
  utils/data/subsegment_data_dir.sh $data \
      $dir/subsegments $sub_data
fi

# Set various variables.
mkdir -p $dir/log
sub_sdata=$sub_data/split$nj;
utils/split_data.sh $sub_data $nj || exit 1;

## Set up features.
aud="${sub_sdata}/JOB/wav.scp"
segs="${sub_sdata}/JOB/segments"
extra="--model $architecture --pretrained_model_path $nnet --batch_size 100"

if [ $stage -le 1 ]; then
  echo "$0: extracting xvectors from nnet"
  if $use_gpu; then
    for g in $(seq $nj); do
      $cmd ${dir}/log/extract.$g.log \
        /home/shreyasr/Displace_baseline_experiments/Baseline/utils/extract_embeddings_for_kaldi.py $extra --use_gpu 1 --input_wav_scp "`echo $aud| sed s/JOB/$g/g`" --segments "`echo $segs| sed s/JOB/$g/g`" --out_path ${dir}/xvector.$g || exit 1 &
    done
    wait
  else
    $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
      /home/shreyasr/Displace_baseline_experiments/Baseline/utils/extract_embeddings_for_kaldi.py $extra --input_wav_scp $aud --segments $segs --out_path ${dir}/xvector.JOB || exit 1;
  fi
fi

if [ $stage -le 2 ]; then
  echo "$0: combining xvectors across jobs"
  for j in $(seq $nj); do cat $dir/xvector.$j.scp; done >$dir/xvector.scp || exit 1;
  cp $sub_data/{segments,spk2utt,utt2spk} $dir
fi

if [ $stage -le 3 ]; then
  echo "$0: Computing mean of xvectors"
  $cmd $dir/log/mean.log \
    ivector-mean scp:$dir/xvector.scp $dir/mean.vec || exit 1;
fi

if [ $stage -le 4 ]; then
  if [ -z "$pca_dim" ]; then
    pca_dim=-1
  fi
  echo "$0: Computing whitening transform"
  $cmd $dir/log/transform.log \
    est-pca --read-vectors=true --normalize-mean=false \
      --normalize-variance=true --dim=$pca_dim \
      scp:$dir/xvector.scp $dir/transform.mat || exit 1;
fi
