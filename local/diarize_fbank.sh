#!/usr/bin/env bash

set -e -u -o pipefail


################################################################################
# Configuration
################################################################################
# Stage to start from IN THE SCRIPT.
stage=0

# Number of parallel jobs to use during extraction/scoring/clustering.
nj=40

# Proportion of energy to retain when performing the conversation-dependent
# PCA projection. Usual default in Kaldi diarization recipes is 10%, but we use
# 30%, which was found to give better performance by Diez et al. (2020).
#
#   Diez, M. et. al. (2020). "Optimizing Bayesian HMM based x-vector clustering
#   for the Second DIHARD Speech Diarization Challenge." Proceedings of
#   ICASSP 2020.
target_energy=0.3

# AHC threshold.
thresh=0.6

# Model for xvector extraction
modeltype=etdnn
# If true, ignore "thresh" and instead tune the AHC threshold using the
# reference RTTM. The tuning stage repeats clustering for a range of thresholds
# and selects the one that  yields the lowest DER. Requires that the data
# directory contains a file named "rttm" with the reference diarization. If this
# file is absent, tuning will be skipped and the threshold will default to
# "thresh".
tune=false
windowlen=2
period=0.5

# clustering strategy
clustering=AHC

scoretype=None
################################################################################
# Parse options, etc.
################################################################################
if [ -f path.sh ]; then
    . ./path.sh;
fi
if [ -f cmd.sh ]; then
    . ./cmd.sh;
fi
. utils/parse_options.sh || exit 1;
if [ $# != 4 ]; then
  echo "usage: $0 <model-dir> <plda-path> <data-dir> <out-dir>"
  echo "e.g.: $0 exp/xvector_nnet_1a exp/xvector_nnet_1a/plda data/dev exp/diarization_nnet_1a_dev"
  echo "  --nj <n|40>         # number of jobs"
  echo "  --stage <stage|0>   # current stage; controls partial reruns"
  echo "  --thresh <t|-0.2>   # AHC threshold"
  echo "  --tune              # tune AHC threshold; data directory must contain reference RTTM"
  exit 1;
fi

# Directory containing the trained x-vector extractor.
model_dir=$1

# Path to the PLDA model to use in scoring.
plda_path=$2

# Data directory containing data to be diarized. If performing AHC tuning (i.e.,
# "--tune true"), must contain a file named "rttm" containing reference
# diarization.
data_dir=$3

# Output directory for x-vectors and diarization.
out_dir=$4


##############################################################################i
# Extract 40-D MFCCs
###############################################################################
name=$(basename $data_dir)
if [ $stage -le 0 ]; then
  echo "$0: Extracting MFCCs...."
  set +e # We expect failures for short segments.
  rm -f $data_dir/utt2dur $data_dir/utt2num_frames
  steps/make_fbank.sh \
    --nj $nj --cmd "$decode_med_cmd" --write-utt2num-frames true  \
    --fbank-config conf/fbank_16k.conf \
    $data_dir  exp/make_fbank/$name exp/make_fbank/$name
  set -e
fi


###############################################################################
# Prepare feats for x-vector extractor by performing sliding window CMN.
###############################################################################
if [ $stage -le 1 ]; then
  echo "$0: Preparing features for x-vector extractor..."
  local/nnet3/xvector/prepare_feats.sh \
    --nj $nj --cmd "$decode_med_cmd" \
    data/$name data/${name}_cmn exp/make_fbank/${name}_cmn/
  if [ -f data/$name/vad.scp ]; then
    echo "$0: vad.scp found; copying it"
    cp data/$name/vad.scp data/${name}_cmn/
  fi
  if [ -f data/$name/segments ]; then
    echo "$0: segments found; copying it"
    cp data/$name/segments data/${name}_cmn/
  fi
  utils/fix_data_dir.sh data/${name}_cmn
fi


###############################################################################
# Extract sliding-window x-vectors for all segments.
###############################################################################
for windowlen in 1.75 2.5 2.75 3 3.5; do
  echo $windowlen
  if [ $stage -le 2 ]; then
    echo "$0: Extracting x-vectors..."
    if [ $modeltype == "etdnn" ];then
      local/diarization/nnet3/xvector/extract_xvectors.sh \
        --nj $nj --cmd "$decode_med_cmd" \
        --window $windowlen --period $period --apply-cmn false \
        --min-segment 0.25 \
        $model_dir data/${name}_cmn $out_dir/xvectors_${period}_${windowlen}
    else
        local/diarization/nnet3/xvector/extract_xvectors_pytorch.sh \
          --nj $nj --cmd "$decode_cmd" \
          --window $windowlen --period $period --apply-cmn false --use-gpu true \
          --min-segment 0.25 \
          $model_dir data/${name}_cmn $out_dir/xvectors_${period}_${windowlen}
    fi
  fi
done
#exit
###############################################################################
# Perform PLDA scoring for x-vectors.
###############################################################################
echo "in stage 3"
for windowlen in 1.75 2.5 2.75 3 3.5; do
  echo $windowlen
  plda_dir=$out_dir/plda_scores_${windowlen}
  if [ $stage -le 3 ]; then
    echo "$0: Performing PLDA scoring..."

    # Use specified PLDA model + whitening computed from actual xvectors.
    plda_model_dir=$out_dir/plda_${period}_${windowlen}
    mkdir -p $plda_model_dir
    cp $plda_path/plda $plda_model_dir/plda
    cp $plda_path/{mean.vec,transform.mat} $plda_model_dir
    local/diarization/nnet3/xvector/score_plda.sh \
      --nj $nj --cmd "$decode_med_cmd" \
      --target-energy $target_energy \
      $plda_model_dir $out_dir/xvectors_${period}_${windowlen} $plda_dir
  fi
done


###############################################################################
# Determine clustering threshold.
###############################################################################
for windowlen in 1.75 2.5 2.75 3 3.5; do
  tuning_dir=$out_dir/tuning_${period}_${windowlen}
  awk '{print $1}' $data_dir/wav.scp >  $data_dir/dataset.list
  plda_dir=$out_dir/plda_scores_${windowlen}
  echo "plda_dir is below"
  echo $plda_dir 
  if [ $clustering == "AHC" ]; then
    if [ $stage -le 4 ]; then
      mkdir -p $tuning_dir
      ref_rttm=$data_dir/rttm
      echo "$0: Determining AHC threshold..."
      if [[ $tune == true ]] && [[ -f $ref_rttm ]]; then
        echo "$0: Tuning threshold using reference diarization stored in: ${ref_rttm}"
        best_der=1000
        best_thresh=0
        # 0.1 0.2 0.3 0.4
        for thresh in -1.2 -1.0 -0.5 0.0 0.5 0.1; do
          echo "$0: Clustering with AHC threshold ${thresh}..."
          cluster_dir=$tuning_dir/plda_scores_t${thresh}
          mkdir -p $cluster_dir
          local/diarization/cluster.sh \
            --nj $nj --cmd "$decode_cmd" \
      --threshold $thresh --rttm-channel 1 \
      $plda_dir $cluster_dir
          perl local/diarization/md-eval.pl \
            -r $ref_rttm -s $cluster_dir/rttm \
            > $tuning_dir/der_t${thresh} \
      2> $tuning_dir/der_t${thresh}.log
          der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
            $tuning_dir/der_t${thresh})
          if [ $(echo $der'<'$best_der | bc -l) -eq 1 ]; then
              best_der=$der
              best_thresh=$thresh
          fi
        done
        echo "$best_thresh" > $tuning_dir/thresh_best
        echo "$0: ***** Results of tuning *****"
        echo "$0: *** Best threshold is: $best_thresh"
        echo "$0: *** DER using this threshold: $best_der"
      else
        echo "$thresh" > $tuning_dir/thresh_best
      fi
    fi


    ###############################################################################
    # Cluster using AHC selected threshold
    ###############################################################################
    if [ $stage -le 5 ]; then
      best_thresh=$(cat $tuning_dir/thresh_best)
      echo "$0: Performing AHC using threshold ${best_thresh}..."
      local/diarization/cluster.sh \
        --nj $nj --cmd "$decode_cmd" \
        --threshold $best_thresh --rttm-channel 1 \
        $plda_dir $out_dir
      local/diarization/split_rttm.py \
        $out_dir/rttm $out_dir/per_file_rttm
    fi

  else
    if [ $stage -le 4 ]; then
      mkdir -p $tuning_dir
      ref_rttm=$data_dir/rttm
      echo "$0: Determining Spectral threshold..."
      if [[ $tune == true ]] && [[ -f $ref_rttm ]]; then
        echo "$0: Tuning threshold using reference diarization stored in: ${ref_rttm}"
        best_der=1000
        best_thresh=0
        # 0.1 0.2 0.3 0.4 0.1 0.2 0.3 0.4 0.5 0.6
        for thresh in 0.6; do
          echo "$0: Clustering with spectral threshold ${thresh}..."
          cluster_dir=$tuning_dir/plda_scores_t${thresh}   #tuning_dir=$out_dir/tuning_${period}_${windowlen}
          mkdir -p $cluster_dir
          echo $cluster_dir
          local/diarization/my_spectral_cluster.sh \
            --nj $nj --cmd "$decode_med_cmd" \
            --threshold $thresh --rttm-channel 1 --score_path $plda_dir \
            --score_file $data_dir/dataset.list \
            --scoretype $scoretype \
            $plda_dir $cluster_dir
          perl local/diarization/md-eval.pl \
            -r $ref_rttm -s $cluster_dir/rttm \
            > $tuning_dir/der_t${thresh} \
            2> $tuning_dir/der_t${thresh}.log
          der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
            $tuning_dir/der_t${thresh})
          if [ $(echo $der'<'$best_der | bc -l) -eq 1 ]; then
            best_der=$der
            best_thresh=$thresh
          fi
        done
        echo "$best_thresh" > $tuning_dir/thresh_best
        echo "$0: ***** Results of tuning *****"
        echo "$0: *** Best threshold is: $best_thresh"
        echo "$0: *** DER using this threshold: $best_der"
      fi
    fi
    ###############################################################################
    # Cluster using selected threshold
    ###############################################################################
    if [ $stage -le 5 ]; then
      best_thresh=0.6
      curout_dir=$out_dir/window_length_${windowlen}
      echo $curout_dir
      echo "$0: Performing Spectral using threshold 0.6"
      local/diarization/my_spectral_cluster.sh \
        --nj $nj --cmd "$decode_med_cmd" \
        --threshold $best_thresh --rttm-channel 1 --score_path $plda_dir \
        --score_file $data_dir/dataset.list \
        --scoretype $scoretype \
        $plda_dir $curout_dir
      local/diarization/split_rttm.py \
        $curout_dir/rttm $curout_dir/per_file_rttm
    fi
  fi
done