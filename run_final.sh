#!/bin/bash

set -e  # Exit on error.

PYTHON=python  # Python to use; defaults to system Python.

################################################################################
# Configuration
################################################################################
nj=4
decode_nj=4 #eval
stage=0
sad_train_stage=0
sad_decode_stage=0
diarization_stage=0
vb_hmm_stage=0
overlap=1
# If following is "true", then SAD output will be evaluated against reference
# following decoding stage. This step requires the following Python packages be
# installed:
#
# - pyannote.core
# - pyannote.metrics
# - pandas
eval_sad=false


################################################################################
# Paths to DATA releases
################################################################################
DISPLACE_DEV_DIR=/data1/priyanshus/Displace2024_baseline/speaker_diarization/track2_cluster/DISPLACE2024_dev
DISPLACE_EVAL_DIR=/data1/priyanshus/Displace2024_baseline/speaker_diarization/track2_cluster/DISPLACE2024_eval
name=displace2024
. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

################################################################################
# Prepare data directories
################################################################################
if [ $stage -le 0 ]; then
  echo "$0: Preparing data directories..."

 
  if [ -d "$DISPLACE_DEV_DIR" ]; then 
  local/make_data_dir.py \
  --rttm-dir $DISPLACE_DEV_DIR/data/rttm_sd \
    data/${name}_dev_fbank \
    $DISPLACE_DEV_DIR/data/wav
  ./create_utt2spk_spk2utt.sh data/${name}_dev_fbank $DISPLACE_DEV_DIR
  ./utils/validate_data_dir.sh \
    --no-text --no-feats data/${name}_dev_fbank/
  else
    echo "${DISPLACE_DEV_DIR} does not exist"
    exit 1
  fi
  
  if [ -d "$DISPLACE_EVAL_DIR" ]; then 
  local/make_data_dir.py \
    --rttm-dir $DISPLACE_EVAL_DIR/data/rttm_sd \
    data/${name}_eval_fbank \
    $DISPLACE_EVAL_DIR/data/wav 
   ./create_utt2spk_spk2utt.sh data/${name}_eval_fbank $DISPLACE_EVAL_DIR
  ./utils/validate_data_dir.sh \
    --no-text --no-feats data/${name}_eval_fbank/
  else
    echo "${DISPLACE_EVAL_DIR} does not exist"
  fi
fi

#####################################
# SAD decoding. Pyannote
#####################################
if [ $stage -le 1 ]; then
  echo "$0: Applying SAD model to DEV/EVAL..."
  for dset in dev_fbank; do #dev_fbank
    ../vad_benchmarking/run_pyannote_SAD.sh \
      --nj $nj --stage $sad_decode_stage \
      data/${name}_${dset} data/${name}_${dset}_seg \
      ../vad_benchmarking/VAD_model/pytorch_model.bin
    done


  for dset in eval_fbank; do #dev_fbank
    ../vad_benchmarking/run_pyannote_SAD.sh \
      --nj $nj --stage $sad_decode_stage \
      data/${name}_${dset} data/${name}_${dset}_seg \
      ../vad_benchmarking/VAD_model/pytorch_model.bin
    done
fi

################################################################################
# Perform first-pass diarization using AHC/SC.
################################################################################
period=0.25


scoretype=laplacian
outdevdir=exp_pyannote_final/${name}_diarization_nnet_1a_dev_fbank_spectral_laplacian_scaling10 # best results for dev
outevaldir=exp_pyannote_final/${name}_diarization_nnet_1a_eval_fbank_spectral_laplacian_scaling10


echo "$0: stage 0 and 1 done."

if [ $stage -le 2 ]; then
 
  echo "$0: Performing first-pass diarization of DEV..."
  local/diarize_fbank.sh \
    --nj $nj --stage $diarization_stage \
    --tune true --period $period --clustering spectral \
    --scoretype $scoretype \
    exp/xvector_nnet_1a_tdnn_fbank/ exp/xvector_nnet_1a_tdnn_fbank/plda_model_full \
    data/${name}_dev_fbank_seg/ $outdevdir
fi

echo "stage 2 done"

if [ $stage -le 3 ]; then
  
  echo "$0: Performing first-pass diarization of EVAL using threshold "
  echo "$0: obtained by tuning on DEV..."
  thresh=0.6
  local/diarize_fbank.sh \
    --nj $decode_nj --stage $diarization_stage \
    --thresh $thresh --tune false --period $period  --clustering spectral \
    --scoretype $scoretype \
    exp/xvector_nnet_1a_tdnn_fbank/ exp/xvector_nnet_1a_tdnn_fbank/plda_model_full \
    data/${name}_eval_fbank_seg/ $outevaldir
fi

echo "stage 3 done"


################################################################################
# Evaluate first-pass diarization.
# ################################################################################

if [ $stage -le 4 ]; then
  echo "$0: Scoring first-pass diarization on DEV"
  local/diarization/score_diarization.sh \
    --scores-dir $outdevdir/scoring \
    $DISPLACE_DEV_DIR $outdevdir/per_file_rttm
fi
echo "stage 4 done"

if [ $stage -le 5 ] && [ -d $DISPLACE_EVAL_DIR/data/rttm_sd ]; then
  echo "$0: Scoring first-pass diarization on EVAL..."
  local/diarization/score_diarization.sh \
    --scores-dir $outevaldir/scoring \
    $DISPLACE_EVAL_DIR $outevaldir/per_file_rttm
fi
echo "stage 5 done"
 
if [ $overlap -eq 0 ]; then
  ################################################################################
  # Refined first-pass diarization using VB-HMM resegmentation
  ################################################################################
  dubm_model=exp/xvec_init_gauss_1024_ivec_400/model/diag_ubm.pkl
  ie_model=exp/xvec_init_gauss_1024_ivec_400/model/ie.pkl

  outdevdir_vb=exp_pyannote_final/displace2024_diarization_nnet_1a_vbhmm_dev_spectral_laplacian_scaling10
  outevaldir_vb=exp_pyannote_final/displace2024_diarization_nnet_1a_vbhmm_eval_spectral_laplacian_scaling10

  if [ $stage -le 6 ]; then
    echo "$0: Performing VB-HMM resegmentation of DEV..."
    statScale=10
    loop=0.45
    maxiters=1
    echo "statScale=$statScale loop=$loop maxiters=$maxiters" 
    local/resegment_vbhmm.sh \
        --nj $nj --stage $vb_hmm_stage --statscale $statScale --loop $loop --max-iters $maxiters \
        data/displace_dev_fbank_full $outdevdir/rttm \
        $dubm_model $ie_model $outdevdir_vb/
  fi


  if [ $stage -le 7 ]; then
    echo "$0: Performing VB-HMM resegmentation of EVAL..."
    local/resegment_vbhmm.sh \
        --nj $decode_nj --stage $vb_hmm_stage \
        data/displace_eval_fbank $outevaldir/rttm \
        $dubm_model $ie_model $outevaldir_vb/
  fi



  ################################################################################
  # Evaluate VB-HMM resegmentation.
  ################################################################################
  if [ $stage -le 8 ]; then
    echo "$0: Scoring VB-HMM resegmentation on DEV..."
    local/diarization/score_diarization.sh \
      --scores-dir $outdevdir_vb/scoring \
      $DISPLACE_DEV_DIR $outdevdir_vb/per_file_rttm
  fi


  if [ $stage -le 9 ] && [ -d $DISPLACE_EVAL_DIR/data/rttm ]; then
    if [ -d $DISPLACE_EVAL_DIR/data/rttm/ ]; then
      echo "$0: Scoring VB-HMM resegmentation on EVAL..."
      local/diarization/score_diarization.sh \
        --scores-dir $outevaldir_vb/scoring \
        $DISPLACE_EVAL_DIR $outevaldir_vb/per_file_rttm
    fi
  fi

else
  #####  WITH OVERLAP ################################
  ################################################################################
  # Refined first-pass diarization using VB-HMM resegmentation with pyannote overlap
  ################################################################################
  dubm_model=exp/xvec_init_gauss_1024_ivec_400/model/diag_ubm.pkl
  ie_model=exp/xvec_init_gauss_1024_ivec_400/model/ie.pkl
  PYTHON=/home/prachis/.conda/envs/pyannote/bin/python


  outdevdir_vb=exp_pyannote_final/displace_diarization_nnet_1a_vbhmm_dev_spectral_laplacian_scaling10_overlap
  outevaldir_vb=exp_pyannote_final/displace_diarization_nnet_1a_vbhmm_eval_spectral_laplacian_scaling10_overlap

  statScale=10
  loop=0.45
  maxiters=1
  echo "statScale=$statScale loop=$loop maxiters=$maxiters" 
  
  if [ $stage -le 6 ]; then
      
    echo "$0: Performing VB-HMM resegmentation of DEV"
    mkdir -p $outdevdir_vb
    pyannote_pretrained_model=../vad_benchmarking/VAD_model/pytorch_model.bin
    local/resegment_vbhmm.sh \
      --nj $nj --stage $vb_hmm_stage --statscale $statScale --loop $loop --max-iters $maxiters \
      --overlap 1 --PYTHON $PYTHON \
      --pyannote_pretrained_model $pyannote_pretrained_model \
      data/${name}_dev_fbank $outdevdir/rttm \
      $dubm_model $ie_model $outdevdir_vb 
  fi


  if [ $stage -le 7 ]; then
    echo "$0: Performing VB-HMM resegmentation of EVAL..."
    mkdir -p $outevaldir_vb
    pyannote_pretrained_model=../vad_benchmarking/VAD_model/pytorch_model.bin
    local/resegment_vbhmm.sh \
      --nj $decode_nj --stage $vb_hmm_stage \
      --statscale $statScale --loop $loop --max-iters $maxiters \
      --overlap 1 --PYTHON $PYTHON \
      --pyannote_pretrained_model $pyannote_pretrained_model \
      data/${name}_eval_fbank $outevaldir/rttm \
      $dubm_model $ie_model $outevaldir_vb

  fi



    ################################################################################
    # Evaluate VB-HMM resegmentation.
    ################################################################################
      # echo "in the VBHMM scoring stage "
  if [ $stage -le 8 ]; then
    echo "$0: Scoring VB-HMM resegmentation on DEV..."
    local/diarization/score_diarization.sh \
      --scores-dir $outdevdir_vb/scoring \
      $DISPLACE_DEV_DIR $outdevdir_vb/per_file_rttm
  fi
    #done

  if [ $stage -le 9 ] && [ -d $DISPLACE_EVAL_DIR/data/rttm_sd ]; then
    if [ -d $DISPLACE_EVAL_DIR/data/rttm_sd/ ]; then
      echo "$0: Scoring VB-HMM resegmentation on EVAL..."
      local/diarization/score_diarization.sh \
        --scores-dir $outevaldir_vb/scoring \
        $DISPLACE_EVAL_DIR $outevaldir_vb/per_file_rttm
    fi
  fi
fi
