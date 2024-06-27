cmd="run.pl"
stage=0
nj=10
cleanup=true
threshold=0.95
rttm_channel=0
lamda=0.1
read_costs=false
reco2num_spk=
reco2utt=exp/${dataset}/spk2utt
score_path=../scores_plda_new/${dataset}_scores/
score_file=my_file
usinginit=0
dataset=ami_dev
p_percentile=0.95
custom_dist=cosine
scoretype=
PYTHON=python
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

echo $reco2num_spk
echo "usinginit $usinginit"

srcdir=$1
dir=$2

mkdir -p $dir/log
mkdir -p $dir/tmp

# echo $srcdir
# echo $score_file

cp $srcdir/spk2utt $dir/tmp/spk2utt
cp $srcdir/utt2spk $dir/tmp/utt2spk
cp $score_file $dir/tmp/dataset.list
cp $srcdir/segments $dir/tmp/segments

# Checks if the given string operand size is not zero; if it is not zero length, then it returns true.
if [ ! -z $reco2num_spk ]; then
  cp $reco2num_spk $dir/tmp/reco2num_spk
fi

utils/fix_data_dir.sh $dir/tmp > /dev/null
# echo "fix"

sdata=$dir/tmp/split$nj;
utils/split_data_mine.sh $dir/tmp $nj || exit 1;

# echo "split"
JOB=1
if [ $stage -le 0 ]; then
  echo "$0: clustering scores"
  
  if [ ! -z $reco2num_spk ]; then
    $cmd JOB=1:$nj $dir/log/spectral.JOB.log \
      $PYTHON local/diarization/spectral_scoring/run_spectralclustering.py  --reco2num $sdata/JOB/reco2num_spk \
      --p-percentile $p_percentile --custom-dist $custom_dist --reco2utt $sdata/JOB/spk2utt \
      --score_path $score_path/scores.$JOB.scp \
      --label-out $dir/labels.JOB 
      # --scoretype $scoretype
      # --score_file $sdata/JOB/dataset.list \


  else
    echo "threshold: "$threshold
    $cmd JOB=1:$nj $dir/log/spectral.JOB.log \
      $PYTHON local/diarization/spectral_scoring/run_spectralclustering.py \
      --p-percentile $p_percentile --custom-dist $custom_dist --reco2utt $sdata/JOB/spk2utt \
      --score_path $score_path/scores.JOB.scp \
      --label-out $dir/labels.JOB \
      --stop_eigenvalue $threshold \
       --scoretype $scoretype


  fi

fi

if [ $stage -le 1 ]; then
  echo "$0: combining labels"
  for j in $(seq $nj); do cat $dir/labels.$j; done > $dir/labels || exit 1;
  for j in $(seq $nj); do $PYTHON local/diarization/make_rttm.py --rttm-channel $rttm_channel $sdata/$j/segments $dir/labels.$j $dir/rttm.$j; done

fi

if [ $stage -le 2 ]; then
  echo "$0: computing RTTM"
  $PYTHON local/diarization/make_rttm.py --rttm-channel $rttm_channel $srcdir/segments $dir/labels $dir/rttm || exit 1;
  
fi

rm -r $dir/tmp
