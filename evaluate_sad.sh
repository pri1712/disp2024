. ./cmd.sh
. ./path.sh
stage=3
eval_sad=true
dset=dev_fbank
nj=20
devdatapath=/data1/prachis/DISPLACE2023/data/DISPLACE_2023_DEV/data

#####################################
# Evaluate SAD output.
#####################################
if [ $stage -eq 3  -a  $eval_sad = "true" ]; then
  echo "$0: Scoring SAD output on FULL DEV set..."
  local/segmentation/score_sad.py \
    --n-jobs $nj --collar 0.0 \
    -u data/displace_$dset/all.uem \
    data/displace_$dset/segments \
    data/displace_${dset}_seg/segments \
    $devdatapath/docs/recordings_filewise.tbl
  echo ""
fi

if [ $stage -eq 4  -a  $eval_sad = "true" ]; then
  echo "$0: Scoring SAD output on FULL EVAL set..."
  local/segmentation/score_sad.py \
    --n-jobs $nj --collar 0.0 \
    data/displace_$dset/segments \
    data/displace_${dset}_seg/segments \
    $evaldatapath/docs/recordings.tbl
  echo ""
fi

