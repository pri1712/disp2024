#dataset=ami_dev_fbank_0.75s
dataset=$2
ref=data/$dataset/rttm
sys=$1/rttm
outpath=$1
perl local/diarization/md-eval.pl -r $ref -s $sys > $outpath/dersplit.txt
cat $outpath/dersplit.txt
