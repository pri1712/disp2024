ref=/data1/prachis/DISPLACE2023/data/DISPLACE_2023_Eval-Phase1_Audio_Release/data/final.rttm
#ref=/data1/prachis/DISPLACE2023/data/DISPLACE_2023_DEV/data/final.rttm
sys=$1/rttm
outpath=$1
perl local/diarization/md-eval.pl -r $ref -s $sys > $outpath/der.txt
cat $outpath/der.txt
