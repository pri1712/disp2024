. ./cmd.sh
. ./path.sh

# dataset=ami_sdm_eval

#DEV set
# dataset=ami_sdm_dev
# dataset=ami_dev_fbank
# kaldi_dataset=ami_dev_fbank_seg

# dataset=displace_dev_fbank #ami_eval_fbank
# kaldi_dataset=displace_pyannote_dev_fbank_seg

eval_sad=false
PYTHON=/home/prachis/.conda/envs/pyannote/bin/python

nj=27
stage=1
# default
min_duration_on=0.0554
min_duration_off=0.0979
onset=0.5
offset=0.5

. utils/parse_options.sh



datasetpath=$1 #/data1/shareefb/track2_cluster/data/$dataset
path_new_kaldi_segs=$2 #/data1/shareefb/track2_cluster/data/$kaldi_dataset
dataset=`basename $datasetpath` #displace_dev_fbank
kaldi_dataset=`basename $path_new_kaldi_segs` #displace_pyannote_dev_fbank_seg

pyannote_pretrained_model=$3 #vad_benchmarking/VAD_model/pytorch_model.bin
# utils/split_data.sh data/displace_dev_fbank $nj
utils/split_data.sh $datasetpath $nj
outputdir=../pyannote_vad_R1/${dataset} #/${hyper} 
mkdir -p $outputdir
echo "dataset=$dataset, kaldi_dataset=$kaldi_dataset"
if [ $stage -le 1 ]; then
    # for audio in $(ls /home1/somil/language_diarization/Displace_data/eval_1/*/*.wav);  
    # for onset in 0.5 ; do
    #  for offset in 0.4 0.5 0.6; do
    #    for min_duration_on in 0.05 0.10 0.25 0.30; do
    #        for min_duration_off in 0.25 0.30; do
                offset=$onset
                # hyper=hyper_onset${onset}_offset${offset}_min_duration_on${min_duration_on}_min_duration_off${min_duration_off}
                JOB=1
                # awk '{print $2}' /data1/prachis/Amrit_sharc/tools_diar/data/$dataset/split$nj/JOB/wav.scp | while read audio
                # do 
                echo "######################################################################"
                # echo "audio=$audio"
                # $decode_med_cmd JOB=1:$nj $outputdir/log/sad.JOB.log \
                # for JOB in 1 3 13 24; do
                  $decode_cmd JOB=1:$nj $outputdir/log/sad.JOB.log \
                    $PYTHON /data1/shareefb/vad_benchmarking/VAD.py \
                    --in-audio=$datasetpath/split$nj/JOB/wav.scp \
                    --in-VAD=Pyannote_VAD \
                    --dataset $dataset \
                    --tuning \
                    --onset $onset \
                    --offset $offset \
                    --min_duration_on $min_duration_on \
                    --min_duration_off $min_duration_off \
                    --outputpath $outputdir \
                    --pyannote_pretrained_model $pyannote_pretrained_model
                    
                # done
                echo "######################################################################"
                    
                
    #  done
     
    # done
fi


#####################################
# Evaluate SAD output.
#####################################
if [ $stage -le 2 ]; then
  echo "$0: Scoring SAD output on FULL set using uem ..."
#   for onset in 0.4 0.5 0.6 0.7 0.8; do
#      for offset in 0.2 0.3 0.4 0.5 0.6; do
    #    for min_duration_on in 0.05 0.10 0.25 0.30; do
    #        for min_duration_off in 0.25 0.30; do
                # offset=$onset
                # hyper=hyper_onset${onset}_offset${offset}_min_duration_on${min_duration_on}_min_duration_off${min_duration_off}
               
                # path_new_kaldi_segs=/data1/shareefb/track2_cluster/data/$kaldi_dataset #/$hyper
                # echo $hyper

                # generate pyannote segments 
                echo utils.py --vad_dir_path $outputdir
                $PYTHON ../vad_benchmarking/utils.py --vad_dir_path $outputdir

                # convert to kaldi style segments
                ../vad_benchmarking/run_kaldi_seg.sh $outputdir $kaldi_dataset $path_new_kaldi_segs/filewise_segments
                cat $path_new_kaldi_segs/filewise_segments/*.segments > $path_new_kaldi_segs/segments
             
fi
####################################################
echo copying wav.scp and creating utt2spk and spk2utt from segments folder
####################################################
if [ $stage -le 3 ]; then
    cp $datasetpath/wav.scp $path_new_kaldi_segs/.
    cp $datasetpath/rttm $path_new_kaldi_segs/
    awk '{print $1,$2}'  $path_new_kaldi_segs/segments >  $path_new_kaldi_segs/utt2spk
    utils/utt2spk_to_spk2utt.pl $path_new_kaldi_segs/utt2spk > $path_new_kaldi_segs/spk2utt
fi
## uncomment this after trails
# if [$stage -le 3 -a  $eval_sad = "true" ]; then
                
#     gndsegments=/data1/shareefb/track2_cluster/data/$dataset/segments_gnd

#     # -u tools_diar/data/$dataset/all.uem \
#     $PYTHON services/score_sad.py \
#         --n-jobs $nj --collar 0.0 \
#         $gndsegments \
#         $path_new_kaldi_segs/segments \
#         track2_cluster/data/$dataset/recordings.tbl
#     echo ""
# fi
## uncomment this till here after trails
