#! /bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=process
#SBATCH --nodelist=ilps-cn002
#SBATCH --time=2-00:00:00
#SBATCH --mem=700G

#SBATCH -o /home/dwu/workplace/logs/iwslt14/out.EC30.o
#SBATCH -e /home/dwu/workplace/logs/iwslt14/err.EC30.e

export PATH=/home/diwu/anaconda3/bin:$PATH
source activate py37
export CUDA_HOME="/usr/local/cuda-11.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/diwu/cudalibs:/usr/lib64/nvidia:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

CORPUS=EC30_dataset

# step-1 prepare data
cd $CORPUS

AR=('de' 'nl' 'sv' 'da' 'is' 'af' 'lb' 'no' 'fr' 'es' 'it' 'pt' 'ro' 'oc' 'ast' 'ca' 'ru' 'cs' 'pl' 'bg' 'uk' 'sr' 'be' 'bs' 'hi' 'bn' 'kn' 'mr' 'sd' 'gu' 'ne' 'ur' 'ar' 'he' 'ha' 'mt' 'ti' 'am' 'kab' 'so')
LANGS=en,de,nl,sv,da,is,af,lb,no,fr,es,it,pt,ro,oc,ast,ca,ru,cs,pl,bg,uk,sr,be,bs,hi,bn,kn,mr,sd,gu,ne,ur,ar,he,ha,mt,ti,am,kab,so

# step-2 TK zero_shot data
ZERO_SHOT_TK_DIR=zero_shot_data_tk

# step-3 apply bpe to zero-shot data
ZERO_SHOT_SPM_DIR=zero_shot_data_spm
rm -rf $ZERO_SHOT_SPM_DIR
mkdir -p $ZERO_SHOT_SPM_DIR

for i in "${!AR[@]}"; do
    ((j=${i}+1))
    while [ $j -lt ${#AR[@]} ]
    do
        SRC=${AR[i]}
        TGT=${AR[j]}
        spm_encode --model spm.bpe.model < $ZERO_SHOT_TK_DIR/test.$SRC-$TGT.$SRC > ${ZERO_SHOT_SPM_DIR}/zero_shot.${SRC}-${TGT}.spm.${SRC} &
        spm_encode --model spm.bpe.model < $ZERO_SHOT_TK_DIR/test.$SRC-$TGT.$TGT > ${ZERO_SHOT_SPM_DIR}/zero_shot.${SRC}-${TGT}.spm.${TGT} &
        ((j=j+1))
    done
    wait
done

# step-4 binary zero-shot data
NAME=data_bin
DICT=dict.txt

for i in "${!AR[@]}"; do
    ((j=${i}+1))
    while [ $j -lt ${#AR[@]} ]
    do
        SRC=${AR[i]}
        TGT=${AR[j]}
        fairseq-preprocess \
          --source-lang ${SRC} \
          --target-lang ${TGT} \
          --testpref ${ZERO_SHOT_SPM_DIR}/zero_shot.${SRC}-${TGT}.spm \
          --destdir zero_shot_${NAME} \
          --thresholdtgt 0 \
          --thresholdsrc 0 \
          --srcdict ${DICT} \
          --tgtdict ${DICT} \
          --workers 70 &
        ((j=j+1))
    done
    wait
done
