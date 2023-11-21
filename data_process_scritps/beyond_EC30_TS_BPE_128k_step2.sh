#! /bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=process
#SBATCH --nodelist=ilps-cn002
#SBATCH --time=2-00:00:00
#SBATCH --mem=700G

#SBATCH -o /home/dwu/workplace/logs/EC30_dataset/out.beyond.step2.o
#SBATCH -e /home/dwu/workplace/logs/EC30_dataset/err.beyond.step2.e

export PATH=/home/diwu/anaconda3/bin:$PATH
source activate py37
export CUDA_HOME="/usr/local/cuda-11.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/diwu/cudalibs:/usr/lib64/nvidia:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"


CORPUS=EC30_dataset
ALIGNMENT_SPM_DIR=alignment_data_spm
AR=('de' 'nl' 'fr' 'es' 'ru' 'cs' 'hi' 'bn' 'ar' 'he' 'sv' 'da' 'it' 'pt' 'pl' 'bg' 'kn' 'mr' 'mt' 'ha' 'af' 'lb' 'ro' 'oc' 'uk' 'sr' 'sd' 'gu' 'ti' 'am')

# step-1 get fasttext
rm ${CORPUS}/fast_text.*
for i in "${!AR[@]}"; do
    SRC=en
    TGT=${AR[i]}
    python util/step2.py -src_sentences ${CORPUS}/${ALIGNMENT_SPM_DIR}/alignment.$SRC-$TGT.spm.$SRC -tgt_sentences ${CORPUS}/${ALIGNMENT_SPM_DIR}/alignment.$SRC-$TGT.spm.$TGT -fast_text ${CORPUS}/fast_text.$SRC-$TGT &
done
wait

# step-2 build equivalence graph
LANGS=en,de,nl,fr,es,ru,cs,hi,bn,ar,he,sv,da,it,pt,pl,bg,kn,mr,mt,ha,af,lb,ro,oc,uk,sr,sd,gu,ti,am
LANG_PAIRS=en-de,en-nl,en-fr,en-es,en-ru,en-cs,en-hi,en-bn,en-ar,en-he,en-sv,en-da,en-it,en-pt,en-pl,en-bg,en-kn,en-mr,en-mt,en-ha,en-af,en-lb,en-ro,en-oc,en-uk,en-sr,en-sd,en-gu,en-ti,en-am
DICT_FILE=${CORPUS}/dict.txt
SENTENCE_PREFIX=${CORPUS}/fast_text
ALGINEMNT_PREFIX=${CORPUS}/final
ALPHA=1.0
OUTPUT=${CORPUS}/alignment_matrix
python util/graph_merge_v2_sparse.py -langs $LANGS -lang_pairs $LANG_PAIRS -dict_file $DICT_FILE -sentence_prefix $SENTENCE_PREFIX -alignment_prefix $ALGINEMNT_PREFIX -alpha $ALPHA -output $OUTPUT


## step-3 binary
#NAME=data_bin
#DICT=dict.txt
#TRAIN_SPM_DIR=train_data_spm
#VALID_SPM_DIR=valid_data_spm
#TEST_SPM_DIR=test_data_spm
#
#cd ${CORPUS}
#
## step-3.1 binarize english-centric data
#for i in "${!AR[@]}"; do
#    SRC=en
#    TGT=${AR[i]}
#    fairseq-preprocess \
#      --source-lang ${SRC} \
#      --target-lang ${TGT} \
#      --trainpref ${TRAIN_SPM_DIR}/train.${SRC}-${TGT}.spm \
#      --validpref ${VALID_SPM_DIR}/valid.${SRC}-${TGT}.spm \
#      --testpref ${TEST_SPM_DIR}/test.${SRC}-${TGT}.spm \
#      --destdir ${NAME} \
#      --thresholdtgt 0 \
#      --thresholdsrc 0 \
#      --srcdict ${DICT} \
#      --tgtdict ${DICT} \
#      --workers 70 &
#done
#wait

