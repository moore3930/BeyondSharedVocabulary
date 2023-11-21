#! /bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=process
#SBATCH --nodelist=ilps-cn001
#SBATCH --time=2-00:00:00
#SBATCH --mem=700G

#SBATCH -o /home/dwu/workplace/logs/EC30_dataset/out.beyond.o
#SBATCH -e /home/dwu/workplace/logs/EC30_dataset/err.beyond.e

export PATH=/home/diwu/anaconda3/bin:$PATH
source activate py37
export CUDA_HOME="/usr/local/cuda-11.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/diwu/cudalibs:/usr/lib64/nvidia:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

CORPUS=EC30_dataset
cd $CORPUS

# step-0 prepare data
# You can find the preprocessed (normalized and tokenized) plain data here: https://drive.google.com/drive/folders/1nZsDnj3mNKynk2D46frnLfmR9qTFzVM9
# Please download the corresponding "EC40-train-set", "Val-set", and "test-set", and rename them as follows, and put them into your $CORPUS

mv EC40-train-set train_data_tk
mv Val-set valid_data_tk
mv test-set/en-test-set test_data_tk
mv test-set/zeroshot-test-set zero_shot_data_tk

#mv EC40-train-set train_data_tk
#mv Ntrex-eval-set valid_data_tk
#mv Flores200-test-set/en-test-set test_data_tk

# langs
AR=('de' 'nl' 'fr' 'es' 'ru' 'cs' 'hi' 'bn' 'ar' 'he' 'sv' 'da' 'it' 'pt' 'pl' 'bg' 'kn' 'mr' 'mt' 'ha' 'af' 'lb' 'ro' 'oc' 'uk' 'sr' 'sd' 'gu' 'ti' 'am')
LANGS=en,de,nl,fr,es,ru,cs,hi,bn,ar,he,sv,da,it,pt,pl,bg,kn,mr,mt,ha,af,lb,ro,oc,uk,sr,sd,gu,ti,am

for i in "${!AR[@]}"; do
    SRC=en
    TGT=${AR[i]}
    mv train_data_tk/$SRC-$TGT.$SRC train_data_tk/train.$SRC-$TGT.$SRC
    mv train_data_tk/$SRC-$TGT.$TGT train_data_tk/train.$SRC-$TGT.$TGT
done

for i in "${!AR[@]}"; do
    SRC=en
    TGT=${AR[i]}
    mv valid_data_tk/test.$SRC-$TGT.$SRC valid_data_tk/valid.$SRC-$TGT.$SRC
    mv valid_data_tk/test.$SRC-$TGT.$TGT valid_data_tk/valid.$SRC-$TGT.$TGT
done


# step-1 get alignment data & temperature sampling
mkdir -p ./alignment_data_tk
python ../util/temperature_sampling.py -langs ${LANGS} -train_dir ./train_data_tk -bpe_dir ./alignment_data_tk

for i in "${!AR[@]}"; do
    SRC=en
    TGT=${AR[i]}
    mv alignment_data_tk/bpe.$SRC-$TGT.$SRC alignment_data_tk/alignment.$SRC-$TGT.$SRC
    mv alignment_data_tk/bpe.$SRC-$TGT.$TGT alignment_data_tk/alignment.$SRC-$TGT.$TGT
done

DEST=data_bin
DICT=dict.txt

TRAIN_TK_DIR=train_data_tk
VALID_TK_DIR=valid_data_tk
TEST_TK_DIR=test_data_tk
ALIGNMENT_TK_DIR=alignment_data_tk

# step-2 learn bpe
TRAIN_SPM_DIR=train_data_spm
rm -rf $TRAIN_SPM_DIR
mkdir -p $TRAIN_SPM_DIR

VALID_SPM_DIR=valid_data_spm
rm -rf $VALID_SPM_DIR
mkdir -p $VALID_SPM_DIR

TEST_SPM_DIR=test_data_spm
rm -rf $TEST_SPM_DIR
mkdir -p $TEST_SPM_DIR

ALIGNMENT_SPM_DIR=alignment_data_spm
rm -rf $ALIGNMENT_SPM_DIR
mkdir -p $ALIGNMENT_SPM_DIR

rm train.all
for i in "${!AR[@]}"; do
    SRC=en
    TGT=${AR[i]}
    cat ${ALIGNMENT_TK_DIR}/alignment.${SRC}-${TGT}.${SRC} ${ALIGNMENT_TK_DIR}/alignment.${SRC}-${TGT}.${TGT} >> train.all
done

spm_train --input=train.all --model_prefix=spm.bpe --vocab_size=128000 --character_coverage=1.0 --model_type=bpe --input_sentence_size=12000000
cut -f1 spm.bpe.vocab | tail -n +4 | sed "s/$/ 100/g" > ${DICT}

# step-2.1 apply bpe to english centric
for i in "${!AR[@]}"; do
    SRC=en
    TGT=${AR[i]}
    spm_encode --model spm.bpe.model < ${TRAIN_TK_DIR}/train.${SRC}-${TGT}.${SRC} > ${TRAIN_SPM_DIR}/train.${SRC}-${TGT}.spm.${SRC} &
    spm_encode --model spm.bpe.model < ${TRAIN_TK_DIR}/train.${SRC}-${TGT}.${TGT} > ${TRAIN_SPM_DIR}/train.${SRC}-${TGT}.spm.${TGT} &
    spm_encode --model spm.bpe.model < ${VALID_TK_DIR}/valid.${SRC}-${TGT}.${SRC} > ${VALID_SPM_DIR}/valid.${SRC}-${TGT}.spm.${SRC} &
    spm_encode --model spm.bpe.model < ${VALID_TK_DIR}/valid.${SRC}-${TGT}.${TGT} > ${VALID_SPM_DIR}/valid.${SRC}-${TGT}.spm.${TGT} &
    spm_encode --model spm.bpe.model < ${TEST_TK_DIR}/test.${SRC}-${TGT}.${SRC} > ${TEST_SPM_DIR}/test.${SRC}-${TGT}.spm.${SRC} &
    spm_encode --model spm.bpe.model < ${TEST_TK_DIR}/test.${SRC}-${TGT}.${TGT} > ${TEST_SPM_DIR}/test.${SRC}-${TGT}.spm.${TGT} &
done
wait

# step-2.2 apply bpe to alignment data
for i in "${!AR[@]}"; do
    SRC=en
    TGT=${AR[i]}
    spm_encode --model spm.bpe.model < ${ALIGNMENT_TK_DIR}/alignment.$SRC-$TGT.$SRC > ${ALIGNMENT_SPM_DIR}/alignment.${SRC}-${TGT}.spm.${SRC} &
    spm_encode --model spm.bpe.model < ${ALIGNMENT_TK_DIR}/alignment.$SRC-$TGT.$TGT > ${ALIGNMENT_SPM_DIR}/alignment.${SRC}-${TGT}.spm.${TGT} &
done
wait


# step-3 get alignments
EFLOMAL_DIR=/ivi/ilps/personal/dwu/tools/eflomal
FASTALIGN_DIR=/ivi/ilps/personal/dwu/tools/fast_align/build
rm *.fwd
rm *.rev
rm *.align
for i in "${!AR[@]}"; do
    SRC=en
    TGT=${AR[i]}
    python ${EFLOMAL_DIR}/align.py -s ${ALIGNMENT_SPM_DIR}/alignment.$SRC-$TGT.spm.$SRC -t ${ALIGNMENT_SPM_DIR}/alignment.$SRC-$TGT.spm.$TGT --model 3 -f $SRC-$TGT.fwd -r $SRC-$TGT.rev &
done
wait

for i in "${!AR[@]}"; do
    SRC=en
    TGT=${AR[i]}
    ${FASTALIGN_DIR}/atools -i $SRC-$TGT.fwd -j $SRC-$TGT.rev -c intersect > final.$SRC-$TGT.align &
done
wait

