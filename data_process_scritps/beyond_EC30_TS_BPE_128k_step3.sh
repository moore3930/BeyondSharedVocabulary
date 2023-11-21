#! /bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=process
#SBATCH --nodelist=ilps-cn001
#SBATCH --time=2-00:00:00
#SBATCH --mem=700G

#SBATCH -o /home/dwu/workplace/logs/EC30_dataset/out.beyond.step3.o
#SBATCH -e /home/dwu/workplace/logs/EC30_dataset/err.beyond.step3.e

export PATH=/home/diwu/anaconda3/bin:$PATH
source activate py37
export CUDA_HOME="/usr/local/cuda-11.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/diwu/cudalibs:/usr/lib64/nvidia:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"


CORPUS=EC30_dataset

# step-3 binary
cd ${CORPUS}
rm -rf spm_sharded
mkdir spm_sharded

SHARD_SUB_DIR=('0' '1' '2' '3' '4')
for i in "${!SHARD_SUB_DIR[@]}"; do
    SUB_NUMBER=${SHARD_SUB_DIR[i]}
    mkdir -p spm_sharded/shard${SUB_NUMBER}
done

HIGH=('de' 'nl' 'fr' 'es' 'ru' 'cs' 'hi' 'bn' 'ar' 'he')
MED=('sv' 'da' 'it' 'pt' 'pl' 'bg' 'kn' 'mr' 'mt') #ha
LOW=('af' 'lb' 'ro' 'oc' 'uk' 'sr' 'sd' 'gu' 'ti' 'am')
#ELOW=('no' 'is' 'ast' 'ca' 'be' 'bs' 'ne' 'ur' 'so') #kab

SPM_DIR=train_data_spm
SPM_SHARD_DIR=spm_sharded


## HIGH 5m each file -> split to 1m for one shard
for i in "${!HIGH[@]}"; do
    LANG=${HIGH[i]}
    split -l 1000000 $SPM_DIR/train.en-$LANG.spm.en -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.en.shard &
    split -l 1000000 $SPM_DIR/train.en-$LANG.spm.$LANG -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard &
done
wait

for i in "${!HIGH[@]}"; do
    LANG=${HIGH[i]}
    for j in "${!SHARD_SUB_DIR[@]}"; do
        SUB_NUMBER=${SHARD_SUB_DIR[j]}
        mv $SPM_SHARD_DIR/train.en-$LANG.en.shard0${SUB_NUMBER} spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.en
        mv $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard0${SUB_NUMBER} spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.$LANG
    done
done

# MED 1m each file -> split to 200K for one shard
for i in "${!MED[@]}"; do
    LANG=${MED[i]}
    split -l 200000 $SPM_DIR/train.en-$LANG.spm.en -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.en.shard &
    split -l 200000 $SPM_DIR/train.en-$LANG.spm.$LANG -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard &
done
wait

for i in "${!MED[@]}"; do
    LANG=${MED[i]}
    for j in "${!SHARD_SUB_DIR[@]}"; do
        SUB_NUMBER=${SHARD_SUB_DIR[j]}
        mv $SPM_SHARD_DIR/train.en-$LANG.en.shard0${SUB_NUMBER} spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.en
        mv $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard0${SUB_NUMBER} spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.$LANG
    done
done

# LOW 100k each file -> split to 20k for one shard
for i in "${!LOW[@]}"; do
    LANG=${LOW[i]}
    split -l 20000 $SPM_DIR/train.en-$LANG.spm.en -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.en.shard &
    split -l 20000 $SPM_DIR/train.en-$LANG.spm.$LANG -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard &
    wait

    for j in "${!SHARD_SUB_DIR[@]}"; do
        SUB_NUMBER=${SHARD_SUB_DIR[j]}
        mv $SPM_SHARD_DIR/train.en-$LANG.en.shard0${SUB_NUMBER} spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.en
        mv $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard0${SUB_NUMBER} spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.$LANG
    done
done

:<<!
## ELOW 50k each file -> split to 10k for one shard
for i in "${!ELOW[@]}"; do
    LANG=${ELOW[i]}
    split -l 10000 $SPM_DIR/train.en-$LANG.spm.en -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.en.shard &
    split -l 10000 $SPM_DIR/train.en-$LANG.spm.$LANG -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard &
    wait

    for j in "${!SHARD_SUB_DIR[@]}"; do
        SUB_NUMBER=${SHARD_SUB_DIR[j]}
        mv $SPM_SHARD_DIR/train.en-$LANG.en.shard0${SUB_NUMBER} spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.en
        mv $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard0${SUB_NUMBER} spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.$LANG
    done
done
!

# SPECIAL HA 344000 -> split to 68800 for one shard
HA=('ha')
for i in "${!HA[@]}"; do
    LANG=${HA[i]}
    split -l 68800 $SPM_DIR/train.en-$LANG.spm.en -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.en.shard &
    split -l 68800 $SPM_DIR/train.en-$LANG.spm.$LANG -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard &
    wait

    for j in "${!SHARD_SUB_DIR[@]}"; do
        SUB_NUMBER=${SHARD_SUB_DIR[j]}
        mv $SPM_SHARD_DIR/train.en-$LANG.en.shard0${SUB_NUMBER} spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.en
        mv $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard0${SUB_NUMBER} spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.$LANG
    done
done

:<<!
# SPECIAL HA 18448 -> split to 3690 for one shard
KAB=('kab')
for i in "${!KAB[@]}"; do
    LANG=${KAB[i]}
    split -l 3690 $SPM_DIR/train.en-$LANG.spm.en -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.en.shard &
    split -l 3690 $SPM_DIR/train.en-$LANG.spm.$LANG -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard &
    wait

    for j in "${!SHARD_SUB_DIR[@]}"; do
        SUB_NUMBER=${SHARD_SUB_DIR[j]}
        mv $SPM_SHARD_DIR/train.en-$LANG.en.shard0${SUB_NUMBER} spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.en
        mv $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard0${SUB_NUMBER} spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.$LANG
    done
done
!

# ------------------------ 4. Fairseq preparation Sharded ------------------------ #
SPM_DATA_DIR=spm_sharded
FAIRSEQ_DIR=fairseq-data-bin-sharded
rm -rf ${FAIRSEQ_DIR}
mkdir ${FAIRSEQ_DIR}

cut -f1 spm.bpe.vocab | tail -n +4 | sed "s/$/ 100/g" > ${FAIRSEQ_DIR}/dict.txt

SHARD_SUB_DIR=('0' '1' '2' '3' '4')
for i in "${!SHARD_SUB_DIR[@]}"; do
    SUB_NUMBER=${SHARD_SUB_DIR[i]}
    mkdir $FAIRSEQ_DIR/shard${SUB_NUMBER}
done

# preprocess with mmap dataset
for SHARD in $(seq 0 4); do
    SRC=en
    for TGT in bg da mt es uk am hi ro ti de cs lb pt nl mr oc ha sv gu ar fr ru it pl sr sd he af kn bn; do
        fairseq-preprocess \
            --dataset-impl mmap \
            --source-lang ${SRC} \
            --target-lang ${TGT} \
            --trainpref ${SPM_DATA_DIR}/shard${SHARD}/train.${SRC}-${TGT} \
            --destdir ${FAIRSEQ_DIR}/shard${SHARD} \
            --thresholdtgt 0 \
            --thresholdsrc 0 \
            --workers 40 \
            --srcdict ${FAIRSEQ_DIR}/dict.txt \
            --tgtdict ${FAIRSEQ_DIR}/dict.txt &
    wait
    cp ${FAIRSEQ_DIR}/dict.txt ${FAIRSEQ_DIR}/shard${SHARD}/dict.txt
    done
done
