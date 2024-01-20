#! /bin/bash

##SBATCH --ntasks=1
##SBATCH --cpus-per-task=5
##SBATCH --nodelist=ilps-cn118
##SBATCH --exclude=ilps-cn101,ilps-cn102,ilps-cn103,ilps-cn104,ilps-cn105,ilps-cn106,ilps-cn107,ilps-cn108,ilps-cn109,ilps-cn110,ilps-cn111,ilps-cn112,ilps-cn113,ilps-cn114,ilps-cn115,ilps-cn116,ilps-cn117,ilps-cn118
##SBATCH --job-name=process
##SBATCH --nodelist=ilps-cn006
##SBATCH --time=2-00:00:00
##SBATCH --mem=10G

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
##SBATCH --nodelist=ilps-cn115
#SBATCH --exclude=ilps-cn111,ilps-cn101,ilps-cn102,ilps-cn103,ilps-cn104,ilps-cn105,ilps-cn106,ilps-cn107,ilps-cn108,ilps-cn109,ilps-cn110,ilps-cn112,ilps-cn113,ilps-cn114,ilps-cn115
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=7-10
##SBATCH --begin=now+1minute
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=d.wu@uva.nl


#SBATCH -o /home/dwu/workplace/logs/common/out.get_bleu.o
#SBATCH -e /home/dwu/workplace/logs/common/err.get_bleu.e

pwd
conda info --envs
source activate knn-test

AR=$1
RESULT_DIR=$2
OUTPUT_FILE=$3

echo $AR
echo $RESULT_DIR
echo $OUTPUT_FILE

IFS=',' read -ra PAIRS <<< "$AR"

for i in "${!PAIRS[@]}"; do
    PAIR=${PAIRS[i]}
    SRC=${PAIR%-*}
    TGT=${PAIR#*-}
    echo $PAIR

    # sacreBleu
    grep ^H ${RESULT_DIR}/${SRC}-${TGT}/generate-test.txt | LC_ALL=C sort -V | cut -f3- | sacremoses -l ${TGT} detokenize > ${RESULT_DIR}/${SRC}-${TGT}/test-sys.txt &
    grep ^T ${RESULT_DIR}/${SRC}-${TGT}/generate-test.txt | LC_ALL=C sort -V | cut -f2- | sacremoses -l ${TGT} detokenize > ${RESULT_DIR}/${SRC}-${TGT}/test-ref.txt &
    grep ^S ${RESULT_DIR}/${SRC}-${TGT}/generate-test.txt | LC_ALL=C sort -V | awk -F'\t' '{ sub(/.*__[a-z]+__/, ""); print }' | sacremoses -l ${SRC} detokenize > ${RESULT_DIR}/${SRC}-${TGT}/test-src.txt &
    wait
    sacrebleu ${RESULT_DIR}/${SRC}-${TGT}/test-ref.txt -i ${RESULT_DIR}/${SRC}-${TGT}/test-sys.txt -l ${SRC}-${TGT} > ${RESULT_DIR}/${SRC}-${TGT}/test_bleu.txt
    sacrebleu ${RESULT_DIR}/${SRC}-${TGT}/test-ref.txt -i ${RESULT_DIR}/${SRC}-${TGT}/test-sys.txt -l ${SRC}-${TGT} -m chrf > ${RESULT_DIR}/${SRC}-${TGT}/test_chrf.txt &
    sacrebleu ${RESULT_DIR}/${SRC}-${TGT}/test-ref.txt -i ${RESULT_DIR}/${SRC}-${TGT}/test-sys.txt -l ${SRC}-${TGT} -m chrf --chrf-word-order 2 > ${RESULT_DIR}/${SRC}-${TGT}/test_chrfpp.txt &
    sacrebleu ${RESULT_DIR}/${SRC}-${TGT}/test-ref.txt -i ${RESULT_DIR}/${SRC}-${TGT}/test-sys.txt -l ${SRC}-${TGT} -m ter > ${RESULT_DIR}/${SRC}-${TGT}/test_ter.txt &
    # comet-score -s ${RESULT_DIR}/${SRC}-${TGT}/test-src.txt -t ${RESULT_DIR}/${SRC}-${TGT}/test-sys.txt -r ${RESULT_DIR}/${SRC}-${TGT}/test-ref.txt --only_system --quiet > ${RESULT_DIR}/${SRC}-${TGT}/test_comet.txt &
    comet-score -s ${RESULT_DIR}/${SRC}-${TGT}/test-src.txt -t ${RESULT_DIR}/${SRC}-${TGT}/test-sys.txt -r ${RESULT_DIR}/${SRC}-${TGT}/test-ref.txt --quiet > ${RESULT_DIR}/${SRC}-${TGT}/test_comet.txt &
    wait
done


# print bleu
PRINT_BLEU=/ivi/ilps/personal/dwu/tools/data_preprocess/print_bleu.py
python ${PRINT_BLEU} -pairs ${AR} -result_dir ${RESULT_DIR} -output ${OUTPUT_FILE}


