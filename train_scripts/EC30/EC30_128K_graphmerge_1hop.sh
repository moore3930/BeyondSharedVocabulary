#! /bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
##SBATCH --nodelist=ilps-cn115
#SBATCH --exclude=ilps-cn111,ilps-cn101,ilps-cn102,ilps-cn103,ilps-cn104,ilps-cn105,ilps-cn106,ilps-cn107,ilps-cn108,ilps-cn109,ilps-cn110,ilps-cn112,ilps-cn113,ilps-cn114,ilps-cn115
#SBATCH --cpus-per-task=11
#SBATCH --mem=64G
#SBATCH --time=14-10
##SBATCH --begin=now+1minute
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=d.wu@uva.nl


#SBATCH -o /home/dwu/workplace/logs/beyond/out.EC30_128K_graphmerge.o
#SBATCH -e /home/dwu/workplace/logs/beyond/out.EC30_128K_graphmerge.e


export PATH=/home/diwu/anaconda3/bin:$PATH
source activate py38cuda11
export CUDA_HOME="/usr/local/cuda-11.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/diwu/cudalibs:/usr/lib64/nvidia:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

DATA_DIR=/ivi/ilps/personal/dwu/release/EC30_dataset
CHECKPOINT_DIR=/ivi/ilps/personal/dwu/checkpoints/beyond/release/EC30-128K-graphmerge

fairseq-train ${DATA_DIR}/fairseq-data-bin-sharded/shard0:${DATA_DIR}/fairseq-data-bin-sharded/shard1:${DATA_DIR}/fairseq-data-bin-sharded/shard2:${DATA_DIR}/fairseq-data-bin-sharded/shard3:${DATA_DIR}/fairseq-data-bin-sharded/shard4 \
    --langs en,de,nl,sv,da,af,lb,fr,es,it,pt,ro,oc,ru,cs,pl,bg,uk,sr,hi,bn,kn,mr,sd,gu,ar,he,ha,mt,ti,am \
    --lang-pairs en-de,en-nl,en-sv,en-da,en-af,en-lb,en-fr,en-es,en-it,en-pt,en-ro,en-oc,en-ru,en-cs,en-pl,en-bg,en-uk,en-sr,en-hi,en-bn,en-kn,en-mr,en-sd,en-gu,en-ar,en-he,en-ha,en-mt,en-ti,en-am,de-en,nl-en,sv-en,da-en,af-en,lb-en,fr-en,es-en,it-en,pt-en,ro-en,oc-en,ru-en,cs-en,pl-en,bg-en,uk-en,sr-en,hi-en,bn-en,kn-en,mr-en,sd-en,gu-en,ar-en,he-en,ha-en,mt-en,ti-en,am-en \
    --task translation_multi_simple_epoch \
    --encoder-langtok tgt --arch transformer_vaswani_wmt_en_de_big_id --encoder-normalize-before --decoder-normalize-before \
    --sampling-method temperature --sampling-temperature 5 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 8192 --update-freq 20 --max-update 900000 \
    --share-decoder-input-output-embed \
    --max-source-positions 256 --max-target-positions 256 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --seed 1234 --patience 10 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --weight-decay 0.0 \
    --dropout 0.1 --attention-dropout 0.1 \
    --ddp-backend no_c10d \
    --save-dir /ivi/ilps/personal/dwu/checkpoints/beyond/release/EC30-128K-graphmerge \
    --checkpoint-suffix _m2m_ \
    --save-interval-updates 2000 --keep-interval-updates 5 --no-epoch-checkpoints --log-interval 20 \
    --distributed-world-size 4 --distributed-num-procs 44 --ddp-comm-hook fp16 \
    --skip-invalid-size-inputs-valid-test \
    --user-dir /home/dwu/workplace/fairseq/graphsage_v3_sparse \
    --graph-path /ivi/ilps/personal/dwu/release/EC30_dataset/alignment_matrix.npz \
    --tie-graph-proj \
    --hop-num 1 \
    --fp16

#Eval
PAIRS=('en-de' 'en-nl' 'en-fr' 'en-es' 'en-ru' 'en-cs' 'en-hi' 'en-bn' 'en-ar' 'en-he' 'en-sv' 'en-da' 'en-it' 'en-pt' 'en-pl' 'en-bg' 'en-kn' 'en-mr' 'en-mt' 'en-ha' 'en-af' 'en-lb' 'en-ro' 'en-oc' 'en-uk' 'en-sr' 'en-sd' 'en-gu' 'en-ti' 'en-am' 'de-en' 'nl-en' 'fr-en' 'es-en' 'ru-en' 'cs-en' 'hi-en' 'bn-en' 'ar-en' 'he-en' 'sv-en' 'da-en' 'it-en' 'pt-en' 'pl-en' 'bg-en' 'kn-en' 'mr-en' 'mt-en' 'ha-en' 'af-en' 'lb-en' 'ro-en' 'oc-en' 'uk-en' 'sr-en' 'sd-en' 'gu-en' 'ti-en' 'am-en')
for i in "${!PAIRS[@]}"; do
    PAIR=${PAIRS[i]}
    SRC=${PAIR%-*}
    TGT=${PAIR#*-}
    fairseq-generate ${DATA_DIR}/data_bin \
        --task translation_multi_simple_epoch \
        --langs en,de,nl,sv,da,af,lb,fr,es,it,pt,ro,oc,ru,cs,pl,bg,uk,sr,hi,bn,kn,mr,sd,gu,ar,he,ha,mt,ti,am \
        --lang-pairs $PAIR \
        --source-lang $SRC \
        --target-lang $TGT \
        --sacrebleu \
        --remove-bpe 'sentencepiece' \
        --arch transformer_vaswani_wmt_en_de_big_id \
        --path ${CHECKPOINT_DIR}/checkpoint_best_m2m_.pt \
        --skip-invalid-size-inputs-valid-test \
        --encoder-langtok tgt \
        --gen-subset test \
        --share-decoder-input-output-embed \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --max-tokens 8192 \
        --beam 5 \
        --seed 222 \
        --results-path ${CHECKPOINT_DIR}/${SRC}-${TGT} \
        --user-dir /home/dwu/workplace/fairseq/graphsage_v3_sparse \
        --graph-path /ivi/ilps/personal/dwu/release/EC30_dataset/alignment_matrix.npz \
        --tie-graph-proj \
        --hop-num 1 \
        --fp16
done
