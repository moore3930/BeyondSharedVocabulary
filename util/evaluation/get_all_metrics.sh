GET_BLEU=/ivi/ilps/personal/dwu/tools/data_preprocess/get_sacre_bleu.sh
PRINT_BLEU=/ivi/ilps/personal/dwu/tools/data_preprocess/print_bleu.py

PAIRS=en-de,en-nl,en-fr,en-es,en-ru,en-cs,en-hi,en-bn,en-ar,en-he,en-sv,en-da,en-it,en-pt,en-pl,en-bg,en-kn,en-mr,en-mt,en-ha,en-af,en-lb,en-ro,en-oc,en-uk,en-sr,en-sd,en-gu,en-ti,en-am,de-en,nl-en,fr-en,es-en,ru-en,cs-en,hi-en,bn-en,ar-en,he-en,sv-en,da-en,it-en,pt-en,pl-en,bg-en,kn-en,mr-en,mt-en,ha-en,af-en,lb-en,ro-en,oc-en,uk-en,sr-en,sd-en,gu-en,ti-en,am-en
RESULT_DIR=/ivi/ilps/personal/dwu/checkpoints/beyond/release/EC30-baseline
OUTPUT_FILE=/home/dwu/workplace/experiments/beyond_multi/release/EC30/rebuttal/output.en_centric.baseline
sbatch $GET_BLEU $PAIRS $RESULT_DIR ${OUTPUT_FILE}



