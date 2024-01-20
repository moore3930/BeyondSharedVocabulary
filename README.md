# Beyond Shared Vocabulary
Here is the repo for our EMNLP23 paper "Beyond Shared Vocabulary: Increasing Representational Word Similarities across Languages for Multilingual Machine Translation".

See [Preprint](https://arxiv.org/pdf/2305.14189.pdf)

See [Poster](materials/EMNLP23%20-%20Poster.pdf)

Training Code is released. See [Run Example](#quick_start)

Data and Model is released. See [Data & Model](#data_model)

## Introduction
![](materials/main.jpg)

Using a shared vocabulary is common practice in Multilingual Systems, like Multilingual Translation, mBERT, or nowadays LLM. In addition to its simple design, shared tokens play an important role in positive knowledge transfer, assuming that shared tokens refer to similar meanings across languages. However, when word overlap is small, especially due to different writing systems, transfer is inhibited.

In this paper, we encourage word-level knowledge transfer via graph networks, which bridge the knowledge sharing among words that have similar meanings but write in different ways (no matter whether they use the same or different writing systems). Broadly speaking, we mine priors of word equivalences and inject them into the embedding table, resulting in better knowledge transfer among multilingual translation systems. 

Multiple advantages of our approach are demonstrated: 
1) Better multilinguality: Embeddings of words with similar meanings are better aligned across languages.
2) Better MMT performance: up to an average of 2.3 BLEU gain is achieved for high- and low-resource MMT.
3) Efficiency: Less than 1.0% additional trainable parameters are required with a limited increase in computational costs, while inference time remains identical to the baseline.

## Experiments

1) For the experiments on the IWSLT-14 dataset: We provide the script in [iwslt14-30k-graphmerge-hop1.sh](https://github.com/research-anonymous/beyond_shared_vocabulary/blob/main/iwslt14-30k-graphmerge-hop1.sh).

2) For the experiments on the WMT30 dataset: We provide the script in [EC30_graphmerge.sh](https://github.com/moore3930/BeyondSharedVocabulary/blob/main/train_scripts/EC30/EC30_graphmerge_1hop.sh). 

<span id="data_model"></span>
### Data and Checkpoints
[EC30_raw](https://drive.google.com/file/d/1HgO278Pxt_B_rS3VISt9jrr2MqWKWss0/view?usp=sharing): Raw data of EC30, which is already tokenized.

[EC30_full](https://drive.google.com/file/d/1e4VxVE_7WSjczPT5SFPMJRIuvmR5wstR/view?usp=sharing): Preprocess data of EC30, including BPE, fairseq binarization, and graph building.

<span id="quick_start"></span>
## Quick Start
### Training
Before training, Please download the dataset and run the graph building scripts.

You can also use our prebuilt dataset [EC30_full](https://drive.google.com/file/d/1e4VxVE_7WSjczPT5SFPMJRIuvmR5wstR/view?usp=sharing).

```angular2html
# install fairseq, fairseq\graphsage_v3_sparse is the model directory.
git clone git@github.com:moore3930/BeyondSharedVocabulary.git
cd BeyondSharedVocabulary/fairseq
pip install --editable ./

# Run directly if you are using slurm systems. Otherwise, please refer to the code within "EC30_graphmerge_1hop.sh"
cd ../train_scripts/EC30
sbatch EC30_graphmerge_1hop.sh

```


### Graph Building
On releasing

