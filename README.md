# CS 4650 Hate Speech Detection Project
Large pretrained language models such as BERT can  easily  achieve  state-of-the-art  results  in most natural language processing tasks. However, for hate speech detection, many classifiers are sensitive to group identifiers which are only hate speech in certain contexts. These terms will be over-associated with hate speech, leading to many false positives during classification. We propose a novel regularization technique to resolve this issue.

## Datasets
We selected three public corpora, GHC (Kennedy et al., 2018), Stormfront (de Gibert et al., 2018) and Latent Hatred (ElSherief et al., 2021), for our experiments.

To generate `stormfront.tsv`:
```
cd datasets/stormfront 
python stormfront.py
```

## Method
We apply our analyses on a fine-tuned BERT model. Our naive baseline is word removal. As the model tends to over-rely on identifier terms, one simple approach would be removing group identifiers altogether. Another approach is to use the attention scores from the last self-attention layer to detect the shortcut token pairs. Based on the number of occurence and attention scores, we can identify if the model develops spurious relationships among certain tokens and the classfication results. This can be used to regularize the model.

## Running experiments
Use `BERT_Hate_Speech_Detection.ipynb` to train and obtain the attention scores. Use `post_hoc.ipynb` to select a curated list of token pairs. 
