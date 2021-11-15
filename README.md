# CS 4650 Hate Speech Detection Project
Large pretrained language model such as BERTcan  easily  achieve  state-of-the-art  results  inmost natural language processing tasks. How-ever, for hate speech detection, many classifiersare sensitive to group identifiers which are onlyhate speech in certain contexts.  These termswill be over-associated with hate speech, leading to many false positives during classification. We proposed a novel regularization technique to resolve this issue

## Datasets
We selected two public corpora, GHC (Kennedyet al., 2018) and Stormfront (de Gibert et al., 2018), for our experiments.

To generate `stormfront.tsv`:
```
cd datasets/stormfront 
python stormfront.py
```

## Method
We apply our analyses on a fine-tuned BERT model. Our naive baseline would be based on wordremoval. As the model tends to overly rely onidentifier terms, one simple approach wouldbe removing group identifiers altogether. Another approach is to use sampling and exclusion (SOC) or the attention scores. By using SOC or the attention scores, we can identify if the model develops spurious relationship amongs certain tokens and the classfication results. This can be used to regularize the model.
