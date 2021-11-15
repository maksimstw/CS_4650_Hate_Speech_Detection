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
We apply our analyses on a fine-tuned BERT model. Our naive baseline would be based on wordremoval. As the model tends to overly rely onidentifier terms, one simple approach wouldbe removing group identifiers altogether. They regularize SOC explanations on the group identifiers. The combined learning objective is written as follows.
    $$ \mathcal{L} = \mathcal{L'} + \alpha \sum_{w \in x \cap S}[\phi(w)]^2 $$
    where $\mathcal{L'}$ denotes the classification objective, $S$ the set of group names, $x$ the input word sequence, $\phi(w)$ the importance score of the word $w$, and $\alpha$ a hyperparameter for the strength of the regularization. The importance score is calculated as follows.
    $$\phi(p) = E_x[s(x) - s(s \setminus p)]$$
    where $p$ denotes a phrase, $s(x)$ the unnormalize prediction score, and $s(x \setminus p)$ the prediction score with phrase $p$ masked. 
