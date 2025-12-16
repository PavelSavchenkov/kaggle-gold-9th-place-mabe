# kaggle-gold-9th-place-mabe
Source code for [MABe Challenge - Social Action Recognition in Mice](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/overview) Kaggle Competition Solution

## Data Structure

```text
data/
├─ annotation/
│  ├─ AdaptableSnail/
|  ...
│  └─ UppityFerret/
├─ tracking/
│  ├─ AdaptableSnail/
|  ...
│  └─ UppityFerret/
├─ train.csv
├─ test.csv
```

## Main Scripts

- **GBDT pipeline:** [`gbdt/train.py`](gbdt/train.py)
- **TCN pipeline:** [`dl/train.py`](dl/train.py)
- **Ensembling:** [`mixed_submission_builder.py`](mixed_submission_builder.py)
