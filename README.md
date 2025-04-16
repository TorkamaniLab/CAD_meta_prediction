<div align="center">
<h1>
Meta-Prediction of Coronary Artery Disease Risk
</h1>

[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41591--025--03648--0-blue)](https://doi.org/10.1038/s41591-025-03648-0)
[![Python version](https://img.shields.io/badge/Python-3.10--3.11-blue.svg)](https://github.com/TorkamaniLab/CAD_meta_prediction)
[![MIT license](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/TorkamaniLab/CAD_meta_prediction/blob/main/LICENSE)
[![Open Source](https://img.shields.io/badge/Open-Source-orange.svg)](https://github.com/TorkamaniLab/CAD_meta_prediction/blob/main/LICENSE) 
[![GitHub latest commit](https://badgen.net/github/last-commit/TorkamaniLab/CAD_meta_prediction)](https://github.com/TorkamaniLab/CAD_meta_prediction/commits)  

</div>

## Description

This repository provides the core **machine learning codebase** used in our *Nature Medicine* publication:  
**“Meta-prediction of coronary artery disease risk”**  
🔗 [https://www.nature.com/articles/s41591-025-03648-0](https://www.nature.com/articles/s41591-025-03648-0)

Our study introduces a novel **meta-prediction framework** that integrates genetic and non-genetic factors - unmodifiable and modifiable risk profiles - into a unified prediction system for 10-year incident coronary artery disease (CAD). This repository shares **key components of the ML pipeline**.

### 📌 Cite us
> Chen SF, et al. *Meta-prediction of coronary artery disease risk*. Nature Medicine. 2025. [DOI: 10.1038/s41591-025-03648-0](https://doi.org/10.1038/s41591-025-03648-0).  
<details>
<summary><strong>📎 BibTeX</strong></summary>

```bibtex
@article{chen2025metapred,
  title={Meta-prediction of coronary artery disease risk},
  author={Chen, Shang-Fu and Lee, Sang Eun and Sadaei, Hossein Javedani and Park, Jun-Bean and Khattab, Ahmed and Chen, Jei-Fu and Henegar, Corneliu and Wineinger, Nathan E. and Muse, Evan D. and Torkamani, Ali},
  journal={Nature Medicine},
  year={2025},
  month={Apr},
  publisher={Nature Portfolio},
  doi={10.1038/s41591-025-03648-0},
  url={https://www.nature.com/articles/s41591-025-03648-0}
}
```
</details>

## 🧬 Study overview

![CAD_meta_prediction](./img/NatMed_metaprediction_fig1b.png?raw=true "CAD_meta_prediction")

<details>
<summary><sub><sup>👨‍💻 Insider trivia</sup></sub></summary>
<sub><sup>The silhouette featured in Figure 1b isn’t just any figure — it’s based on Shaun (Chen SF), the first author and lead developer of this codebase. Fitting, since the meta-prediction includes both features about the individual… and of the individual. 😉</sup></sub>
</details>


- We developed a meta-prediction framework that combines unmodifiable and modifiable factors to predict 10-year risk of CAD.
- The UK Biobank dataset was partitioned into two cohorts, each serving a distinct role in the pipeline:
  - A prevalent CAD cohort, used to train baseline models for predicting biomarker levels and diagnostic categories.
  - An incident CAD cohort, used to build the final CAD risk model based on meta-features derived from baseline model outputs.
- The framework generated 296 meta-features from ~2,000 variables, including clinical biomarkers, diagnostic categories, and >1,000 polygenic risk scores (PRSs).
- The final model used **50 selected features** (13 measured variables, 22 PRSs, and 15 meta-features), achieving **AUROC 0.84 in UK Biobank** and **AUROC 0.81 in All of Us**
- The framework supports **individualized intervention simulation** and **identifies subgroups with differential benefit**, offering new opportunities for precision prevention.




## 🧰 Assets

This codebase includes components for **training individual prediction models**, such as CAD diagnoses and biomarker estimations. A consistent pipeline used across multiple prediction tasks to enable meta-feature generation, which feeds into our final CAD risk model and trained the final model.

Key features include:
- Compatible with tested tree-based ML models: **XGBoost**, **LightGBM**, **CatBoost**
- Custom utilities:
  - [`zoish`](https://github.com/TorkamaniLab/zoish): SHAP-based feature importance wrapper built on `fasttreeshap`
  - [`lohrasb`](https://github.com/TorkamaniLab/lohrasb): Optuna-based hyperparameter tuner (TPE + Hyperband)
 

## ⚙️ Environment Configuration

This project requires **Python >=3.10** and uses [Poetry](https://python-poetry.org/) for dependency management. If you prefer `pip`, a `requirements.txt` is also provided.

### Option 1: Using Poetry
```bash
poetry install
```

### Option 2: Using pip
```bash
pip install -r requirements.txt
```

### Dependencies
Main runtime dependencies:
```
catboost==1.2.5
category-encoders==2.6.3
fasttreeshap==0.1.6
lightgbm==4.5.0
lohrasb==4.2.0
matplotlib==3.8.4
numpy==1.21.6
optuna-integration==3.6.0
pandas>=1.3.5
ray==2.7.1
scikit-learn==1.0.2
seaborn
shap==0.42.1
tune-sklearn==0.5.0
xgboost==1.7.5
zoish==5.0.4
```

## 🚀 Usage

You may either run the provided notebook or invoke the command-line interface:

### Option 1: Run the tutorial notebook
A reproducible example is provided in [`Tutorial_Notebook.ipynb`](./Tutorial_Notebook.ipynb), demonstrating model training and SHAP analysis.

### Option 2: Run the pipeline via CLI
The CLI expects a `.pkl` file containing a preprocessed `pandas.DataFrame` with appropriate data types. It must include:
- A target column matching `--y_label`
- An ID column matching `--id_col`

<details>
<summary><strong>Sample Data</strong></summary>

This project uses the publicly available [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) as an example. The input file `data/typed_cardio_train.pkl` is generated by `data/make_cardio_train_pickle.py` and tracked with Git LFS.

</details>

To explore all available options and their default values, run:
```bash
python -u meta_prediction_estimator.py -h
```

#### Example command:
```bash
python -u meta_prediction_estimator.py \
  --y_label "cardio" \
  --input_pickle_fp "data/typed_cardio_train.pkl" \
  --id_col "id" \
  --pkg "xgb" \
  --estimator_type "classifier" \
  --n_features 5 \
  --n_trials 100
```

The pipeline will produce:

1. A trained pipeline object (including the best estimator)  
   ↳ `final_pipeline__xgb_classifier__cardio.joblib`

2. A SHAP-based feature importance file (mean absolute SHAP values per feature)  
   ↳ `shap__xgb_classifier__cardio__preselect.tsv`

See [`expected_output/`](./expected_output/) for example outputs.
