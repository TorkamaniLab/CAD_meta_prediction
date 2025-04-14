<div align="center">
<h1>
Meta-Prediction of Coronary Artery Disease Risk
</h1>

[![DOI](https://img.shields.io/badge/DOI-pending-blue.svg)](https://doi.org/)
[![Python version](https://img.shields.io/badge/Python-3.7%E2%80%933.12-blue.svg)](https://github.com/TorkamaniLab/CAD_meta_prediction)
[![MIT license](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/TorkamaniLab/CAD_meta_prediction/blob/main/LICENSE)
[![Open Source](https://img.shields.io/badge/Open-Source-orange.svg)](https://github.com/TorkamaniLab/CAD_meta_prediction/blob/main/LICENSE) 
[![GitHub latest commit](https://badgen.net/github/last-commit/TorkamaniLab/CAD_meta_prediction)](https://github.com/TorkamaniLab/CAD_meta_prediction/commits)  

[![GitHub forks](https://img.shields.io/github/forks/TorkamaniLab/CAD_meta_prediction.svg?style=social&label=Fork&maxAge=2592000)](https://github.com/TorkamaniLab/CAD_meta_prediction/network/) 
[![GitHub stars](https://img.shields.io/github/stars/TorkamaniLab/CAD_meta_prediction.svg?style=social&label=Star&maxAge=2592000)](https://github.com/TorkamaniLab/CAD_meta_prediction/stargazers/) 
[![GitHub watchers](https://img.shields.io/github/watchers/TorkamaniLab/CAD_meta_prediction.svg?style=social&label=Watch&maxAge=2592000)](https://github.com/TorkamaniLab/CAD_meta_prediction/watchers/)
</div>


## Description

This repository provides the core **machine learning codebase** used in our *Nature Medicine* publication:  
**“Meta-prediction of coronary artery disease risk”**  
🔗 [https://www.nature.com/articles/s41591-025-03648-0](https://www.nature.com/articles/s41591-025-03648-0)

Our study introduces a novel **meta-prediction framework** that integrates genetic and non-genetic factors - unmodifiable and modifiable risk profiles - into a unified prediction system for 10-year incident CAD. This repository shares **key components of the ML pipeline**.

![CAD_meta_prediction](./img/NatMed_metaprediction_fig1b.png?raw=true "CAD_meta_prediction")


### Cite us
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


## Assets

This codebase includes components for **training individual prediction models**, such as CAD diagnoses and biomarker estimations. A consistent pipeline used across multiple prediction tasks to enable meta-feature generation, which feeds into our final CAD risk model and trained the final model:

- Compatible with tested tree-based ML models: **XGBoost**, **LightGBM**, **CatBoost**
- Custom utilities:
  - [`zoish`](https://github.com/TorkamaniLab/zoish): SHAP-based feature importance wrapper built on `fasttreeshap`
  - [`lohrasb`](https://github.com/TorkamaniLab/lohrasb): Optuna-based hyperparameter tuner (TPE + Hyperband)

These individual models were used to generate meta-features feeding into our final CAD risk prediction framework.
 

## Instructions