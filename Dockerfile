# hash:sha256:6b212c24ad89ce60c5ad110a4239f9ecf823f2ab367e9fff719539640ecd4b5b
FROM registry.codeocean.com/codeocean/py-r:python3.10.12-R4.3.2-JupyterLab4.0.10-RStudiorstudio-server-2023.12.0-369-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U --no-cache-dir \
    catboost==1.2.5 \
    category-encoders==2.6.3 \
    fasttreeshap==0.1.6 \
    lightgbm==4.5.0 \
    lohrasb==4.2.0 \
    matplotlib==3.8.4 \
    numpy==1.21.6 \
    optuna-integration==3.6.0 \
    pandas==1.3.5 \
    ray==2.7.1 \
    scikit-learn==1.0.2 \
    seaborn \
    shap==0.42.1 \
    tune-sklearn==0.5.0 \
    xgboost==2.0.1 \
    zoish==5.0.4
