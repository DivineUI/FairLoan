# FairLoan
A machine learning system that supports loan approval decisions using historical application data

## Problem Statement
Loan officers must decide whether to approve a loan application before granting it, often under limited review time, which can lead to rushed or inconsistent decisions. FairLoan is a machine learning system that predicts loan approval outcomes from historical application data, giving loan officers a consistent, data-driven reference to support (not replace) their judgment. The project also conducts a fairness audit to assess whether predictions are biased with respect to features such as gender or race.

## Team Members
- Serge Ishimwe
- Lyse Claudia Irera
- Divine Uwase Ingabire
- Monia Nijimbere

Note: All members contribute collaboratively to data preparation, modeling, and evaluation.

## Dataset
Loan Approval Classification Dataset - 45,000 records, 14 variables, sourced from Kaggle: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data

## Tools & Tech Stack
- **Python** with **pandas** / **numpy** for data cleaning and preparation
- **Scikit-learn** for building and evaluating classification models (logistic regression baseline, then a tuned second model)
- **Google Colab** for interactive development
