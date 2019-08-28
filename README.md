# A cross validated logistic regression solution for highly skewed data.

This repository contains codes on multiple approaches exploring performance of machine learning models on highly skewed data upto 99.6% of data belonging to only class only. It is a solution submitted on India ML Hiring Hiring Hackathon 2019 on Analytics Vidhya. This solution was ranked with 408 on public score board with only logistic regression implementation with minor feature engineering. Also, other approaches explored and implemented are also specified in the repository. 

Let's dissect multiple models and analyze different approaches for the given 'loan deliquency dataset'. Also, we try to analyze why other models like xgboost, lightgbm and even voting or stack based ensembling approaches didn't performed that great for this challenge. This in-depth explanatory analysis will take time and hence it'll be added with time.

But, for the time being as an immediate commit the logistic regression model file along with datasets is made completely available to replicate results. Plus, other standalone models will also be made available along it.

# Current Model List

* Logisitic Regression: It is specifically good at handling Low Precision/High Recall or High Precision/Low Recall cases. Where `Precision` being defined as "How many selected items(TP+FP) are relevant(TP) ?" and `Recall` being defined as "How many relevant items(TP+FN) are selected(TP) ?"


__Note:__ The `data.zip` contains the dataset from the given competition: India ML Hiring Hiring Hackathon 2019 on Analytics Vidhya for download.