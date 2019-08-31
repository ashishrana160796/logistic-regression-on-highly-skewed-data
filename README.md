#### Note: View this text under a mardown reader. Click on the following ![link](https://github.com/ashishrana160796/logistic-regression-on-highly-skewed-data/blob/master/README.md) to view directly on ![github](https://github.com/ashishrana160796/logistic-regression-on-highly-skewed-data).

## Approch analysis document

The jupyter notebook does cover the approach on the fly with the mentioned coding cells mentioned. For solving the problem f1-score as the evaluation metric was considered and the models were trained accordingly only.

### A solution approach for the loan deliquency challenge. 

The first observation without any EDA from the given data was that one class is definitely predominant similar to like detecting outliers in any scenario for a machine learning model. Hence, it is problem of __binary classification with highly imbalanced target data__.

* In this solution file, first step that was followed is to do some EDA analysis on different features and create a summary report for our analysis. Also, check scope converting category variables, find missing value features etc.  

* Secondly, determine if relations between different features and if high correlation is present then do some feature engineering. After that observing the increase in accuracy of the results or so.
    * Approach tried: combining deliquency columns `mi's` for result improvement. But, the results stayed the same or no significant improvement.

* Third, we tried to mitigate the skewness in our variables ( _observed from EDA_ ) with different standardization methodologies like log, log(1+x), tan and percentile linearization methods. Where results from percentile linearlization outperforms other transformations.  

* Finally, build model pipelines of different types to check if which one performed the best on leader-board. The models that we tried were xgboost, lightgbm, extra-trees, ensemble of first three, stacked first three models & fed to another xgboost model and finally settling for logistic regression model. Below we highlight approximate score for some of the models were trained before settling for logistic regression.
    * __lightgbm__, f1-score: 0.028
    * __vote-ensemble(lightgbm, xgboost, extra-trees)__, f1-score: 0.29
    * __stacked-ensemble(lightgbm, random-forest, extra-trees, svm -> xgboost )__, f1-score: 0.23
    * Finally, __logistic regression__, f1-score: 0.315


#### Hence, my solution provides a model pipeline with of cross validated light weight minimalist logistic regression solution for highly skewed data which yields decently good results without any complexity data generation approaches like SMOTE.


### Pre-processing Steps Followed and it's analysis: 
  
  * Observing for outlier values with IQR value analysis and checking missing values percentages. In case of any missing values following imputations can be made.
    * No such scenario was observed with the analysis.
  * Plotting correlation variable plots to find out redundant variable. But, most importantly finding correlation with target variables to get the estimate about importnat feartures.
  * Converting categorical feature values into their equivalent numerical representations. After, that generating pandas_profiling reports which gave all the EDA details needed for any dataframe.
    
    * Inferences made from the above analysis. 
      * High correlation of `number_of_borrowers` feature with `co-borrower_credit_score`. Hence, removing that column from the analysis.  
      * Majority of the features being highly skewed in nature upto 99.6% of storing only one value.  
      * Definitely high correlation observed between all the `mi` values.  
      * Relatively large distinct counts for `interest_rate`, `unpaid_principal_bal` features.

    * Mitigating solutions used and applied.
      * Dropping off the `number_of_borrower` feature.
      * Using different transformations like log, log(1+x), tanh etc. Finally, using percentile linearization standardization giving best results.
      * Feature engineering of `mi` values to create new features didn't improved performance on given model. Hence, omitting the feature engineered columns. 
      * Binning values for `interest_rate`, `unpaid_principal_bal` features to make the model more lucid in nature.


### Final Model and Process of achieving it.

Final model selected is a minimal logistic regression model with optimized parameters searched via GridSearch algorithm out of a decently large solution space. Approaches tried before selecting this model are as follow:
  * XGBoost, LightGbm, ExtraTrees, RandomForest classifier and regressor standalone model with Z-score standardization. All these models didn't gave any good results with Z-score standardization.
  * Changed the standardization to different ones like log, log(1+x), sigmoid, tanh etc. Finally, settling for percentile linearlization.
  * For further exploration tried stacking models created from stratified sampled out data with significant contribution from minor class also. This stacked model approach and the single created models didn't gave any accuracy boost.
  * After that, shifted to majority vote ensembling from XGBoost, lightgbm, randomforest. Also, tried multiple different combinations from different models like ada-boost, svm etc.
  * Tried stacking the models mentioned above but in their weak learner form with final predictor as XGBoost regression and classifier model. This stacking approach failed miserably.
  * Finally, moved to traning best model from each algorithm like lightgbm, XGBoost, RandomForest, LogisitcRegression etc.
    * Logisitc Regression model outperformed any of the above approaches with percentile linearlization standardization.

#### Final Model Building 

Since, f1-score is the final evaluation criteria models like logisitic regression performs significantly well for this classification problem even better than the multiple ensemble models created with decision tree based algorithms. Hence, a lucid and light logistic regression is trained with cross-validation on the most optimized parameters for achieving the final results.

* Since, data is highly skewed for this binary class prediction problem we have used `class_weight` to regularize on the distribution issue which led to significant result improvement.
* Normalization part of data is not done as it didn't lead to any performance improvement after percentile linearlization.
* We carry out grid search to optimize our model of logistic regression. Also, we carry out OOF cross-validation for this model both while searching parameters and training the final model on optimized parameters.
* Finally, the area under ROC curve was also considered for selecting the best model. As, f1-score was the criteria area under ROC curve is also an equivalently good loss function for optimization.
* In the end we carried out 5 fold CV for our logistic regression model and converted the predictions into classifier results accordingly.


#### Improvement Scope

The approaches that could have resulted in the improved results with this model architecture.

* Data generation techniques like SMOTE got skipped out of the model creation pipeline analysis.
* Also, feature generation and elimination recursive techniques or data transformation from PCA also weren't explored.
* Outlier detection approach with One class SVM model also wasn't explored.

---

### Repository Description

This repository contains codes on multiple approaches exploring performance of machine learning models on highly skewed data upto 99.6% of data belonging to only class only. It is a solution submitted on India ML Hiring Hiring Hackathon 2019 on Analytics Vidhya. This solution was ranked 410 on public score board with only logistic regression implementation with minor feature engineering. Also, other approaches explored and implemented are also specified in the repository. 

### Model Elaboration

* Logisitic Regression: It is specifically good at handling Low Precision/High Recall or High Precision/Low Recall cases. Where `Precision` being defined as "How many selected items(TP+FP) are relevant(TP) ?" and `Recall` being defined as "How many relevant items(TP+FN) are selected(TP) ?"


__Note:__ The `data.zip` contains the dataset from the given competition: India ML Hiring Hiring Hackathon 2019 on Analytics Vidhya for download.