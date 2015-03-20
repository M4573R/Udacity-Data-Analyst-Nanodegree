# Identifying Fraud from Enron Email and Financial Data
Mofei Li  
March 14, 2015  




###<font color='#068EDB'>Project Overview</font>
This the fourth project of [Nanodegree of Data Analyst of Udacity](https://www.udacity.com/course/nd002). It's for the course Intro to Machine Learning and you can find the related code and dataset [here on github](https://github.com/lmf90409/Udacity-Data-Analyst-Nanodegree/tree/master/P4).  

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives.  
In this project, I tried four supervised methods -- **Gaussian Naive Bayes**, **Random Forest**, **Logistic Regression** and **Support Vector Classifier** -- to detect person of interest (POI) in the fraud case. POIs are individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange for prosecution immunity based on financial and email data made public as a result of the Enron scandal.  

###<font color='#068EDB'>Features and Outliers</font>

####<font color=#4A79D9'>Remove Outliers</font>
![](https://raw.githubusercontent.com/lmf90409/Udacity-Data-Analyst-Nanodegree/master/P4/plots%20of%20outliers-1.png) 

I removed two subjects from the dataset -- 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK'. 'TOTAL' is a spreadsheet quirk that can be easily detected using the point plot "salary" vs "bonus" (the blue point). After removing 'TOTAL', I expected all the other subjects in the dataset are qualified individuals, however, I found 'THE TRAVEL AGENCY IN THE PARK', which I don't have more information to explain why it was in the dataset per se, but removed it anyway since it's more like a group rather than an individual.

####<font color=#4A79D9'>Add New Features</font>
I added four new features based on the existing features:  
 - '**fraction_from_poi**': the ratio of 'from_poi_to_this_person' to 'to_messages'  
 - '**fraction_to_poi**': the ratio of 'from_this_person_to_poi' to 'from_messages'  
 - '**shared_poi_per_email**' the ratio of 'shared_receipt_with_poi' to 'to_messages'  
 - '**email_exists**': indicator of whether one person has a valid 'email_address'  
 

|    | variable                  | score |
|--- | :------------------------ | :-------:|
|1    |exercised_stock_options    |24.72 |
|2    |total_stock_value          |24.40 |
|3    |fraction_to_poi            |22.98 |
|4    |bonus                      |22.07 |
|5    |salary                     |16.54 |
|6    |restricted_stock           |12.12 |
|7    |long_term_incentive        |11.32 |
|8    |deferred_income            |11.26 |
|9    |expenses                   |9.41  |
|10   |total_payments             |9.24  |
|11   |loan_advances              |7.49  |
|12   |shared_receipt_with_poi    |7.20  |
|13   |shared_poi_per_email       |6.96  |
|14   |email_exists               |5.35  |
|15   |other                      |4.62  |
|16   |director_fees              |2.02  |
|17   |fraction_from_poi          |1.84  |
|18   |to_messages                |1.35  |
|19   |from_messages              |0.15  |
|20   |deferral_payments          |0.09  |

Then using the `SelectKBest` function in `sklearn.feature_selection` module, I got the highest 20 features and their scores. 'fraction_to_poi' ranks the 3th, 'shared_poi_per_email', 'email_exists' and 'fraction_from_poi' in the 13th, 14th and 17th place, respectively.

###<font color='#068EDB'>Four Classifiers Q&A</font>

####<font color=#4A79D9'>Gaussian Naive Bayes</font>
```{python}
clf = GaussianNB()
```

 - **Q: Scaled the features or not?**  
 - **A:** No, I didn't scale the features since Naive Bayes Clssifier is famous for 'naively' treating all features independently and therefore there's no need to scale features.  
 
 - **Q: How many features did you use? What are they and how did you choose them?**  
 - **A:** I ended up using eight features in total and they are:  
 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'fraction_to_poi', 'deferred_income', 'long_term_incentive' and 'restricted_stock'.   
 I tried different number of features ranging from 5 to 15 and the model with 8 features had the best performance.  
  
####<font color=#4A79D9'>Random Forest</font>
```{python}
clf = RandomForestClassifier(max_depth = 3, 
                             max_features = 'sqrt', 
                             n_estimators = 10, 
                             random_state = 42)
```

 - **Q: Scaled the features or not?**  
 - **A:** No, I didn't scale the features since random forest classifier make predictions based on information gain of each individual feature.  
 
 - **Q: How many features did you use? What are they and how did you choose them?**   
 - **A:** I ended up using eight features in total and they are:  
 'exercised_stock_options', 'total_stock_value', 'fraction_to_poi', 'bonus', 'salary', 'restricted_stock', 'long_term_incentive', 'deferred_income', 'expenses'.  
 For the random forest classifier, I wanted to only features of relatively big score (say > 9) so I kept 9 features as following:  
 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'fraction_to_poi', 'deferred_income', 'long_term_incentive', 'restricted_stock' and 'shared_poi_per_email'  
 However, the performance of the model was not so good. This is the only model I tried but failed to get both precision and recall reach at least 0.3. I recently read an [article](http://jmlr.csail.mit.edu/papers/v15/delgado14a.html) saying from 179 classifiers arising from 17 families, the classifiers most likely to be the bests in terms of accuracy are the random forest. But this made me think maybe it's a different story for precision and/or recall.
 
####<font color=#4A79D9'>Logistic Regression [FINAL MODEL]</font>
```{python}
clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', 
                                          random_state = 42))
])
```

 - **Q: Scaled the features or not?**  
 - **A:** Yes, I used `StandardScaler` since feature scaling is critical for the later regularization of logistic regression.  
 - **Q: How many features did you use? What are they and how did you choose them?**   
 - **A:** I used 16 features in total:  
 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'fraction_to_poi', 'deferred_income', 'long_term_incentive', 'restricted_stock', 'shared_poi_per_email', 'total_payments', 'shared_receipt_with_poi', 'loan_advances', 'email_exists', 'expenses', 'other', 'fraction_from_poi' 
 I tried to include more features and hopefully more information into the model and let regularization take care the rest.  
 
####<font color=#4A79D9'>Support Vector Machine</font>
```{python}
clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel = 'rbf', C = 1000, gamma = 0.0001, 
                           random_state = 42, class_weight = 'auto'))
])
```

 - **Q: Scaled the features or not?**  
 - **A:** Yes, I used `StandardScaler` to avoid larger numeric range of some features dominating smaller ones.  
 
 - **Q: How many features did you use? What are they and how did you choose them?**   
 - **A:** I ended up using eight features in total and they are:  
 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'fraction_to_poi', 'deferred_income', 'long_term_incentive' and 'restricted_stock'.   
 I tried different number of features ranging from 5 to 10 and the model with 8 features had the best performance.  


Classifier                Accuracy    Precision    Recall     F1       F2   
-----------------------  ----------  -----------  --------  -------  -------
GaussianNB                 0.846        0.454      0.365     0.405    0.380 
RandomForestClassifier     0.856        0.484      0.169     0.250    0.194 
LogisticRegression         0.792        0.330      0.542     0.410    0.480 
SVC                        0.751        0.326      0.693     0.443    0.566 

###<font color='#068EDB'>Thoughts on the Final Model and How I got there</font>
I held out 20% of the data as test set and put 80% into training set. Since the classe of labels are unbalanced -- 18 poi and 126 non-poi after removing "TOTAL" and "THE TRAVEL AGENCY IN THE PARK", I used stratified sampling method `StratifiedShuffleSplit` to make sure there are both of the classes in the training and test set.  

Parameter Tuning is of great importance in machine learning algorithms for the sake of bringing the best (performance) of model. From the table below, we can see how parameter tuning makes a big difference on the same model.  

While there's only one option of scoring when tuning parameters and different scoring parameter gives different model evaluation rules, the goal of the classifier is to get both precision and recall above 0.3 at the same time. To give precision and recall an even chance, I ran `GridSearchCV` on 'precision', 'recall' and 'f1' all seperately and then compared the result of the tuned parameters. If the parameters were tuned to the same values on all three scoring parameter (only happened to logistic regression model), then there you have it -- the undoubted champion set of parameters. But if different scoring parameter gives different tuned parameters (for the rest three classifiers), then I would campare the different result and choose the set of parameters: 1. with precision $\geq$ 0.3 and/or recall $\geq$ 0.3 2. a good balance between precision and recall

```{python}
lr_org = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
])

lr_tn = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', 
                                          random_state = 42))
])

lr_cla = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', 
                                          random_state = 42, class_weight = 'auto'))
])
```


LogisticRegression                     Accuracy    Precision    Recall     F1       F2   
------------------------------------  ----------  -----------  --------  -------  -------
Not tuned (lr_org)                      0.850        0.367      0.170     0.233    0.191 
Tuned (lr_tn) [FINAL]                   0.789        0.325      0.541     0.406    0.477 
Tuned w/ auto class_weight (lr_cla)     0.666        0.274      0.911     0.421    0.621 



**I chose the Logistic Regression as the final model** for three reasons. Firstly, the overall performance of the model is pretty good and well balanced among different scores (accuracy:0.789, precision: 0.325 and recall: 0.541). Secondly, it is the only model achieved the highest precision, recall and f1 at the same time when parameter tuning. Last but not the least, logistic regression is heavily used in text classification, so the model could be easily extented if email text features would be added in the future.


###<font color='#068EDB'>References</font>
[1]http://www.strath.ac.uk/media/departments/eee/cesip/cesipseminar/Combination_of_Support_Vector_Machine_and_Principal_Component_Analysis_for_classification_tasks.pdf  

[2]http://scikit-learn.org/stable/auto_examples/grid_search_text_feature_extraction.html  

[3]http://discussions.udacity.com/t/when-to-use-feature-scaling/12923  

[4]http://jmlr.csail.mit.edu/papers/v15/delgado14a.html

[5]http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
