---
title: "Identifying Fraud from Enron Email and Financial Data"
author: "Mofei Li"
date: "March 14, 2015"
output: html_document
---

```{r knitr global settings, echo=FALSE, message=FALSE, warning=FALSE}
library(knitr)
library(dplyr)
library(ggplot2)
opts_chunk$set(cache=TRUE, echo=FALSE, eval=TRUE, 
               message=FALSE, warning=FALSE)

### Set working directory
setwd("~/Documents/Udacity-Projects/P4")

### Read the dataset and remove the first column (which is row number)
enron.fne <- tbl_df(
    read.csv("./data/final_project_dataset.csv", #enron [f]inancial a[n]d [e]mails
             header = TRUE, stringsAsFactors = FALSE) )[ ,-1]

### Make "name" as the first column for easy identification
enron.fne <- enron.fne[c("name", names(enron.fne)[names(enron.fne) != "name"])] 

row.names(enron.fne) <- enron.fne$name
enron.fne$poi <- factor(enron.fne$poi, levels = c("True", "False"))
enron.fne$email_exists <- ifelse(enron.fne$email_address == "NaN", 0, 1)
```


###<font color='#068EDB'>Project Overview</font>
This the fourth project of [Nanodegree of Data Analyst of Udacity](https://www.udacity.com/course/nd002). It's for the course [Intro to Machine Learning] and you can find the related code and dataset [here on github](https://github.com/lmf90409/Udacity-Data-Analyst-Nanodegree/tree/master/P4).  

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives.  
In this project, I tried four supervised methods -- **Gaussian Naive Bayes**, **Random Forest**, **Logistic Regression** and **Support Vector Classifier** -- to detect person of interest (POI) in the fraud case. POIs are individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange for prosecution immunity based on financial and email data made public as a result of the Enron scandal.  

###<font color='#068EDB'>Features and Outliers</font>

####<font color=#4A79D9'>Remove Outliers</font>
```{r plots of outliers, fig.height=4, fig.width=10}

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  require(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

enron.fne_tot <- data.frame(
    salary = c(enron.fne$salary, 26704229),
    bonus = c(enron.fne$bonus, 97343619),
    total = factor(c(rep(0,145), 1))
    )

p1 <- ggplot(aes(x=salary, y=bonus, color=total), data=enron.fne_tot) +
    geom_point() +
    ggtitle('All the data')

p2 <- ggplot(aes(x=salary, y=bonus, color=total), 
       data=subset(enron.fne_tot,total==0)) +
    geom_point() +
    ggtitle("After removing 'TOTAL'")

multiplot(p1, p2, cols=2)
```

I removed two subjects from the dataset -- 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK'. 'TOTAL' is a spreadsheet quirk that can be easily detected using the point plot "salary" vs "bonus" (the blue point). After removing 'TOTAL', I expected all the other subjects in the dataset are qualified individuals, however, I found 'THE TRAVEL AGENCY IN THE PARK', which I don't have more information to explain why it was in the dataset per se, but removed it anyway since it's more like a group rather than an individual.

####<font color=#4A79D9'>Add New Features</font>
I added four new features based on the existing features:  
 - '**fraction_from_poi**': the ratio of 'from_poi_to_this_person' to 'to_messages'  
 - '**fraction_to_poi**': the ratio of 'from_this_person_to_poi' to 'from_messages'  
 - '**shared_poi_per_email**' the ratio of 'shared_receipt_with_poi' to 'to_messages'  
 - '**email_exists**': indicator of whether one person has a valid 'email_address'  
 
```{r scores of features}

feature.importance <- tbl_df(
    read.csv("./data/importance.csv", 
             header = TRUE, stringsAsFactors = FALSE) )[ ,-1]

feature.importance <- feature.importance %>%
    distinct %>%
    slice(1:20)


kable(feature.importance, align = c("l", "c"), digits = 2,row.names = TRUE)
```

Then using the `SelectKBest` function in `sklearn.feature_selection` module, I got the highest 20 features and their scores. 'fraction_to_poi' ranks the `r which(feature.importance$variable == 'fraction_to_poi')`th, 'shared_poi_per_email', 'email_exists' and 'fraction_from_poi' in the `r which(feature.importance$variable == 'shared_poi_per_email')`th, `r which(feature.importance$variable == 'email_exists')`th and `r which(feature.importance$variable == 'fraction_from_poi')`th place, respectively.

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

```{r performance table}
clf.name <- c("GaussianNB", "RandomForestClassifier","LogisticRegression", "SVC")
clf.acc <- c(0.84650, 0.85550, 0.79200, 0.75143)
clf.prc <- c(0.45370, 0.48355, 0.32968, 0.32596)
clf.rcl <- c(0.36500, 0.16900, 0.54200, 0.69300)
clf.f1 <- c(0.40454, 0.25046, 0.40998, 0.44338)
clf.f2 <- c(0.37985, 0.19428, 0.48016, 0.56562)

clf.pfc <- data.frame(Classifier = clf.name, 
                      Accuracy = clf.acc,
                      Precision = clf.prc,
                      Recall = clf.rcl,
                      F1 = clf.f1,
                      F2 = clf.f2)
kable(clf.pfc, digits = 3, align = c("l", rep("c", 5)))

```

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

```{r logistic regression performance table}
lrg.name <- c("Not tuned (lr_org)", "Tuned (lr_tn) [FINAL]", "Tuned w/ auto class_weight (lr_cla)")
lrg.acc <- c(0.85027, 0.78880, 0.66567)
lrg.prc <- c(0.36746, 0.32473, 0.27361)
lrg.rcl <- c(0.17050, 0.54100, 0.91100)
lrg.f1 <- c(0.23292, 0.40585, 0.42083)
lrg.f2 <- c(0.19097, 0.47741, 0.62146)

lrg.pfc <- data.frame(LogisticRegression = lrg.name, 
                      Accuracy = lrg.acc,
                      Precision = lrg.prc,
                      Recall = lrg.rcl,
                      F1 = lrg.f1,
                      F2 = lrg.f2)
kable(lrg.pfc, digits = 3, align = c("l", rep("c", 5)))

```



**I chose the Logistic Regression as the final model** for three reasons. Firstly, the overall performance of the model is pretty good and well balanced among different scores (accuracy:0.789, precision: 0.325 and recall: 0.541). Secondly, it is the only model achieved the highest precision, recall and f1 at the same time when parameter tuning. Last but not the least, logistic regression is heavily used in text classification, so the model could be easily extented if email text features would be added in the future.


###<font color='#068EDB'>References</font>
[1]http://www.strath.ac.uk/media/departments/eee/cesip/cesipseminar/Combination_of_Support_Vector_Machine_and_Principal_Component_Analysis_for_classification_tasks.pdf  

[2]http://scikit-learn.org/stable/auto_examples/grid_search_text_feature_extraction.html  

[3]http://discussions.udacity.com/t/when-to-use-feature-scaling/12923  

[4]http://jmlr.csail.mit.edu/papers/v15/delgado14a.html

[5]http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

```{r, include=FALSE}
   # add this chunk to end of mycode.rmd
   file.rename(from="Questions for Scaled Project.Rmd", 
               to="trytry.md")
```