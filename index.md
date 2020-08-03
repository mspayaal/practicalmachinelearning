---
title: "Practical Machine Learning Course Project"
author: "Mahendra S Payaal"
date: "8/3/2020"
output: 
  html_document: 
    keep_md: yes
---
#  Practical Machine Learning Assignment to Predict the Manner of Exercise 

## Synopsis

The human activity recognition research thorugh fitbit etc. has traditionally focused on "which" activity was performed at a specific point in time. In the data provided the purpose was to investigate  "how (well)" an activity was performed by the wearer of a device. Six young healthy participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A) and in  incorrect mannner( Class B to E). The purpose of the project was to predict the type of activity i.e. from 'A' to 'E' through the training set provided and then use it to predict for a test set of 20 samples. The data set was downloaded, cleaned and then the prediction model giving at least 99% accuracy was adopted. Finally the test set was predicted.

## Reading and Cleaning data set 

### Reading the training and test data 


```r
trainWts <- read.csv("C:/Users/mahendra.payaal/Documents/coursera/data/pml-training.csv")
testWts <- read.csv("C:/Users/mahendra.payaal/Documents/coursera/data/pml-testing.csv")
names(testWts) <- names(trainWts)
dim(trainWts)
```

```
## [1] 19622   160
```

```r
dim(testWts)
```

```
## [1]  20 160
```

There are 19,622 samples in training data 'trainWts' and 20 in testing data(testWts) with 160 variables in both.

### Cleaning the Data

The data was opened in Notepad++ and it was noticed that there were a number of blank spaces, "#DIV/0!" and NAs. The first two were also replaced with NA and thereafter those columns with NAs of 50% or more were deleted. The columns with 50% or more NAs do not serve any credible purpose even after imputing values. Columns 1 to 7 were also removed as these were about row number, name of person, time stamps etc. and not relevant to analysis.


```r
# replacing blanks and 'DIV/0' by NA
trainWts[trainWts==""]<- NA;trainWts[trainWts=="#DIV/0!"]<- NA
testWts[testWts==""]<- NA;testWts[testWts=="#DIV/0!"]<- NA
# finding columns with >50% NAs in training data and dropping them
findNA_train <- apply(trainWts,2,is.na)
findNA_train <- apply(findNA_train,2,sum)
findNA_train <- findNA_train[findNA_train < 9811]
# Making a new data frame after dropping column with > 50% NAs and  # columns 1 to7 
trainWts1 <- subset(trainWts,select = names(findNA_train))
dim(trainWts1)
```

```
## [1] 19622    60
```

```r
# dropping first 7 columns
trainWts1 <- trainWts1[,-(1:7)]
# same dropping of columns for test data set 
testWts1 <- subset(testWts,select=names(trainWts1))
dim(trainWts1);dim(testWts1)
```

```
## [1] 19622    53
```

```
## [1] 20 53
```

we now see that the cleaned  training datasets 'trainWts1' is reduced to 53 variables from 160. Same is the case  for cleaned test data set'testWts1'.

## Model development

Since Random Forest and Boosting are the two most accurate prediction models we will first look at these two models as our aim is to get 99% accuracy. To train the modelwe will divide the  'trainwts1' into  training and testing datasets


```r
library(randomForest);library(caret)
```

```
## Warning: package 'randomForest' was built under R version 3.6.3
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Warning: package 'caret' was built under R version 3.6.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.6.3
```

```
## 
## Attaching package: 'ggplot2'
```

```
## The following object is masked from 'package:randomForest':
## 
##     margin
```

```r
inTrain <- createDataPartition(y= trainWts1$classe,p=0.7,list= FALSE)
training <- trainWts1[inTrain,];testing <- trainWts1[-inTrain,]
modFit <- randomForest(classe ~ ., data = training, ntree = 1000)
pred <- predict(modFit,testing)
confusionMatrix(pred,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    3    0    0    0
##          B    0 1136    6    0    0
##          C    0    0 1019    9    0
##          D    0    0    1  953    5
##          E    0    0    0    2 1077
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9956          
##                  95% CI : (0.9935, 0.9971)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9944          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9974   0.9932   0.9886   0.9954
## Specificity            0.9993   0.9987   0.9981   0.9988   0.9996
## Pos Pred Value         0.9982   0.9947   0.9912   0.9937   0.9981
## Neg Pred Value         1.0000   0.9994   0.9986   0.9978   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1930   0.1732   0.1619   0.1830
## Detection Prevalence   0.2850   0.1941   0.1747   0.1630   0.1833
## Balanced Accuracy      0.9996   0.9981   0.9957   0.9937   0.9975
```

We find that 99% accuracy in the prediction model has been achieved through 'Random Forest'.

##  Predicting on actual test set 
 

```r
pred_final <- predict(modFit,testWts1)
pred_final
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

The prediction on actual 20 samples can be seen above.
