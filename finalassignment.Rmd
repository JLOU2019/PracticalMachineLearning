---
title: "Practical Machine Learning: Final Assignment"
author: "JL"
date: "30/04/2020"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
## Overview

A substantial amount of data about personal actively is collectable now relatively inexpensively by using devices such as Jawbone Up, Nike FuelBand and Fitbit. These type of devices are part of the quantified self-movement - a group of enthusiasts who take measurements about themselves regularly to improve their health or to find patterns in their behavior. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumb bell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The data consists of two data sets: Training data and Test data (to be used to validate the selected model).

The goal of the project is to predict the manner in which they did the exercise. This is the “classe” variable in the training data set. Other variables may also be used for prediction.

Note: The dataset used in this project is a courtesy of “Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers’ Data Classification of Body Postures and Movements”

## Loading and Processing Data and the relevant packages

```{r}
library(mnormt)
library(Rcpp)
library(lava)
library(numDeriv)
library(caret)
library(corrplot)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(rpart)
library(rpart.plot)
library(gbm)
library(e1071)
```

## Data access, cleaning and exploration

```{r}
setwd("~/Modelling team - tasks/Training/Coursera - Data Science Specilaisation/Course - Practical machine learning")
train_in <- read.csv('./pml-training.csv', header=T)
valid_in <- read.csv('./pml-testing.csv', header=T)
dim(train_in)
dim(valid_in)
```
The functions above provide some information about the dimensions of the data sets: there are 19,622 observations and 160 variables in the Training dataset, whereas the Testing dataset comprises 20 observations and 160 variables.

### Removing missing data
The variables that contained missing values were removed from the data sets.  After this data cleaning process, the dimensions of both datasets have changed compared to the original dimensions of these data sets.

```{r}
trainData<- train_in[, colSums(is.na(train_in)) == 0]
validData <- valid_in[, colSums(is.na(valid_in)) == 0]
dim(trainData)
dim(validData)
```

### Removing the first seven variables which have limited effect on the 'classe' outcome

```{r}
trainData <- trainData[, -c(1:7)]
validData <- validData[, -c(1:7)]
dim(trainData)
dim(validData)
```

## Prepare the datasets for analysis
Took the following steps to prepare the datasets : spilt the training data which comprised 70% of train data and 30% of test data. This arrangement also helped to compute the out-of-sample errors.

The test data was used later to test the production algorithm on the 20 cases. Note the dimensions of the datasets have changed after the spilt.

```{r}
set.seed(1234) 
inTrain <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
trainData <- trainData[inTrain, ]
testData <- trainData[-inTrain, ]
dim(trainData)
dim(testData)
```

### Removed the variables that are near zero variance
The datasets have 53 variables (rather than 86) after the 'near zero variance' variables have been removed.

```{r}
NZV <- nearZeroVar(trainData)
trainData <- trainData[, -NZV]
testData  <- testData[, -NZV]
dim(trainData)
dim(testData)
```

The correlation plot below (source: corrplot) provides some clues of the correlated predictors (or variables), which are those with dark colour intersection.

```{r}
cor_mat <- cor(trainData[, -53])
corrplot(cor_mat, order = "FPC", method = "color", type = "upper", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

In order to identify the names of the relevant variables as depicted in the plot above, the 'findCorrelation' function was performed as the first step to search for highly correlated attributes, with a cut off rate equal to 0.75.

```{r}
highlyCorrelated = findCorrelation(cor_mat, cutoff=0.75)
```

The names of highly correlated attributes were obtained by the function below.

```{r}
names(trainData)[highlyCorrelated]
```

## Building the prediction model

For this project, three algorithms were used to predict the outcome: Classification Tree, Generalised Boosted Model and Random Forest.

### Classification Tree: the prediction
The 'fancyRpartPlot' function was used to plot the classification tree as a dendrogram.

```{r}
set.seed(12345)
decisionTreeMod1 <- rpart(classe ~ ., data=trainData, method="class")
fancyRpartPlot(decisionTreeMod1)
```

The model was validated on the testData to find out the level of accuracy by looking at the 'Accuracy' variable.

```{r}
predictTreeMod1 <- predict(decisionTreeMod1, testData, type = "class")
cmtree <- confusionMatrix(predictTreeMod1, testData$classe)
cmtree
```

Plot matrix results by the following function.

```{r}
plot(cmtree$table, col = cmtree$byClass, 
     main = paste("Decision Tree - Accuracy =", round(cmtree$overall['Accuracy'], 4)))
```

The graph above shows that the accuracy rate of the model is 0.7642, with the out-of-sample error around 0.24.

### Generalised Boosted Model: the prediction

```{r}
set.seed(12345)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modGBM  <- train(classe ~ ., data=trainData, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
modGBM$finalModel
print(modGBM)
```

Validate the Generalised Boosted Model

```{r}
predictGBM <- predict(modGBM, newdata=testData)
cmGBM <- confusionMatrix(predictGBM, testData$classe)
cmGBM
```

The accuracy rate using the Generalised Boosted Model is fairly high (0.9731), with the out-of-sample error around 0.0269.

### Random Forest: the prediction

```{r}
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modRF1 <- train(classe ~ ., data=trainData, method="rf", trControl=controlRF)
modRF1$finalModel
```

Validate the Random Forest Model to find out the level of accuracy.

```{r}
predictRF1 <- predict(modRF1, newdata=testData)
cmrf <- confusionMatrix(predictRF1, testData$classe)
cmrf
```

## Applied the best model to the 'Test Data'
Given the accuracy rates of the three models, it is obvious that the 'Random Forest' model produced the best prediction with an accuracy rate of 1.  The model was therefore used to validate the test data accordingly and used to answer the 'Prediction Quiz'.

```{r}
Results <- predict(modRF1, newdata=validData)
Results
```

