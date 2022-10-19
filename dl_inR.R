########## Step 1: Installing and Loading the Packages ##########

# install.packages("mlbench")
# install.packages("deepnet")

library(mlbench)
library(deepnet)

########## Step 2: Choosing Dataset ########## 

data("BreastCancer")

# Clean off rows with missing data
BreastCancer = BreastCancer[which(complete.cases(BreastCancer)
                                  == TRUE),]
head(BreastCancer)
names(BreastCancer)


########## Step 3: Applying the deepnet package to the dataset ########## 

y = as.matrix(BreastCancer[, 11])
y[which(y == "benign")] = 0
y[which(y == "malignant")] = 1
y = as.numeric(y)
x = as.numeric(as.matrix(BreastCancer[, 2:10]))
x = matrix(as.numeric(x), ncol = 9)


########## Step 4: Modeling NN ########## 

# Applying nn.train() function
nn <- nn.train(x, y, hidden = c(5))
yy = nn.predict(nn, x)
print(head(yy))
yhat = matrix(0,length(yy), 1)
yhat[which(yy > mean(yy))] = 1
yhat[which(yy <= mean(yy))] = 0


########## Step 5: Creating a confusion matrix ########## 

# Applying table() function
cm = table(y, yhat)
print(cm)
print(sum(diag(cm))/sum(cm))


