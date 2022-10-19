# 安装R包
# install.packages("randomForest")
# install.packages("party")

# 语法在R中创建随机林的基本语法是 

# randomForest(formula, data) # formula-是描述预测变量和响应变量的公式。data-是使用的数据集的名称

# 输入数据我们将使用名为readingSkills的R内置数据集来创建一个决策树 
# 如果我们知道变量："age","shoesize","score"以及"nativeSpeaker"表示该人员是否为讲母语的人，那么它描述某个人员的阅读技能的得分

# Load the party package. It will automatically load other required packages.
library(party)
library(pROC)

# Print some records from data set readingSkills.
print(head(readingSkills))


# setwd("F:/worksp/R")
# Load the party package. It will automatically load other required packages.
library("party")
library("randomForest")

# Create the forest.
output.forest <- randomForest(group~age+dishistory+index1+index2+index3+index4+index5+index6,data = train)


# View the forest results.
print(output.forest) 
output.forest$importance

# Importance of each predictor.
print(importance(output.forest,type = 2))

#变量重要性的判断
varImpPlot(output.forest, main = "variable importance")




