rm(list = ls())

# 安装R包
if(!require(rpart))install.packages('rpart',update = F,ask = F)
if(!require(tibble))install.packages('tibble',update = F,ask = F)
if(!require(bitops))install.packages('bitops',update = F,ask = F)
if(!require(rpart.plot))install.packages('rpart.plot',update = F,ask = F)
if(!require(RColorBrewer))install.packages('RColorBrewer',update = F,ask = F)
if(!require(pROC))install.packages('pROC',update = F,ask = F)
if(!require(mice))install.packages('mice',update = F,ask = F)
if(!require(klaR))install.packages('klaR',update = F,ask = F)

# 加载R包
library(rpart)
library(tibble)
library(bitops)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(data.table)
library(pROC)
library(e1071)
library(mice)

######################## Decision Tree ########################

# 读取数据
mydata <- read.csv("data.csv",header=T) 

# 训练集与验证集划分,50个样例作为训练集，其余作为测试集
sub<-sample(1:100,50)
train<-mydata[sub,]
test<-mydata[-sub,]

#构建决策树并绘制图形,利用训练集构建决策树：
model <- rpart(group~age+dishistory+index1+index2+index3+index4+index5+index6,data = train)
fancyRpartPlot(model)

#测试模型,利用验证集对模型结果进行验证：
x<-subset(test,select=-group)
pred<-predict(model,x,type="class")
k<-test[,"group"]
table(pred,k)


######################## SVM ########################
# 读取数据
data=fread('data.csv')
data=data.frame(data)
data$group =factor(data$group)

# 数据预处理
imputed_Data <- mice(data) # 利用mice包填补缺失值
data=complete(imputed_Data)

# 按 7:3 分训练集和测试
set.seed(1234) # 随机抽样设置种子
train <- sample(nrow(data),0.7*nrow(data)) # 抽样函数
tdata <- data[train,] # 根据抽样参数列选择样本，都好逗号是选择行
vdata <- data[-train,] # 删除抽样行

# 模型构建
cats_svm_model <- svm(group~.,data = tdata,
                        type="C-classification", kernel="linear",cost=10,scale=FALSE)

# 模型预测
predtree <- predict(cats_svm_model,newdata=vdata,type="group") # 利用预测集进行预测

# 输出混淆矩阵
table(vdata$group,predtree,dnn=c("true", "predit")) # 输出混淆矩阵

# 绘制ROC曲线
ran_roc <-roc(vdata$group ,as.numeric(predtree))
png(file = "ROC.png")
plot(ran_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"),max.auc.polygon=TRUE,auc.polygon.col="skyblue", 
     print.thres=TRUE,main='SVM')
dev.off()


######################## naive bayesian ########################

library(e1071); library(klaR)
naiveBayes.model <- naiveBayes(Species ~ ., data = iris)
iris_predict <- predict(naiveBayes.model, newdata = iris)
table_iris <- table(actual = iris$Species, predict = iris_predict)
1 - sum(diag(table_iris))/sum(table_iris) # 计算误差



######################## Xgboost ########################

url <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
redwine <- read.csv(url,sep = ';')
url1 <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
whitewine <- read.csv(url1,sep = ';')
data <- rbind(redwine,whitewine)

#训练集、测试集划分
set.seed(17)  
index <-  which( (1:nrow(data))%%3 == 0 )
train <- data[-index,]
test <- data[index,]

# 数据预处理,将数据转化为gb.DMatrix类型，并对label进行处理，超过6分为1，否则为0。
library("xgboost")
library("Matrix")
library(Ckmeans.1d.dp)

train_matrix <- sparse.model.matrix(quality ~ .-1, data = train)
test_matrix <- sparse.model.matrix(quality ~ .-1, data = test)
train_label <- as.numeric(train$quality>6)
test_label <-  as.numeric(test$quality>6)
train_fin <- list(data=train_matrix,label=train_label) 
test_fin <- list(data=test_matrix,label=test_label) 
dtrain <- xgb.DMatrix(data = train_fin$data, label = train_fin$label) 
dtest <- xgb.DMatrix(data = test_fin$data, label = test_fin$label)

#模型训练
xgb <- xgboost(data = dtrain,max_depth=6, eta=0.5,  
               objective='binary:logistic', nround=25)
#重要重要性排序 
importance <- xgb.importance(train_matrix@Dimnames[[2]], model = xgb)  
head(importance)
xgb.ggplot.importance(importance)

#混淆矩阵
pre_xgb = round(predict(xgb,newdata = dtest))
table(test_label,pre_xgb,dnn=c("true","pre"))

#ROC曲线
xgboost_roc <- roc(test_label,as.numeric(pre_xgb))
plot(xgboost_roc, print.auc=TRUE, auc.polygon=TRUE, 
     grid=c(0.1, 0.2),grid.col=c("green", "red"), 
     max.auc.polygon=TRUE,auc.polygon.col="skyblue", 
     print.thres=TRUE,main='ROC curve')


######################## PCA ########################

#将R自带的范例数据集iris储存为变量data;
data<-iris
head(data)
#对原数据进行z-score归一化；
dt<-as.matrix(scale(data[,1:4]))
head(dt)

#计算相关系数矩阵；
rm1<-cor(dt)
rm1

# 求解特征值和相应的特征向量
rs1<-eigen(rm1)
rs1

#提取结果中的特征值，即各主成分的方差；
val <- rs1$values
#换算成标准差(Standard deviation);
(Standard_deviation <- sqrt(val))
#计算方差贡献率和累积贡献率；
(Proportion_of_Variance <- val/sum(val))
(Cumulative_Proportion <- cumsum(Proportion_of_Variance))


#碎石图绘制;
par(mar=c(6,6,2,2))
plot(rs1$values,type="b",
     cex=2,
     cex.lab=2,
     cex.axis=2,
     lty=2,
     lwd=2,
     xlab = "主成分编号",
     ylab="特征值(主成分方差)")


# 计算主成分得分
#提取结果中的特征向量(也称为Loadings,载荷矩阵)；
(U<-as.matrix(rs1$vectors))
#进行矩阵乘法，获得PC score；
PC <-dt %*% U
colnames(PC) <- c("PC1","PC2","PC3","PC4")
head(PC)

# 绘制主成分散点图
#将iris数据集的第5列数据合并进来；
df<-data.frame(PC,iris$Species)
head(df)

#载入ggplot2包；
library(ggplot2)
#提取主成分的方差贡献率，生成坐标轴标题；
xlab<-paste0("PC1(",round(Proportion_of_Variance[1]*100,2),"%)")
ylab<-paste0("PC2(",round(Proportion_of_Variance[2]*100,2),"%)")
#绘制散点图并添加置信椭圆；
p1<-ggplot(data = df,aes(x=PC1,y=PC2,color=iris.Species))+
        stat_ellipse(aes(fill=iris.Species),
                     type ="norm", geom ="polygon",alpha=0.2,color=NA)+
        geom_point()+labs(x=xlab,y=ylab,color="")+
        guides(fill=F)
p1

# 尝试使用3个主成分绘制3D散点图
# 载入scatterplot3d包；
library(scatterplot3d)
color = c(rep('purple',50),rep('orange',50),rep('blue',50))
scatterplot3d(df[,1:3],color=color,
              pch = 16,angle=30,
              box=T,type="p",
              lty.hide=2,lty.grid = 2)
legend("topleft",c('Setosa','Versicolor','Virginica'),
       fill=c('purple','orange','blue'),box.col=NA)


######################## 聚类算法 ########################

#################### k-means ####################

# kmeans对iris进行聚类分析 

iris2<-iris[,1:4]
iris.kmeans<-kmeans(iris2,3)
?kmeans
iris.kmeans

#用table函数查看分类结果情况
table(iris$Species,iris.kmeans$cluster)

#下边我们将分类以及中心点打印出来

plot(iris2$Sepal.Length,iris2$Sepal.Width,
     col=iris.kmeans$cluster,pch="*")
points(iris.kmeans$centers,pch="X",cex=1.5,col=4)


#-----使用K-mediods方法来进行聚类分析
#k-mediods中包含pam、clara、pamk三种算法，我们通过iris数据集来看看三者表现

# install.packages("cluster")
library(cluster)

## pam
iris2.pam<-pam(iris2,3)
table(iris$Species,iris2.pam$clustering)
layout(matrix(c(1,2),1,2)) #每页显示两个图
plot(iris2.pam)
layout(matrix(1))

## clara
iris2.clara<-clara(iris2,3)
table(iris$Species,iris2.clara$clustering)
layout(matrix(c(1,2),1,2)) #每页显示两个图
plot(iris2.clara)
layout(matrix(1))

## pamk
install.packages("fpc")
library(fpc)
iris2.pamk<-pamk(iris2)
table(iris2.pamk$pamobject$clustering,iris$Species)
layout(matrix(c(1,2),1,2)) #每页显示两个图
plot(iris2.pamk$pamobject)
layout(matrix(1))

#通过上述分类结果可以看到，pam和calra算法分类结果基本类似，但是pamk将三类分为了两类。


#################### DBSCAN ####################

# install.packages("fpc")
# install.packages("dbscan")

#---基于密度的聚类分析
library(fpc)
iris2<-iris[-5]
ds<-dbscan(iris2,eps=0.42,MinPts = 5)
table(ds$cluster,iris$Species)

#打印出ds和iris2的聚类散点图
plot(ds,iris2)

#打印出iris第一列和第四列为坐标轴的聚类结果
plot(ds,iris2[,c(1,4)])

#另一个表示聚类结果的函数，plotcluster
plotcluster(iris2,ds$cluster)


#################### 层次聚类 ####################

#---层次聚类 
dim(iris)#返回行列数

idx<-sample(1:dim(iris)[1],40)
iris3<-iris[idx,-5]
iris3
hc<-hclust(dist(iris3),method = "ave")  #注意hcluster里边传入的是dist返回值对象

plot(hc,hang=-1,labels=iris$Species[idx])  #这里的hang=-1使得树的节点在下方对齐
#将树分为3块
rect.hclust(hc,k=3)  
groups<-cutree(hc,k=3)   










