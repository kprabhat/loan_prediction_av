setwd("F:/DA Projects/AV_Loan Prediction 3")


#initialization
train=read.csv("train.csv")
test=read.csv("test.csv")

train$Loan_ID=NULL
Loan_ID=test$Loan_ID
test$Loan_ID=NULL

#filling all the missing values with NA annd converting it to numerics
train$Gender[train$Gender==""]=NA
train$Gender=as.numeric(train$Gender)-2
train$Married[train$Married==""]=NA
train$Married=as.numeric(train$Married)-2
train$Dependents[train$Dependents==""]=NA
train$Dependents=as.numeric(train$Dependents)-1
train$Self_Employed[train$Self_Employed==""]=NA
train$Self_Employed=as.numeric(train$Self_Employed)-2
train$Education=as.numeric(train$Education)-1
train$Property_Area=as.numeric(train$Property_Area)

test$Gender[test$Gender==""]=NA
test$Gender=as.numeric(test$Gender)-2
test$Married[test$Married==""]=NA
test$Married=as.numeric(test$Married)-2
test$Dependents[test$Dependents==""]=NA
test$Dependents=as.numeric(test$Dependents)-1
test$Self_Employed[test$Self_Employed==""]=NA
test$Self_Employed=as.numeric(test$Self_Employed)-2
test$Education=as.numeric(test$Education)-1
test$Property_Area=as.numeric(test$Property_Area)



#multiple imputation by chained reaction

require(mice)
#simple=train[c("ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History")]
#simple=train[c("Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area")]
set.seed(145)
train=complete(mice(train))
test=complete(mice(test))

train_f=train


#new variable = income
train_f$income=train_f$ApplicantIncome+train_f$CoapplicantIncome
test$income=test$ApplicantIncome+test$CoapplicantIncome


#new variable = loan amount/income
train_f$ratio=(train_f$LoanAmount*1000)/train_f$income
test$ratio=(test$LoanAmount*1000)/test$income





#svm not fruitfull
install.packages("e1071")
require(e1071)
svm=svm(Loan_Status~Credit_History+Property_Area+income+ratio,data = train_f,kernel="radial")
pred_svm=predict(svm,newdata=test)
table(pred_svm)
MySubmission = data.frame(Loan_ID=Loan_ID,Loan_Status=pred_svm)
write.csv(MySubmission, "SubmissionSVM.csv", row.names=FALSE)


#randomforest (overfits the data)
require(randomForest)
forest=randomForest(Loan_Status~.,data=train,ntree=100,nodesize=21,cp=.39)
pred=predict(forest,newdata = test)
table(pred)
MySubmission = data.frame(Loan_ID=Loan_ID,Loan_Status=pred)
write.csv(MySubmission, "SubmissionForest.csv", row.names=FALSE)



#xgboost(overfits the data)
train_x=train_f[,c("Credit_History","Property_Area","income","ratio")]
label=train$Loan_Status
require(xgboost)
label=as.numeric(label)-1
bst=xgboost(data = as.matrix(train_x),label = label,max.depth = 20, eta = 1, nthread = 4, nround = 10, objective = "binary:logistic")
pred=predict(bst,as.matrix(test))
pred_bst = as.numeric(pred > 0.31)
table(pred_bst)
pred_bst=factor(pred_bst,levels = c(0,1),labels = c("N","Y"))
table(pred_bst)
MySubmission = data.frame(Loan_ID=Loan_ID,Loan_Status=pred_bst)
write.csv(MySubmission, "SubmissionXG.csv", row.names=FALSE)





#tree(best result with accuracy of .79167, submitted model)
#============================================================
require(rpart)
require(rpart.plot)
tree=rpart(Loan_Status~Credit_History+Property_Area+ApplicantIncome+ratio,data = train_f,minbucket=5,cp=.01)
prp(tree)
pred_tree=predict(tree,newdata = test,type = "class")
table(pred_tree)
MySubmission = data.frame(Loan_ID=Loan_ID,Loan_Status=pred_tree)
write.csv(MySubmission, "Submissiontree.csv", row.names=FALSE)






#tree with cv(cv comes out t0 be .39 but that made tree two simple)
train$income=train$ApplicantIncome+train$CoapplicantIncome-train$Loan_Amount_Term
test$income=test$ApplicantIncome+test$CoapplicantIncome-test$Loan_Amount_Term
install.packages("caret")
library(caret)
library(e1071)
numfolds=trainControl(method = "cv",number = 10)
cpGrid=expand.grid(.cp=seq(0.01,0.5,0.01))
train(Loan_Status~Credit_History+Property_Area,data=train,method="rpart",trControl=numfolds,tuneGrid=cpGrid)
tree=rpart(Loan_Status~Credit_History+Property_Area,data = train,minbucket=21,cp=.39)
prp(tree)
pred_tree=predict(tree,newdata = test,type = "class")
table(pred_tree)
MySubmission = data.frame(Loan_ID=Loan_ID,Loan_Status=pred_tree)
write.csv(MySubmission, "Submissiontree.csv", row.names=FALSE)


#gbm(also gives an accuracy .79167)
#===================================
require(gbm)
boost=gbm(Loan_Status~Credit_History+Property_Area+income+ratio,data = train_f,distribution = "gaussian",n.trees = 1000,shrinkage = .01,interaction.depth = 4)
pred=predict(boost,newdata = test,type = "response",n.trees = 100)
pred_gbm=rep("Y",nrow(test))
for (i in 1:nrow(test))
{ if(pred[i]>=1.6)
{next()}
  else 
  {
    pred_gbm[i]="N"
  }
}
table(pred_gbm)
MySubmission = data.frame(Loan_ID=Loan_ID,Loan_Status=pred_gbm)
write.csv(MySubmission, "Submissiongbm.csv", row.names=FALSE)








#logistic(also gives an accuracy of .79167)
#===========================================
log=glm(Loan_Status~Credit_History+Property_Area+income+ratio,data = train_f,family = binomial)
predlog=predict(log,type = "response",newdata = test)
pred_log=rep("Y",nrow(test))
for (i in 1:nrow(test))
{
  if(predlog[i]>=.73)
  {next()}
  else 
  {
    pred_log[i]="N"
  }
}
table(pred_log)
MySubmission = data.frame(Loan_ID=Loan_ID,Loan_Status=pred_log)
write.csv(MySubmission, "Submissionlog.csv", row.names=FALSE)
