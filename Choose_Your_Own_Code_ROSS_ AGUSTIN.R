# Hello to everyone, this is my code for the "Choose your own" part of the capstone.
# I have decided to create different models to try to predict the Default (the fail in payment)
# for a debt in a credit card. 
# The Data is a very small sample provided by a financial company from Argentina (my country).

# So, let's start.

#First, I load the required packages:


detach(package:neuralnet,unload = T)
library(corrplot)
library(tidytable)
library(readxl)
library(plyr)
library(dplyr)
library(car)
library(stats)
library(reshape)
library(lmtest)
library(Rsolnp)
library(openxlsx)
library(memisc)
library(foreign)
library(ROCR)
library(InformationValue)
library(pscl)
library(MASS)
library(ggplot2)
library(ggthemes)
library(rpart)
library(partykit)
library(nnet)

# I set the path.
DB_path <- "https://raw.githubusercontent.com/agustinross/Capstone_Harvard/main/Choose_Your_Own_database.xlsx"

# Then, I get the data.
DataBase <- as.data.frame(read.xlsx(DB_path))

# The DB has 12 columns. 11 independent variables (X) and 1 dependent (Y). The dependent variable is a flag that
# shows if the person has paid the credit card debt or not. It is a binary variable.
ncol(DataBase)


## UNIVARIATE ANALYSIS OF THE INDEPENDET VARIABLES

# Let's now study the distribution of the independent variables, a simple way to do this is with a histogram:

# The first variable is "Commercial References", it shows the amount of times the person under analysis
# did not pay some expenses like electricity and cellphone bills (here in Argentina you can buy this type of 
# information).
hist((DataBase$CR))
# The majority of the observations are distributed around the "0", as expected.

# The second variable is "Quantity of credit cards the person owns".
hist(DataBase$QCC)
# As expected, the distribution of the variable is also around "0" .


# The third variable is a very interesting one, it is "Basic Needs". It shows the percentage of basic needs
# that are covered in the region the person lives (it can be a huge indicator for the good or bad behavior 
# in payments).
hist(DataBase$BN)
# It is distributed around 91 (it's mean). We can also see a low variance, and it is almost impossible to
# find an observation below the 60 points. In this case, tt could be suitable to standardize the variable, 
# so I subtracted  the mean and divided by the SD.

BN_Stand <- (BN_Stand=((DataBase$BN-mean(DataBase$BN))/sd(DataBase$BN)))
hist(BN_Stand)

# Now it is distributed around "0".

# The fourth variable is "Age".
hist(DataBase$Age )
# Since this is supposed to apply to people that are not underage I truncated 
# the data to 18 years old.
# It also can be seen how we have just a few observations over the 80 years old, so it could be appropriated
# to truncated it to 80.
# We can also see that the data has high variance, and we can also see how it is not normally distributed. So, I 
# proceeded to apply the LN function to try to catch better the prediction power of the variable. 
Age_LN <- (Age_LN=log(ifelse(DataBase$Age<18,18,DataBase$Age)-17))

hist(Age_LN)


# The fifth variable is the sector where the person work. PRIV = Private sector, PUB = Public sector , 
# JUB = Retired person, UNIV = University teacher , RN = Worker for the state "Rio Negro"  in Argentina.
ggplot(as.data.frame(DataBase$Work), aes(x=reorder(DataBase$Work, DataBase$Work, function(x)-length(x)))) +
  geom_bar(fill='pink') + labs(x='work sector')

# The sixth variable is the amount of money spent in Debit Card
hist(DataBase$DC)
# Since this is a variable in money units, it could be appropriated to apply LN function here, it should
# make the distribution of the variable more normal.
DC_LN <- (DC_LN=log(ifelse(DataBase$DC<1,1,DataBase$DC)))

hist(DC_LN)


# The seventh variable is Monthly Expenses.
hist(DataBase$ME)
# Same as DebitCard, I applied LN function.
ME_LN <- (ME_LN=log(ifelse(DataBase$ME<1,1,DataBase$ME)))

hist(ME_LN)


# The eighth variable is Wage, this is a very important variable to estimate Default.
hist(DataBase$Wage)
# With the same logic as before, I applied the LN function
Wage_LN <- (Wage_LN=log(ifelse(DataBase$Wage<1,1,DataBase$Wage)))


hist(Wage_LN)

# The ninth variable is a flag that shows if the person has used a passive product (like debit card) 12 month
# in a row. (It could be a good indicator of how much use the person gives to the bank's products)  
hist(DataBase$M12_PP)

# The tenth variable is the limit the person has in his/her credit card in the financial sector in Argentina.
hist(DataBase$CCLim)
# I applied the LN function.
CCLim_LN <- (CCLim_LN=log(ifelse(DataBase$CCLim<1,1,DataBase$CCLim)))
hist(CCLim_LN)

# The last variable is Interest, and it shows the amount of interest (it could be in fix rent) the person
# won the last year.
hist(DataBase$I)
# With the same logic as before, I applied LN function.
I_LN <- (I_LN=log(ifelse(DataBase$I<1,1,DataBase$I)))
hist(I_LN)

# Then, I added the new variables to the data set.
DataBase <- cbind(DataBase, BN_Stand, Age_LN, I_LN, DC_LN, ME_LN, CCLim_LN ,Wage_LN)
DataBase <- DataBase[,-c(which(colnames(DataBase) %in% c("BN","Age","DC","ME","Wage", "I", "CCLim")))]


## BIVARIATE ANALYSIS  

# Let's now see how much the independents variables explain of the dependent variable.
# To do this I applied a logistic regression (remember that the dependent variable is a flag that
# shows if the person has paid the credit card debt or not, it is a binary variable, so it is appropriated 
# to use logistic)

# In these types of models, it is mainly used the KS (kolmogorov smirnov) and the area under the ROC curve as 
# performance measures. In both cases, a higher number denote a better performance.

for (i in 2:length(DataBase)) {
  
  bivariate_regression <- glm(DataBase[,1] ~ 1+ DataBase[,i], family = "binomial")
  prediction <- predict(bivariate_regression, DataBase, type = "response")
  
  m1_pred <- prediction(prediction , DataBase$Default)
  m1_perf <- performance(m1_pred,"tpr","fpr")
  
  KS <- round(max(attr(m1_perf,'y.values')[[1]]-
                    attr(m1_perf,'x.values')[[1]])*100, 2)
  ROC <- round(performance(m1_pred, measure =
                             "auc")@y.values[[1]]*100, 2)
  
  assign(paste0("KS_",colnames(DataBase[i])),KS,.GlobalEnv)
  assign(paste0("ROC_",colnames(DataBase[i])),ROC,.GlobalEnv)
  
}

# Let's unify the results
KSs <- data.frame("varaible"=c("KS_Wage_LN","KS_CR","KS_QCC","KS_DC_LN","KS_Age_LN","KS_BN_Stand","KS_ME_LN","KS_Work","KS_I_LN","KS_M12_PP","KS_CCLim_LN"),"KS"=c(KS_Wage_LN,KS_CR,KS_QCC,KS_DC_LN,KS_Age_LN,KS_BN_Stand,KS_ME_LN,KS_Work,KS_I_LN,KS_M12_PP,KS_CCLim_LN))
ROCs <- data.frame("variable"=c("ROC_Wage_LN","ROC_CR","ROC_QCC","ROC_DC_LN","ROC_Age_LN","ROC_BN_Stand","ROC_ME_LN","ROC_Work","ROC_I_LN","ROC_M12_PP","ROC_CCLim_LN"),"AUC-ROC"=c(ROC_Wage_LN,ROC_CR,ROC_QCC,ROC_DC_LN,ROC_Age_LN,ROC_BN_Stand,ROC_ME_LN,ROC_Work,ROC_I_LN,ROC_M12_PP,ROC_CCLim_LN))
rm(KS_Wage_LN,KS_CR,KS_QCC,KS_DC_LN,KS_Age_LN,KS_BN_Stand,KS_ME_LN,KS_Work,ROC_Wage_LN,ROC_CR,ROC_QCC,ROC_DC_LN,ROC_Age_LN,ROC_BN_Stand,ROC_ME_LN,ROC_Work,ROC_I_LN , KS_I_LN, ROC_M12_PP,KS_M12_PP, ROC_CCLim_LN,KS_CCLim_LN, prediction, m1_perf, m1_pred, bivariate_regression, KS, ROC)

KSs
ROCs
# It can be seen how "M12_PP" and "I" have a nearly 0 KS and nearly 50 ROC. It means they did not explain
# anything of the dependent variable, so I proceeded to eliminate them from the database.
DataBase <- DataBase[,-c(which(colnames(DataBase) %in% c("M12_PP", "I")))]


# Let's now make a corrplot to study correlations between the variables. To do this, I excluded the "Work" 
# variable because it was a "factor" variable.  
factor_variable <- c("Work")
DataBase_withoutFactor <- DataBase[,-c(which(colnames(DataBase) %in% factor_variable))]
Mat_cor <- round(cor(DataBase_withoutFactor),2)

corrplot(Mat_cor, method="color",
         type="upper",
         addcoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=45, #Text label color and rotation
         # Combine with significance
         #p.mat = p.mat$p, sig.level = 0.01, insig = "blank",
         # hide correlation coefficient on the principal diagonal
         diag=FALSE
)

# There is not a strong correlation between independent variables.

rm(factor_variable,DataBase_withoutFactor, Mat_cor)

## MULTIVARIATE MODELS

# Let's now start with the different models that will be tested. I have test 6 different models:
# 1_ Linear Regression with the intercept only
# 2_ Multivariate Linear Regression
# 3_ Logistic Regression with the intercept only
# 4_ Multivariate Logistic Regression
# 5_ Decision tree
# 6_ Neural Network


# First, I have to convert to dummies the factor variable
factor_variable <- "Work"
DataBase_wDummies <- as.data.frame(get_dummies.(DataBase))
DataBase_wDummies <- DataBase_wDummies[,-c(which(colnames(DataBase_wDummies) == "Work"))]


# Then, I set a seed
seed = 1
set.seed(seed)

# I divided the database between train and test.
Muestra = 0.8
Coord = sample(nrow(DataBase_wDummies),nrow(DataBase_wDummies)*Muestra)
Test = DataBase_wDummies[-Coord,]
Train = DataBase_wDummies[Coord,]


## 1ST MODEL: LINEAL MODEL - Intercept Only

# 
Model_Lineal_null <- glm(Default ~ 1, data=Train)

# With the created model, I estimated the estimated probability of default
Prob_est_null <- predict(Model_Lineal_null,
                              Test,type = 'response')

# Then I calculated the KS and the ROC
m1_pred_null <- prediction(Prob_est_null , Test$Default )
m1_perf_null <- performance(m1_pred_null,"tpr","fpr")
KS_lm_null <- round(max(attr(m1_perf_null,'y.values')[[1]]-
                          attr(m1_perf_null,'x.values')[[1]])*100, 2)
ROC_lm_null <- round(performance(m1_pred_null, measure =
                                   "auc")@y.values[[1]]*100, 2)

KS_lm_null
ROC_lm_null

# This model does not explain the default. A KS = 0 and a ROC = 50 are similar to an aleatory decision between
# default and no default

## 2ND MODEL: MULTIVARIATE LINEAR MODEL
Modelo_Lineal <- lm(Default ~ 1 +  . ,data = Train)

Prob_estimada <- predict(Modelo_Lineal,
                         Test,type = 'response')
m1_pred <- prediction(Prob_estimada , Test$Default )
m1_perf <- performance(m1_pred,"tpr","fpr")
KS_lm <- round(max(attr(m1_perf,'y.values')[[1]] -
                     attr(m1_perf,'x.values')[[1]])*100, 2)
ROC_lm <- round(performance(m1_pred, measure =
                              "auc")@y.values[[1]]*100, 2)

KS_lm 
ROC_lm 

# This model is much better than the one with only the intercept. Let's try to improve it with a logistic.

## 3RD MODEL: LOGISTIC REGRESSION - Intercept Only
Model_Logistic_null <- glm(Default ~ 1, data=Train, family = "binomial")

Prob_est_null <- predict(Model_Logistic_null,
                              Test,type = 'response')
m1_pred_null <- prediction(Prob_est_null , Test$Default )
m1_perf_null <- performance(m1_pred_null,"tpr","fpr")

KS_glmL_null <- round(max(attr(m1_perf_null,'y.values')[[1]]-
                            attr(m1_perf_null,'x.values')[[1]])*100, 2)
ROC_glmL_null <- round(performance(m1_pred_null, measure =
                                     "auc")@y.values[[1]]*100, 2)

KS_glmL_null
ROC_glmL_null
# The same as the first model, a model with only an intercept does not explain the default.

## 4TH MODEL: MULTIVARIATE LOGISTIC MODEL

Model_Logistic <- glm(Default ~ 1 +  . ,data = Train , family = "binomial")

Prob_est <- predict(Model_Logistic,
                         Test,type = 'response')
m1_pred <- prediction(Prob_est , Test$Default )
m1_perf <- performance(m1_pred,"tpr","fpr")
KS_glmL_log <- round(max(attr(m1_perf,'y.values')[[1]]-
                           attr(m1_perf,'x.values')[[1]])*100, 2)
ROC_glmL_log <- round(performance(m1_pred, measure =
                                    "auc")@y.values[[1]]*100, 2)

KS_glmL_log 
ROC_glmL_log

# Here we can see how the logistic model improves the linear one. This is mainly because the dependent 
# variable is binary and the logistic model applies much better in these cases.

# An extra analysis could be to take a look at the p-value
pvalue <- as.data.frame(summary(Model_Logistic)$coefficients)
pvalue <- pvalue[order(pvalue$`Pr(>|z|)`),]
pvalue
# Here we can see that all the p-values are quite small, so all the variables are significant to the model. 

## 5TH MODEL: DECISION TREES

# Creation of the model
tree<-rpart(Default~., Train)

# Let's plot the tree
pfit <- as.party(tree)
plot(pfit[1], main = "Decision Tree",type =  "simple",
     tp_args = list(),
     inner_panel = node_inner, 
     edge_panel = edge_simple, ep_args = list(),
     drop_terminal = NULL,
     tnex = NULL, pop = FALSE, gp = gpar(fontsize = 8))


# Performance
Prob_estimada <- (predict(tree,Test,type = 'vector'))
m1_pred <- prediction(as.numeric(Prob_estimada), as.numeric(Test$Default))
m1_perf <- performance(m1_pred,"tpr","fpr")

KS_tree <- round(max(attr(m1_perf,'y.values')[[1]]-attr(m1_perf,'x.values')[[1]])*100, 2)
ROC_tree <- round(performance(m1_pred, measure ="auc")@y.values[[1]]*100, 2)

KS_tree
ROC_tree

# The KS and the ROC are much higher with this model. It can discriminate very good the defaults.


## 6TH MODEL: NEURAL NETWORK

# Set the formula to use
formula <- as.formula(paste("Default ~ ."))

# To create this model I use the neuralnet package
library(neuralnet)

# This model takes a while to run (2 or 3 minutes), so I added a timer
t <- proc.time()  ; Modelo_Redes_prueba <- neuralnet(formula , hidden = c(3,3) ,data =
                                   Train ,linear.output = FALSE, algorithm =
                                   'rprop+',likelihood = TRUE,threshold = 0.5, stepmax = 100000) ; proc.time()-t

#  Now, let's see the performance
Variables <- Modelo_Redes_prueba$model.list$variables
model_results3 <- neuralnet::compute(Modelo_Redes_prueba, Test[,c(which(colnames(Test) %in% Variables))])
Prob_estimada <- as.vector(model_results3$net.result)
detach(package:neuralnet,unload = T)
m1_pred <- prediction(Prob_estimada , Test$Default)
m1_perf <- performance(m1_pred,"tpr","fpr")
KS_nn <- round(max(attr(m1_perf,'y.values')[[1]]-
                     attr(m1_perf,'x.values')[[1]])*100, 2)
ROC_nn <- round(performance(m1_pred, measure =
                              "auc")@y.values[[1]]*100, 2)
KS_nn
ROC_nn

# The neural network is a good model, as good as the logistic regression in this case. 


## RESUME

# Let's analyze all models results

results <- data.frame("model"=c("Linear - only Intercept","Multivariate Linear", "Logistic - only Intercept", "Multivariate Logistic","Decision Tree","Neural Network"),"KS"=c(KS_lm_null,KS_lm,KS_glmL_null,KS_glmL_log,KS_tree,KS_nn),"ROC"=c(ROC_lm_null,ROC_lm,ROC_glmL_null,ROC_glmL_log,ROC_tree,ROC_nn))
results

# A model with KS above 60 and ROC above 80 can be consider as a good one. A model with 
# KS above 70 and ROC above 90 can be consider as a very good one.

# Best KS
results$model[which.is.max(results$KS)]

# Best ROC
results$model[which.is.max(results$ROC)]

# In this particular case, the best model is decision trees. As it has KS = 77 and ROC = 91 it
# can be consider as an excellent model. It can do a really good job discriminating defaults.


