Using gbm linear logistic regression, this script reached .80861 accuracy on Kaggle. 

train <- read.csv("train.csv")
test <- read.csv("test.csv")
# combine train and test set for feature transformations
combined<-merge(train, test, all.x =T, all.y = T)

# reset the column names for easier indexing
names(combined)<-c("id","class","name","sex","age","sibsp","parch","ticket","fare","cabin","embarked","sur")

# the id, class, sur should be factors
combined[,1]<-as.factor(combined[,1])
combined[,2]<-as.factor(combined[,2])
combined[,12]<-as.factor(combined[,12])

#the NA's for cabin and embarked are marked as empty values, change them to NA. 
combined$cabin[which(combined$cabin == "")] <- NA
combined$embarked[which(combined$embarked == "")] <- NA

# its seems that cabins are from letters "A to G" + a special T cabin, 
# they can be regrouped and simplified into letter-only factor levels
combined$cabin<-as.character(combined$cabin)
cabin.levels<-LETTERS[1:7]
for(i in cabin.levels){
    combined$cabin[grep(i, combined$cabin)]<-i
}
combined$cabin<-as.factor(combined$cabin) 

# there are a empty level in "embarked", drop the empty level
combined$embarked<-droplevels(combined$embarked) 
missingembarked<-which(is.na(combined$embarked)) #62, 830 were missing. since most people embarked at S, impute with S
combined$embarked[c(62,830)] <- "S"

# missing age can be guessed from person titles, extracting unique titles from name
combined$title<-character(length = nrow(combined))
titles<-c("Mr.","Mrs.","Miss","Master","Special") # special is for Dr., Capt., Major., etc.

# replace all common titles
for(i in titles[1:4]){
    combined$title[grep(i, combined$name)] <- i
}

# replace others with all special titles
combined$title[which(combined$title == "")] <- "Special"
combined$title<-factor(combined$title, c("Mr.","Mrs.","Miss", "Master","Special"))

# find mean, median, mode of age for each group
mean.age<-aggregate(combined$age, by = list(combined$title), mean, na.rm = T)
median.age<-aggregate(combined$age, by = list(combined$title), median, na.rm = T)

# a function to calculate mode
Mode <- function(x, na.rm = FALSE) {
    if(na.rm){
        x = x[!is.na(x)]
    }
    ux <- unique(x)
    return(ux[which.max(tabulate(match(x, ux)))])
}
mode.age<-aggregate(combined$age, by = list(combined$title), Mode, na.rm = T)

# merge the results into one df
dt1<-merge(mean.age, median.age, by = "Group.1", all.x = T)
info<-merge(dt1, mode.age, by = "Group.1", all.x = T)
names(info)<-c("title","mean","median","mode")


#for "Special", normal, median for imputation
#for "Master", highly positively skewed, mode for imputation
#for "Miss", normal, take the median
#for "Mr", slightly skewed, take the median 
#for "Mrs" normal, taken the median

# imputation
naage.id<-which(is.na(combined$age))
naage.title<-combined$title[naage.id]
age.imputed<-numeric()
for(i in 1:length(naage.title)){
    if(naage.title[i] == "Master"){age.imputed[i] = info[1,4]} else
        if(naage.title[i] == "Miss"){age.imputed[i] = info[2,3]} else
            if(naage.title[i] == "Mr."){age.imputed[i] = info[3,3]} else
                if(naage.title[i] == "Mrs."){age.imputed[i] = info[4,3]} else
                {age.imputed[i] = info[5,3]}
}
combined$age.imputed<-combined$age #create a new column for imputed age
combined$age.imputed[naage.id]<-age.imputed

# find the missing observations for fare
missingfare<-which(is.na(combined$fare)) # case 1044 is missing, find which class is this case
combined$class[1044]                     # it is a 3rd class ticket, find the average fare for 3rd clas
mean.fare<-aggregate(combined$fare, by = list(combined$class), mean, na.rm = T)
mode.fare<-aggregate(combined$fare, by = list(combined$class), Mode, na.rm = T) #8.05 for 3rd class
combined$fare.imputed<-combined$fare
combined$fare.imputed[1044]<-mode.fare[3,2]

# reorganize family features. family size = # of parent and child + # of sibings + 1
combined$fam_size<-combined$sibsp + combined$parch + 1

# ---------------------finished preprocessing, splitting into new train & test sets----------------------
# ------------------------------------------rf ~= .78--------------------------#
rfdata <- combined[, c(2, 4, 11, 12, 13, 14, 15, 16)]
rf <- randomForest(sur~., data = rfdata, subset = 1:891, ntree = 10000, mtry = 2, do.trace = T,
                   importance = T)

pred <- predict(rf, newdata = rfdata[892:1309,], type = "response")
result<-combined[892:1309, c(1,12)]
result[,2]<-pred
names(result)<-c("Passengerid","Survived")
write.csv(result, file = "solution_rf.csv", row.names = F)

#-----------------------------------------boosted logistic regression ~.80861--------------------------#
train<-combined[1:891, c(2, 4, 11, 12, 13, 14, 15, 16)]
test<-combined[892:1309,c(2, 4, 11, 13, 14, 15, 16)]
test <- sparse.model.matrix(~.-1, data = test)

X <- sparse.model.matrix(sur~.-1, data = train)
Y <- as.numeric(train[, "sur"]) - 1

params <- list("booster" = "gblinear", 
               "lambda" = .04,
               "lambda_bias" =.05,
               "objective" = "reg:logistic",
               "eval_metric" = "error")

niter <- 15
# leave one-out cv, cv error minimized at 8th iteraction
bst.cv <-  xgb.cv(params = params, data = X, label = Y, nfold = 891, nround = niter, verbost = T)

# plot cv error agianst iteration number
plot(1:niter, bst.cv$test.error.mean, type = "l")
min <- which.min(bst.cv$test.error.mean)
points(min, bst.cv$test.error.mean[min], col = "red")
text(min, bst.cv$test.error.mean[min], labels = bst.cv$test.error.mean[min])

bst  <- xgboost(data = X, label = Y, params = params, nround = 8)
pred <- predict(bst, newdata = test)
cutoff<-.55
sur<-vector()
for(i in 1:nrow(test)){
   ifelse(pred[i] < cutoff, sur[i] <- 0, sur[i] <- 1)
}

result<-combined[892:1309, c(1,12)]
result[,2]<-sur
names(result)<-c("Passengerid","Survived")

write.csv(result, file = "solution_xgboost.csv", row.names = F)
