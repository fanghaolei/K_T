train <- read.csv("train.csv")
test <- read.csv("test.csv")

library(caret)
library(ggplot2)
library(mice)
library(Rmisc)

# combine train and test set for feature transformations
combined<-merge(train, test, all.x =T, all.y = T)

# reset the column names for easier indexing
names(combined)<-c("id","class","name","sex","age","sibsp","parch","ticket","fare","cabin","embarked","sur")

# check the structure of the data
str(combined)
summary(combined)

# the id, class, sur should be factors
combined[,1]<-as.factor(combined[,1])
combined[,2]<-as.factor(combined[,2])
combined[,12]<-as.factor(combined[,12])

# a table for missing patterns
md.pattern(combined)

#the NA's for cabin and embarked are marked as empty values, change them to NA. 
combined$cabin[which(combined$cabin == "")] <- NA
combined$embarked[which(combined$embarked == "")] <- NA

# check the specific cabin levels
lvls<-levels(combined$cabin)

# its seems that cabins are from letters "A to G" + a special T cabin, 
# they can be regrouped and simplified into letter-only factor levels
combined$cabin<-as.character(combined$cabin)
cabin.levels<-LETTERS[1:7]
for(i in cabin.levels){
  combined$cabin[grep(i, combined$cabin)]<-i
}
combined$cabin<-as.factor(combined$cabin) 

# check the structure now
table(combined$cabin)
str(combined)
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
table(combined$title)

# plot title against age colored by survival
qplot(title, age, color = sur, data = combined, geom = "jitter")  

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
info

qplot(age, color = title, data = combined, geom = "density") # age distribution according to titles

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
qplot(fare, color = class, data = combined, geom = "density") # it is skewed with a high kurtosis, impute with mode
mode.fare<-aggregate(combined$fare, by = list(combined$class), Mode, na.rm = T) #8.05 for 3rd class
mode.fare
combined$fare.imputed<-combined$fare
combined$fare.imputed[1044]<-mode.fare[3,2]

# reorganize family features. family size = # of parent and child + # of sibings + 1
combined$fam_size<-combined$sibsp + combined$parch + 1
qplot(fam_size, fill = class, data = combined)

# ---------------------finished preprocessing, splitting into new train & test sets----------------------
train<-combined[1:891, ]
test<-combined[892:1309,]

# feature selection
# exploratory on categorical features:
p1<-qplot(class, fill = sur, data = train)
p2<-qplot(sex, fill = sur, data = train)
p3<-qplot(embarked, fill = sur, data = train)
p4<-qplot(cabin, fill = sur, data =train)  #too many missing values for cabin, not include in the model
multiplot(p1,p2,p3,p4, cols = 2)

# exploratory on continuous features:
p5<-qplot(age.imputed, color = sur, data = train, geom = "density") 
p6<-qplot(fam_size, color = sur, data = train, geom = "density")
p7<-qplot(fare.imputed, color = sur, data = train, geom = "density")
multiplot(p5,p6,p7, cols = 2)


# train a logistic regression model
model<-glm(sur~class+sex+age.sqrt+fam_size+title+fare.imputed, 
           family = "binomial", data = train)
summary(model)

# apply model 
prediction<-predict.glm(model, newdata = test, type  = "response")

# set cutoff, >.55 survived, <.55 dead
cutoff<-.55
sur<-vector()
for(i in 1:nrow(test)){
  if(prediction[i] < cutoff) {sur[i] = 0} else
  {sur[i] = 1}
}

result<-test[, c(1,12)]
result[,2]<-sur
names(result)<-c("Passengerid","Survived")

# output
write.csv(result, file = "solution.csv", row.names = F)
