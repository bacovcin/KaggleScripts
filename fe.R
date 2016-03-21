library('caret')
library('lme4')

# Assumes the following data structure:
# train.csv has columns ID,target,features...
# test.csv has columns ID,features...

# Choose the R2 cutoff for eliminating overly correlated variables

r2cutoff = 0.8
cat.level.min = 150
# Load the data set

# train<-read.csv('train.csv')
# test<-read.csv('test.csv')

# Identify the original number of columns

# varNum = dim(train)[2]

# Add a new column equal to the number of NAs per row

# train$NumNA<-apply(train,1,function(x) sum(is.na(x)|x==''))
# test$NumNA<-apply(test,1,function(x) sum(is.na(x)|x==''))


# For every variable in the data set add a new variable tracking NA values (1 if NA, 0 if not)

# for (i in 3:(dim(train)[2]-1)){
#  train <- as.data.frame(cbind(train,(is.na(train[,i])|train[,i]=='')*1))
#  test<- as.data.frame(cbind(test,(is.na(test[,i-1])|test[,i-1]=='')*1))
# }

# Rename the new columns to be 'isNA' + variable name

# names(train)<-c('ID','target',paste('v',seq(1,varNum,sep=''),'NumNA',paste('isNAv',varNum,sep=''))
# names(test)<-c('ID',paste('v',seq(1,varNum),sep=''),'NumNA',paste('isNAv',seq(1,varNum),sep=''))

# Save the new datasets for easier future access

# save(test,file='test.RData')
# save(train,file='train.RData')

# Reload the data

load('../train.RData')
load('../test.RData')

# Separate numerical and categorical predictors

nums<-sapply(train,is.numeric)
train.num<-train[,nums]
train.cat<-train[,nums==F]

nums<-sapply(test,is.numeric)
test.num<-test[,nums]
test.cat<-test[,nums==F]

# Label origin of categorical variables

train.cat$origin<-'train'
test.cat$origin<-'test'

#Combine test and train categorical data for feature engineering
cat<-as.data.frame(rbind(train.cat,test.cat))

#See if the variable levels exist in both the training and test set
#If levels aren't shared, save the variables to be used as random effects

cat.vars<-names(cat)[names(cat)!='origin']

tests <- data.frame(y=cat.vars,inBoth=NA,uniques=NA)

for (i in 1:dim(tests)[1]) {
  print(i)
  tests$uniques[i] <- length(unique(factor(cat[,names(cat)==as.character(tests$y[i])])))
  tests$inBoth[i] <- 0 %in% table(cat[,names(cat)==as.character(tests$y[i])],cat$origin)
}

rands<-as.character(tests$y[(tests$inBoth)|tests$uniques>cat.level.min])
train.rands<-cat[cat$origin=='train',rands]
test.rands<-cat[cat$origin=='test',rands]

train.cat2<-as.data.frame(cbind(train.num[1],cat[cat$origin=='train',!(names(cat)%in%rands)&(names(cat)!='origin')]))
test.cat2<-as.data.frame(cbind(test.num[1],cat[cat$origin=='test',!(names(cat)%in%rands)&(names(cat)!='origin')]))

names(train.cat2)[1]<-'ID'
names(test.cat2)[1]<-'ID'

#Convert categorical variables into dummy variables
dummy.mod<-dummyVars(ID ~ ., data = train.cat2)

train.dummies<-predict(dummy.mod,newdata=train.cat2)
test.dummies<-predict(dummy.mod,newdata=test.cat2)

# Convert some categorical variables to their training set frequencies

train.rand.cols<-as.data.frame(rep(0,dim(train.rands)[1]))
test.rand.cols<-as.data.frame(rep(0,dim(test.rands)[1]))
for (i in 1:(length(rands)-1)) {
  train.rand.cols <- as.data.frame(cbind(train.rand.cols,as.data.frame(rep(0,dim(train.rands)[1]))))
  test.rand.cols <- as.data.frame(cbind(test.rand.cols,as.data.frame(rep(0,dim(test.rands)[1]))))
}

for (i in 1:length(rands)) {
  names(train.rand.cols)[i]<-rands[i]
  names(test.rand.cols)[i]<-rands[i]
  print(i)
  temp.tbl <- prop.table(table(train.rands[,i]))
  train.rands[,i] <- as.character(train.rands[,i])
  for (j in 1:length(names(temp.tbl))) {
    print(j)
    name <- names(temp.tbl)[j]
    train.rand.cols[train.rands[,i] == name,i] <- temp.tbl[j]
    test.rand.cols[test.rands[,i] == name,i] <- temp.tbl[j]
  }
}


# Median impute NAs in the continuous data (since NAhood has been stored in separate variables)

impMod <- preProcess(train.num,method='medianImpute')

train.num <- predict(impMod, newdata=train.num)
test.num <- predict(impMod, newdata=test.num)

# Add the predictions form the random effects models and the dummy variables to
# the categorical variables from before

train.num<-as.data.frame(cbind(train.num,train.rand.cols[names(train.rand.cols)!='ID'],train.dummies))
test.num<-as.data.frame(cbind(test.num,test.rand.cols[names(test.rand.cols)!='ID'],test.dummies))

# Label origin of numerical variables and merge train and test set

train.num$origin<-'train'
test.num$origin<-'test'

num <- as.data.frame(rbind(train.num[,3:dim(train.num)[2]],test.num[,2:dim(test.num)[2]]))

#Engineer the variables
# Remove any variables with zero variance
zv.mod<-preProcess(num,method='zv')
num.zv<-predict(zv.mod,newdata=num)

# Find the pairwise correlation of remaining variables (start with the continuous variables)
# Categorical variables will be added once the correlated continuous variables have been removed
cor.names = names(num.zv)[!(grepl('isNA|\\.|origin',names(num.zv)))]
tests<-expand.grid(x=cor.names,y=cor.names,cor=NA)

tests<-subset(tests,x!=y)
 
start = Sys.time()
for (i in 1:dim(tests)[1]) {
 r = dim(tests[1])-i
 print(paste('Remaining: ',r,',',round((as.numeric(Sys.time()-start)/i)*r,digits=2)))
 mod<-NA
 x<-num.zv[,names(num.zv)==tests$x[i]]
 y<-num.zv[,names(num.zv)==tests$y[i]]
 try(tests$cor[i]<-cor(x,y,use='pairwise.complete.obs')[1])
}
# # Look at results 
# 
high <- tests[(tests$cor>.1|tests$cor< -0.1)&!is.na(tests$cor),]
levels(factor(high$y))
# 
# #Check lms of highly correlated predictors to find calculated variables
new.tests<-expand.grid(y=levels(factor(high$y)),R2=NA)
 
for (i in 1:dim(new.tests)[1]) {
 print(i)
 new.tests$R2[i]<-summary(lm(data=num.zv,as.formula(paste0(new.tests$y[i],'~',paste(high[as.character(high$y)==as.character(new.tests$y[i]),]$x,collapse='+')))))$r.squared
}
# store the results in decreasing order of R2

new.tests2 = new.tests[order(-new.tests$R2),]


#Keep removing variables and identify highly predicatable predictors for removal

removes<-NULL

for (i in 1:dim(new.tests2)[1]) {
  temp.mod <- lm(data=train.num,as.formula(paste0(new.tests2$y[i],'~',paste(high[as.character(high$y)==as.character(new.tests2$y[i])&!(high$x %in% removes),]$x,collapse='+'))))
  print(i)
  print(summary(temp.mod)$adj.r.squared)
  if (summary(temp.mod)$adj.r.squared >= r2cutoff) {
    removes <- c(removes, as.character(new.tests2$y[i]))
  }
}

num2<-num.zv[,!(names(num.zv)%in%removes)]

# Now include the categorical and isNA variables in correlation and other calculations
cor.names = c(names(num2)[names(num2)!='origin'])
tests<-expand.grid(x=as.character(cor.names),y=as.character(cor.names),cor=NA,xiny=NA,yinx=NA)

for (i in (1:dim(tests)[1])) {
  print(i)
  tests$xiny[i] <- grepl(tests$x[i],tests$y[i])
  tests$yinx[i] <- grepl(tests$y[i],tests$x[i])
}

tests<-subset(tests,x!=y&!xiny&!yinx)

start = Sys.time()
for (i in 1:dim(tests)[1]) {
  r = dim(tests[1])-i
  print(paste('Remaining: ',r,',',round((as.numeric(Sys.time()-start)/i)*r,digits=2)))
  mod<-NA
  x<-num2[,names(num2)==tests$x[i]]
  y<-num2[,names(num2)==tests$y[i]]
  try(tests$cor[i]<-cor(x,y,use='pairwise.complete.obs')[1])
}
# # Look at results 
# 
high <- tests[(tests$cor>.1|tests$cor< -0.1)&!is.na(tests$cor),]
levels(factor(high$y))
# 
# #Check lms of highly correlated predictors to find calculated variables
new.tests<-expand.grid(y=levels(factor(high$y)),R2=NA)

for (i in 1:dim(new.tests)[1]) {
  print(i)
  try(new.tests$R2[i]<-summary(lm(data=num.zv,as.formula(paste0(new.tests$y[i],'~',paste(high[as.character(high$y)==as.character(new.tests$y[i]),]$x,collapse='+')))))$r.squared)
}

new.tests2 = new.tests[order(-new.tests$R2),]


#Keep removing variables and identify highly predicatable predictors for removal

removes<-NULL

for (i in 1:dim(new.tests2)[1]) {
  try(temp.mod <- lm(data=num2,as.formula(paste0(new.tests2$y[i],'~',paste(high[as.character(high$y)==as.character(new.tests2$y[i])&!(as.character(high$x) %in% removes),]$x,collapse='+')))))
  print(i)
  try(print(summary(temp.mod)$adj.r.squared))
  try(if (summary(temp.mod)$adj.r.squared >= r2cutoff) {
    removes <- c(removes, as.character(new.tests2$y[i]))
  })
}

num3<-num2[,!(names(num2)%in%c(removes))]

write.csv(num3[num3$origin=='train',names(num3)!='origin'],file='../trainfe.csv',quote=F,row.names=F)
write.csv(train.num[,2],file='../yfe.csv',quote=F,row.names=F)
write.csv(num3[num3$origin=='test',names(num3)!='origin'],file='../testfe.csv',quote=F,row.names=F)
write.csv(test.num[,1],file='../testids.csv',quote=F,row.names=F)
