library(MASS)
library(RSNNS)
library(nnet)
library(data.table)
library(tictoc)

# Set ford number for CV
foldNum <- 5

# Read state parameters
state <- scan("state.dat")
dimNum <- state[1]
classNum <- state[2]
trainDataNum <- state[3]
testDataNum <- state[4]
trialNum <- state[5]
learnRate <- state[6]
maxIter <- state[7]

trainLabel <- numeric(trainDataNum*classNum)
testLabel <- numeric(testDataNum*classNum)

learnTime <- numeric(trialNum)
testTime <- numeric(trialNum)
accuracy <- numeric(trialNum)
accuracyCV <- numeric(foldNum)
bestHid <-numeric(trialNum)

for(c in 1:classNum){
  trainLabel[((c-1)*trainDataNum+1):(c*trainDataNum)] <- c
  testLabel[((c-1)*testDataNum+1):(c*testDataNum)] <- c
}
trainTeach <- decodeClassLabels(trainLabel)
testTeach <- decodeClassLabels(testLabel)

# Repeat training and testing
for(t in 1:trialNum){
  cat("------------------------------\n")
  cat(sprintf("Trial: %d\n", t))
  
  trainfName <- sprintf("./train/learn_data%d.dat",t)
  testfName <- sprintf("./test/discriminate_data%d.dat",t)
  
  trainData <- fread(trainfName, stringsAsFactors=FALSE, data.table = FALSE)
  testData <- fread(testfName, stringsAsFactors=FALSE, data.table = FALSE)
  
  # CV
  CVdata <- cbind(trainData, trainTeach)
  CVdata <- CVdata[sample.int(dim(CVdata)[1]),]

  folds <- cut(seq(1,nrow(CVdata)),breaks=foldNum,labels=FALSE)
  
  cat("start CV: \n")
  
  maxAccuracy <- 0
  for(hiddenCV in (dimNum/2):(dimNum/2+5)){
    for(i in 1:foldNum){
      #Segement your data by fold using the which() function 
      testIndexes <- which(folds==i,arr.ind=TRUE)
      testDataCV <- CVdata[testIndexes, ]
      trainDataCV <- CVdata[-testIndexes, ]
      
      modelCV <- nnet(trainDataCV[,1:dimNum], trainDataCV[,(dimNum+1):(dimNum+classNum)], size = 2*hiddenCV, rang = 1/(max(trainDataCV)), decay = 5e-4, maxit = 800);
      
      pred <- predict(modelCV, testDataCV[,1:dimNum])
      testLabelCV <- apply(testDataCV[,(dimNum+1):(dimNum+classNum)], 1, which.is.max)
      confMat <- table(testLabelCV, apply(pred, 1, which.is.max))
      accuracyCV[i] <- sum(confMat[row(confMat)==col(confMat)])/sum(confMat)*100
    }
    meanAccuracyCV <- mean(accuracyCV)
    if(meanAccuracyCV == 100){
      bestHid[t] = 2*hiddenCV
      break;
    }
    
    if(meanAccuracyCV > maxAccuracy){
      maxAccuracy = meanAccuracyCV
      bestHid[t] = 2*hiddenCV
    }
  }
  
  cat("Finish!\n")
  
  bef <- tic()
  model <- nnet(trainData, trainTeach, size = bestHid[t], rang = 1/(max(trainData)), decay = 5e-4, maxit = 2000)
  learnTime[t] <- toc()$toc - bef
  
  bef <- tic()
  pred <- predict(model, testData)
  testTime[t] <- toc()$toc - bef
  
  predLabel <- apply(pred, 1, which.is.max)
  confMat <- caret::confusionMatrix(
    factor(predLabel,levels = 1:classNum),
    factor(testLabel,levels = 1:classNum)
  )
  accuracy[t] <- confMat$overall["Accuracy"]*100
  
  cat(sprintf("Accuracy: %f\n\n", accuracy[t]))
}

write.matrix(accuracy, file = "accuracy.dat")
write.matrix(bestHid, file ="bestHiddenNum.dat")
write.matrix(learnTime, file = "trainTime.dat")
write.matrix(testTime, file = "testTime.dat")

