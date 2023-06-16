# set of functions to do thinks specifically for shiny

# Add a parameter based on a click ####

# add a parameter based on x-location in shiny plot click (DY)
t.add.DY.parameter <- function(resObj, xloc){
  
  #can click outside of parameters, don't want anything to be done in this case  
 
  #too small
  if(xloc <= 1){ 
    stop()
  }
  
  #too large
  if(xloc > length(resObj$DYGp$DY)){ 
    stop()
  }
  
  #get variables together
  myDY <- resObj$DYGp$DY
  group <- resObj$DYGp$gp
  mygp <- c()
  
  #I just want the parameter numbers (not the letters)
  group.num <- as.numeric(substr(group, start = 2, stop = 4)) #assumes fewer than 1000 parameters
  
  #add 1 to every parameter number starting with the xloc-th, and continuing to the oldest DY
  for(i in 1:length(myDY)){
    if(i >= xloc){
      group.num[i] <- group.num[i] + 1
    }
    
    #make mygp vector of new variable names
    mygp[i] <- paste(c("G", group.num[i]), sep = "", collapse = "")
  }
  
  #set up the new resObj$DYGp data frame
  DYGp <- data.frame(DY = myDY, gp = mygp)
  
  #update resObj
  resObj$DYGp <<- DYGp
  
  #this ensures that the parameters in ROCKY match the parameters in Excel
  push.DYGp()
  push.AYGp()
  push.CYGp()  

  #add a log entry
  myComment <- paste(c("Added DY trend parameter ", mygp[xloc], " in between DY's ", xloc-1, " and ", xloc), sep = "", collapse = "")
  update.trend.parameters(comment = myComment, writeToLog = TRUE, plots = FALSE, msgbox = FALSE)
}

#same for CY
t.add.CY.parameter <- function(resObj, xloc){
#   browser()
  #get variables together
  myCY <- resObj$CYGp$CY
  group <- resObj$CYGp$gp
  mygp <- c()
  
  #too small
  if(xloc <= ROCKY.settings$StartCY){ 
    stop()
  }
  
  #too large
  if(xloc > max(myCY)){ 
    stop()
  }
  
  #I just want the parameter numbers (not the letters)
  group.num <- as.numeric(substr(group, start = 2, stop = 4)) #assumes fewer than 1000 parameters
  
  #add 1 to every parameter number starting with the xloc-th, and continuing to the most recent CY
  for(i in min(myCY):max(myCY)){
    if(i >= xloc){
      group.num[i-min(myCY)+1] <- group.num[i-min(myCY)+1] + 1
    }
    
    #make mygp vector of new variable names
    mygp[i-min(myCY)+1] <- paste(c("I", group.num[i-min(myCY)+1]), sep = "", collapse = "")
  }
  
  #set up the new resObj$DYGp data frame
  CYGp <- data.frame(CY = myCY, gp = mygp)
  
  #update resObj
  resObj$CYGp <<- CYGp
  
  #this ensures that the parameters in ROCKY match the parameters in Excel
  push.DYGp()
  push.AYGp()
  push.CYGp()  
  
  #add a log entry
  myComment <- paste(c("Added CY trend parameter ", mygp[xloc-min(myCY)+1], " in between CY's ", xloc-1, " and ", xloc), sep = "", collapse = "")
  update.trend.parameters(comment = myComment, writeToLog = TRUE, plots = FALSE, msgbox = FALSE)
}


#same for CY
t.add.AY.parameter <- function(resObj, xloc){
  #   browser()
  #get variables together
  myAY <- resObj$AYGp$AY
  group <- resObj$AYGp$gp
  mygp <- c()
  
  #too small
  if(xloc <= ROCKY.settings$StartAY){ 
    stop()
  }
  
  #too large
  if(xloc > max(myAY)){ 
    stop()
  }
  
  #I just want the parameter numbers (not the letters)
  group.num <- as.numeric(substr(group, start = 2, stop = 4)) #assumes fewer than 1000 parameters
  
  #add 1 to every parameter number starting with the xloc-th, and continuing to the most recent AY
  for(i in min(myAY):max(myAY)){
    if(i >= xloc){
      group.num[i-min(myAY)+1] <- group.num[i-min(myAY)+1] + 1
    }
    
    #make mygp vector of new variable names
    mygp[i-min(myAY)+1] <- paste(c("A", group.num[i-min(myAY)+1]), sep = "", collapse = "")
  }
  
  #set up the new resObj$AYGp data frame
  AYGp <- data.frame(AY = myAY, gp = mygp)
  
  #update resObj
  resObj$AYGp <<- AYGp
  
  #this ensures that the parameters in ROCKY match the parameters in Excel
  push.DYGp()
  push.AYGp()
  push.CYGp()  
  
  #add a log entry
  myComment <- paste(c("Added AY level parameter ", mygp[xloc-min(myAY)+1], " at AY ", xloc), sep = "", collapse = "")
  update.trend.parameters(comment = myComment, writeToLog = TRUE, plots = FALSE, msgbox = FALSE)
}

# add a parameter based on x-location in shiny plot click (DY)
t.add.hetero.parameter <- function(resObj, xloc){
  
  #can click outside of parameters, don't want anything to be done in this case  
#   browser()
  #too small
  if(xloc <= 0){ 
    stop()
  }
  
  #too large
  if(xloc > length(resObj$DYGp$DY)){ 
    stop()
  }
  
  #get variables together
  myDYwGp <- resObj$DYwGp$DY
  group <- resObj$DYwGp$gp
  mygp <- c()
  
  #I just want the parameter numbers (not the letters)
  group.num <- as.numeric(substr(group, start = 2, stop = 4)) #assumes fewer than 1000 parameters
  
  #add 1 to every parameter number starting with the xloc-th, and continuing to the oldest DY
  for(i in 1:length(myDYwGp)){
    if(i >= xloc){
      group.num[i] <- group.num[i] + 1
    }
    
    #make mygp vector of new variable names
    mygp[i] <- paste(c("W", group.num[i]), sep = "", collapse = "")
  }
  
  #set up the new resObj$DYwGp data frame
  DYwGp <- data.frame(DY = myDYwGp, gp = mygp)
  
  #update resObj
  resObj$DYwGp <<- DYwGp
  
  #this ensures that the parameters in ROCKY match the parameters in Excel
  push.DYwGp() 
  
  #add a log entry
  myComment <- paste(c("Adjusted hetero by adding DY weight parameter ", mygp[xloc], " at DY ", xloc-1), sep = "", collapse = "")
#   update.trend.parameters(comment = myComment, writeToLog = TRUE, plots = FALSE, msgbox = FALSE)
  custom.group.hetero.adjustment(myComment = myComment, reset = FALSE, plots = FALSE, includeMessage = FALSE)
}

# Remove a parameter based on a (double) click ####

t.remove.DY.parameter <- function(resObj, xloc){
  
  #can click outside of parameters, don't want anything to be done in this case  
  
  #too small
  if(xloc <= 1){ 
    stop()
  }
  
  #too large
  if(xloc > length(resObj$DYGp$DY)){ 
    stop()
  }
  
  #get variables together
  myDY <- resObj$DYGp$DY
  group <- resObj$DYGp$gp
  mygp <- c()
  
  #current xloc parameter
  curPara <- group[xloc]
  
  #don't want to do anything if there isn't a parameter here (eg if the parameter at xloc is the same as at (xloc - 1) )
  if(group[xloc]==group[xloc - 1]){
    stop()
  }
  
  #I just want the parameter numbers (not the letters)
  group.num <- as.numeric(substr(group, start = 2, stop = 4)) #assumes fewer than 1000 parameters
  
  #subtract 1 from every parameter number starting with the xloc-th, and continuing to the oldest DY
  for(i in 1:length(myDY)){
    if(i >= xloc){
      group.num[i] <- group.num[i] - 1
    }
    
    #make mygp vector of new variable names
    mygp[i] <- paste(c("G", group.num[i]), sep = "", collapse = "")
  }
  
  #set up the new resObj$DYGp data frame
  DYGp <- data.frame(DY = myDY, gp = mygp)
  
  #update resObj
  resObj$DYGp <<- DYGp
  
  #this ensures that the parameters in ROCKY match the parameters in Excel
  push.DYGp()
  push.AYGp()
  push.CYGp()  
  
  #add a log entry
  myComment <- paste(c("Removed DY trend parameter ", curPara, " from in between DY's ", xloc-1, " and ", xloc), sep = "", collapse = "")
  update.trend.parameters(comment = myComment, writeToLog = TRUE, plots = FALSE, msgbox = FALSE)
}

#same for CY
t.remove.CY.parameter <- function(resObj, xloc){
  #   browser()
  #get variables together
  myCY <- resObj$CYGp$CY
  group <- resObj$CYGp$gp
  mygp <- c()
  
  #location index
  myLoc <- xloc - min(myCY) + 1
  
  #current parameter (for log comment)
  curPara <- group[myLoc]
  
  #too small
  if(xloc <= ROCKY.settings$StartCY){ 
    stop()
  }
  
  #too large
  if(xloc > max(myCY)){ 
    stop()
  }
  
  #do nothing if there is no parameter
  if(group[myLoc]==group[myLoc-1]){
    stop()
  }
  
  #I just want the parameter numbers (not the letters)
  group.num <- as.numeric(substr(group, start = 2, stop = 4)) #assumes fewer than 1000 parameters
  
  #remove 1 from every parameter number starting with the xloc-th, and continuing to the most recent CY
  for(i in min(myCY):max(myCY)){
    if(i >= xloc){
      group.num[i-min(myCY)+1] <- group.num[i-min(myCY)+1] - 1
    }
    
    #make mygp vector of new variable names
    mygp[i-min(myCY)+1] <- paste(c("I", group.num[i-min(myCY)+1]), sep = "", collapse = "")
  }
  
  #set up the new resObj$DYGp data frame
  CYGp <- data.frame(CY = myCY, gp = mygp)
  
  #update resObj
  resObj$CYGp <<- CYGp
  
  #this ensures that the parameters in ROCKY match the parameters in Excel
  push.DYGp()
  push.AYGp()
  push.CYGp()  
  
  #add a log entry
  myComment <- paste(c("Removed CY trend parameter ", curPara, " in between CY's ", xloc-1, " and ", xloc), sep = "", collapse = "")
  update.trend.parameters(comment = myComment, writeToLog = TRUE, plots = FALSE, msgbox = FALSE)
}


#same for AY
t.remove.AY.parameter <- function(resObj, xloc){
  #   browser()
  #get variables together
  myAY <- resObj$AYGp$AY
  group <- resObj$AYGp$gp
  mygp <- c()
  
  #location index
  myLoc <- xloc - min(myAY) + 1
  
  #current parameter (for log comment)
  curPara <- group[myLoc]
  
  #too small
  if(xloc <= ROCKY.settings$StartAY){ 
    stop()
  }
  
  #too large
  if(xloc > max(myAY)){ 
    stop()
  }
  
  #do nothing if there is no parameter
  if(group[myLoc]==group[myLoc-1]){
    stop()
  }
  
  #I just want the parameter numbers (not the letters)
  group.num <- as.numeric(substr(group, start = 2, stop = 4)) #assumes fewer than 1000 parameters
  
  #remove 1 from every parameter number starting with the xloc-th, and continuing to the most recent AY
  for(i in min(myAY):max(myAY)){
    if(i >= xloc){
      group.num[i-min(myAY)+1] <- group.num[i-min(myAY)+1] - 1
    }
    
    #make mygp vector of new variable names
    mygp[i-min(myAY)+1] <- paste(c("A", group.num[i-min(myAY)+1]), sep = "", collapse = "")
  }
  
  #set up the new resObj$AYGp data frame
  AYGp <- data.frame(AY = myAY, gp = mygp)
  
  #update resObj
  resObj$AYGp <<- AYGp
  
  #this ensures that the parameters in ROCKY match the parameters in Excel
  push.DYGp()
  push.AYGp()
  push.CYGp()  
  
  #add a log entry
  myComment <- paste(c("Removed AY level parameter ", curPara, " from AY ", xloc), sep = "", collapse = "")
  update.trend.parameters(comment = myComment, writeToLog = TRUE, plots = FALSE, msgbox = FALSE)
}

# remove a parameter based on x-location in shiny plot click (DY)
t.remove.hetero.parameter <- function(resObj, xloc){
  
  #can click outside of parameters, don't want anything to be done in this case  
  #   browser()
  #too small
  if(xloc <= 0){ 
    stop()
  }
  
  #too large
  if(xloc > length(resObj$DYGp$DY)){ 
    stop()
  }
  
  #get variables together
  myDYwGp <- resObj$DYwGp$DY
  group <- resObj$DYwGp$gp
  mygp <- c()
  
  #do nothing if there is no parameter
  if(group[myLoc]==group[myLoc-1]){
    stop()
  }
  
  #I just want the parameter numbers (not the letters)
  group.num <- as.numeric(substr(group, start = 2, stop = 4)) #assumes fewer than 1000 parameters
  
  #add 1 to every parameter number starting with the xloc-th, and continuing to the oldest DY
  for(i in 1:length(myDYwGp)){
    if(i >= xloc){
      group.num[i] <- group.num[i] - 1
    }
    
    #make mygp vector of new variable names
    mygp[i] <- paste(c("W", group.num[i]), sep = "", collapse = "")
  }
  
  #set up the new resObj$DYwGp data frame
  DYwGp <- data.frame(DY = myDYwGp, gp = mygp)
  
  #update resObj
  resObj$DYwGp <<- DYwGp
  
  #this ensures that the parameters in ROCKY match the parameters in Excel
  push.DYwGp() 
  
  #add a log entry
  myComment <- paste(c("Adjusted hetero by removing DY weight parameter ", mygp[xloc], " at DY ", xloc-1), sep = "", collapse = "")
  #   update.trend.parameters(comment = myComment, writeToLog = TRUE, plots = FALSE, msgbox = FALSE)
  custom.group.hetero.adjustment(myComment = myComment, reset = FALSE, plots = FALSE, includeMessage = FALSE)
}

# Hover Output ####
t.hover.output <- function(resObj, xloc, yloc, typeOfGraph, typeOfOutput){
  #get data for DY graph
  if(typeOfGraph == "DY"){
    myDY <- round(xloc, digits = 0)
    myResidStd <- round(yloc, digits = 2)
    
    #get table of all points with myDY
    myDat <- resObj$dat[resObj$dat$DY==myDY,]
    
    #remove NA values
    myDat <- myDat[!is.na(myDat$residStd),]
    
    #get everything that matches up to 2 decimal places
    myDat <- myDat[round(myDat$residStd, digits = 2) == myResidStd,]
  }
  
  #output whatever
  if(typeOfOutput=="AY"){
    return(myDat$AY)
  }
  if(typeOfOutput=="DY"){
    return(myDat$DY)
  }
  if(typeOfOutput=="CY"){
    return(myDat$CY)
  }
  if(typeOfOutput=="value"){
    return(myDat$value)
  }
  if(typeOfOutput=="logvalue"){
    return(myDat$logvalue)
  }
  if(typeOfOutput=="residStd"){
    return(myDat$residStd)
  }
  
}


# ###### resplot (Weighted standardized residuals) Function with outlier selection (AW special specifically for shiny) ################################
# resPlotAW <- function(resObj, newWindow = FALSE, option = 'all', locator = FALSE) {
#   # plot std weighted residuals vs AY, CY, DY, and actual
#   # newWindow = TRUE; allows user to have plot created in new window
#   #   option = NULL; allows user to select single plot
#   #     browser()
#   # check if a valid option was selected. If not, stop with message
#   if ( !(option %in% c('AY','DY','CY','fitted','all')) ) { 
#     stop("Error: Select option in ('AY','DY','CY','fitted','all')")
#   }
#   
#   # Make sure only 'DY', 'AY', or 'CY' were selected if locator is TRUE
#   if (locator == TRUE){
#     if ( !(option %in% c('AY','DY','CY')) ) { 
#       stop("Error: If using locator only select option in ('AY','DY','CY')")
#     }
#   }
#   
#   if (newWindow == TRUE) {
#     x11()  # Create plot in new window
#   } 
#   
#   if (option == 'all'){
#     op <- par(mfrow=c(2,2),oma = c(0, 1, 3, 0))
#   } else {
#     op <- par(mfrow=c(1,1),oma = c(0, 1, 3, 0))
#   }
#   
#   model = resObj$model  
#   # Data used to fit the model omits NAs
#   dat <- resObj$dat
#   
#   #use logvalue if loglinear, value if GLM
#   if(ROCKY.settings$selected.model=="loglinear"){
#     fitDat <- dat[!is.na(dat$logvalue),]
#   }
#   else if(ROCKY.settings$selected.model=="GLM"){
#     fitDat <- dat[!is.na(dat$value),]
#   }
#   
#   wt <- model$weights
#   wgt <- sqrt(wt)
#   dgf<-model$df.residual
#   
#   # Using standardized residuals (pearson)
#   # Normally we could obtain the standardized residuals using rstandard(model),
#   # but in the case of filtered AY values we need to calculate the standardized residuals using the unfiltered logvalues.
#   
#   # standardized residuals
#   residStd <- fitDat$residStd
#   
#   # function: UBE variance -> MLE standard deviation
#   sdp<-function(x)(sqrt(var(x)*(length(x)-1)/length(x)))
#   
#   
#   #set ymin and ymax based on ROCKY settings, default to 1.1 if no setting used
#   #ymin
#   if(is.null(ROCKY.settings$outliers$yminFactor)){
#     ymin <- min(residStd)*1.1
#   }
#   else{
#     ymin <- min(residStd)*ROCKY.settings$outliers$yminFactor
#   }
#   
#   #ymax
#   if(is.null(ROCKY.settings$outliers$ymaxFactor)){
#     ymax <- max(residStd)*1.1 
#   }
#   else{
#     ymax <- max(residStd)*ROCKY.settings$outliers$ymaxFactor 
#   }
#   #   browser()
#   
#   
#   if (option %in% c('DY','all')){
#     plot(residStd ~ fitDat[['DY']],
#          main=paste("Wtd.Std. Residuals vs. Dev. Years"),
#          xlab='DY', ylab="Wtd. Standardised Residuals", ylim=c(ymin,ymax),
#          pch = 21, col = 'blue', bg = 'blue',xaxt = 'n')  
#     axis(side = 1, at=sort(unique(fitDat[['DY']])))
#     res.mean<- tapply(residStd, as.factor(fitDat[['DY']]),mean)
#     lines(min(fitDat$DY):(max(fitDat$DY)),res.mean, col=2) 
#     abline(h=0, lty=2)
#     res.std <- tapply(residStd, as.factor(fitDat[['DY']]),sdp)
#     # remove NAs from std dev data
#     res.stdRmNA <- res.std[!is.na(res.std)]
#     res.mean <- res.mean[!is.na(res.std)]
#     xline1 <- as.numeric(names(res.stdRmNA))
#     polygon(c(xline1,rev(xline1)),c(res.mean + 0.25*res.stdRmNA,rev(res.mean - 0.25*res.stdRmNA)), col = "#0000A040", border = NA)
#     polygon(c(xline1,rev(xline1)),c(res.mean + 1*res.stdRmNA,rev(res.mean - 1*res.stdRmNA)), col = "#0000A010", border = NA)
#     polygon(c(xline1,rev(xline1)),c(res.mean + 2*res.stdRmNA,rev(res.mean - 2*res.stdRmNA)), col = "#0000A010", border = NA)
#     polygon(c(xline1,rev(xline1)),c(res.mean + 3*res.stdRmNA,rev(res.mean - 3*res.stdRmNA)), col = "#0000A010", border = NA)
#     polygon(c(xline1,rev(xline1)),c(res.mean + 4*res.stdRmNA,rev(res.mean - 4*res.stdRmNA)), col = "#0000A020", border = NA)
#     
#     # Use locator to select outliers
#     if (locator == TRUE){
#       outlierSelect = identify(fitDat[[option]],residStd,labels = row.names(fitDat))
#       outliers <- fitDat[outlierSelect,][,c('id','logvalue')]
#     }
#   }
#   
#   if (option %in% c('AY','all')){
#     plot(residStd ~ fitDat[['AY']],
#          main=paste("Wtd.Std. Residuals vs. Acc. Years"), 
#          xlab='AY', ylab="Wtd. Standardised Residuals", ylim=c(ymin,ymax),
#          pch = 21, col = 'blue', bg = 'blue',xaxt = 'n')    
#     axis(side = 1, at=seq(min(fitDat[['AY']]),max(fitDat[['AY']]),2))
#     res.mean<- tapply(residStd, as.factor(fitDat[['AY']]),mean)
#     lines(min(fitDat$AY):(max(fitDat$AY)),res.mean, col=2) 
#     abline(h=0, lty=2)
#     res.std <- tapply(residStd, as.factor(fitDat[['AY']]),sdp)
#     # remove NAs from std dev data
#     res.stdRmNA <-res.std[!is.na(res.std)]
#     res.mean <- res.mean[!is.na(res.std)]
#     xline1 <- as.numeric(names(res.stdRmNA))
#     polygon(c(xline1,rev(xline1)),c(res.mean + 0.25*res.stdRmNA,rev(res.mean - 0.25*res.stdRmNA)), col = "#0000A040", border = NA)
#     polygon(c(xline1,rev(xline1)),c(res.mean + 1*res.stdRmNA,rev(res.mean - 1*res.stdRmNA)), col = "#0000A010", border = NA)
#     polygon(c(xline1,rev(xline1)),c(res.mean + 2*res.stdRmNA,rev(res.mean - 2*res.stdRmNA)), col = "#0000A010", border = NA)
#     polygon(c(xline1,rev(xline1)),c(res.mean + 3*res.stdRmNA,rev(res.mean - 3*res.stdRmNA)), col = "#0000A010", border = NA)
#     polygon(c(xline1,rev(xline1)),c(res.mean + 4*res.stdRmNA,rev(res.mean - 4*res.stdRmNA)), col = "#0000A020", border = NA)
#     
#     # Use locator to select outliers
#     if (locator == TRUE){
#       outlierSelect = identify(fitDat[[option]],residStd,labels = row.names(fitDat))
#       outliers <- fitDat[outlierSelect,][,c('id','logvalue')]
#     }
#   }
#   
#   if (option %in% c('CY','all')){
#     
#     
#     plot.default(residStd ~ fitDat[['CY']],
#                  main=paste("Wtd.Std. Residuals vs. Cal. Years"),
#                  xlab='CY', ylab="Wtd. Standardised Residuals", ylim=c(ymin,ymax),
#                  pch = 21, col = 'blue', bg = 'blue',xaxt = 'n')  
#     axis(side = 1, at=seq(min(fitDat[['CY']]),max(fitDat[['CY']]),2))
#     res.mean<- tapply(residStd, as.factor(fitDat[['CY']]),mean)
#     lines(min(fitDat$CY):(max(fitDat$CY)),res.mean, col=2) 
#     abline(h=0, lty=2)
#     res.std <- tapply(residStd, as.factor(fitDat[['CY']]),sdp)
#     # remove NAs from std dev data
#     res.stdRmNA <-res.std[!is.na(res.std)]
#     res.mean <- res.mean[!is.na(res.std)]
#     xline1 <- as.numeric(names(res.stdRmNA))
#     polygon(c(xline1,rev(xline1)),c(res.mean + 0.25*res.stdRmNA,rev(res.mean - 0.25*res.stdRmNA)), col = "#0000A040", border = NA)
#     polygon(c(xline1,rev(xline1)),c(res.mean + 1*res.stdRmNA,rev(res.mean - 1*res.stdRmNA)), col = "#0000A010", border = NA)
#     polygon(c(xline1,rev(xline1)),c(res.mean + 2*res.stdRmNA,rev(res.mean - 2*res.stdRmNA)), col = "#0000A010", border = NA)
#     polygon(c(xline1,rev(xline1)),c(res.mean + 3*res.stdRmNA,rev(res.mean - 3*res.stdRmNA)), col = "#0000A010", border = NA)
#     polygon(c(xline1,rev(xline1)),c(res.mean + 4*res.stdRmNA,rev(res.mean - 4*res.stdRmNA)), col = "#0000A020", border = NA)
#     
#     # Use locator to select outliers
#     if (locator == TRUE){
#       outlierSelect = identify(fitDat[[option]],residStd,labels = row.names(fitDat))
#       outliers <- fitDat[outlierSelect,][,c('id','logvalue')]
#     }
#   }
#   
#   if (option %in% c('fitted','all')){
#     yname <- names(model.frame(model))[1]
#     plot.default(residStd ~ fitted(model),
#                  main="Wtd.Std.Residuals vs. fitted",
#                  xlab='fitted logvalue', ylab="Wtd. Standardised Residuals", ylim=c(ymin,ymax),
#                  pch = 21, col = 'blue', bg = 'blue')  
#     panel.smooth(y=residStd, x=fitted(model))
#     abline(h=0, lty=2)
#   }
#   
#   
#   par(op) 
#   
#   # Return any selected outliers
#   if (locator == TRUE){
#     return(outliers)
#   }
# }
# #### End of resplot Function# # # # # # # # # # # # # # # # # # # # # #  # # # # # # # # # # # # # # # # 

# # Code to reset hetero plots (re-render them in shiny)
# t.reset.hetero.plots <- function(){
#   output$hetero.residual.plot <- renderPlot({
#     resPlot(resObj, option = 'DY', newWindow = FALSE)
#   })
#   output$hetero.structure.plot <- renderPlot({
#     trendVarPlot(resObj, newWindow = FALSE, option = 'weight')
#   })
# }

#### Create all VBScripts On-the-fly ##################################################################################

create.all.VB.Scripts <- function(selectionFile = 'Reserve Model Selections.xlsx'){
  VB.AddToLog_Comment(selectionFile)
  VB.AddToLog_CopyCurrentID(selectionFile)
  VB.AddToLog_G_f(selectionFile)
  VB.AddToLog_I_f(selectionFile)
  VB.AddToLog_ID(selectionFile)
  VB.AddToLog_LoadModel(selectionFile)
  VB.AddToLog_Loglikelihood(selectionFile)
  VB.AddToLog_N(selectionFile)
  VB.AddToLog_NewModelID(selectionFile)
  VB.AddToLog_NumberOfParameters(selectionFile)
  VB.AddToLog_R2(selectionFile)
  VB.AddToLog_ReserveStats(selectionFile)
  VB.Check_HasAYGp(selectionFile)
  VB.Check_HasAYgpFilters(selectionFile)
  VB.Check_HasCYGp(selectionFile)
  VB.Check_HasCYpt(selectionFile)
  VB.Check_HasDYGp(selectionFile)
  VB.Check_HasDYw(selectionFile)
  VB.Check_HasDYwGp(selectionFile)
  VB.Check_NewModel(selectionFile)
  VB.Clear_AVResults(selectionFile)
  VB.Clear_AYFilters(selectionFile)
  VB.Clear_Outliers(selectionFile)
  VB.Load_ClearLog(selectionFile)
  VB.Load_StartAY(selectionFile)
  VB.Load_StartCY(selectionFile)
  VB.PastePlot_AVResults(selectionFile)
  VB.Push_AVResults(selectionFile)
  VB.Push_AYGp(selectionFile)
  VB.Push_AYgpFilters(selectionFile)
  VB.Push_CYGp(selectionFile)
  VB.Push_CYpt(selectionFile)
  VB.Push_DYGp(selectionFile)
  VB.Push_DYw(selectionFile)
  VB.Push_DYwGp(selectionFile)
  VB.Push_Outliers(selectionFile)
  VB.Push_RiskFreeSummaryOutput(selectionFile)
  VB.Push_SummaryOutput(selectionFile)
  VB.SaveWorkbook(selectionFile) 
  VB.Load_Images(selectionFile)
  VB.Load_AVPlot(selectionFile)
  VB.Push_ReserveTable(selectionFile)
}

#### Create VBScripts On-the-fly:  'AddToLog' Folder ##################################################################

# ADD TO LOG FOLDER 
VB.AddToLog_Comment <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'AddToLog_Comment',
    'Sub AddToLog_Comment() ',
    
    'Dim xlApp ',
    'Dim NumLog',
    'Dim CommentCol, CommentOffset',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Log").Activate',
    '.Range("A1").Activate',
    '.ScreenUpdating = FALSE',
    'End With',
    
    'NumLog = xlApp.Range("NumLog").Value',
    'CommentCol = xlApp.Range("CommentCol").Value',
    'CommentOffset = CommentCol - 1',
    
    'With xlApp',
    '.ActiveCell.Offset(NumLog, CommentOffset).PasteSpecial',
    '.Range("A1").Activate',
    '.ScreenUpdating = TRUE',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AddToLog/AddToLog_Comment.vbs")
}

VB.AddToLog_CopyCurrentID <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'AddToLog_CopyCurrentID',
    
    'Sub AddToLog_CopyCurrentID() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Log").Activate',
    '.Range("NumLog").Copy',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AddToLog/AddToLog_CopyCurrentID.vbs")
}

VB.AddToLog_G_f <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'AddToLog_G_f',
    'Sub AddToLog_G_f() ',
    
    'Dim xlApp ',
    'Dim NumLog',
    'Dim Gf_Col, Gf_Offset',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Log").Activate',
    '.Range("A1").Activate',
    'End With',
    
    'NumLog = xlApp.Range("NumLog").Value',
    
    'Gf_Col = xlApp.Range("FutureDYCol_Log").Value',
    'Gf_Offset = Gf_Col - 1',
    
    'With xlApp',
    '.ActiveCell.Offset(NumLog, Gf_Offset).PasteSpecial',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AddToLog/AddToLog_G_f.vbs")
}

VB.AddToLog_I_f <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'AddToLog_I_f',
    'Sub AddToLog_I_f() ',
    
    'Dim xlApp ',
    'Dim NumLog',
    'Dim If_Col, If_Offset',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Log").Activate',
    '.Range("A1").Activate',
    'End With',
    
    'NumLog = xlApp.Range("NumLog").Value',
    
    'If_Col = xlApp.Range("FutureCYCol_Log").Value',
    'If_Offset = If_Col - 1',
    
    'With xlApp',
    '.ActiveCell.Offset(NumLog, If_Offset).PasteSpecial',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AddToLog/AddToLog_I_f.vbs")
}

VB.AddToLog_ID <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'AddToLog_ID',
    
    'Sub AddToLog_ID() ',
    
    'Dim xlApp ',
    'Dim PrevID, NewID',
    'Dim NumLog',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Log").Activate',
    '.Range("A1").Activate',
    'End With',
    
    'NumLog = xlApp.Range("NumLog").Value',
    
    'With xlApp',
    '.ActiveCell.Offset(NumLog, 0).Activate',
    'End With',
    
    'PrevID = xlApp.ActiveCell.Value',
    'NewID = PrevID + 1',
    
    'With xlApp',
    '.ActiveCell.Offset(1, 0).Activate',
    '.ActiveCell.Value = NewID',
    '.Range("A1").Activate',
    'End With',
    
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AddToLog/AddToLog_ID.vbs")
}

VB.AddToLog_LoadModel <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'AddToLog_LoadModel',
    'Sub AddToLog_LoadModel() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Log").Activate',
    '.Range("A2").PasteSpecial',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AddToLog/AddToLog_LoadModel.vbs")
}

VB.AddToLog_Loglikelihood <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'AddToLog_Loglikelihood',
    'Sub AddToLog_Loglikelihood() ',
    
    'Dim xlApp ',
    'Dim NumLog',
    'Dim LLCol, LLOffset',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Log").Activate',
    '.Range("A1").Activate',
    'End With',
    
    'NumLog = xlApp.Range("NumLog").Value',
    
    'LLCol = xlApp.Range("LoglikelihoodCol").Value',
    'LLOffset = LLCol - 1',
    
    'With xlApp',
    '.ActiveCell.Offset(NumLog, LLOffset).PasteSpecial',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AddToLog/AddToLog_Loglikelihood.vbs")
}

VB.AddToLog_N <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'AddToLog_N',
    'Sub AddToLog_N() ',
    
    'Dim xlApp ',
    'Dim NumLog',
    'Dim N_Col, N_Offset',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Log").Activate',
    '.Range("A1").Activate',
    'End With',
    
    'NumLog = xlApp.Range("NumLog").Value',
    
    'N_Col = xlApp.Range("NCol_Log").Value',
    'N_Offset = N_Col - 1',
    
    'With xlApp',
    '.ActiveCell.Offset(NumLog, N_Offset).PasteSpecial',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AddToLog/AddToLog_N.vbs")
}

VB.AddToLog_NewModelID <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'AddToLog_NewModelID',
    
    'Sub AddToLog_NewModelID() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Log").Activate',
    '.Range("A2").Value = 1',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AddToLog/AddToLog_NewModelID.vbs")
}

VB.AddToLog_NumberOfParameters <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'AddToLog_NumberOfParameters',
    'Sub AddToLog_NumberOfParameters() ',
    
    'Dim xlApp ',
    'Dim NumLog',
    'Dim NPara_Col, NPara_Offset',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Log").Activate',
    '.Range("A1").Activate',
    'End With',
    
    'NumLog = xlApp.Range("NumLog").Value',
    
    'NPara_Col = xlApp.Range("NParaCol_Log").Value',
    'NPara_Offset = NPara_Col - 1',
    
    'With xlApp',
    '.ActiveCell.Offset(NumLog, NPara_Offset).PasteSpecial',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AddToLog/AddToLog_NumberOfParameters.vbs")
}

VB.AddToLog_R2 <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'AddToLog_R2',
    'Sub AddToLog_R2() ',
    
    'Dim xlApp ',
    'Dim NumLog',
    'Dim R2_Col, R2_Offset',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Log").Activate',
    '.Range("A1").Activate',
    'End With',
    
    'NumLog = xlApp.Range("NumLog").Value',
    
    'R2_Col = xlApp.Range("RSqCol_Log").Value',
    'R2_Offset = R2_Col - 1',
    
    'With xlApp',
    '.ActiveCell.Offset(NumLog, R2_Offset).PasteSpecial',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AddToLog/AddToLog_R2.vbs")
}

VB.AddToLog_ReserveStats <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'AddToLog_ReserveStats',
    'Sub AddToLog_ReserveStats() ',
    
    'Dim xlApp ',
    'Dim NumLog',
    'Dim ForecastCol, ForecastOffset',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Log").Activate',
    '.Range("A1").Activate',
    'End With',
    
    'NumLog = xlApp.Range("NumLog").Value',
    
    'ForecastCol = xlApp.Range("ForecastCol").Value',
    'ForecastOffset = ForecastCol - 1',
    
    'With xlApp',
    '.ActiveCell.Offset(NumLog, ForecastOffset).PasteSpecial',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AddToLog/AddToLog_ReserveStats.vbs")
}

#### Create VBScripts On-the-fly:  'AutoValidation' Folder ##################################################################

VB.Clear_AVResults <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Clear_AVResults',
    
    'Sub Clear_AVResults() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    '.ScreenUpdating = FALSE',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Validation").Activate',
    '.Range("A2").Activate',
    '.ActiveCell.PasteSpecial',
    '.ScreenUpdating = TRUE',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AutoValidation/Clear_AVResults.vbs")
}

VB.PastePlot_AVResults <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'PastePlot_AVResults',
    
    'Sub PastePlot_AVResults() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    '.ScreenUpdating = FALSE',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Validation").Activate',
    '.Range("A12").Activate',
    '.ActiveCell.PasteSpecial',
    '.ScreenUpdating = TRUE',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AutoValidation/PastePlot_AVResults.vbs")
}

VB.Push_AVResults <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Push_AVResults',
    
    'Sub Push_AVResults() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    '.ScreenUpdating = FALSE',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Validation").Activate',
    '.Range("A2").Activate',
    '.ActiveCell.PasteSpecial',
    '.ScreenUpdating = TRUE',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/AutoValidation/Push_AVResults.vbs")
}

#### Create VBScripts On-the-fly:  'Push' Folder ##################################################################

VB.Push_AYGp <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Push_AYGp',
    
    'Sub Push_AYGp()',
    
    'Dim xlApp',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    
    '.Visible = True',
    
    '.ScreenUpdating = FALSE',
    
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    
    '.Worksheets("AYGp").Activate',
    
    '.Range("A1").Activate',
    
    '.ActiveCell.PasteSpecial',
    
    '.ScreenUpdating = TRUE',
    
    '.Range("A1").Activate',
    
    'End With',
    
    'End Sub'
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Push/Push_AYGp.vbs")
}

VB.Push_AYgpFilters <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Push_AYgpFilters',
    
    'Sub Push_AYgpFilters() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    '.ScreenUpdating = FALSE',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("AYgpFilters").Activate',
    '.Range("A1").Activate',
    '.ActiveCell.PasteSpecial',
    '.ScreenUpdating = TRUE',
    '.Worksheets("Log").Activate',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Push/Push_AYgpFilters.vbs")
}

VB.Push_CYGp <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Push_CYGp',
    
    'Sub Push_CYGp() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    '.ScreenUpdating = FALSE',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("CYGp").Activate',
    '.Range("A1").Activate',
    '.ActiveCell.PasteSpecial',
    '.ScreenUpdating = TRUE',
    '.Worksheets("CYGp").Range("A1").Activate',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Push/Push_CYGp.vbs")
}

VB.Push_CYpt <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Push_CYpt',
    
    'Sub Push_CYpt() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    '.ScreenUpdating = FALSE',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("CYpt").Activate',
    '.Range("A1").Activate',
    '.ActiveCell.PasteSpecial',
    '.ScreenUpdating = TRUE',
    '.Worksheets("CYpt").Range("A1").Activate',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Push/Push_CYpt.vbs")
}

VB.Push_DYGp <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Push_DYGp',
    
    'Sub Push_DYGp() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    '.ScreenUpdating = FALSE',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("DYGp").Activate',
    '.Range("A1").Activate',
    '.ActiveCell.PasteSpecial',
    '.ScreenUpdating = TRUE',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Push/Push_DYGp.vbs")
}

VB.Push_DYw <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Push_DYw',
    
    'Sub Push_DYw() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    '.ScreenUpdating = FALSE',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("DYw").Activate',
    '.Range("A1").Activate',
    '.ActiveCell.PasteSpecial',
    '.ScreenUpdating = TRUE',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Push/Push_DYw.vbs")
}

VB.Push_DYwGp <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Push_DYwGp',
    
    'Sub Push_DYwGp() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    '.ScreenUpdating = FALSE',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("DYwGp").Activate',
    '.Range("A1").Activate',
    '.ActiveCell.PasteSpecial',
    '.ScreenUpdating = TRUE',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Push/Push_DYwGp.vbs")
}

VB.Push_Outliers <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Push_Outliers',
    
    'Sub Push_Outliers() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    '.ScreenUpdating = FALSE',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Outliers").Activate',
    '.Range("A2").Activate',
    '.ActiveCell.PasteSpecial',
    '.ScreenUpdating = TRUE',
    '.Worksheets("Log").Activate',
    '.Range("A2").Activate',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Push/Push_Outliers.vbs")
}

VB.Push_RiskFreeSummaryOutput <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Push_RiskFreeSummaryOutput',
    
    'Sub Push_RiskFreeSummaryOutput() ',
    
    'Dim xlApp',
    'Dim PasteLoc',  
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    '.ScreenUpdating = FALSE',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    'End With',
    
    'PasteLoc = xlApp.Range("SummaryPasteLoc").Value',
    
    'With xlApp',
    '.Worksheets("Risk-Free Summary Output").Activate',
    '.Range("A" & PasteLoc).PasteSpecial',
    '.ScreenUpdating = TRUE',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Push/Push_RiskFreeSummaryOutput.vbs")
}

VB.Push_SummaryOutput <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Push_SummaryOutput',
    
    'Sub Push_SummaryOutput() ',
    
    'Dim xlApp',
    'Dim PasteLoc',  
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    '.ScreenUpdating = FALSE',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    'End With',
    
    'PasteLoc = xlApp.Range("SummaryPasteLoc").Value',
    
    'With xlApp',
    '.Worksheets("Summary Output").Activate',
    '.Range("A" & PasteLoc).PasteSpecial',
    '.ScreenUpdating = TRUE',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Push/Push_SummaryOutput.vbs")
}

VB.Push_ReserveTable <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Push_ReserveTable',
    
    'Sub Push_ReserveTable()',
    
    'Dim xlApp',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    
    '.Visible = True',
    
    '.ScreenUpdating = FALSE',
    
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    
    paste(c('.Run "\'', selectionFile, '\'!ClearResTable.clearReserveTable"'), sep = "", collapse = ""),
    
    '.Worksheets("Reserve Table").Activate',
    
    '.Range("AM1").Activate',
    
    '.ActiveCell.PasteSpecial',
    
    '.ScreenUpdating = TRUE',
    
    '.Range("A1").Activate',
    
    '.Worksheets("Log").Activate',
    
    'End With',
    
    'End Sub'
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Push/Push_ReserveTable.vbs")
}



#### Create VBScripts On-the-fly:  'Util' Folder ##################################################################

VB.SaveWorkbook <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'SaveWorkbook',
    
    'Sub SaveWorkbook() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.ActiveWorkbook.Save',
    'End With',
    
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Util/SaveWorkbook.vbs")
}


#### Create VBScripts On-the-fly:  Main Folder ##################################################################

VB.Check_HasAYGp <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Check_HasAYGp',
    
    'Sub Check_HasAYGp() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Range("HasAYGp").Copy',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Check_HasAYGp.vbs")
}

VB.Check_HasAYgpFilters <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Check_HasAYgpFilters',
    
    'Sub Check_HasAYgpFilters() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Range("HasAYgpFilters").Copy',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Check_HasAYgpFilters.vbs")
}

VB.Check_HasCYGp <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Check_HasCYGp',
    
    'Sub Check_HasCYGp() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Range("HasCYGp").Copy',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Check_HasCYGp.vbs")
}

VB.Check_HasCYpt <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Check_HasCYpt',
    
    'Sub Check_HasCYpt() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Range("HasCYpt").Copy',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Check_HasCYpt.vbs")
}

VB.Check_HasDYGp <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Check_HasDYGp',
    
    'Sub Check_HasDYGp() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Range("HasDYGp").Copy',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Check_HasDYGp.vbs")
}

VB.Check_HasDYw <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Check_HasDYw',
    
    'Sub Check_HasDYw() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Range("HasDYw").Copy',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Check_HasDYw.vbs")
}

VB.Check_HasDYwGp <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Check_HasDYwGp',
    
    'Sub Check_HasDYwGp() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Range("HasDYwGp").Copy',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Check_HasDYwGp.vbs")
}

VB.Check_NewModel <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Check_NewModel',
    
    'Sub Check_NewModel() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Range("NewModel").Copy',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Check_NewModel.vbs")
}

VB.Clear_AYFilters <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'On Error Resume Next',
    
    'Clear_AYFilters',
    
    'Sub Clear_AYFilters() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True', 
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("AYgpFilters").Activate',
    '.Range("AY_gp_Filters").ClearContents',
    '.Range("A2").Activate',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Clear_AYFilters.vbs")
}

VB.Clear_Outliers <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'On Error Resume Next',
    
    'Clear_Outliers',
    
    'Sub Clear_Outliers() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Outliers").Activate',
    '.Range("Outliers").ClearContents',
    '.Range("A2").Activate',
    '.Worksheets("Log").Activate',
    '.Range("A2").Activate',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Clear_Outliers.vbs")
}

VB.Load_ClearLog <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'On Error Resume Next',
    
    'Load_ClearLog',
    'Sub Load_ClearLog() ',
    
    'Dim xlApp ',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Worksheets("Log").Activate',
    '.Range("A1").Activate',
    'End With',
    
    'With xlApp',
    '.Range("Log").ClearContents',
    '.Range("A1").Activate',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Load_ClearLog.vbs")
}

VB.Load_StartAY <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Load_StartAY',
    
    'Sub Load_StartAY() ',
    
    'Dim xlApp ',
    'Dim StartAY',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Range("StartAY").Copy',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Load_StartAY.vbs")
}

VB.Load_StartCY <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Load_StartCY',
    
    'Sub Load_StartCY() ',
    
    'Dim xlApp ',
    'Dim StartCY',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    '.Range("StartCY").Copy',
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Load_StartCY.vbs")
}

#### Create VBScripts On-the-fly:  'Images' Folder ###########################################

VB.Load_Images <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Load_Images',
    
    'Sub Load_Images() ',
    
    'Dim xlApp ',
    'Dim StartCY',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    paste(c('.Run "\'', selectionFile, '\'!update_all_plots"'), sep = "", collapse = ""),
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Images/Load_Images.vbs")
}

VB.Load_AVPlot <- function(selectionFile = 'Reserve Model Selections.xlsm'){
  script <- paste0(c(
    'Option Explicit',
    
    'Load_AVPlot',
    
    'Sub Load_AVPlot() ',
    
    'Dim xlApp ',
    'Dim StartCY',
    
    'Set xlApp = GetObject(,"Excel.Application")',
    
    'With xlApp',
    '.Visible = True',
    paste(c('.Workbooks("', selectionFile, '").Activate'), sep = "", collapse = ""),
    paste(c('.Run "\'', selectionFile, '\'!update_auto_validation_plot"'), sep = "", collapse = ""),
    'End With',
    
    'End Sub '
    
  ), sep = "\n" 
  )
  
  write(x = script, file = "./VBScripts/Images/Load_AVPlot.vbs")
}

#### Come up with model name ##################################################################
create.model.name <- function(myString){
  pos <- regexpr(".xls", text = myString)[1]
  modelName <- substr(x = myString, start = 1, stop = pos-1)
  if(modelName == "Reserve Model Selections"){
    return("Model Name?")
  }
  else{
    return(modelName)
  }
}
