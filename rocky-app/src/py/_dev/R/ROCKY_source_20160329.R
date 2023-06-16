## Run the following functions ## (I will put them as .R functions later so we
## can use the source() function to load function)

# AW ROCKY II notes:
# - does 'logLikCompare' function calculate the 'logLikAct' correctly? (also modelVarMLE)


## Libraries ###############################################################################################
library(dplyr)
library(tidyr)
library(readxl)
library(ggplot2)
library(gridExtra)
############################################################################################################

suppressPackageStartupMessages(suppressWarnings(source("ADRIAN_20160108.R")))
suppressPackageStartupMessages(suppressWarnings(source("MR T.R")))

# Basic Functions ---------------------------------------------------------

# copy table to clipboard
copy.table <- function(obj, size = ROCKY.settings$data$ClipboardSize, row.names = FALSE, col.names = TRUE) {
  clip <- paste('clipboard-', size, sep = '')
  f <- file(description = clip, open = 'w')
  write.table(obj, f, row.names = row.names, col.names = col.names, sep = '\t')
  close(f)  
}

# Paste data into R
paste.table <- function(header = TRUE) {
  f <- file(description = 'clipboard', open = 'r')
  df <- read.table(f, sep = '\t', header = header)
  close(f)
  return(df)
}

lagFnc <- function(vec,nlag = 1){
  # creates a lagged vector lagged by nlag
  l = length(vec)  
  return(c(rep(NA,nlag),vec[1:(l-nlag)]))
}


# Population Variance -----------------------------------------------------
popVar <- function(y){  
  y <- y[!is.na(y)]
  return(sum((y - mean(y))^2)/length(y))
}



## 1.Function to input data and transfer to matrix table #####################################
tritomatrix <- function(Triangle, beginAY = NA, beginCY = NA){
  # This function take the input triangle and converts it to the long format suitable for modeling
  # beginAY holds the minimum AY to keep
  
  # Keep AYs >= beginAY
  if (!is.na(beginAY)){
    Triangle <- Triangle[Triangle$AY >= beginAY,]
  }
  
  
  I <- dim(Triangle)[1]           # number of accident years
  D1 <- dim(Triangle)[2] - 1       # number of development years + 1, gives column number for last AY
  
  initay<- min(Triangle[,1])      # The first accident year
  curray<- max(Triangle[,1])      # The current year
  
  # adj for exposure, remove AY and Exposure columns
  # sweep takes the triangle and divides '/' by the exposure vector accross rows
  TriAdj = sweep(x = Triangle[,c(2:D1)],MARGIN = 1,STATS = Triangle[,dim(Triangle)[2]],FUN = '/')
  
  D <- dim(TriAdj)[2]            # number of development year
  
  TriAdj <-as.matrix(TriAdj)
  dimnames(TriAdj)=list(AY=initay:curray, DY=0:(D-1))  # data is ready as input triangle
  
  # convert to data frame 
  mymatx <- data.frame(
    AY=rep(initay:curray, D),
    DY=rep(0:(D-1), each=I),
    value=as.vector(TriAdj))
  
  # If incremental<=0 set NA
  mymatx$value[mymatx$value<=0] <- NA
  
  # Add dimensions as factors
  mydat <- with(mymatx, data.frame(AY, DY, CY=AY+DY, 
                                   AYf=factor(AY),
                                   DYf=as.factor(DY),
                                   CYf=as.factor(AY+DY),value, 
                                   logvalue=log(value)))
  
  rownames(mydat) <- with(mydat, paste(AY, DY, sep="-"))
  
  # remove CYs before the beginCY
  if (!is.na(beginCY)){
    mydat <- mydat[mydat$CY >= beginCY,]
  }
  
  # relevel CYf incase any levels were removed
  mydat$CYf <- factor(as.character(mydat$CYf))
  
  mydat <- mydat[order(mydat$AY),]
  
  mydat <- inner_join(x = mydat, y = Triangle, by = c("AY"))
  mydat <- mydat[c('AY','DY','CY','AYf','DYf','CYf', 'value','logvalue', 'Exposure')]
  rownames(mydat) <- paste(mydat$AY,mydat$DY,sep='-')
  mydat$id = rownames(mydat)
  mydat <- mydat[order(mydat$AY),]
  
  return(mydat)
  
}
#########End of tritomatrix Function##


#### inc.plot funtion #################################################################################
inc.plot <- function(dat, newWindow = TRUE) {
  # Creats interaction plot of inc payments and log inc payments vs. DY
  #
  # Args:  
  #   dat: data frame containing plot data
  #   newWindow: Choose to have the plot show up in a new window
  
  if (newWindow == TRUE) {
    x11()  # Create plot in new window
  }
  
  # Set window parameters
  op <- par(mfrow=c(2,1),oma = c(0, 0, 3, 0))
  
  with(dat, 
       interaction.plot(DY, AY, value,col=1:nlevels(AYf),fixed=1,main="Incremental Payments",legend=F))
  with(dat,points(1+DY, value, pch=16, cex=0.8))
  with(dat, 
       interaction.plot(DY, AY, logvalue,col=1:nlevels(AYf),fixed=1,main="Log Incremental Payments",legend=F))
  with(dat,points(1+DY, logvalue, pch=16, cex=0.8))
  par(op)
}
#### End of inc.plot funtion ###############################################


## 2.Design matrix function ########################################

designmatrix <- function(YY,varb) {
  varb<-as.character(varb)
  YY<- data.frame(YY)
  n<- nlevels(as.factor(YY[,1]))     ## number of levels of calandar(payment) year
  L<- min(YY)                        ## minimum calendar(payment) year
  S<- max(YY)
  FF<- apply(YY, 1, function(x) t(as.vector(c(rep(1,x-L),rep(0,n-1-(x-L))))))
  FF<- t(as.matrix(FF))
  colnames(FF)<- paste(varb, (L+1):S, sep = "")
  return(FF)
}

## End of Design matrix function## # # # # # # # # # # # # # # # # # # # # #  # # # # # # # # # #
##### updateTrends ########################################################################
updateTrends <- function(resObj,plots = TRUE, customFutureCYTrend = FALSE, customFutureCYStdError = FALSE){
  # Function takes selected groups definitions and joins appropriate variables to reserve data
  #
  # Args
  #     resObj: which contains the following pertenant members
  #     plots: after updating the data we fit the model; when polts == TRUE plots are created in fitModel
  
#   browser()
  ### Initialize variables from reserve object
  dat <- resObj$dat
  AYGp <- resObj$AYGp
  DYGp <- resObj$DYGp
  CYGp <- resObj$CYGp
  trendVars <- resObj$trendVars
  
  ### Remove current variable definitions
  # create vector containing each current variable name, created from last process of this funtion
  groupsToDrop <- unique(trendVars)
  
  columnsToRemove = which(names(dat) %in% groupsToDrop)
  
  # Remove variables from data frame, check first to make sure there are variables to remove
  if (length(columnsToRemove) > 0 ) {dat <- dat[-columnsToRemove]}
  
  # reset trendVars to NULL; variables will be added to trendVars below
  trendVars <- NULL
  
  ##### (1) Accident Year Trend: ALPHA
  # Form dataframe contiaing all levels with placeholders for groups
  alphadm <- (outer(dat$AYf, levels(dat$AYf), `==`)*1)
  rownames(alphadm)<-rownames(dat)
  colnames(alphadm) <- paste("alpha",sep="",levels(dat$AYf))
  
  # Get unique groups
  gps = unique(AYGp$gp)
  
  # Set placeholder matrix of number of rows in data x number of vars (groups)
  ALPHA<-matrix(0,ncol=length(gps),nrow=nrow(dat))
  
  # for each element of the group we multiply full design matrix against a vector corresponding to AY groupings
  for (i in 1:length(gps)){
    
    # create placeholder vector corresponding to levels of AYf and set levels corresponding to current group to 1
    A<- as.vector(rep(0,nlevels(dat$AYf)))
    pos = AYGp[AYGp$gp == gps[i],1]-min(dat$AY)+1
    A[pos] = 1
    # collopse full design matrix by position group vector
    ALPHA[,i]<-alphadm %*% A
  }
  
  colnames(ALPHA)<- paste(gps)
  rownames(ALPHA)<-rownames(dat)
  trendVars <- c(trendVars,colnames(ALPHA))
  
  ##### (2) Development Year Trend: GAMMA
  DYM<- dat['DY']
  # Create full design matrix
  DYdm<- designmatrix(DYM,"gamma")
  # Drop 0 level
  gps = unique(DYGp[DYGp$gp != 0,]$gp)[drop = TRUE]
  GAMMA<-matrix(0,ncol=length(gps),nrow=nrow(dat))
  for (i in 1:length(gps)) {
    g<- as.vector(rep(0,nlevels(dat$DYf)-1))
    pos <- DYGp[DYGp$gp == as.character(gps[i]), 1]
    g[pos]<-1
    GAMMA[,i]<-DYdm %*% g
  }
#   colnames(GAMMA)<- paste("GAMMA",1:length(gps),sep="")  ## input AY names in a group
  colnames(GAMMA)<- paste(gps)  ## input AY names in a group
  rownames(GAMMA)<-rownames(dat)
  trendVars <- c(trendVars,colnames(GAMMA))
  
  ##### (3) Payment Year Trend: IOTA 
  CYM<- dat['CY']
  
  # rownames(CYM)<-rownames(dat)
  CYdm<- designmatrix(CYM,"iota")
  
  
  gps = unique(CYGp[CYGp$gp != 0,]$gp)[drop = TRUE]
  IOTA<-matrix(0,ncol=length(gps),nrow=nrow(dat))

  # in case of zero trend check if there is at least one group
  if (length(gps)>0){
    for (i in 1:length(gps)) {
      I <- as.vector(rep(0,length(unique(dat$CY))-1))
      pos <- CYGp[CYGp$gp == as.character(gps[i]), 1] - min(dat$CY)
      I[pos]<-1
      IOTA[,i]<-CYdm %*% I
    }
    colnames(IOTA)<- paste(gps)
    rownames(IOTA)<- rownames(dat)
    trendVars <- c(trendVars,colnames(IOTA))
  }
  
  dat  <-  cbind(dat,ALPHA,GAMMA,IOTA)
  rownames(dat) <- dat$id
  resObj$dat  <-  dat
  resObj$trendVars <- trendVars

  # update the AYgpFilters based on the updated AY variable groups
  
  AYvars <- unique(AYGp$gp)
  AYgpFilters <- data.frame(gp = AYvars, filter = rep('none',length(AYvars)))
  resObj$AYgpFilters <- AYgpFilters
  resObj <- fitModel(resObj,plots = plots, customFutureCYTrend = customFutureCYTrend, customFutureCYStdError = customFutureCYStdError,
                     UserSelectedModel = ROCKY.settings$selected.model, UserSelectedGLMModel = ROCKY.settings$GLM$selected.model)
  return(resObj)
  
}
##### End of updateTrends function ########################################################################

##### fitModel ########################################################################
fitModel <- function(resObj,updateWeights = FALSE,plots = TRUE,forecast = TRUE,calcCYstability = FALSE, customFutureCYTrend = FALSE, 
                     customFutureCYStdError = FALSE, UserSelectedModel = "loglinear", UserSelectedGLMModel = "odp"){
  # fit the model for reserve object
  # updateWeights = FALSE; when TRUE will update data frame with new weights before fitting
#     browser()
  if (updateWeights == TRUE){
    # find position of weight column and remove
    weightPos <- which(names( resObj$dat) %in% c('w'))
    resObj$dat <-  resObj$dat[-weightPos]
    # join updated weights to datas
    resObj$dat <- inner_join( resObj$dat,  resObj$DYw, by = 'DY') 
    rownames(resObj$dat)  <-  resObj$dat$id
  }  
  
  # Get list of trend variables and convert to formula
  # First we obtain non-zero group names
  AYgps = as.character(unique(resObj$AYGp$gp))
  DYgps = as.character(unique(resObj$DYGp[resObj$DYGp$gp != 0,]$gp)[drop = TRUE])
  CYgps = as.character(unique(resObj$CYGp[resObj$CYGp$gp != 0,]$gp)[drop = TRUE])
  
  # create formula
  vars <- resObj$trendVars
  vars2 <- paste(vars,collapse = ' + ')
  
  # different formula based on selection of loglinear model or GLM
  if(UserSelectedModel=="loglinear"){
    modelFormula <- as.formula(paste('logvalue ~ ',vars2,' - 1'))
  }
  else if(UserSelectedModel=="GLM"){
    modelFormula <- as.formula(paste('value ~ ',vars2,' - 1'))
  }
  
  # We restrict the fitting data to non NA logvalue. This is done automatically by lm,
  # but we make it clear here and fitDat is used below
  dat <- resObj$dat
  
#   browser()
  
  # do non NA logvalue if using loglinear model--if fitting GLM, non-NA regular data is fine
#     fitDat <- dat[ !is.na(dat$logvalue),]
  if(ROCKY.settings$selected.model=="loglinear"){
    fitDat <- dat[ !is.na(dat$logvalue),]
  }
  else if(ROCKY.settings$selected.model=="GLM"){
    fitDat <- dat[!is.na(dat$value),]
    fitDat <- fitDat[fitDat$value > 0,]
  } 
  
# here actually fit the model--standard linear regression for loglinear, GLM for GLM
#   model <- lm(modelFormula,data = fitDat,weights = w)
  if(ROCKY.settings$selected.model=="loglinear"){
    model <- lm(modelFormula,data = fitDat,weights = w)
  }
  else if(ROCKY.settings$selected.model=="GLM"){
    if(ROCKY.settings$GLM$selected.model=="odp"){
      model <- glm(modelFormula, data = fitDat, family = quasipoisson(), weights = w)
    }
    else if(ROCKY.settings$GLM$selected.model=="gamma"){
      model <- glm(modelFormula, data = fitDat, family = Gamma(), weights = w)
    }
    else if(ROCKY.settings$GLM$selected.model=="gaussian"){
      model <- glm(modelFormula, data = fitDat, family = gaussian(), weights = w)
    }
  } 

  ### A lot of the below code won't work with GLM as of right now (6/16/2016)

  resObj$model <- model
  


  # If the model was fitted on filtered data, then the process variance returned
  # by the fit is not the process variance of the data.
  # This step calculates the unfiltered data total process variance. The sigma squared.
  # Let y = act - fit, w equal the weights vector, and dgf be the degress of freedom.
  # Then the process (vectors calculated elementwise) var = sum(y^2 * w)/dgf
  # Individual process variance = (total process var)/(weight for observation)
  # UBE represent unbiased estimator due to dividing by the degrees of freedom.
  # MLE estimator would be found by dividing by the number of observations used to fit the model.
  procVarUBE <- sum((fitDat$logvalueAct - fitted(model))^2 * fitDat$w)/model$df.residual
  resObj$procVarUBE <- procVarUBE
  
  # Calculate standardized residuals (pearson) and add to resObj$dat
  # Normally we could obtain the standardized residuals using rstandard(model),
  # but in the case of filtered AY values we need to calculate the standardized residuals using the unfiltered logvalues.
  
  # model matrix
  X <- model.matrix(model)
  
  # weight matrix
  W <- diag(fitDat$w)
  
  # Difference between actual (unfiltered) logvalue and fitted
  if(ROCKY.settings$selected.model=="loglinear"){
    act_fit <- fitDat$logvalueAct - fitted(model)
  }
  else if(ROCKY.settings$selected.model=="GLM"){
    act_fit <- fitDat$value - fitted(model)
  }
  
  # Record actual variance of unfiltered data to scale standard errors below
  procVarUBE <- resObj$procVarUBE
  
  # calculate standard errors
  varMat <- (solve(W) - X %*% solve(t(X) %*% W %*% X) %*% t(X))*procVarUBE

  # use abs below to eliminate numbers that are just slightly negative, but should be 0 (due to precision issues)
  se <- sqrt(abs(diag(varMat)))

# browser()
  
  # standardized residuals
  residStd <- act_fit/se
  residDF <- data.frame(id = fitDat$id, residStd = residStd, stringsAsFactors = FALSE)
  
  # remove residStd from dat and then join back the updated residStd
  dat$residStd <- NULL
  dat <- left_join(dat, residDF, by = 'id')
  # left_join removes rownames - add back
  rownames(dat) <- dat$id
  resObj$dat <- dat
  
  # Update variable differences
  assign(x = 'resObj', value = get_var_diffs(resObj), inherits = TRUE)
#   resObj <<- get_var_diffs(resObj)
#   browser()
  # Create plots
  if (plots == TRUE){
    resPlot(resObj)
    trendVarPlot(resObj)
  }
#   browser()
  # Update forecast
  if (forecast){
#     resObj <<- resForecast(resObj, customCyTrend = customFutureCYTrend, customCyStdErr = customFutureCYStdError)
    # Changed this since there are too many levels, and the global assignment was not sticking
    assign(x = 'resObj', value = resForecast(resObj, customCyTrend = customFutureCYTrend, customCyStdErr = customFutureCYStdError), inherits = TRUE)
  }

  # Record projected CY trend and standard error
  # Obtain min and max project CYs
  minCY = max(dat$AY)+1
  maxCY = max(dat$CY)
  # Projected CY parameter is the last CY parameter in CYgp
  projCyParam <- filter(resObj$CYGp,CY == maxCY)$gp
  # Get corresponding coefficient
  if(projCyParam == 0){
    projCyTrend <- 0
  }
  else{
    projCyTrend <- coef(model)[names(coef(model)) == projCyParam]
  }
  # Form data frame of all projected CYs and proj CY trend
  CyProjTrendModel <- data.frame(CY = minCY:maxCY, trend = rep(projCyTrend,maxCY-minCY+1))

  # Record std error associated with projected parameter
  if(projCyParam == 0){
    CyProjStdErrorModel <- data.frame(param = 0, stdError = 0)
  }
  else{
    paramVarMat <- vcov(model)
    pos <- which(colnames(paramVarMat) == projCyParam)
    projCyStdError <- paramVarMat[pos,pos]^0.5
    CyProjStdErrorModel <- data.frame(param = projCyParam, stdError = projCyStdError)
  }

  # Record to reserve object
  resObj$CyProjTrendModel  <- CyProjTrendModel
  resObj$CyProjStdErrorModel  <- CyProjStdErrorModel
  
# browser()

  return(resObj)  
}
##### End of fitModel function ########################################################################

#### initializeModel #################################################################################
initializeModel <- function(data, plots = TRUE, customFutureCYTrend = FALSE, customFutureCYStdError = FALSE){
  # This function take reserve data in the long data frame format and creates an initial reserve model object
#   browser()
  # Select a single grouping for AY, DY, and CY
  AYGp = data.frame(AY = unique(data$AY),gp = rep('A1',length(unique(data$AY))))
  DYGp = data.frame(DY = sort(unique(data$DY))[-1],gp = rep('G1',length(unique(data$DY)) - 1))
  DYwGp = data.frame(DY = sort(unique(data$DY)),gp = rep('W1',length(unique(data$DY))))
  CYGp = data.frame(CY = sort(unique(data$CY))[-1],gp = rep('I1',length(unique(data$CY)) - 1))
  # Set all initial DY weights to 1 and join to data
  DYw  = data.frame(DY = sort(unique(data$DY)),w = rep(1,length(unique(data$DY))))  
  dat = inner_join(data,DYw,by = 'DY')
  
  rownames(dat) = dat$id
  # the logvalue in dat will be filtered, we will make a backup copy of dat to hold unfiltered data
  # we also add an unfiltered column in dat labeled logvalueAct
  datBackup <- dat
  dat$logvalueAct <- dat$logvalue
  # We will make a backup logvalue in datBackup inorder to undo outlier removal
  datBackup$logvalueBackup <- datBackup$logvalue
  
  # Initial filter set to none. None has no special purpose other than to signify that no filter has been selected 
  AYgpFilters <- data.frame(gp = 'A1', filter = 'none',stringsAsFactors = FALSE)
  
  # initialize trendVars to NULL since we haven't added trendVars to the data (dat) yet
  # the function updateTrends will add the AYGp, DYGp, CYGp vars to trendVars
  trendVars <- character(0)
  
  # combine data with groupings
  resObj = list(dat = dat,
                    datBackup = datBackup,
                    AYGp = AYGp,
                    DYGp = DYGp,
                    CYGp = CYGp,
                    DYwGp = DYwGp,
                    DYw = DYw,
                    trendVars = trendVars,
                    AYgpFilters = AYgpFilters)
   
  # add initial trend variables to data frame
  resObj = updateTrends(resObj, plots = FALSE)
  
  # add filter data - data that is lagged and leaded to apply filter to
  resObj <- createFilterDat(resObj,nLag = ROCKY.settings$ay.filter$NumberToLag)
  
  # generate default filters
  resObj$filters <- createFilters(ROCKY.settings$ay.filter$filterVector,3)
  
  # defualt outlier list to NULL
  resObj$outliers <- data.frame(id = character(0), logvalue = numeric(0))
  
  # fit initial model
  resObj  <- fitModel(resObj, plots = plots, customFutureCYTrend = customFutureCYTrend, customFutureCYStdError = customFutureCYStdError, 
                      UserSelectedModel = ROCKY.settings$selected.model, UserSelectedGLMModel = ROCKY.settings$GLM$selected.model)
  
  #initial var diffs
  resObj <- get_var_diffs(resObj)
  
  # place holder for selected CY projections
  resObj$CyProjTrendSelect  <- resObj$CyProjTrendModel
  resObj$CyProjStdErrorSelect  <- resObj$CyProjStdErrorModel
  
  return(resObj)
}
#### End of initializeModel funtion #################################################################################


######### calcweights ##################################################################################################
calcWeights <- function(resObj){
  # The function uses the selected groups to update weights (resObj$DYw)
  # The updated weights = current weights * inverse of variance of std residuals in each group
  # Returns resObj with updated resObj$DYw
  # Note that the weights in the reserve data (resObj$dat$w) are not updated
  # running fitModel(resObj,updateWeights = TRUE) will update the weights in $dat and $model
#       browser()
  dat = resObj$dat
  model = resObj$model
  DYwGp <- resObj$DYwGp
  names(DYwGp)[2] <- "gp"
  DYw <- resObj$DYw
  
  # join weight groups to data frame
  dat$gp <- NULL
  dat <- inner_join(dat,DYwGp,by = 'DY')
  rownames(dat) = dat$id
  

#   names(dat)[ncol(dat)] <- 'wGp'
  
  
  # For calculating standardized residuals we need to restrict to data used to fit the model.
  fitDat <- dat[!is.na(dat$logvalue),]
  
  # calculate variance of residuals over each group then format to data frame
  resGps <- fitDat[,c('id','DY','gp','residStd')] 
  
  # subract mean for each DY
  DYmean <- resGps %>% group_by(DY) %>% summarize( DYmu = mean(residStd))
  resGps <- inner_join(resGps,DYmean,by = 'DY')
  resGps$residStd <- resGps$residStd - resGps$DYmu
  wGpVar <- resGps %>% group_by(gp) %>% summarise(var = popVar(residStd)) 
  
  
  # Check if there were errors in calculating varaince. If so, exit.
  if (any(is.na(wGpVar)) | any(wGpVar == 0) == TRUE){
    stop("Variance of group either 0 or NA")
  }
  
  # the adjustment weight is the inverse of the variance
  wGpVar$wAdj <- wGpVar$var^(-1)
  
  # rescale
  wGpVar$wAdj <- wGpVar$wAdj/wGpVar$wAdj[1]
  
  # join weight adj to DY on the DYwGp. Arrange to ensure DY are lined up
  DYwAdj <- inner_join(DYwGp,wGpVar,by = 'gp') 
  DYwAdj <- arrange(DYwAdj, DY)
  # Update weights in DYW
  DYw$w <- DYw$w * DYwAdj$wAdj
  
  #note that the weights in dat have not been updated, running fitModel with updateWeights = TRUE updates the model weights
  # update resObj
  resObj$dat <- dat
  resObj$DYw <- DYw
  return(resObj)
}
######### End of calcweights ##################################################################################################3

## iterateWts function ################################################# 
iterateWts <- function(resObj, maxIter = 10, returnIter = FALSE, plots = TRUE, customFutureCYTrend = FALSE, customFutureCYStdError = FALSE){ #AW -- added plots = TRUE
  # This fuction uses the iterates the calcWeights function to arrive at weights to equalize the variance of the std residuals
  # args:
  #      maxIter is the max number of iterations for adjusting weights
  #      returnIter - if TRUE appends to iterated weights and AIC to resObj under wIterResults
#     browser()
  
  # First reset weights to 1
  resObj$DYw[,c('w')] <- 1
  resObj$dat$w <- 1
  # refit model with weights = 1
  resObj <- fitModel(resObj,updateWeights = TRUE,plots = FALSE, customFutureCYTrend = customFutureCYTrend, customFutureCYStdError = customFutureCYStdError,
                     UserSelectedModel = ROCKY.settings$selected.model, UserSelectedGLMModel = ROCKY.settings$GLM$selected.model)
  
  # variables to save iteration results
  wIter <- resObj$DYw
  AICIter <- data.frame(initialAIC = AIC(resObj$model))
  
  #set maxIter = value in ROCKY.settings
  maxIter <- ROCKY.settings$hetero$MaxIter
  
  for (i in 1:maxIter){
    # save currents wts and AIC for comparison
    priorWts <- resObj$DYw
    priorAIC <- AIC(resObj$model)
    
    # update weights and fit model
    resObj <- calcWeights(resObj)
    resObj <- fitModel(resObj,updateWeights = TRUE,plots = FALSE, customFutureCYTrend = customFutureCYTrend, customFutureCYStdError = customFutureCYStdError,
                       UserSelectedModel = ROCKY.settings$selected.model, UserSelectedGLMModel = ROCKY.settings$GLM$selected.model)
    
    # record new weights and AIC
    
    currWts <- resObj$DYw
    currAIC <- AIC(resObj$model)
    
    # add latest iteration weight
    wIter <- cbind(wIter,currWts$w)
    # assign iteration name
    names(wIter)[ncol(wIter)] = paste('w',i,sep='')
    
    # add later iteration AIC
    AICIter <- cbind(AICIter,currAIC)
    colnames(AICIter)[ncol(AICIter)] = paste('w',i,sep='')
    
    
    # if the change in weights is small we exit the loop
    wChange  <-  sum(abs(currWts$w - priorWts$w))
    if (wChange < ROCKY.settings$hetero$MinWtChange) break
  }
  
  if (returnIter == TRUE){
    resObj$wIterResults = list(w = wIter,AIC = AICIter,nIter = i)
  }
  
  if (i == maxIter){
    warnTxt = paste("Did not coverage in ",i," iterations.",sep='')
    warning(warnTxt)
  }
  resObj <- fitModel(resObj,updateWeights = TRUE,plots = plots, customFutureCYTrend = customFutureCYTrend, 
                     customFutureCYStdError = customFutureCYStdError,
                     UserSelectedModel = ROCKY.settings$selected.model, UserSelectedGLMModel = ROCKY.settings$GLM$selected.model)
  return(resObj)
}
## End of iterateWts function ################################################# 

## Incremental to cumulative ###########################
incr2cum <- function(Triangle, na.rm=FALSE){
  if(na.rm){
    upper <- col(Triangle) <= ncol(Triangle) + 1 - row(Triangle)
    upperna <- which(is.na(Triangle[upper]), arr.ind=TRUE)
    
    Triangle[upper][upperna] <- 0
  }
  cum <- t(apply(Triangle,1, cumsum))
  dimnames(cum) <- dimnames(Triangle)
  expos <- attr(Triangle,"exposure")
  if (!is.null(expos))
    attr(cum,"exposure") <- expos
  class(cum) <- c("triangle", "matrix")
  cum
}
## End of Incremental to cumulative ###########################


## Cumulative to Incremental #####################################
cum2incr <- function(Triangle){
  incr <- cbind(Triangle[,1], t(apply(Triangle,1,diff)))
  dimnames(incr) <- dimnames(Triangle)
  expos <- attr(Triangle,"exposure")
  if (!is.null(expos))
    attr(incr,"exposure") <- expos    
  class(incr) <- c("triangle", "matrix")
  incr
}
## End of Cumulative to Incremental #####################################


## resForecast #################################################
resForecast <- function(resObj, varEst = ROCKY.settings$forecast$varEst, varType = ROCKY.settings$forecast$varType, customCyTrend = FALSE, customCyStdErr = FALSE){
  # This function calculates forecasted reserves and se for inc, AY, and CY paryments
  # varEst can be set as follows
  #   varEst = 'MLE' corresponds to the population est of process variance which is baised; ICRFS defaults to baised estimate
  #   varEst = 'UBE' unbiased estimate of variance
  # varType can be set as follows
  #   varType = 'param_process' estimate lognorm sigma^2 by including param and process variance
  #   varType = 'process' just process variance
  #   varType = 'none' no variance, corresponds to median estimate
  #  customCyTrend = FALSE; if TRUE the custom selected CY trend table (resObj$CYpt) is used to forecast
  #  customCyStdErr = FALSE; if TRUE the custom selected CY param variance is used
#   browser()
  
#   # assign parameters based on ROCKY settings
#   varEst <- ROCKY.settings$forecast$varEst
#   varType <- ROCKY.settings$forecast$varType
  
  
  model <- resObj$model ## lm output
  AYmax <- max(resObj$dat$AY)
  # Only predict NA values greater than latest CY diag
  newdat <- resObj$dat[is.na(resObj$dat$value) & resObj$dat$CY > AYmax,]
  w <- newdat[['w']]
  dgf<-model$df.residual
  nobs<- nobs(model)
  procVarUBE <- resObj$procVarUBE
  
  # if (customCyTrend = TRUE) read  in CY custum trend
  if (customCyTrend == TRUE){
    read.custom.CY.trend()
  }  
  
  # if (customCyStdErr = TRUE) read in CY custum std err
  if (customCyStdErr == TRUE){
    assign(x = 'resObj$CyProjStdErrorSelect$stdError', value = resObj$ModelSettings$FutureCYstdError$Input, inherits = TRUE)
  }
  
  # Use predict function to forecast in linear space, se.fit = TRUE allows us to obtain sigma for each value
  if(ROCKY.settings$selected.model=="loglinear"){
    Pred <- predict(model, newdata = newdat, se.fit=TRUE)  ## weight is considered
  }
  else if(ROCKY.settings$selected.model=="GLM"){
    Pred <- predict(model, newdata = newdat, type="response", se.fit = TRUE)  
  }
  
  # if (customCyTrend = TRUE) update fitted values based on select CY trends
  if (customCyTrend == TRUE){
    
    # join fitted values to newdat
    fitDf <- data.frame(id = newdat$id,CY = newdat$CY,mFit =  Pred$fit)
    
    # list current cofficients/trends
    CyProjTrendModel <- resObj$CyProjTrendModel
    CyProjTrendSelect <- resObj$CyProjTrendSelect
      
    
    # form cumulative sum OF current CY trends as we need to detrend each CY back to starting CY by subtracting
    # the cumulative sum from the linear predictor and adding the cumulative updated trend
    cumCurrCY <- data.frame(CY = CyProjTrendModel$CY, currTrendCY = cumsum(CyProjTrendModel$trend),newTrendCY = cumsum(CyProjTrendSelect$trend))
    
    # fit change = new trend - old trend
    cumCurrCY$fitChange = cumCurrCY$newTrendCY - cumCurrCY$currTrendCY
          
    #join change to fitted data
    fitDf <- inner_join(fitDf,cumCurrCY[, c('CY','fitChange')], by = 'CY')
    fitDf$newFit <- fitDf$mFit + fitDf$fitChange
  }
  
  ## Recreate formula to derive the model.frame and future design matrix   
  model.formula <- as.formula(paste("~", as.character(formula(model)[3])))
  
  ### calculate parameter error
  fdm <- model.matrix(model.formula, data=newdat)
  
  # If custom proj CY std err is selected adjust here
  if (customCyStdErr == TRUE){
    # we replace the variance entry in the vcov matrix corresponding to the projected CY parameter
    vcovModel <- vcov(model)
    # assign selected CY trend parameter and variance
    param <- resObj$CyProjStdErrorSelect$param
    CyParamVar <- resObj$CyProjStdErrorSelect$stdError^2
    # get position of the CY trend param and update with selected variance
    pos <- which(rownames(vcovModel) == param)
    vcovModel[pos,pos] <- CyParamVar
    
  } else {
    vcovModel <- vcov(model)
  }
  varcovar <- fdm %*% vcovModel %*% t(fdm)
  param.err <- as.vector(diag(varcovar))
  
  # custom CY var selection can result in negative parameter variance
  # in this case we issue an error and exit the function
  if (any(param.err < 0)){
    stop("Some values of the parameter variance are negative. This is most likely cuased by a custom CY std Error selection. Try values that are larger than the model CY std error. Check 'CyProjStdErrorModel' and 'CyProjStdErrorSelect'.")
  }
  
  # Process error based on MLE or UBE estimation
  if (varEst == 'MLE') {
    proc.error<-(dgf/nobs)*(procVarUBE)/w
  } else if (varEst == 'UBE') {
    proc.error <- (procVarUBE)/w
  }
  
  # custom future trend, note for fitDf$newFit we are assuming inner_join preserves order (which it does)
  # Will this hold? May want to add control in case this changes.
  # Source that unchanged row order is intended by the author: https://groups.google.com/forum/#!topic/manipulatr/OuAPC4VyfIc
  if (customCyTrend == TRUE){
    Y <- fitDf$newFit
    names(Y) <- fitDf$id
  } else {
    Y <- Pred$fit
  }
  
  # lognorm sigma^2 calc; VarY <- 0 corresponds to median
  if (varType == 'param_process') {
    VarY <- param.err + proc.error
  } else if (varType == 'process') {
    VarY <- proc.error
  } else {
    VarY  <- 0
  }
  
  if(ROCKY.settings$selected.model=="loglinear"){
    P <- exp(Y + VarY/2)
  }
  else if(ROCKY.settings$selected.model=="GLM"){
    P <- Y
  }
  
  P <- P*newdat$Exposure
  if(ROCKY.settings$selected.model=="loglinear"){
    VarP <-  P^2*(exp(VarY)-1)
  }
  else if(ROCKY.settings$selected.model=="GLM"){
    VarP <- VarY
  }
#   VarP <-  P^2*(exp(VarY)-1)
  seP <- sqrt(VarP)
  Results <- cbind(newdat[,c('AY','DY','CY')],Y,VarY,P,VarP,seP,CV=seP/P)
  
  CoVar <-  sweep(sweep((exp(varcovar)-1), 1, P, "*"), 2, P, "*")
  CoVar[col(CoVar)==row(CoVar)] <- 0
  
  seTotal <- sqrt(sum(CoVar) + sum(VarP))
  ResTotal <- sum(P)
  ResultsTotal <- cbind(Reserve = ResTotal,se = seTotal,CV = seTotal/ResTotal)
  
  covarDf <- cbind(newdat[,c('AY','DY','CY')],CoVar)
  numCol <- ncol(covarDf)
  covarDfLong <- gather(data = covarDf,key = corrVar,value = cov,4:numCol)
  covarDfLong$corrVarAY = as.numeric(substr(covarDfLong$corrVar,start = 1,stop = 4))
  covarDfLong$corrVarDY = as.numeric(substr(covarDfLong$corrVar,start = 6,stop = 8))
  covarDfLong$corrVarCY <- covarDfLong$corrVarAY + covarDfLong$corrVarDY
  covarAY <- filter(covarDfLong,AY == corrVarAY)
  
  ResAY <- tapply(Results$P, Results$AY,sum)
  paramCovAY <- tapply(covarAY$cov, covarAY$AY, sum)
  VarAY <- tapply(Results$VarP,Results$AY,sum)
  seAY <- sqrt(paramCovAY + VarAY)
  ResultsAY <- cbind(AY = as.numeric(names(ResAY)), Reserve = ResAY, se = seAY, CV = seAY/ResAY)
  
  covarCY <- filter(covarDfLong,CY == corrVarCY)
  ResCY <- tapply(Results$P, Results$CY,sum)
  paramCovCY <- tapply(covarCY$cov, covarCY$CY, sum)
  VarCY <- tapply(Results$VarP,Results$CY,sum)
  seCY <- sqrt(paramCovCY + VarCY)
  ResultsCY <- cbind(CY = as.numeric(names(ResCY)), Reserve = ResCY, se = seCY, CV = seCY/ResCY)
  
  forecast <- list(inc = Results, AY = ResultsAY, CY = ResultsCY, Total = ResultsTotal)
  resObj$forecast  <- forecast
#   browser()
  return(resObj)
}
## End of resForecast ################################################# 

###### resplot (Weighted standardized residuals) Function with outlier selection ################################
resPlot <- function(resObj, newWindow = TRUE, option = 'all', locator = FALSE, usingShiny = FALSE, includeParameters = TRUE, includeHetero = FALSE) {
  # plot std weighted residuals vs AY, CY, DY, and actual
  # newWindow = TRUE; allows user to have plot created in new window
  #   option = NULL; allows user to select single plot
  # 6/21/2016 (AW) -- started working on greying out/highlighting in some way the parameters, for use in
  #                   DOLPH LUNDGREN (so you know where you have already clicked)
#     browser()
  # check if a valid option was selected. If not, stop with message
  if ( !(option %in% c('AY','DY','CY','fitted','all')) ) { 
    stop("Error: Select option in ('AY','DY','CY','fitted','all')")
  }
  
  # Make sure only 'DY', 'AY', or 'CY' were selected if locator is TRUE
  if (locator == TRUE){
    if ( !(option %in% c('AY','DY','CY')) ) { 
      stop("Error: If using locator only select option in ('AY','DY','CY')")
    }
  }
  
  if (newWindow == TRUE) {
    x11()  # Create plot in new window
  } 
  
  if (option == 'all'){
    op <- par(mfrow=c(2,2),oma = c(0, 1, 3, 0))
  } else {
    op <- par(mfrow=c(1,1),oma = c(0, 1, 3, 0))
  }
  
  model <- resObj$model  
  # Data used to fit the model omits NAs
  dat <- resObj$dat
  
  #use logvalue if loglinear, value if GLM
  if(ROCKY.settings$selected.model=="loglinear"){
    fitDat <- dat[!is.na(dat$logvalue),]
  }
  else if(ROCKY.settings$selected.model=="GLM"){
    fitDat <- dat[!is.na(dat$value),]
  }
  
  wt <- model$weights
  wgt <- sqrt(wt)
  dgf<-model$df.residual
  
  # Using standardized residuals (pearson)
  # Normally we could obtain the standardized residuals using rstandard(model),
  # but in the case of filtered AY values we need to calculate the standardized residuals using the unfiltered logvalues.
  
  # standardized residuals
  residStd <- fitDat$residStd
  
  # function: UBE variance -> MLE standard deviation
  sdp<-function(x)(sqrt(var(x)*(length(x)-1)/length(x)))
  
  
  #set ymin and ymax based on ROCKY settings, default to 1.1 if no setting used
  #ymin
  if(is.null(ROCKY.settings$outliers$yminFactor)){
    ymin <- min(residStd)*1.1
  }
  else{
    ymin <- min(residStd)*ROCKY.settings$outliers$yminFactor
  }
  
  #ymax
  if(is.null(ROCKY.settings$outliers$ymaxFactor)){
    ymax <- max(residStd)*1.1 
  }
  else{
    ymax <- max(residStd)*ROCKY.settings$outliers$ymaxFactor 
  }
#   browser()

  #get parameters & locations
  myDYGp <- resObj$DYGp
  myCYGp <- resObj$CYGp
  myAYGp <- resObj$AYGp
  myDYwGp <- resObj$DYwGp
  myDYw <- resObj$DYw

  # get coefficients in a form that we can join to the variable groups
  coeffs <- data.frame(summary(model)$coefficients[,1:2])
  coeffs$gp <- rownames(coeffs)
    
  # join DY, CY, AY coeffients to myDYGp, myCYGp, and myAYGp
  suppressWarnings(coeffsYr.DY <- left_join(myDYGp,coeffs, by = 'gp'))
  suppressWarnings(coeffsYr.CY <- left_join(myCYGp,coeffs, by = 'gp'))
  suppressWarnings(coeffsYr.AY <- left_join(myAYGp,coeffs, by = 'gp'))
  
  #join DY weights to myDYwGp
  suppressWarnings(coeffsYr.DYwGp <- left_join(myDYwGp,myDYw, by = 'DY'))

  # if there are NA
  if (any(is.na(coeffsYr.DY$Estimate)) == TRUE) {
    coeffsYr.DY[is.na(coeffsYr.DY$Estimate), ][,c('Estimate','Std..Error')] <- 0
  }
  if (any(is.na(coeffsYr.CY$Estimate)) == TRUE) {
    coeffsYr.CY[is.na(coeffsYr.CY$Estimate), ][,c('Estimate','Std..Error')] <- 0
  }
  if (any(is.na(coeffsYr.AY$Estimate)) == TRUE) {
    coeffsYr.AY[is.na(coeffsYr.AY$Estimate), ][,c('Estimate','Std..Error')] <- 0
  }
  if (any(is.na(coeffsYr.DYwGp$w)) == TRUE) {
    coeffsYr.DYwGp[is.na(coeffsYr.DYwGp$w), ][,c('w')] <- 0
  }

  # cumulative sums
  coeffsYr.DY$sum <- cumsum(coeffsYr.DY$Estimate)
  coeffsYr.CY$sum <- cumsum(coeffsYr.CY$Estimate)
  coeffsYr.AY$sum <- cumsum(coeffsYr.AY$Estimate)

  coeffsYr.DY <- rbind(rep(0,4),coeffsYr.DY)
  coeffsYr.CY <- rbind(rep(0,4),coeffsYr.CY)
  coeffsYr.AY <- rbind(rep(0,4),coeffsYr.AY)
  
  # make labels for each unique group
  labelDF.DY <- coeffsYr.DY[,c('DY','gp','Estimate','sum')]
  labelDF.CY <- coeffsYr.CY[,c('CY','gp','Estimate','sum')]
  labelDF.AY <- coeffsYr.AY[,c('AY','gp','Estimate','sum')]
  labelDF.DYwGp <- coeffsYr.DYwGp[,c('DY','gp','w')]
  
  labelDF.DY <- labelDF.DY[labelDF.DY$gp != 0, ]
  labelDF.CY <- labelDF.CY[labelDF.CY$gp != 0, ]
  labelDF.AY <- labelDF.AY[labelDF.AY$gp != 0, ]
  labelDF.DYwGp <- labelDF.DYwGp[labelDF.DYwGp$gp != 0, ]
  
  labelDF.DY <- labelDF.DY[!duplicated(labelDF.DY$gp),]
  labelDF.CY <- labelDF.CY[!duplicated(labelDF.CY$gp),]
  labelDF.AY <- labelDF.AY[!duplicated(labelDF.AY$gp),]
  labelDF.DYwGp <- labelDF.DYwGp[!duplicated(labelDF.DYwGp$gp),]

  if (option %in% c('DY','all')){
    plot(residStd ~ fitDat[['DY']],
         main=paste(ROCKY.settings$plots$residualPlots$DY$title),
         xlab=ROCKY.settings$plots$residualPlots$DY$xAxisTitle, 
         ylab=ROCKY.settings$plots$residualPlots$DY$yAxisTitle, 
         ylim=c(ymin,ymax),
         pch = ROCKY.settings$plots$residualPlots$residuals$symbol, 
         col = ROCKY.settings$plots$residualPlots$residuals$color, 
         bg = ROCKY.settings$plots$residualPlots$residuals$backgroundColor,
         xaxt = 'n')  
    axis(side = 1, at=sort(unique(fitDat[['DY']])))
    res.mean<- tapply(residStd, as.factor(fitDat[['DY']]),mean)
    lines(min(fitDat$DY):(max(fitDat$DY)),res.mean, 
          col=ROCKY.settings$plots$residualPlots$meanLine$color) 
    abline(h=0, 
           lty=ROCKY.settings$plots$residualPlots$zeroLine$lineType, 
           col=ROCKY.settings$plots$residualPlots$zeroLine$color)
    res.std <- tapply(residStd, as.factor(fitDat[['DY']]),sdp)
    # remove NAs from std dev data
    res.stdRmNA <- res.std[!is.na(res.std)]
    res.mean <- res.mean[!is.na(res.std)]
    xline1 <- as.numeric(names(res.stdRmNA))
    polygon(c(xline1,rev(xline1)),c(res.mean + 0.25*res.stdRmNA,rev(res.mean - 0.25*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$oneFourth$color, border = NA)
    polygon(c(xline1,rev(xline1)),c(res.mean + 1*res.stdRmNA,rev(res.mean - 1*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$one$color, border = NA)
    polygon(c(xline1,rev(xline1)),c(res.mean + 2*res.stdRmNA,rev(res.mean - 2*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$two$color, border = NA)
    polygon(c(xline1,rev(xline1)),c(res.mean + 3*res.stdRmNA,rev(res.mean - 3*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$three$color, border = NA)
    polygon(c(xline1,rev(xline1)),c(res.mean + 4*res.stdRmNA,rev(res.mean - 4*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$four$color, border = NA)
    
    if(includeParameters){
      abline(v = labelDF.DY$DY-0.5, 
             lty = ROCKY.settings$plots$residualPlots$parameterLine$type, 
             lwd = ROCKY.settings$plots$residualPlots$parameterLine$width,
             col = ROCKY.settings$plots$residualPlots$parameterLine$color)
    }
    
    if(includeHetero){
      abline(v = labelDF.DYwGp$DY, 
             lty = ROCKY.settings$plots$residualPlots$heteroLine$type, 
             lwd = ROCKY.settings$plots$residualPlots$heteroLine$width, 
             col = ROCKY.settings$plots$residualPlots$heteroLine$color)
    }
    
    # Use locator to select outliers
    if (locator == TRUE){
      outlierSelect = identify(fitDat[[option]],residStd,labels = row.names(fitDat))
      outliers <- fitDat[outlierSelect,][,c('id','logvalue')]
    }
  }
  
  if (option %in% c('AY','all')){
    plot(residStd ~ fitDat[['AY']],
         main=paste(ROCKY.settings$plots$residualPlots$AY$title), 
         xlab=ROCKY.settings$plots$residualPlots$AY$xAxisTitle, 
         ylab=ROCKY.settings$plots$residualPlots$AY$yAxisTitle, 
         ylim=c(ymin,ymax),
         pch = ROCKY.settings$plots$residualPlots$residuals$symbol, 
         col = ROCKY.settings$plots$residualPlots$residuals$color, 
         bg = ROCKY.settings$plots$residualPlots$residuals$backgroundColor,
         xaxt = 'n')    
    axis(side = 1, at=seq(min(fitDat[['AY']]),max(fitDat[['AY']]),2))
    res.mean<- tapply(residStd, as.factor(fitDat[['AY']]),mean)
    lines(min(fitDat$AY):(max(fitDat$AY)),res.mean, 
          col=ROCKY.settings$plots$residualPlots$meanLine$color) 
    abline(h=0, 
           lty=ROCKY.settings$plots$residualPlots$zeroLine$lineType, 
           col=ROCKY.settings$plots$residualPlots$zeroLine$color)
    res.std <- tapply(residStd, as.factor(fitDat[['AY']]),sdp)
    # remove NAs from std dev data
    res.stdRmNA <-res.std[!is.na(res.std)]
    res.mean <- res.mean[!is.na(res.std)]
    xline1 <- as.numeric(names(res.stdRmNA))
    polygon(c(xline1,rev(xline1)),c(res.mean + 0.25*res.stdRmNA,rev(res.mean - 0.25*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$oneFourth$color, border = NA)
    polygon(c(xline1,rev(xline1)),c(res.mean + 1*res.stdRmNA,rev(res.mean - 1*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$one$color, border = NA)
    polygon(c(xline1,rev(xline1)),c(res.mean + 2*res.stdRmNA,rev(res.mean - 2*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$two$color, border = NA)
    polygon(c(xline1,rev(xline1)),c(res.mean + 3*res.stdRmNA,rev(res.mean - 3*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$three$color, border = NA)
    polygon(c(xline1,rev(xline1)),c(res.mean + 4*res.stdRmNA,rev(res.mean - 4*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$four$color, border = NA)
    
    if(includeParameters){
      abline(v = labelDF.AY$AY-0.5, 
             lty = ROCKY.settings$plots$residualPlots$parameterLine$type, 
             lwd = ROCKY.settings$plots$residualPlots$parameterLine$width,
             col = ROCKY.settings$plots$residualPlots$parameterLine$color)
    }
    
    # Use locator to select outliers
    if (locator == TRUE){
      outlierSelect = identify(fitDat[[option]],residStd,labels = row.names(fitDat))
      outliers <- fitDat[outlierSelect,][,c('id','logvalue')]
    }
  }
  
  if (option %in% c('CY','all')){
    
    
    plot.default(residStd ~ fitDat[['CY']],
                 main=paste(ROCKY.settings$plots$residualPlots$CY$title),
                 xlab=ROCKY.settings$plots$residualPlots$CY$xAxisTitle, 
                 ylab=ROCKY.settings$plots$residualPlots$CY$yAxisTitle, 
                 ylim=c(ymin,ymax),
                 pch = ROCKY.settings$plots$residualPlots$residuals$symbol, 
                 col = ROCKY.settings$plots$residualPlots$residuals$color, 
                 bg = ROCKY.settings$plots$residualPlots$residuals$backgroundColor,
                 xaxt = 'n')  
    axis(side = 1, at=seq(min(fitDat[['CY']]),max(fitDat[['CY']]),2))
    res.mean<- tapply(residStd, as.factor(fitDat[['CY']]),mean)
    lines(min(fitDat$CY):(max(fitDat$CY)),res.mean, 
          col=ROCKY.settings$plots$residualPlots$meanLine$color) 
    abline(h=0, 
           lty=ROCKY.settings$plots$residualPlots$zeroLine$lineType, 
           col=ROCKY.settings$plots$residualPlots$zeroLine$color)
    res.std <- tapply(residStd, as.factor(fitDat[['CY']]),sdp)
    # remove NAs from std dev data
    res.stdRmNA <-res.std[!is.na(res.std)]
    res.mean <- res.mean[!is.na(res.std)]
    xline1 <- as.numeric(names(res.stdRmNA))
    polygon(c(xline1,rev(xline1)),c(res.mean + 0.25*res.stdRmNA,rev(res.mean - 0.25*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$oneFourth$color, border = NA)
    polygon(c(xline1,rev(xline1)),c(res.mean + 1*res.stdRmNA,rev(res.mean - 1*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$one$color, border = NA)
    polygon(c(xline1,rev(xline1)),c(res.mean + 2*res.stdRmNA,rev(res.mean - 2*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$two$color, border = NA)
    polygon(c(xline1,rev(xline1)),c(res.mean + 3*res.stdRmNA,rev(res.mean - 3*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$three$color, border = NA)
    polygon(c(xline1,rev(xline1)),c(res.mean + 4*res.stdRmNA,rev(res.mean - 4*res.stdRmNA)), 
            col = ROCKY.settings$plots$residualPlots$SE$four$color, border = NA)
    
    if(includeParameters){
      abline(v = labelDF.CY$CY-0.5, 
             lty = ROCKY.settings$plots$residualPlots$parameterLine$type, 
             lwd = ROCKY.settings$plots$residualPlots$parameterLine$width,
             col = ROCKY.settings$plots$residualPlots$parameterLine$color)
    }
    
    # Use locator to select outliers
    if (locator == TRUE){
      outlierSelect = identify(fitDat[[option]],residStd,labels = row.names(fitDat))
      outliers <- fitDat[outlierSelect,][,c('id','logvalue')]
    }
  }
  
  if (option %in% c('fitted','all')){
    yname <- names(model.frame(model))[1]
    plot.default(residStd ~ fitted(model),
                 main=ROCKY.settings$plots$residualPlots$fitted$title,
                 xlab=ROCKY.settings$plots$residualPlots$fitted$xAxisLabel, 
                 ylab=ROCKY.settings$plots$residualPlots$fitted$yAxisLabel, 
                 ylim=c(ymin,ymax),
                 pch = ROCKY.settings$plots$residualPlots$residuals$symbol, 
                 col = ROCKY.settings$plots$residualPlots$residuals$color, 
                 bg = ROCKY.settings$plots$residualPlots$residuals$backgroundColor)  
    panel.smooth(y=residStd, x=fitted(model))
    abline(h=0, 
           lty=ROCKY.settings$plots$residualPlots$zeroLine$lineType,
           col=ROCKY.settings$plots$residualPlots$zeroLine$color)
  }
  
  
  par(op) 
  
  # Return any selected outliers
  if (locator == TRUE){
    return(outliers)
  }
}
#### End of resplot Function# # # # # # # # # # # # # # # # # # # # # #  # # # # # # # # # # # # # # # # 


###### remove outliers ################################
removeOutliers <- function(resObj, newWindow = TRUE,option = 'DY', select.method = 'locator', outlier.select) {
  # This function removes outlier logvalues from the resObj data
  # option allows the user to select preferred window for outlier removal
#   browser()
  # check if a valid option was selected. If not, stop with message
  if ( !(option %in% c('AY','DY','CY','fitted')) ) { 
    stop("Error: Select option in ('AY','DY','CY','fitted')")
  }
  
  
  
  dat  <- resObj$dat
  outliers <- resObj$outliers
  dataFit <- dat[!is.na(dat$logvalue),]
  wtdrstandard <- rstandard(resObj$model)
  
  # Select outliers using the rePlot function with locator = TRUE
  if(select.method == 'locator'){
    outlierSelect <- resPlot(resObj,option = option,locator = TRUE)
    outlier.select <- outlierSelect
  }
  
  
  
  outliers <- unique(rbind(outliers,outlier.select))
  
  dataWOut <- dat
  dataWOut[dataWOut$id %in% outliers$id,]$logvalue  <- NA
  dataWOut[dataWOut$id %in% outliers$id,]$logvalueAct  <- NA
  resObj$outliers <- outliers
  resObj$dat <- dataWOut
  
  # since we have updated datFiltered we also need to update the filterDat
  # nLag normally defaults to 5; the formula below calculates the value based on the current number of lags in filterDat
  resObj <- createFilterDat(resObj,nLag = (ncol(resObj$filterDat) - 2)/2)
  return(resObj)
}




# Add back outleirs to the data -------------------------------------------


addBackOutliers <- function(resObj){
  # This function adds back to the data outliers that have been removed
#   browser()
  if (length(resObj$outliers$id) > 1){
    # resObj$datBackup$logvalueBackup never changes or has outliers removed
    resObj$dat$logvalue <- resObj$datBackup$logvalue
    resObj$dat$logvalueAct <- resObj$datBackup$logvalueAct
    resObj$outliers <- data.frame(id = character(0), logvalue = numeric(0))
    
    # update filterDat
    resObj <- createFilterDat(resObj,nLag = (ncol(resObj$filterDat) - 2)/2)
  }
  return(resObj)
}

# function: createFilters -------------------------------------------------
createFilters <- function(filterVec,lagN){
  # This function creates discrete exponetial decreasing filters
  # filterVec: vector contianing various exponential decay values
  # lagN: maximum distance to fiter over
  # The filter value is of the form c^lagDistance, where c is a filter value from filterVec
  # This function iterates through all possible combination of 1:lagN and the elements of filterVec
  
  #   browser()
  
  # check that lagN is an integer in 1:5
  if (!(lagN %in% 1:5)){
    stop("Choose lagN in {1,2,3,4,5}")
  }
  
  # check that filter values are in (0,1]
  if (any(filterVec > 1) | any(filterVec <= 0)){
    stop("Make sure filter values are in (0,1]")
  }
  
  # create combination of filter distance (d) and weights (w)
  numFilter <- length(filterVec)
  d <- rep(1:lagN,each = numFilter)
  w <- rep(filterVec,lagN)
  
  # create vector represting distance from filter value
  lagLead <- c(lagN:1,0,1:lagN)
  
  # create placeholder matrix with 1's where there are nonzero filter values
  # that is we set to zero any value that is a distance d further than the lag lead width
  # we accomplish this by comparing the ration of d/lagLead
  # when lagLead is less than 1, then the element is set to 0
  mtmp <- outer(d,1/lagLead)
  mtmp[mtmp<1] <- 0
  mtmp[mtmp>=1] <- 1
  
  # the filter is of the form w^LagLead
  mAdj <- outer(w,lagLead,FUN = '^')
  filterMat <- mtmp*mAdj
  
  # convert to dataframe with descriptive column and rownames
  rownames(filterMat) <- paste('d',d,'_w_',w,sep='')
  colnames(filterMat) <- c(paste('lag',lagN:1,sep=''),'logvalue',paste('lead',1:lagN,sep=''))
  filterDf <- as.data.frame(filterMat)
  filterDf$filter <- rownames(filterDf)
  return(filterDf)
}



# Function: create filtering data -----------------------------------------
createFilterDat <- function(resObj,nLag = ROCKY.settings$ay.filter$NumberToLag){
  # This function takes the data from teh resObj and create lag/lead data to be filtered
  # nLag: select the number of lad/leads. Defaults to 5.
  # Add the resulting filterDat to the resObj and returns the reserve object
  
  dat <- resObj$dat
  DYs <- sort(unique(dat$DY))
  
  laggedAY <- NULL
  
  for (DYindex in 1:length(DYs)){
    
    # select logvalues corresponding to DY
    AYdat <- filter(dat, DY ==  DYs[DYindex])[c('AY','DY','logvalue')]
    lagged <- AYdat['logvalue'] 
    
    # add lag and lead values
    for (i in 1:nLag){
      lagged$lag <- lagFnc(lagged$logvalue,i)
      
      names(lagged)[length(names(lagged))] <- paste('lag',i,sep = '')
      
      numCol <- ncol(lagged)
      lagged <- lagged[,c(numCol,1:(numCol-1))]
      lagged$lead <- rev(lagFnc(rev(lagged$logvalue),i))
      names(lagged)[length(names(lagged))] <- paste('lead',i,sep = '')
    }
    
    # add back AY and DY indentifiers
    laggedID <- cbind(AYdat[c('AY','DY')],lagged)
    
    # attach (rbind) to laggedAY object
    laggedAY <- rbind(laggedAY,laggedID)
  }
  
  # for rows whose logvalue is NA set the entire row to NA (except the AY and DY ids)
  laggedAY[is.na(laggedAY$logvalue),-c(1,2)] <- NA
  
  # order to match ordering on main data
  laggedAY <- arrange(laggedAY, AY, DY)
  resObj$filterDat <- laggedAY
  return(resObj)
}



# Function: calculate t values for differences between factor levels --------
get_var_diffs <- function(resObj){
  # this function calculates the variance of the difference between levels within AY, DY, and CY variable groups
  # var(A - B) = var(A) + var(B) - 2*cov(A,B)
#   browser()
  var_diff <- NULL
  model <- resObj$model
  
  # From the variance covariance matrix we obtain the pieces needed to the variance calculation.
  varMat <- vcov(model)
  
  # List group names to cycle through
  AllGpNames = c('AY','DY','CY')
  
  for (i in 1:3){
    
    # get current group out of c('AY','DY','CY')
    GpName = AllGpNames[i]
    
    # Obtain group elements
    GpNameFull <- paste(GpName,'Gp',sep='')
    Gp  <- as.character(unique(resObj[[GpNameFull]]$gp))
    Gp <- Gp[Gp != '0']
    
    # if there is only one group level then proceed to next loop
    if (length(Gp) < 2){
      next
    }
    pos <- which(colnames(varMat) %in% Gp)
    GPcovar <- varMat[pos,pos]
    
    # we use the sweep function to add the variance first by rows then by columns.
    # The result is that each non-diagonal represents var(A - B)
    varVec <- diag(GPcovar)
    varDiffMat <- sweep(sweep(-2*GPcovar,MARGIN = 1,STATS = varVec,FUN = '+'),MARGIN = 2,STATS = varVec,FUN='+')
    
    # take the sqrt to get the standard error
    seDiffMat <- sqrt(varDiffMat)
    
    # get coefficients
    GPcoef <- coefficients(model)[pos]
    
    # We want to take all possible differences between levels of the variables.
    # To do this we first form a dummy 0 matrix, then use sweep to add all coefficients row wise. Use sweep a second time to subtract all coefficient row wise.
    mat0 <- matrix(0,ncol = length(GPcoef),nrow = length(GPcoef))
    diffMat <- sweep(mat0,1,GPcoef,'+') - sweep(mat0,2,GPcoef,'+')
    rownames(diffMat) = Gp
    colnames(diffMat) = Gp
    diff_T_Ratio <- abs(diffMat/seDiffMat)
    seDiffMat[!upper.tri(seDiffMat)] <- 0
    diff_T_Ratio[!upper.tri(diff_T_Ratio)] <- 0
    diffMat[!upper.tri(diffMat)] <- 0
    
    
    # for data frame to hold the group name (DY, AY, or CY), the two variable levels of the difference,
    # The difference between the coefficients, the standard errorr, and the t value.
    t_ratioDF <- NULL
    t_ratioDF <- data.frame(
      type = rep(GpName,length(diffMat)),
      var1 = c(matrix(rownames(diff_T_Ratio),nrow = nrow(diff_T_Ratio), ncol = ncol(diff_T_Ratio))),
      var2 = rep(rownames(diff_T_Ratio),each = length(rownames(diff_T_Ratio))),
      Diff = c(diffMat),
      se = c(seDiffMat),
      t_value = c(diff_T_Ratio),
      stringsAsFactors=FALSE
    )
    
    # calculate how many periods apart the levels of the group are, then join to t_ratioDf
    yrGps <- resObj[[GpNameFull]]
    names(yrGps)[1] <- 'yr'
    len <- nrow(yrGps)
    
    yrDiffDf <- NULL
    yrDiffDf <- data.frame(var1 = as.character(rep(yrGps$gp)),
                           var2 = as.character(rep(yrGps$gp,each = len)),
                           yrDiff = abs(rep(yrGps$yr,len) - rep(yrGps$yr,each = len)),
                           stringsAsFactors=FALSE
    )
    
    # keep minimum distance and drop combinations with zero distance as this represents the same group level
    yrDiffDf <- aggregate(yrDiff ~ var1 + var2, data = yrDiffDf, min)
    yrDiffDf <- yrDiffDf[yrDiffDf$yrDiff > 0,]
    
    # join yrDiff to t_ratioDF
    t_ratioDF <- inner_join(t_ratioDF,yrDiffDf, by = c('var1','var2'))    
    
    # bind current var differences to var_diff
    var_diff <- rbind(var_diff,t_ratioDF)
  }
  
  # remove zero rows
  var_diff <- var_diff[c(var_diff$t_value != 0),]
  
#   browser()
  
#   assign(x = 'resObj$var_diff', value = var_diff, inherits = TRUE, )
  resObj$var_diff <- var_diff
  return(resObj)
}



# Function: Eliminate insignificant factor level --------------------------
eliminate_var_diff <- function(resObj, t_cutoff = ROCKY.settings$optimizer$tCutoff, maxDist = ROCKY.settings$optimizer$maxDist){
  # This function will elimiate an AY, DY, or CY variable level based
  # on the least significat variable difference found in resObj$var_diff
  # The defualt t value cutoff for two levels being insignificant is 1, but can be increased.
  # maxDist is maximum period distance at which two levels will be grouped. For example if levels
  # A1 = (2002,2003,2004) and A2 = (2008,2009). Then A1 and A2 are 4 periods apart and will not be grouped 
  # with a maxDist = 3.
#   browser()
  var_diff <- resObj$var_diff
  
  # keep rows of var_diff that have t values below the cutoff
  reduce_var <- var_diff[var_diff$t_value < t_cutoff,]
  
  # keep rows <= maxDist
  reduce_var <- reduce_var[reduce_var$yrDiff <= maxDist,]
  
  # keep the most insignificant level
  reduce_var <- arrange(reduce_var, t_value)
  # if there is at least one row, grab the first
  if (nrow(reduce_var) > 0 ){
    reduce_var <- reduce_var[1,]
  }else {
    print(paste("All variable levels significant at cutoff of ",t_cutoff," and distance ",maxDist,sep=''))
    
    #Addition by Andy (to allow this to run automatically)
    variableOptimizer.ExistsInsignificantVariables <<- FALSE
    variableOptimizer.optimizeComment <<- paste("All variable levels significant at cutoff of ",t_cutoff," and distance ",maxDist,sep='')
    variableOptimizer.commentCounter <<- variableOptimizer.commentCounter + 1
  }
  
  
  
  # check if there is a variable to reduce
  if (nrow(reduce_var) > 0 ){
    # In the variable grouping (E.g., DYGp) we set the label of var2 equal to var1
    # First get the group label
    GpName <- reduce_var$type
    GpNameFull <- paste(GpName,'Gp',sep='')
    
    # Set changeVar equal to baseVar; use as.character to remover factor level reference
    baseVar <- as.character(reduce_var$var1)
    changeVar <- as.character(reduce_var$var2)
    
    # change level name
    # same group DF in temp dataframe
    tmpGpDF <- resObj[[GpNameFull]]
    tmpGpDF[ tmpGpDF$gp == changeVar,]$gp <- baseVar
    # save tmpGpDF back to resObj
    resObj[[GpNameFull]] <- tmpGpDF
    resObj <- updateTrends(resObj,plots = FALSE)
    print(paste('Replaced',changeVar,'with',baseVar,'in',GpNameFull))
    print(reduce_var)
    
    #Addition by Andy (to get comment for log (what you get depends on one-step or auto)):
    variableOptimizer.optimizeComment <<- paste('Replaced',changeVar,'with',baseVar,'in',GpNameFull)
    variableOptimizer.commentList <<- append(variableOptimizer.commentList, list(changeVar, baseVar))
    variableOptimizer.commentCounter <<- variableOptimizer.commentCounter + 1
    
  }  
  return(resObj)  
}


# Apply filter ------------------------------------------------------------

applyFilter <- function(toFilter,filterSelect){
  # This function applies a selected filter by value from resObj$filters to
  # a subset of the data resObj$filterDat.
  # This function is purpose built apply the actually filtering inside of the more
  # comprehensive filtering function.
  # The data toFilter needs to be of the form AY DY ... lag/lead values ... This is a subset of resObj$filterDat
  # filter should be a single row from restObj$filterDat
  # E.g., dat = filterDat[filterDat$gp %in% c(2008:2011),]; fulterSelect = 'd4_w_0.5']
  
  # remove filter name and unlist
  pos = which(names(filterSelect) == 'filter')
  filterSelect <- filterSelect[-pos]
  filterSelect <- c(unlist(filterSelect))
  
  # remove AY DY labels from toFilter data
  pos <- which(names(toFilter) %in% c('AY','DY'))
  toFilterTmp <-   toFilter[-pos]
  
  # apply filter accross rows
  
  AYxFilter <- sweep(x = toFilterTmp, MARGIN = 2, STATS = filterSelect, FUN = '*')
  
  # sum up filtered values
  AYSum <- apply(AYxFilter,1,sum,na.rm = TRUE)
  
  # sum up filter weights
  # first set non NA data values equal to 1 then multiply by filter weights 
  forWtSum <- AYxFilter
  forWtSum[!is.na(forWtSum)] <- 1
  forWtSum <- sweep(x = forWtSum,MARGIN = 2,STATS = filterSelect,FUN = '*')
  WtSum <- apply(forWtSum,1,sum,na.rm = TRUE)
  
  # filtered value = sum(wt * value)/sum(wt)
  filteredAYs <- AYSum/WtSum
  
  # some logvalues were NAs before filtering, set these NA logvalues back to NA
  filteredAYs[is.na(toFilter$logvalue)] <- NA
  
  # combine filtered values with the intial AY and DY values
  # note the relative order between AY, DY, and logvalue has been maintained
  
  datFiltered <- cbind(toFilter[c('AY','DY')], logvalue = filteredAYs)
  return(datFiltered)
  
}



# Apply custom filters ----------------------------------------------------
applySelectedAYfilters <-  function(resObj,plots = TRUE, customFutureCYTrend = FALSE, customFutureCYStdError = FALSE){
  # This function applies the selected filters in resObj$AYgpFilters
#   browser()
  # Reset filtered logvalues in dat to unfiltered
  dat <- resObj$dat
  dat$logvalue <- dat$logvalueAct
  
  # Obtain the AYgpFilters and filtering data
  AYgpFilters <- resObj$AYgpFilters
  filterDat <- resObj$filterDat
  
  # join AY gp to filterDat by AY
  AYGp <- resObj$AYGp
  filterDat <- inner_join(filterDat, AYGp, by = 'AY')
  
  # Cycle through all the AY variables. First we record the number of AY vars.
  numVars <- nrow(AYgpFilters)
  
  for (i in 1:numVars){
    
    # get current AY group variable and restrict filterDat to just the corresponding AYs
    currentGp <- AYgpFilters[i,1]    
    toFilter <- filterDat[filterDat$gp == currentGp,]
    
    # remove AYgp columns
    pos <- which(names(toFilter) %in% 'gp')
    toFilter <- toFilter[-pos]
    
    filterName  <-  AYgpFilters[i,2]
    filters <- resObj$filters
    
    
    # check if the selected filter is in the list of filters
    # if it is a valid filter then we filter the data
    if (filterName %in% filters$filter) {
      
      # apply filter using applyFilter function
      filterSelect <- filters[filters$filter == filterName,]
      filteredValues <- applyFilter(toFilter, filterSelect)
      
      # Join filtered data to dat. The result is the old and new logvalues labeled
      # as logvalue.x and logvalue.y
      dat <- left_join(dat,filteredValues,by = c('AY','DY'))
      
      # Order by AY, DY incase the join mixed up the order
      dat <- arrange(dat,AY,DY)
      rownames(dat) <- dat$id
      
      # Change name of logvalue.x to logvalue. First find position, then change name.
      pos <- which(names(dat) %in% 'logvalue.x')
      names(dat)[pos] <- 'logvalue'
      
      # Replace logvalue with logvalue.y wherever logvalue.y is not NA
      dat[!is.na(dat$logvalue.y),]$logvalue <- dat[!is.na(dat$logvalue.y),]$logvalue.y
      
      # Drop logvalue.y
      pos <- which(names(dat) %in% 'logvalue.y')
      dat <- dat[-pos]
      
    }
  }
  
  # Update resObj$dat with filtered data, fit model, and return
  resObj$dat <- dat
  resObj <- fitModel(resObj,plots = plots, customFutureCYTrend = customFutureCYTrend, customFutureCYStdError = customFutureCYStdError,
                     UserSelectedModel = ROCKY.settings$selected.model, UserSelectedGLMModel = ROCKY.settings$GLM$selected.model)
  return(resObj)  
}



# Unfilter AY data --------------------------------------------------------

unfilterDat <- function(resObj, customFutureCYTrend = FALSE, customFutureCYStdError = FALSE){
  # This function returns the data in resObj$dat back to the unfiltered values
  resObj$dat$logvalue <- resObj$dat$logvalueAct
  resObj <- fitModel(resObj, customFutureCYTrend = customFutureCYTrend, customFutureCYStdError = customFutureCYStdError,
                     UserSelectedModel = ROCKY.settings$selected.model, UserSelectedGLMModel = ROCKY.settings$GLM$selected.model)
  return(resObj)
}


# CY Parameter Stability --------------------------------------------------
CYstabilityCalc <- function(resObj, CYholdout = ROCKY.settings$autoValidation$NumOfYears){
  # This function calculates the stability of the latest CY parameter by holding out recent CY data
  # and calculating the projected trend and total reserve
  # The defualt is to holdout up to eight periods:  CYholdout = 8
#       browser() 
  
  # Get most recent CY with data. (equal to the most recent AY)
  latestCY <- max(resObj$dat$AY)
  firstCY <- latestCY - CYholdout + 1
  
  CYstability <- NULL
  startCY <- firstCY
  for (startCY in firstCY:latestCY){
    
    # Save tmp resObj to perform calendar holdout  
    resTmp <- resObj
    dat <- resTmp$dat
    # replace holdout calendar year data with NAs
    dat[dat$CY %in% startCY:latestCY,]$logvalue <- NA
    dat[dat$CY %in% startCY:latestCY,]$logvalueAct <- NA
    resTmp$dat <- dat
    # restrict trend vars to data left to fit
    # first get latest parameter yaer with data
    #     maxAYfit <- max(dat[!is.na(dat$logvalue),]$AY)
    #     maxDYfit <- max(dat[!is.na(dat$logvalue),]$DY)
    maxCYfit <- startCY - 1
    
    
    
    # replace any variables names that have NA data. This step is needed since sometimes we restrict
    # the data in a way such that a given variable level only has NA values
    ZeroGpSearch <- inner_join(dat[,c('AY','logvalue')],resTmp$AYGp, by = 'AY')
    ZeroGpAgg <- tapply(ZeroGpSearch$logvalue,ZeroGpSearch$gp,function(x) sum(!is.na(x)))
    ZeroGp <- ZeroGpAgg[ZeroGpAgg ==0]
    if (length(ZeroGp) > 0){
      ZeroGp <- names(ZeroGp)
      # Get a nonZeroGp to replace the zero group. We pick the first since we only need one
      nonZeroGp <- ZeroGpAgg[ZeroGpAgg != 0]
      nonZeroGp <- names(nonZeroGp)[1]
      resTmp$AYGp[resTmp$AYGp$gp %in% ZeroGp,]$gp <- nonZeroGp
    }
    
    ZeroGpSearch <- inner_join(dat[,c('DY','logvalue')],resTmp$DYGp, by = 'DY')
    ZeroGpAgg <- tapply(ZeroGpSearch$logvalue,ZeroGpSearch$gp,function(x) sum(!is.na(x)))
    ZeroGp <- ZeroGpAgg[ZeroGpAgg ==0]
    if (length(ZeroGp) > 0){
      ZeroGp <- names(ZeroGp)
      # Get a nonZeroGp to replace the zero group. We pick the first since we only need one
      nonZeroGp <- ZeroGpAgg[ZeroGpAgg != 0 & names(ZeroGpAgg) != 0]
      nonZeroGp <- names(nonZeroGp)[1]
      resTmp$DYGp[resTmp$DYGp$gp %in% ZeroGp,]$gp <- nonZeroGp
    }
    
    ZeroGpSearch <- inner_join(dat[,c('CY','logvalue')],resTmp$CYGp, by = 'CY')
    ZeroGpAgg <- tapply(ZeroGpSearch$logvalue,ZeroGpSearch$gp,function(x) sum(!is.na(x)))
    ZeroGp <- ZeroGpAgg[ZeroGpAgg ==0]
    if (length(ZeroGp) > 0){
      ZeroGp <- names(ZeroGp)
      # Get a nonZeroGp to replace the zero group. We pick the first since we only need one
      nonZeroGp <- ZeroGpAgg[ZeroGpAgg != 0 & names(ZeroGpAgg) != 0]
      nonZeroGp <- names(nonZeroGp)[1]
      resTmp$CYGp[resTmp$CYGp$gp %in% ZeroGp,]$gp <- nonZeroGp
    }
    
    # get latest variable group name
    #     lastAYvar <- resTmp$AYGp[resTmp$AYGp$AY == maxAYfit,]$gp
    #     lastDYvar <- resTmp$DYGp[resTmp$DYGp$DY == maxDYfit,]$gp
    lastCYvar <- resTmp$CYGp[resTmp$CYGp$CY == maxCYfit,]$gp
    
    resTmp <- updateTrends(resTmp,plots = FALSE)
    
    # Get coefficients without p values. Add rownames to be the coefficient labal. Reorder to put parameter names first.
    coeffs <- data.frame(summary(resTmp$model)$coefficients[,c(1:2)])
    coeffs$param <- rownames(coeffs)
    coeffs <- coeffs[,c(3,1:2)]
    
    # Get coeff for latestCYgp
    coeffs <- coeffs[coeffs$param == lastCYvar,]
    # if there is no coeff for CY (zero trend) then attatch zeros
    if (nrow(coeffs) == 0){
      coeffs <- rbind(coeffs,data.frame(param = '0',Estimate=0,Std..Error=0))
    }
    # Add holdout label
    holdoutLabel <- paste(startCY,latestCY,sep = '_')
    coeffs$holdout <- holdoutLabel
    coeffs <- coeffs[,c(4,1:3)]
    coeffs <- cbind(coeffs,resTmp$forecast[4])
    
    # Combine with CYstability dataframe
    CYstability <- rbind(coeffs,CYstability)
  }
  
  # Add full model fit
  coeffs <- data.frame(summary(resObj$model)$coefficients[,c(1:2)])
  coeffs$param <- rownames(coeffs)
  coeffs <- coeffs[,c(3,1:2)]
  
  # Get coeff for latestCYgp
  coeffs <- coeffs[coeffs$param == lastCYvar,]
  # if there is no coeff for CY (zero trend) then attatch zeros
  if (nrow(coeffs) == 0){
    coeffs <- rbind(coeffs,data.frame(param = '0',Estimate=0,Std..Error=0))
  }
  # Add holdout label
  holdoutLabel <- 'full_model'
  coeffs$holdout <- holdoutLabel
  coeffs <- coeffs[,c(4,1:3)]
  coeffs <- cbind(coeffs,resObj$forecast[4])
  
  # Combine with CYstability dataframe
  CYstability <- rbind(coeffs,CYstability)
  rownames(CYstability) <- NULL
  names(CYstability) <- c('holdout', 'CYparam', 'Estimate', 'Std_Error', 'Total_Reserve','Reserve_SE','Reserve_CV')
  resObj$CYstability <- CYstability
  return(resObj)
}


# Calculate total model variance: parameter + process variance ------------
calcModelVar <- function(resObj){
  # This function calculates the total (sum) process and parameter variance of a fitted model
  
  model <- resObj$model
  procVarUBE <- resObj$procVarUBE
  dat <- resObj$dat
  fitDat <- dat[!is.na(dat$logvalue),]
  
  X <- model.matrix(model)
  
  paramCovar <- X %*% vcov(model) %*% t(X)
  
  paramVarTot <- sum(paramCovar)
  procVar <- procVarUBE/fitDat$w
  procVarTot <- sum(procVar)
  
  varVec <- cbind(paramVarTot, procVarTot, varTot = paramVarTot + procVarTot)
  return(varVec)
}

# Calculate log likelihood of actual (unfiltered) data in smoothed --------

logLikCompare <- function(resObj){
  # This function calculates the log likelihood of the unsmoothed data in the smoothed model
  # and compares to the smoothed model loglihood.
#   browser()
  model <- resObj$model
  dat <- resObj$dat
  if(ROCKY.settings$selected.model=="loglinear"){
    fitDat <- dat[!is.na(dat$logvalue),]
  }
  else if(ROCKY.settings$selected.model=="GLM"){
    fitDat <- dat[!is.na(dat$value),]
  }
  w <- fitDat$w
  if(ROCKY.settings$selected.model=="loglinear"){
    res <- fitDat$logvalueAct - fitted(model)
  }
  else if(ROCKY.settings$selected.model=="GLM"){
    res <- fitDat$value - fitted(model)
  }
  N  <- length(res)
  dgf <- summary(model)$df[2]  
  
  #this is the residual standard error squared * residual degrees of freedom
  #this is available in the lm model summary, but not the glm model summary
  if(ROCKY.settings$selected.model=="loglinear"){
    modelVarMLE <- summary(model)$s^2 * dgf/N
  }
  else if(ROCKY.settings$selected.model=="GLM"){
    modelVarMLE <- sqrt(deviance(model)/df.residual(model))
  }
  
  # calculate the log likelhood of the actual residuals on the smoothed model
  # note in the formula below modelVarMLE is the model variance and res = logvalueAct (unfiltered) - fitted
  logLikAct <- -(N/2)*log(2*pi*modelVarMLE) + 0.5 * sum(log(w)) - sum(w * res^2)/(2*modelVarMLE)
  
#   browser()
  
  ####I need to figure out why this doesn't work (6/16/2016)
  
  logLiks <- data.frame(logLikModel = logLik(model), logLikAct, combinedLogLik = logLik(model) + logLikAct)
  return(logLiks)
}


# # Trend variable plots ----------------------------------------------------
# 
# trendVarPlot <- function(resObj, newWindow = TRUE, option = 'all'){
#   # plot trend variable levels vs AY, CY, DY
#   # newWindow = TRUE; allows user to have plot created in new window
#   #   option = 'all'; allows user to select single plot
# #       browser()
#   # check if a valid option was selected. If not, stop with message
#   if ( !(option %in% c('AY','DY','CY','weights','all')) ) { 
#     stop("Error: Select option in ('AY','DY','CY','weight','all')")
#   }
#   
#   if (newWindow == TRUE) {
#     x11()  # Create plot in new window
#   } 
#   
#   # Variable Plots
#   
#   model <- resObj$model
#   AYGp <- resObj$AYGp
#   CYGp <- resObj$CYGp
#   DYGp <- resObj$DYGp
#   DYw <- resObj$DYw
#   DYwGp <- resObj$DYwGp
#   
#   # get coefficients in a form that we can join to the variable groups
#   coeffs <- data.frame(summary(model)$coefficients[,1:2])
#   coeffs$gp <- rownames(coeffs)
#   
#   # DY trendplot
#   
#   if (option %in% c('DY','all')){
#     # join DY coeffients to DYGp
#     suppressWarnings(coeffsYr <- left_join(DYGp,coeffs, by = 'gp'))
#     # if there are NA
#     if (any(is.na(coeffsYr$Estimate)) == TRUE) {
#       coeffsYr[is.na(coeffsYr$Estimate), ][,c('Estimate','Std..Error')] <- 0
#     }
#     
#     coeffsYr$sum <- cumsum(coeffsYr$Estimate)
#     coeffsYr <- rbind(rep(0,4),coeffsYr)
#     coeffsYr$ubar <- coeffsYr$sum + 1.96 * coeffsYr$Std..Error
#     coeffsYr$lbar <- coeffsYr$sum - 1.96 * coeffsYr$Std..Error
#     
#     # get zero selections
#     zeroDF <- coeffsYr[coeffsYr$gp == 0,]
#     # remove starting row; we don't want to show the starting row with a zero selection
#     zeroDF <- zeroDF[-c(1),]
#     
#     # make labels for each unique group
#     labelDF <- coeffsYr[,c('DY','gp','Estimate','sum')]
#     labelDF <- labelDF[labelDF$gp != 0, ]
#     # keep first 
#     labelDF <- labelDF[!duplicated(labelDF$gp),]
#     labelDF$label <- paste(labelDF$gp,': ',round(labelDF$Estimate,2),sep='')
#     
#     
#     DYplot <- ggplot(coeffsYr, aes(x = DY, y = sum)) +
#       geom_ribbon(aes(x = DY,y = sum, ymin = lbar,ymax = ubar), fill = 'orange', alpha = 0.2) +
#       geom_point(aes(color = gp),size = 3,shape = 18,ylim = c(-2,1)) + labs(color = 'DYgp') + geom_line(size = 0.5) + 
#       geom_vline(data = zeroDF,aes(xintercept = DY), color = c('black'),alpha = 0.1, size = 2) + 
#       geom_vline(data = labelDF,aes(xintercept = DY), color = c('blue'),alpha = 0.1, size = 2) +
#       labs(title ="DY Trend", y = "Cumulative Trend") + scale_x_continuous(breaks = coeffsYr$DY) +
#       theme(panel.grid.major = element_line(linetype = "dotted")) +
#       coord_cartesian(ylim = c(min(coeffsYr$lbar),max(coeffsYr$ubar))) +
#       geom_text(data = labelDF,aes(x = DY, y = sum),label = labelDF$label,hjust=1.25, vjust=1, size = 3)
#   }
#   
#   # AY trendplot
#   
#   if (option %in% c('AY','all')){
#     # join AY coeffients to DYGp
#     suppressWarnings(coeffsYr <- left_join(AYGp,coeffs, by = 'gp'))
#     if (any(is.na(coeffsYr$Estimate)) == TRUE) {
#       coeffsYr[is.na(coeffsYr$Estimate), ][,c('Estimate','Std..Error')] <- 0
#     }
#     
#     coeffsYr$ubar <- coeffsYr$Estimate + 1.96 * coeffsYr$Std..Error
#     coeffsYr$lbar <- coeffsYr$Estimate - 1.96 * coeffsYr$Std..Error
#     
#     # make labels for each unique group
#     labelDF <- coeffsYr[,c('AY','gp','Estimate')]
#     # keep first 
#     labelDF <- labelDF[!duplicated(labelDF$gp),]
#     labelDF$label <- paste(labelDF$gp,': ',round(labelDF$Estimate,2),sep='')
#     
#     AYplot <- ggplot(coeffsYr, aes(x = AY, y = Estimate)) +
#       geom_ribbon(aes(x = AY,y = Estimate, ymin = lbar,ymax = ubar), fill = 'orange', alpha = 0.2) +
#       geom_point(aes(color = gp),size = 3,shape = 18,ylim = c(-2,1)) + labs(color = 'AYgp') + geom_line(size = 0.5) + 
#       geom_vline(data = labelDF,aes(xintercept = AY), color = c('blue'),alpha = 0.1, size = 2) +
#       labs(title ="AY Level", y = "Cumulative Trend") + scale_x_continuous(breaks = seq(min(coeffsYr$AY),max(coeffsYr$AY),by = 2)) +
#       theme(panel.grid.major = element_line(linetype = "dotted")) +
#       coord_cartesian(ylim = c(min(coeffsYr$lbar),max(coeffsYr$ubar) + 0.4)) +
#       geom_text(data = labelDF,aes(x = AY, y = Estimate),label = labelDF$label,hjust=-0.25, vjust=-1, size = 3)
#   }
#   
#   
#   # CY trendplot
#   if (option %in% c('CY','all')){
#     # join CY coeffients to CYGp
#     # check if there is only a zero trend. If so we plot 0s, otherwise we plot the coefficient values
#     if (length(unique(CYGp$gp)) == 1 && any(unique(CYGp$gp) == 0)){
#       coeffsYr <- CYGp
#       CYplot <- ggplot(coeffsYr, aes(x = CY, y = gp)) +
#         geom_point(aes(color = gp),size = 3,shape = 18,ylim = c(-2,1)) + geom_line(size = 0.5) + 
#         labs(title ="CY Trend", y = "Cumulative Trend") + scale_x_continuous(breaks = seq(min(coeffsYr$CY),max(coeffsYr$CY),by = 2)) +
#         theme(panel.grid.major = element_line(linetype = "dotted"))
#     } else{
#       suppressWarnings(coeffsYr <- left_join(CYGp,coeffs, by = 'gp'))
#       # if there are NA
#       if (any(is.na(coeffsYr$Estimate)) == TRUE) {
#         coeffsYr[is.na(coeffsYr$Estimate), ][,c('Estimate','Std..Error')] <- 0
#       }
#       
#       # Only show plot through max AY
#       maxAY <- max(AYGp$AY)
#       coeffsYr <- coeffsYr[coeffsYr$CY <= maxAY,]
#       
#       
#       # get starting CY row
#       minCY <- min(coeffsYr$CY) - 1
#       startRow <- c(minCY,rep(0,3))
#       coeffsYr <- rbind(startRow,coeffsYr)
#       coeffsYr$sum <- cumsum(coeffsYr$Estimate)
#       
#       coeffsYr$ubar <- coeffsYr$sum + 1.96 * coeffsYr$Std..Error
#       coeffsYr$lbar <- coeffsYr$sum - 1.96 * coeffsYr$Std..Error
#       
#       # get zero selections
#       zeroDF <- coeffsYr[coeffsYr$gp == 0,]
#       # remove starting row; we don't want to show the starting row with a zero selection
#       zeroDF <- zeroDF[-c(1),]
#       
#       # make labels for each unique group
#       labelDF <- coeffsYr[,c('CY','gp','Estimate','sum')]
#       labelDF <- labelDF[labelDF$gp != 0, ]
#       # keep first 
#       labelDF <- labelDF[!duplicated(labelDF$gp),]
#       labelDF$label <- paste(labelDF$gp,': ',round(labelDF$Estimate,2),sep='')
#       
#       
#       CYplot <- ggplot(coeffsYr, aes(x = CY, y = sum)) +
#         geom_ribbon(aes(x = CY,y = sum, ymin = lbar,ymax = ubar), fill = 'orange', alpha = 0.2) +
#         geom_point(aes(color = gp),size = 3,shape = 18,ylim = c(-2,1)) + labs(color = 'CYgp') + geom_line(size = 0.5) + 
#         geom_vline(data = zeroDF,aes(xintercept = CY), color = c('black'),alpha = 0.1, size = 2) + 
#         geom_vline(data = labelDF,aes(xintercept = CY), color = c('blue'),alpha = 0.1, size = 2) +
#         labs(title ="CY Trend", y = "Cumulative Trend") + scale_x_continuous(breaks = seq(min(coeffsYr$CY),max(coeffsYr$CY),by = 2)) +
#         theme(panel.grid.major = element_line(linetype = "dotted")) +
#         coord_cartesian(ylim = c(min(coeffsYr$lbar),max(coeffsYr$ubar))) +
#         geom_text(data = labelDF,aes(x = CY, y = sum),label = labelDF$label,hjust=1.25, vjust=-1, size = 3)
#     }
#     
#   }
#   
#   # DYw trendplot
#   if (option %in% c('weights','all')){
#     DYw
#     DYwGp
#     coeffsYr <- inner_join(DYw,DYwGp, by = 'DY')
#     
#     # make labels for each unique group
#     labelDF <- coeffsYr
#     # keep first 
#     labelDF <- labelDF[!duplicated(labelDF$wGp),]
#     labelDF$label <- paste(labelDF$wGp,': ',round(labelDF$w,2),sep='')
#     
#     
#     wplot <- ggplot(coeffsYr, aes(x = DY, y = w)) +
#       geom_point(aes(color = wGp),size = 3,shape = 18,ylim = c(-2,1)) + labs(color = 'wGp') + geom_line(size = 0.5) + 
#       labs(title ="DY Weights", y = "Weight") + scale_x_continuous(breaks = coeffsYr$DY) +
#       theme(panel.grid.major = element_line(linetype = "dotted")) +
#       geom_text(data = labelDF,aes(x = DY, y = w),label = labelDF$label,hjust=1.25, vjust=1, size = 3)
#   }
#   
#   # check if we are plotting all, if not check which plot exists and plot
#   if (option == 'all'){
#     grid.arrange(DYplot,AYplot,CYplot,wplot,main = 'Trend Variable Plots')  
#   } else{
#     if( exists('DYplot',inherits = FALSE)) {print(DYplot)}
#     if( exists('AYplot',inherits = FALSE)) {print(AYplot)}
#     if( exists('CYplot',inherits = FALSE)) {print(CYplot)}
#     if( exists('wplot',inherits = FALSE)) {print(wplot)}
#   }
# }

# Trend variable plots ----------------------------------------------------

trendVarPlot <- function(resObj, newWindow = TRUE, option = 'all', firstModel = FALSE){
  # plot trend variable levels vs AY, CY, DY
  # newWindow = TRUE; allows user to have plot created in new window
  #   option = 'all'; allows user to select single plot
#         browser()
  # check if a valid option was selected. If not, stop with message
  if ( !(option %in% c('AY','DY','CY','weight','all')) ) { 
    stop("Error: Select option in ('AY','DY','CY','weight','all')")
  }
  
  if (newWindow == TRUE) {
    x11()  # Create plot in new window
  } 
  
  # Variable Plots
  
  model <- resObj$model
  AYGp <- resObj$AYGp
  CYGp <- resObj$CYGp
  DYGp <- resObj$DYGp
  DYw <- resObj$DYw
  DYwGp <- resObj$DYwGp
  
  # get coefficients in a form that we can join to the variable groups
  coeffs <- data.frame(summary(model)$coefficients[,1:2])
  coeffs$gp <- rownames(coeffs)
  
  # DY trendplot
  
  if (option %in% c('DY','all')){
    # join DY coeffients to DYGp
    suppressWarnings(coeffsYr <- left_join(DYGp,coeffs, by = 'gp'))
    # if there are NA
    if (any(is.na(coeffsYr$Estimate)) == TRUE) {
      coeffsYr[is.na(coeffsYr$Estimate), ][,c('Estimate','Std..Error')] <- 0
    }
    
    coeffsYr$sum <- cumsum(coeffsYr$Estimate)
    coeffsYr <- rbind(rep(0,4),coeffsYr)
    coeffsYr$ubar <- coeffsYr$sum + 1.96 * coeffsYr$Std..Error
    coeffsYr$lbar <- coeffsYr$sum - 1.96 * coeffsYr$Std..Error
    
    # get zero selections
    zeroDF <- coeffsYr[coeffsYr$gp == 0,]
    # remove starting row; we don't want to show the starting row with a zero selection
    zeroDF <- zeroDF[-c(1),]
    
    # make labels for each unique group
    labelDF <- coeffsYr[,c('DY','gp','Estimate','sum')]
    labelDF <- labelDF[labelDF$gp != 0, ]
    # keep first 
    labelDF <- labelDF[!duplicated(labelDF$gp),]
    labelDF$label <- paste(labelDF$gp,': ',round(labelDF$Estimate,2),sep='')
    
    
    DYplot <- ggplot(coeffsYr, aes(x = DY, y = sum)) +
      geom_ribbon(aes(x = DY,y = sum, ymin = lbar,ymax = ubar), fill = ROCKY.settings$plots$trendVarPlot$SE$fillColor,
                  alpha = ROCKY.settings$plots$trendVarPlot$SE$alpha) +
      geom_point(aes(color = gp),size = ROCKY.settings$plots$trendVarPlot$points$size,
                 shape = ROCKY.settings$plots$trendVarPlot$points$shape) + labs(color = 'DYgp') + 
      geom_line(size = ROCKY.settings$plots$trendVarPlot$connectingLine$size) + 
      #       geom_point(aes(color = gp),size = 3,shape = 18,ylim = c(-2,1)) + labs(color = 'DYgp') + geom_line(size = 0.5) + 
      geom_vline(data = zeroDF,aes(xintercept = DY), color = c(ROCKY.settings$plots$trendVarPlot$zeroTrendVertLine$color),
                 alpha = ROCKY.settings$plots$trendVarPlot$zeroTrendVertLine$alpha, 
                 size = ROCKY.settings$plots$trendVarPlot$zeroTrendVertLine$size) + 
      geom_vline(data = labelDF,aes(xintercept = DY), color = c(ROCKY.settings$plots$trendVarPlot$nonZeroTrendVertLine$color),
                 alpha = ROCKY.settings$plots$trendVarPlot$nonZeroTrendVertLine$alpha, 
                 size = ROCKY.settings$plots$trendVarPlot$nonZeroTrendVertLine$size) +
      labs(title =ROCKY.settings$plots$trendVarPlot$DY$title,
           y = ROCKY.settings$plots$trendVarPlot$DY$yAxisTitle) + scale_x_continuous(breaks = coeffsYr$DY) +
      theme(panel.grid.major = element_line(linetype = ROCKY.settings$plots$trendVarPlot$theme$lineType)) +
      coord_cartesian(ylim = c(min(coeffsYr$lbar),max(coeffsYr$ubar))) +
      geom_text(data = labelDF,aes(x = DY, y = sum),label = labelDF$label,hjust=1.25, vjust=1, size = 3)
  }
  
  # AY trendplot
  
  if (option %in% c('AY','all')){
    # join AY coeffients to DYGp
    suppressWarnings(coeffsYr <- left_join(AYGp,coeffs, by = 'gp'))
    if (any(is.na(coeffsYr$Estimate)) == TRUE) {
      coeffsYr[is.na(coeffsYr$Estimate), ][,c('Estimate','Std..Error')] <- 0
    }
    
    coeffsYr$ubar <- coeffsYr$Estimate + 1.96 * coeffsYr$Std..Error
    coeffsYr$lbar <- coeffsYr$Estimate - 1.96 * coeffsYr$Std..Error
    
    # make labels for each unique group
    labelDF <- coeffsYr[,c('AY','gp','Estimate')]
    # keep first 
    labelDF <- labelDF[!duplicated(labelDF$gp),]
    labelDF$label <- paste(labelDF$gp,': ',round(labelDF$Estimate,2),sep='')
    
    AYplot <- ggplot(coeffsYr, aes(x = AY, y = Estimate)) +
      geom_ribbon(aes(x = AY,y = Estimate, ymin = lbar,ymax = ubar), fill = ROCKY.settings$plots$trendVarPlot$SE$fillColor, 
                  alpha = ROCKY.settings$plots$trendVarPlot$SE$alpha) +
      geom_point(aes(color = gp),size = ROCKY.settings$plots$trendVarPlot$points$size,
                 shape = ROCKY.settings$plots$trendVarPlot$points$shape) + labs(color = 'AYgp') + 
      geom_line(size = ROCKY.settings$plots$trendVarPlot$connectingLine$size) + 
      geom_vline(data = labelDF,aes(xintercept = AY), color = c(ROCKY.settings$plots$trendVarPlot$nonZeroTrendVertLine$color),
                 alpha = ROCKY.settings$plots$trendVarPlot$nonZeroTrendVertLine$alpha, 
                 size = ROCKY.settings$plots$trendVarPlot$nonZeroTrendVertLine$size) +
      labs(title =ROCKY.settings$plots$trendVarPlot$AY$title, 
           y = ROCKY.settings$plots$trendVarPlot$AY$yAxisTitle) + scale_x_continuous(breaks = seq(min(coeffsYr$AY),max(coeffsYr$AY),by = 2)) +
      theme(panel.grid.major = element_line(linetype = ROCKY.settings$plots$trendVarPlot$theme$lineType)) +
      coord_cartesian(ylim = c(min(coeffsYr$lbar),max(coeffsYr$ubar) + 0.4)) +
      geom_text(data = labelDF,aes(x = AY, y = Estimate),label = labelDF$label,hjust=-0.25, vjust=-1, size = 3)
  }
  
  
  # CY trendplot
  if (option %in% c('CY','all')){
    # join CY coeffients to CYGp
    # check if there is only a zero trend. If so we plot 0s, otherwise we plot the coefficient values
    if (length(unique(CYGp$gp)) == 1 && any(unique(CYGp$gp) == 0)){
      coeffsYr <- CYGp
      CYplot <- ggplot(coeffsYr, aes(x = CY, y = gp)) +
        geom_point(aes(color = gp),size = ROCKY.settings$plots$trendVarPlot$points$size,
                   shape = ROCKY.settings$plots$trendVarPlot$points$shape) + 
        geom_line(size = ROCKY.settings$plots$trendVarPlot$connectingLine$size) + 
        labs(title =ROCKY.settings$plots$trendVarPlot$CY$title, 
             y = ROCKY.settings$plots$trendVarPlot$CY$yAxisTitle) + scale_x_continuous(breaks = seq(min(coeffsYr$CY),max(coeffsYr$CY),by = 2)) +
        theme(panel.grid.major = element_line(linetype = "dotted"))
    } else{
      suppressWarnings(coeffsYr <- left_join(CYGp,coeffs, by = 'gp'))
      # if there are NA
      if (any(is.na(coeffsYr$Estimate)) == TRUE) {
        coeffsYr[is.na(coeffsYr$Estimate), ][,c('Estimate','Std..Error')] <- 0
      }
      
      # Only show plot through max AY
      maxAY <- max(AYGp$AY)
      coeffsYr <- coeffsYr[coeffsYr$CY <= maxAY,]
      
      
      # get starting CY row
      minCY <- min(coeffsYr$CY) - 1
      startRow <- c(minCY,rep(0,3))
      coeffsYr <- rbind(startRow,coeffsYr)
      coeffsYr$sum <- cumsum(coeffsYr$Estimate)
      
      coeffsYr$ubar <- coeffsYr$sum + 1.96 * coeffsYr$Std..Error
      coeffsYr$lbar <- coeffsYr$sum - 1.96 * coeffsYr$Std..Error
      
      # get zero selections
      zeroDF <- coeffsYr[coeffsYr$gp == 0,]
      # remove starting row; we don't want to show the starting row with a zero selection
      zeroDF <- zeroDF[-c(1),]
      
      # make labels for each unique group
      labelDF <- coeffsYr[,c('CY','gp','Estimate','sum')]
      labelDF <- labelDF[labelDF$gp != 0, ]
      # keep first 
      labelDF <- labelDF[!duplicated(labelDF$gp),]
      labelDF$label <- paste(labelDF$gp,': ',round(labelDF$Estimate,2),sep='')
      
      
      CYplot <- ggplot(coeffsYr, aes(x = CY, y = sum)) +
        geom_ribbon(aes(x = CY,y = sum, ymin = lbar,ymax = ubar), fill = ROCKY.settings$plots$trendVarPlot$SE$fillColor, 
                    alpha = ROCKY.settings$plots$trendVarPlot$SE$alpha) +
        geom_point(aes(color = gp),size = ROCKY.settings$plots$trendVarPlot$points$size,
                   shape = ROCKY.settings$plots$trendVarPlot$points$shape) + labs(color = 'CYgp') + 
        geom_line(size = ROCKY.settings$plots$trendVarPlot$connectingLine$size) + 
        geom_vline(data = zeroDF,aes(xintercept = CY), color = c(ROCKY.settings$plots$trendVarPlot$zeroTrendVertLine$color),
                   alpha = ROCKY.settings$plots$trendVarPlot$zeroTrendVertLine$alpha, 
                   size = ROCKY.settings$plots$trendVarPlot$zeroTrendVertLine$size) + 
        geom_vline(data = labelDF,aes(xintercept = CY), color = c(ROCKY.settings$plots$trendVarPlot$nonZeroTrendVertLine$color),
                   alpha = ROCKY.settings$plots$trendVarPlot$nonZeroTrendVertLine$alpha, 
                   size = ROCKY.settings$plots$trendVarPlot$nonZeroTrendVertLine$size) +
        labs(title =ROCKY.settings$plots$trendVarPlot$CY$title, 
             y = ROCKY.settings$plots$trendVarPlot$CY$yAxisTitle) + scale_x_continuous(breaks = seq(min(coeffsYr$CY),max(coeffsYr$CY),by = 2)) +
        theme(panel.grid.major = element_line(linetype = ROCKY.settings$plots$trendVarPlot$theme$lineType)) +
        coord_cartesian(ylim = c(min(coeffsYr$lbar),max(coeffsYr$ubar))) +
        geom_text(data = labelDF,aes(x = CY, y = sum),label = labelDF$label,hjust=1.25, vjust=-1, size = 3)
    }
    
  }
  
#   browser()
  
  # DYw trendplot
  if (option %in% c('weight','all')){
#     browser()
    DYw
    DYwGp
    coeffsYr <- inner_join(DYw,DYwGp, by = 'DY')
    
    # make labels for each unique group
    labelDF <- coeffsYr
    # keep first 
    labelDF <- labelDF[!duplicated(labelDF$gp),]
    
    if(firstModel){
      labelDF <- NULL
      labelDF$DY <- 0
      labelDF$w <- 1
      labelDF$gp <- "W1"
    }
    else{
    labelDF$label <- paste(labelDF$gp,': ',round(labelDF$w,2),sep='')
    }
    
    
    wplot <- ggplot(coeffsYr, aes(x = DY, y = w)) +
      geom_point(aes(color = gp),size = ROCKY.settings$plots$trendVarPlot$points$size,
                 shape = ROCKY.settings$plots$trendVarPlot$points$shape) + labs(color = 'gp') + 
      geom_line(size = ROCKY.settings$plots$trendVarPlot$connectingLine$size) + 
      labs(title =ROCKY.settings$plots$trendVarPlot$weights$title, 
           y = ROCKY.settings$plots$trendVarPlot$weights$yAxisTitle) + scale_x_continuous(breaks = coeffsYr$DY) +
      theme(panel.grid.major = element_line(linetype = "dotted")) +
      geom_text(data = labelDF,aes(x = DY, y = w),label = labelDF$label,hjust=1.25, vjust=1, size = 3)
  }
  
  # check if we are plotting all, if not check which plot exists and plot
  if (option == 'all'){
    grid.arrange(grobs = list(DYplot,AYplot,CYplot,wplot), top = 'Trend Variable Plots')  
  } else{
    if( exists('DYplot',inherits = FALSE)) {print(DYplot)}
    if( exists('AYplot',inherits = FALSE)) {print(AYplot)}
    if( exists('CYplot',inherits = FALSE)) {print(CYplot)}
    if( exists('wplot',inherits = FALSE)) {print(wplot)}
  }
}


# Plot CY stability -------------------------------------------------------
plotCYstability <- function(resObj, newWindow = TRUE){
  if (newWindow == TRUE) {
    # We don't want to close a previously opened window, so we have to do some window gynastics. 
    # Create new window, turn window off, then create window of custom size    
    x11(); dev.off(); windows(record=TRUE, width=10, height=5)
  }
#     browser()
  plotDat <- resObj$CYstability
  nRow <- nrow(plotDat)
  # reverse order
  plotDat <- plotDat[c(nRow:1),]
  plotDat$index <- 1:nRow
  fullModelRes <- plotDat[plotDat$holdout == 'full_model',]$Total_Reserve
  plotDat$fullModelRes <- fullModelRes
  
  plotDat$ubar1 <- plotDat$Total_Reserve + 1 * plotDat$Reserve_SE
  plotDat$lbar1 <- plotDat$Total_Reserve - 1 * plotDat$Reserve_SE
  plotDat$ubar2 <- plotDat$Total_Reserve + 2 * plotDat$Reserve_SE
  plotDat$lbar2 <- plotDat$Total_Reserve - 2 * plotDat$Reserve_SE
  
  
  print(ggplot(data = plotDat, aes(x=holdout,y=Total_Reserve)) + geom_point() + 
          geom_line(aes(x=index,y=Total_Reserve),
                    col = ROCKY.settings$plots$autoValidationPlot$line$color,
                    size = ROCKY.settings$plots$autoValidationPlot$line$size) +
    geom_ribbon(aes(x = index,y = Total_Reserve, ymin = lbar1,ymax = ubar1), 
                fill = ROCKY.settings$plots$autoValidationPlot$SE$color, 
                alpha = ROCKY.settings$plots$autoValidationPlot$SE$alpha2SE) + 
    geom_ribbon(aes(x = index,y = Total_Reserve, ymin = lbar2,ymax = ubar2), 
                fill = ROCKY.settings$plots$autoValidationPlot$SE$color, 
                alpha = ROCKY.settings$plots$autoValidationPlot$SE$alpha1SE) +
    geom_hline(data = plotDat,aes(yintercept = fullModelRes), 
               linetype = ROCKY.settings$plots$autoValidationPlot$centerLine$type, 
               color = ROCKY.settings$plots$autoValidationPlot$centerLine$color))
}

# Calculate best weights based on combined log likelihood -----------------

AYsmoothLogLik <- function(resObj,IMax = ROCKY.settings$ay.filter$MaxIterations){
#     browser()
  # cycle through all filters
  resObj$AYgpFilters <- data.frame(gp = unique(resObj$AYGp$gp), filter = 'none',stringsAsFactors = FALSE)
  resObj <- applySelectedAYfilters(resObj,plots = FALSE)
  AYgp <- resObj$AYgpFilters$gp
  filters <- c('none',resObj$filters$filter)
  
  # for the filtering process first we filter all groups the same amount,
  # then we adjust the filtering for each group individually
  
  M <- length(AYgp)
  N <- length(filters)
  priorAYgpFilters <- resObj$AYgpFilters
  bestFilters <- resObj$AYgpFilters
  logLikVar <- NULL
  maxVar <- NULL
  logTmp <- NULL
  
#   IMax = 5
  iterlogLikCompare <- data.frame(iter = -1, logLikCompare(resObj))
  priorLogLik <- logLikCompare(resObj)[1,3]
  for (n in 1:N){
    resObj$AYgpFilters[,c('filter')] <- filters[n]
    resObj <- applySelectedAYfilters(resObj,plots = FALSE)
    logTmp <- data.frame(filter = filters[n],logLikCompare(resObj),stringsAsFactors = FALSE)
    logLikVar <- rbind(logLikVar,logTmp)
  }
  maxLogLik <- max(logLikVar$combinedLogLik)
  maxVar <- logLikVar[logLikVar$combinedLogLik ==maxLogLik,]
  # keep first in case two log likelihoods are equal
  maxVar <- maxVar[1,]
  
  # set best filter
  bestFilters[,c('filter')] <- maxVar$filter
  
  # set tmp filters for next variable iteration
  resObj$AYgpFilters[,c('filter')] <-  maxVar$filter
  resObj <- applySelectedAYfilters(resObj,plots = FALSE)
  iterlogLikCompare <- rbind(iterlogLikCompare,data.frame(iter = 0, logLikCompare(resObj)))
  
  # Now we adjust each parameter
  priorAYgpFilters <- resObj$AYgpFilters
  bestFilters <- resObj$AYgpFilters
  
  priorLogLik <- logLikCompare(resObj)[1,3]
  
  for (iter in 1:IMax){
    for (m in 1:M){
      logLikVar <- NULL
      maxVar <- NULL
      logTmp <- NULL
      
      # cycle through selected variable
      for (n in 1:N){
        resObj$AYgpFilters[m,c('filter')] <- filters[n]
        resObj <- applySelectedAYfilters(resObj,plots = FALSE)
        logTmp <- data.frame(filter = filters[n],logLikCompare(resObj),stringsAsFactors = FALSE)
        logLikVar <- rbind(logLikVar,logTmp)
      }
      maxLogLik <- max(logLikVar$combinedLogLik)
      maxVar <- logLikVar[logLikVar$combinedLogLik ==maxLogLik,]
      # keep first in case two log likelihoods are equal
      maxVar <- maxVar[1,]
      
      # set best filter
      bestFilters[m,c('filter')] <- maxVar$filter
      
      # set tmp filters for next variable iteration
      resObj$AYgpFilters[m,c('filter')] <-  maxVar$filter
      
    }
    resObj <- applySelectedAYfilters(resObj,plots = FALSE)
    iterlogLikCompare <- rbind(iterlogLikCompare,data.frame(iter = iter, logLikCompare(resObj)))
    print(iterlogLikCompare)
    currLogLik <- logLikCompare(resObj)[1,3]
    diffLogLik <- abs(currLogLik- priorLogLik)
    if (diffLogLik < ROCKY.settings$ay.filter$LogLikCutoff) {break}
    priorLogLik <- currLogLik
  }
  
  
  resObj$AYgpFilters <- bestFilters
#   resObj <- applySelectedAYfilters(resObj,plots = FALSE)
  resObj$iterlogLikCompare <- iterlogLikCompare
  return(resObj)
}


# Normal diagnostics plots of std residuals -------------------------------

normPlots <- function(resObj, newWindow = TRUE){
  # Plot various versions of normal diagnostics.
  
  dat <- resObj$dat
  dat <- dat[!is.na(dat$logvalue),]
  if(newWindow) {x11()}
  op <- par(mfrow=c(2,2),oma = c(0, 1, 3, 0))
  
  plot(resObj$model,2,col=ROCKY.settings$plots$normalDiagnostics$QQPlot$color)
  hist(dat$residStd,xlab = ROCKY.settings$plots$normalDiagnostics$histogram$xlabel, 
       ylab = ROCKY.settings$plots$normalDiagnostics$histogram$ylabel,
       main = ROCKY.settings$plots$normalDiagnostics$histogram$title,
       freq = ROCKY.settings$plots$normalDiagnostics$histogram$freq,
       nclass = ROCKY.settings$plots$normalDiagnostics$histogram$nclass,
       col = ROCKY.settings$plots$normalDiagnostics$histogram$color)
  boxplot(dat$residStd,
          horizontal = ROCKY.settings$plots$normalDiagnostics$boxplot$isHorizontal,
          main = ROCKY.settings$plots$normalDiagnostics$boxplot$title,
          col=ROCKY.settings$plots$normalDiagnostics$boxplot$color)
  plot(density(dat$residStd),col = ROCKY.settings$plots$normalDiagnostics$density$color,
       main = ROCKY.settings$plots$normalDiagnostics$density$title,
       xlab = ROCKY.settings$plots$normalDiagnostics$density$xlab)
  
  par(op) 
  
}


# Automatic DY weight group selections ------------------------------------

DYweightGroups <- function(resObj, selWeights = 'logval', nbrGps = NULL, customFutureCYTrend = FALSE, customFutureCYStdError = FALSE){
  # This function using a heirarchical k-means clustering to select DY weight groupings
  # The user has the option of selecting the following weights by DY 'logval', 'count', and 'uniform'
  # representing the sum of logvalue, the count of entries by DY, and uniform weights respectively.
  # The function defaults to selecting the number of groups via amount of variance explained. Specifically,
  #   when the increase in variance explained by adding a group drops below of threshold then the prior number of groups is selected.
  # The user can instead select the number of groups desired by setting nbrGps.
#   browser()
  # Reset weights to 1
  resObj$DYw[,c('w')] <- 1
  resObj$dat$w <- 1
  # refit model with weights = 1
  resObj <- fitModel(resObj,updateWeights = TRUE,plots = FALSE, customFutureCYTrend = customFutureCYTrend, customFutureCYStdError = customFutureCYStdError,
                     UserSelectedModel = ROCKY.settings$selected.model, UserSelectedGLMModel = ROCKY.settings$GLM$selected.model)
  
  dat = resObj$dat
  # Restrict to non-NA values
  fitDat <- dat[!is.na(dat$logvalue),]
  
  # calculate variance of residuals over each group then format to data frame
  resDY <- fitDat[,c('id','DY','logvalue','residStd')]
  
  # get weights and values for clustering
  # weights = logvalues
  wt1 <- resDY %>% group_by(DY) %>% summarize( logval = sum(logvalue))
  # weights = count
  wt2 <- resDY %>% group_by(DY) %>% summarize( count = length(logvalue))
  # population variance by DY - values to cluster
  valDF <- resDY %>% group_by(DY) %>% summarize( var = popVar(residStd))
  toClusterDat <- inner_join(inner_join(valDF,wt1, by = 'DY'),wt2,by='DY')
  
  # select weights base on selWeights function argument
  if (selWeights == 'count'){
    weight <- toClusterDat$count
  } else if (selWeights == 'logval'){
    weight <- toClusterDat$logval
  } else {
    weight <- rep(1,nrow(toClusterDat))
  }
  
  # Form data frame to hold DY, groups, values, and weights for clustering
  val <- toClusterDat$var  
  clust <- data.frame(DY = toClusterDat$DY,gp = toClusterDat$DY, val = val, w = weight)
  
  # Records total mean, sum of squares (wtd), and number of DYs
  totMean <- sum(clust$val*clust$w)/sum(clust$w)
  totSS <- sum((clust$val - totMean)^2 * clust$w )  
  totN <- length(unique(clust$gp))
  
  # Initialize clustering sets
  # clustIter will hold the current best cluster
  clustIter <- clust
  # gpsSS records all best SS
  gpsSS <- NULL
  # holds all the best clusters (for each k)
  kmeansGps <- NULL
  
  # We cycle over all possible number of groups gpCt (the k in k-means cluster)
  # We start with each DY being a group and find which adjacent combination increases the within SS
  # the least amount. The minimum SS grouping the then carried into the next step where
  # we once again try to find the adjacent groups to combine that increase the within SS the least.
  for (gpCt in totN:3){
    
    # record the unique groups and initialize gpWtSS
    # In the following for loop we test combining each adjacent group and record the resuslts in gpWtSS
    gps <- unique(clustIter$gp)
    gpWtdSS <- NULL
    
    
    for (i in 1:(gpCt-1)){
      # This for loop will combine the (i-1)th group with the ith group and records the within SS (wtd)
      # Note the groups may not be labeled in ascending order which is why we index gps by i.
      clustTmp <- clustIter
      gpToCombine <- gps[i + 1]
      gpBase <- gps[i]
      clustTmp[clustTmp$gp == gpToCombine,]$gp  <-  gpBase
      wtdMean <- clustTmp %>% group_by(gp) %>% summarize(mean = sum(val*w)/sum(w))
      clustwMean <- inner_join(clustTmp,wtdMean,by = 'gp')
      wtdSS <- sum((clustwMean$val-clustwMean$mean)^2 * clustwMean$w)
      gpWtdSS <- rbind(gpWtdSS,data.frame(iter = i,wtdSS= wtdSS))
    }
    
    # Record the minimum within SS grouping
    minWtdSS = min(gpWtdSS$wtdSS)
    minWtdSSdf <- filter(gpWtdSS,wtdSS == minWtdSS)
    # Select the first in case of ties
    minWtdSSdf <- minWtdSSdf[1,]
    gpToCombine <- gps[minWtdSSdf$iter + 1]
    gpBase <- gps[minWtdSSdf$iter]
    gpsSS <- rbind(gpsSS,data.frame(gps = gpCt - 1, withinSS = minWtdSSdf$wtdSS))
    clustIter[clustIter$gp == gpToCombine,]$gp  <-  gpBase
    kmeansGps[[gpCt-1]] <- clustIter
  }
  
  gpsSS <- rbind(gpsSS,data.frame(gps = 1, withinSS = totSS))
  # calculate fraction of variance explained
  gpsSS$fracExpl <- 1 - gpsSS$withinSS/totSS
  
  # select optimal number of groups by stopping when the amount of variance explained by adding another group drops below 2.5% 
  
  # incremental explained variance
  incExpl <- data.frame(gps = gpsSS[-(totN - 1),c('gps')],inc =gpsSS[-(totN - 1),c('fracExpl')] - gpsSS[-1,c('fracExpl')])
  
  # values below thresh
  incAboveThresh <- incExpl[incExpl$inc >= ROCKY.settings$hetero$ClusterThresh,]
  
  # there could be gaps from above so we cycle starting with three groups 
  
  for (j in 2:totN){
    if( !( (j+1) %in% incAboveThresh$gps)){
      selGp <- j
      break
    }  
  }
  
  # If the user entered a number for nbrGps select the corresponding groupings from
  # kmeansGps, otherwise select the minumum CH stat grouping.
  if (length(nbrGps) > 0){
    choosenGps <- kmeansGps[[nbrGps]]
  } else {
    choosenGps <- kmeansGps[[selGp]]
  }  
  
  # format group labels
  gps <- unique(choosenGps$gp)
  newGps <- 1:length(gps)
  newDYgps <- newGps[match(choosenGps$gp,gps)]
  newDYgps <- paste('W',newDYgps,sep='')
  DYwGp <- data.frame(DY = choosenGps$DY,gp = newDYgps) 
  
  # Update DYwGp and return
  resObj$DYwGp <- DYwGp
  #   return(resObj$DYwGp)
#   print(gpsSS)
  return(resObj)  
}




##End load function### # # # # # # # # # # # # # # # # # # # #  # 
