################################################################
##### Additional Functions to support ROCKY using VBScripts ####
#####                   ADRIAN v.0.0.2                      ####
################################################################

############################
##### TABLE OF CONTENTS ####
############################

#   1. Functions to load inputs from Excel workbook
#   2. Functions to push R model parameters to Excel
#   3. Functions to Update Trend Parameters
#   4. Functions to add an entry to the log or to load a model from the log
#   5. Functions to adjust for heteroscedasticity, and to clear or reset the adjustment
#   6. Functions to optimize the number of parameters
#   7. Functions to support the validation feature
#   8. Functions to produce diagnostic/residual plots
#   9. Functions to remove outliers, then add them back in
#   10. Functions to run ROCKY's AY Filtering
#   11. Functions to support ROCKY's Custom Future Trend Hetero Fixed Feature
#   12. Output for summary files
#   13. Save Function
#   14. Load Function


##### 1. Funcitons to load inputs from Excel workbook ####


# function that calls VBScript to return "StartAY" from Excel workbook
load.StartAY <- function(){
  #first, run the shell command that runs the VBScript and copies the cell with the StartAY value
  shell(cmd = "./VBScripts/Load_StartAY.vbs", translate = TRUE)
  
  #the proper cell (and value) is now on the clipboard on a 1x1 data frame
  #read it in
  temp <- read.table(file = "clipboard", header = FALSE, sep = "\t")
  
  #since just read in a data frame, convert variable to a simple integer
  temp <- as.numeric(temp)
  
  #now return the value
  return(temp)
}

# function that calls VBScript to return "StartCY" from Excel workbook
load.StartCY <- function(){
  #first, run the shell command that runs the VBScript and copies the cell with the StartCY value
  shell(cmd = "./VBScripts/Load_StartCY.vbs", translate = TRUE)
  
  #the proper cell (and value) is now on the clipboard on a 1x1 data frame
  #read it in
  temp <- read.table(file = "clipboard", header = FALSE, sep = "\t")
  
  #since just read in a data frame, convert variable to a simple integer
  temp <- as.numeric(temp)
  
  #now return the value
  return(temp)
}

#single call to load both StartAY and StartCY
load.FileInputs <- function(){
#   StartAY <<- load.StartAY()     # blackout data (truncate) before the StartAY (value comes from inputs tab on Excel worksheet)
#   StartCY <<- load.StartCY()     # blackout data before the StartCY (value comes from inputs tab on Excel worksheet) 
  ROCKY.settings$StartAY <<- load.StartAY()
  ROCKY.settings$StartCY <<- load.StartCY()
}

load.ModelBooleans <- function(){
  #creates a vector (NewModel) of boolean values that tell us whether or not the model is empty or not 
  NewModel <- data.frame(c(1))  
  
  #NewModel
  shell(cmd = "./VBScripts/Check_NewModel.vbs", translate = TRUE)
  temp <- read.table(file = "clipboard", header = FALSE, sep = "\t")
  temp <- as.logical(temp)
  NewModel$NewModel <- temp
  
  #HasAYGp
  shell(cmd = "./VBScripts/Check_HasAYGp.vbs", translate = TRUE)
  temp <- read.table(file = "clipboard", header = FALSE, sep = "\t")
  temp <- as.logical(temp)
  NewModel$HasAYGp <- temp
  
  #HasAYgpFilters
  shell(cmd = "./VBScripts/Check_HasAYgpFilters.vbs", translate = TRUE)
  temp <- read.table(file = "clipboard", header = FALSE, sep = "\t")
  temp <- as.logical(temp)
  NewModel$HasAYgpFilters <- temp
  
  #HasCYGp
  shell(cmd = "./VBScripts/Check_HasCYGp.vbs", translate = TRUE)
  temp <- read.table(file = "clipboard", header = FALSE, sep = "\t")
  temp <- as.logical(temp)
  NewModel$HasCYGp <- temp
  
  #HasCYpt
  shell(cmd = "./VBScripts/Check_HasCYpt.vbs", translate = TRUE)
  temp <- read.table(file = "clipboard", header = FALSE, sep = "\t")
  temp <- as.logical(temp)
  NewModel$HasCYpt <- temp
  
  #HasDYGp
  shell(cmd = "./VBScripts/Check_HasDYGp.vbs", translate = TRUE)
  temp <- read.table(file = "clipboard", header = FALSE, sep = "\t")
  temp <- as.logical(temp)
  NewModel$HasDYGp <- temp
  
  #HasDYw
  shell(cmd = "./VBScripts/Check_HasDYw.vbs", translate = TRUE)
  temp <- read.table(file = "clipboard", header = FALSE, sep = "\t")
  temp <- as.logical(temp)
  NewModel$HasDYw <- temp
  
  #HasDYwGp
  shell(cmd = "./VBScripts/Check_HasDYwGp.vbs", translate = TRUE)
  temp <- read.table(file = "clipboard", header = FALSE, sep = "\t")
  temp <- as.logical(temp)
  NewModel$HasDYwGp <- temp
  
  #get this data outside this function
  ModelBooleans <<- NewModel
}


##### 2. Functions to push R model parameters to Excel ####


#push AYGp, DYGp, CYGp, DYwGp, DYw in one function call
push.ModelParameters <- function(){
  push.AYGp()
  push.DYGp()
  push.CYGp()
  push.DYwGp()
  push.DYw()
  push.AYgpFilters()
}

#push AYGp parameters to Excel
push.AYGp <- function(){
  copy.table(resObj$AYGp)
  shell(cmd = "./VBScripts/Push/Push_AYGp.vbs", translate = TRUE)
}

#push DYGp parameters to Excel
push.DYGp <- function(){
  copy.table(resObj$DYGp)
  shell(cmd = "./VBScripts/Push/Push_DYGp.vbs", translate = TRUE)
}

#push CYGp parameters to Excel
push.CYGp <- function(){
  copy.table(resObj$CYGp)
  shell(cmd = "./VBScripts/Push/Push_CYGp.vbs", translate = TRUE)
}

#push CYpt parameters to Excel
push.CYpt <- function(){
  copy.table(resObj$CyProjTrendModel)
  shell(cmd = "./VBScripts/Push/Push_CYpt.vbs", translate = TRUE)
}

#push DYwGp parameters to Excel
push.DYwGp <- function(){
  copy.table(resObj$DYwGp)
  shell(cmd = "./VBScripts/Push/Push_DYwGp.vbs", translate = TRUE)
}

#push DYw parameters to Excel
push.DYw <- function(){
  copy.table(resObj$DYw)
  shell(cmd = "./VBScripts/Push/Push_DYw.vbs", translate = TRUE)
}

#push AYgpFilters parameters to Excel
push.AYgpFilters <- function(){
  shell(cmd = "./VBScripts/Clear_AYFilters.vbs", translate = TRUE) #clear AY filter table before paste in new one
  copy.table(resObj$AYgpFilters)
  shell(cmd = "./VBScripts/Push/Push_AYgpFilters.vbs", translate = TRUE)
}

#this one checks to see whether or not there are already parameters, if so, it doesn't replace them, if not, it puts them
#in the worksheet
load.ModelParameters <- function(){
  #first, check what is/isn't there
  load.ModelBooleans()
  
  if(ModelBooleans$NewModel) push.ModelParameters() #there are more of these!!!
  else {
    if(! ModelBooleans$HasAYGp)  push.AYGp()
    if(! ModelBooleans$HasCYGp)  push.CYGp()
    if(! ModelBooleans$HasCYpt)  push.CYpt()
    if(! ModelBooleans$HasDYGp)  push.DYGp()
    if(! ModelBooleans$HasDYw)   push.DYw()
    if(! ModelBooleans$HasDYwGp) push.DYwGp()
    if(! ModelBooleans$HasAYgpFilters) push.AYgpFilters()
  }
}

#does everything we need to do to ensure Excel & ROCKY are in sync
  #for a brand-new model:
new.model <- function(plots = TRUE, selectedModel = "loglinear", selectedGLMModel = "odp"){
#   browser()
  #selectedModel can be one of "loglinear" or "GLM" (6/16/2016)
  #selectedGLMModel can be one of "odp" or "gamma"
  
  graphics.off() #close graphics windows
  save.workbook() #run VBScript to save WB
  triDataRawAll <<- data.frame(read_excel(path = selectionsFile,sheet = 'tri_Data')) #read in data from the triangle
#   build.settings(selectedModel1 = selectedModel, selectedGLMModel1 = selectedGLMModel)
#   set.default.settings()
#   load.FileInputs() #load in StartAY and StartCY from Excel
  mydataraw <<- tritomatrix(data.frame(triDataRawAll),beginAY =  ROCKY.settings$StartAY,beginCY = ROCKY.settings$StartCY) #incremental payment triangle
  resObj  <<-  initializeModel(mydataraw, plots = plots)
  resObj$log <<- NULL #create the log
  resObj$excelLog <<- NULL #create the excel log
  resObj$ModelSettings$FutureCYTrend$Source <<- "New" #create the list of settings
  resObj$ModelSettings$FutureCYstdError$Source <<- "New" #create the list of settings
  resObj$forecast <<- resForecast(resObj, customCyTrend = FALSE, customCyStdErr = FALSE)
  graphics.off() #close graphics windows
  shell(cmd = "./VBScripts/Load_ClearLog.vbs", translate = TRUE) #clear anything already in the log
  load.ModelParameters() #push whatever parameter tables are needed in excel
  resObj <<- updateTrends(resObj, plots = plots) #fit the model (no selected parameters)
  add.to.log(IsNewModel = TRUE) #add first log entry
  shell(cmd = "./VBScripts/Messages/LoadModel_Done.vbs", translate = TRUE) #message box
}

#this is a list of the settings we are starting to put together--putting it as its own thing rather than bury it inside of resObj
build.settings <- function(selectedModel1, selectedGLMModel1){
  ROCKY.settings <<- NULL
  ROCKY.settings$selected.model <<- selectedModel1
  ROCKY.settings$GLM$selected.model <<- selectedGLMModel1
}

update.settings <- function(Data.StartingAY = NULL, Data.StartingCY = NULL, Data.ClipboardSize = NULL, 
                            Hetero.SelectWeights = NULL, Hetero.MaxIter = NULL, Hetero.MinWtChange = NULL,
                            Forecast.varEst = NULL, Forecast.varType = NULL,
                            Outliers.yminFactor = NULL, Outliers.ymaxFactor = NULL,
                            AYFilter.LogLikCutoff = NULL, AYFilter.NumberToLag = NULL, AYFilter.MaxIterations = NULL,
                            Optimizer.tCutoff = NULL, Optimizer.maxDist = NULL,
                            AutoValidation.NumOfYears = NULL, Output.PlotWidth = NULL, Output.PlotQuality = NULL,
                            Plots.Norm.QQ.Color = NULL, 
                            Plots.Norm.Hist.Title = NULL, Plots.Norm.Hist.Freq = NULL, Plots.Norm.Hist.nClass = NULL, Plots.Norm.Hist.Color = NULL,
                            Plots.Norm.Box.isHoriz = NULL, Plots.Norm.Box.Title = NULL, Plots.Norm.Box.Color = NULL,
                            Plots.Norm.Density.Color = NULL, Plots.Norm.Density.Title = NULL,
                            Plots.Structure.SE.Color = NULL, Plots.Structure.SE.Alpha = NULL,
                            Plots.Structure.Points.Size = NULL, Plots.Structure.Points.Shape = NULL,
                            Plots.Structure.ConnectingLine.Size = NULL, 
                            Plots.Structure.ZeroTrendLine.Color = NULL, Plots.Structure.ZeroTrendLine.Alpha = NULL, Plots.Structure.ZeroTrendLine.Size = NULL,
                            Plots.Structure.nonZeroTrendLine.Color = NULL, Plots.Structure.nonZeroTrendLine.Alpha = NULL, Plots.Structure.nonZeroTrendLine.Size = NULL,
                            Plots.Structure.DY.Title = NULL, Plots.Structure.DY.yAxisTitle = NULL,
                            Plots.Structure.CY.Title = NULL, Plots.Structure.CY.yAxisTitle = NULL,
                            Plots.Structure.AY.Title = NULL, Plots.Structure.AY.yAxisTitle = NULL,
                            Plots.Structure.weights.Title = NULL, Plots.Structure.weights.yAxisTitle = NULL,
                            Plots.Structure.Theme.LineType = NULL,
                            Plots.autoVal.SE.Color = NULL, Plots.autoVal.SE.Alpha2 = NULL, Plots.autoVal.SE.Alpha1 = NULL,
                            Plots.autoVal.CenterLine.Type = NULL, Plots.autoVal.CenterLine.Color = NULL,
                            Plots.autoVal.Line.Color = NULL, Plots.autoVal.Line.Size = NULL,
                            Plots.Residual.Res.Symbol = NULL, Plots.Residual.Res.Color = NULL, Plots.Residual.Res.BackgroundColor = NULL,
                            Plots.Residual.DY.Title = NULL, Plots.Residual.DY.xAxisTitle = NULL, Plots.Residual.DY.yAxisTitle = NULL,
                            Plots.Residual.AY.Title = NULL, Plots.Residual.AY.xAxisTitle = NULL, Plots.Residual.AY.yAxisTitle = NULL,
                            Plots.Residual.CY.Title = NULL, Plots.Residual.CY.xAxisTitle = NULL, Plots.Residual.CY.yAxisTitle = NULL,
                            Plots.Residual.ZeroLine.Color = NULL, Plots.Residual.ZeroLine.LineType = NULL,
                            Plots.Residual.MeanLine.Color = NULL,
                            Plots.Residual.SE.OneFourthColor = NULL, Plots.Residual.SE.OneColor = NULL, Plots.Residual.SE.TwoColor = NULL,
                            Plots.Residual.SE.ThreeColor = NULL, Plots.Residual.SE.FourColor = NULL,
                            Plots.Residual.ParameterLine.Type = NULL, Plots.Residual.ParameterLine.Width = NULL, Plots.Residual.ParameterLine.Color = NULL,
                            Plots.Residual.HeteroLine.Type = NULL, Plots.Residual.HeteroLine.Width = NULL, Plots.Residual.HeteroLine.Color = NULL,
                            Plots.Residual.Fitted.Title = NULL, Plots.Residual.Fitted.xAxisTitle = NULL, Plots.Residual.Fitted.yAxisTitle = NULL){
  
  if(!is.null(Data.StartingAY)){ROCKY.settings$data$StartingAY <<- Data.StartingAY}
  if(!is.null(Data.StartingCY)){ROCKY.settings$data$StartingCY <<- Data.StartingCY}
  if(!is.null(Data.ClipboardSize)){ROCKY.settings$data$ClipboardSize <<- Data.ClipboardSize}
  if(!is.null(Hetero.SelectWeights)){ROCKY.settings$hetero$SelectWeights <<- Hetero.SelectWeights}
  if(!is.null(Hetero.MaxIter)){ROCKY.settings$hetero$MaxIter <<- Hetero.MaxIter}
  if(!is.null(Hetero.MinWtChange)){ROCKY.settings$hetero$MinWtChange <<- Hetero.MinWtChange}
  if(!is.null(Forecast.varEst)){ROCKY.settings$forecast$varEst <<- Forecast.varEst}
  if(!is.null(Forecast.varType)){ROCKY.settings$forecast$varType <<- Forecast.varType}
  if(!is.null(Outliers.yminFactor)){ROCKY.settings$outliers$yminFactor <<- Outliers.yminFactor}
  if(!is.null(Outliers.ymaxFactor)){ROCKY.settings$outliers$ymaxFactor <<- Outliers.ymaxFactor}
  if(!is.null(AYFilter.LogLikCutoff)){ROCKY.settings$ay.filter$LogLikCutoff <<- AYFilter.LogLikCutoff}
  if(!is.null(AYFilter.NumberToLag)){ROCKY.settings$ay.filter$NumberToLag <<- AYFilter.NumberToLag}
  if(!is.null(AYFilter.MaxIterations)){ROCKY.settings$ay.filter$MaxIterations <<- AYFilter.MaxIterations}
  if(!is.null(Optimizer.tCutoff)){ROCKY.settings$optimizer$tCutoff <<- Optimizer.tCutoff}
  if(!is.null(Optimizer.maxDist)){ROCKY.settings$optimizer$maxDist <<- Optimizer.maxDist}
  if(!is.null(AutoValidation.NumOfYears)){ROCKY.settings$autoValidation$NumOfYears <<- AutoValidation.NumOfYears}
  if(!is.null(Output.PlotWidth)){ROCKY.settings$output$plotWidth <<- Output.PlotWidth}
  if(!is.null(Output.PlotQuality)){ROCKY.settings$output$plotQuality <<- Output.PlotQuality}
  
  #plots
  #normal diagnostics
  if(!is.null(Plots.Norm.QQ.Color)){ROCKY.settings$plots$normalDiagnostics$QQPlot$color <<- Plots.Norm.QQ.Color}
  if(!is.null(Plots.Norm.Hist.Title)){ROCKY.settings$plots$normalDiagnostics$histogram$title <<- Plots.Norm.Hist.Title}
  if(!is.null(Plots.Norm.Hist.Freq)){ROCKY.settings$plots$normalDiagnostics$histogram$freq <<- Plots.Norm.Hist.Freq}
  if(!is.null(Plots.Norm.Hist.nClass)){ROCKY.settings$plots$normalDiagnostics$histogram$nclass <<- Plots.Norm.Hist.nClass}
  if(!is.null(Plots.Norm.Hist.Color)){ROCKY.settings$plots$normalDiagnostics$histogram$color <<- Plots.Norm.Hist.Color}
  if(!is.null(Plots.Norm.Density.Color)){ROCKY.settings$plots$normalDiagnostics$density$color <<- Plots.Norm.Density.Color}
  if(!is.null(Plots.Norm.Density.Title)){ROCKY.settings$plots$normalDiagnostics$density$title <<- Plots.Norm.Density.Title}
  if(!is.null(Plots.Norm.Box.isHoriz)){ROCKY.settings$plots$normalDiagnostics$boxplot$isHorizontal <<- Plots.Norm.Box.isHoriz}
  if(!is.null(Plots.Norm.Box.Title)){ROCKY.settings$plots$normalDiagnostics$boxplot$title <<- Plots.Norm.Box.Title}
  if(!is.null(Plots.Norm.Box.Color)){ROCKY.settings$plots$normalDiagnostics$boxplot$color <<- Plots.Norm.Box.Color}
  
  #structure plots
  if(!is.null(Plots.Structure.SE.Color)){ROCKY.settings$plots$trendVarPlot$SE$fillColor <<- Plots.Structure.SE.Color}
  if(!is.null(Plots.Structure.SE.Alpha)){ROCKY.settings$plots$trendVarPlot$SE$alpha <<- Plots.Structure.SE.Alpha}
  if(!is.null(Plots.Structure.Points.Size)){ROCKY.settings$plots$trendVarPlot$points$size <<- Plots.Structure.Points.Size}
  if(!is.null(Plots.Structure.Points.Shape)){ROCKY.settings$plots$trendVarPlot$points$shape <<- Plots.Structure.Points.Shape}
  if(!is.null(Plots.Structure.ConnectingLine.Size)){ROCKY.settings$plots$trendVarPlot$connectingLine$size <<- Plots.Structure.ConnectingLine.Size}
  if(!is.null(Plots.Structure.ZeroTrendLine.Color)){ROCKY.settings$plots$trendVarPlot$zeroTrendVertLine$color <<- Plots.Structure.ZeroTrendLine.Color}
  if(!is.null(Plots.Structure.ZeroTrendLine.Alpha)){ROCKY.settings$plots$trendVarPlot$zeroTrendVertLine$alpha <<- Plots.Structure.ZeroTrendLine.Alpha}
  if(!is.null(Plots.Structure.ZeroTrendLine.Size)){ROCKY.settings$plots$trendVarPlot$zeroTrendVertLine$size <<- Plots.Structure.ZeroTrendLine.Size}
  if(!is.null(Plots.Structure.nonZeroTrendLine.Color)){ROCKY.settings$plots$trendVarPlot$nonZeroTrendVertLine$color <<- Plots.Structure.nonZeroTrendLine.Color}
  if(!is.null(Plots.Structure.nonZeroTrendLine.Alpha)){ROCKY.settings$plots$trendVarPlot$nonZeroTrendVertLine$alpha <<- Plots.Structure.nonZeroTrendLine.Alpha}
  if(!is.null(Plots.Structure.nonZeroTrendLine.Size)){ROCKY.settings$plots$trendVarPlot$nonZeroTrendVertLine$size <<- Plots.Structure.nonZeroTrendLine.Size}
  if(!is.null(Plots.Structure.DY.Title)){ROCKY.settings$plots$trendVarPlot$DY$title <<- Plots.Structure.DY.Title}
  if(!is.null(Plots.Structure.DY.yAxisTitle)){ROCKY.settings$plots$trendVarPlot$DY$yAxisTitle <<- Plots.Structure.DY.yAxisTitle}
  if(!is.null(Plots.Structure.AY.Title)){ROCKY.settings$plots$trendVarPlot$AY$title <<- Plots.Structure.AY.Title}
  if(!is.null(Plots.Structure.AY.yAxisTitle)){ROCKY.settings$plots$trendVarPlot$AY$yAxisTitle <<- Plots.Structure.AY.yAxisTitle}
  if(!is.null(Plots.Structure.CY.Title)){ROCKY.settings$plots$trendVarPlot$CY$title <<- Plots.Structure.CY.Title}
  if(!is.null(Plots.Structure.CY.yAxisTitle)){ROCKY.settings$plots$trendVarPlot$CY$yAxisTitle <<- Plots.Structure.CY.yAxisTitle}
  if(!is.null(Plots.Structure.weights.Title)){ROCKY.settings$plots$trendVarPlot$weights$title <<- Plots.Structure.weights.Title}
  if(!is.null(Plots.Structure.weights.yAxisTitle)){ROCKY.settings$plots$trendVarPlot$weights$yAxisTitle <<- Plots.Structure.weights.yAxisTitle}
  if(!is.null(Plots.Structure.Theme.LineType)){ROCKY.settings$plots$trendVarPlot$theme$lineType <<- Plots.Structure.Theme.LineType}
  
  #autovalidation plot
  if(!is.null(Plots.autoVal.SE.Color)){ROCKY.settings$plots$autoValidationPlot$SE$color <<- Plots.autoVal.SE.Color}
  if(!is.null(Plots.autoVal.SE.Alpha2)){ROCKY.settings$plots$autoValidationPlot$SE$alpha2SE <<- Plots.autoVal.SE.Alpha2}
  if(!is.null(Plots.autoVal.SE.Alpha1)){ROCKY.settings$plots$autoValidationPlot$SE$alpha1SE <<- Plots.autoVal.SE.Alpha1}
  if(!is.null(Plots.autoVal.CenterLine.Type)){ROCKY.settings$plots$autoValidationPlot$centerLine$type <<- Plots.autoVal.CenterLine.Type}
  if(!is.null(Plots.autoVal.CenterLine.Color)){ROCKY.settings$plots$autoValidationPlot$centerLine$color <<- Plots.autoVal.CenterLine.Color}
  if(!is.null(Plots.autoVal.Line.Color)){ROCKY.settings$plots$autoValidationPlot$line$color <<- Plots.autoVal.Line.Color}
  if(!is.null(Plots.autoVal.Line.Size)){ROCKY.settings$plots$autoValidationPlot$line$size <<- Plots.autoVal.Line.Size}
  
  #residual plot
  if(!is.null(Plots.Residual.Res.Symbol)){ROCKY.settings$plots$residualPlots$residuals$symbol <<- Plots.Residual.Res.Symbol}
  if(!is.null(Plots.Residual.Res.Color)){ROCKY.settings$plots$residualPlots$residuals$color <<- Plots.Residual.Res.Color}
  if(!is.null(Plots.Residual.Res.BackgroundColor)){ROCKY.settings$plots$residualPlots$residuals$backgroundColor <<- Plots.Residual.Res.BackgroundColor}
  if(!is.null(Plots.Residual.DY.Title)){ROCKY.settings$plots$residualPlots$DY$title <<- Plots.Residual.DY.Title}
  if(!is.null(Plots.Residual.DY.xAxisTitle)){ROCKY.settings$plots$residualPlots$DY$xAxisTitle <<- Plots.Residual.DY.xAxisTitle}
  if(!is.null(Plots.Residual.DY.yAxisTitle)){ROCKY.settings$plots$residualPlots$DY$yAxisTitle <<- Plots.Residual.DY.yAxisTitle}
  if(!is.null(Plots.Residual.AY.Title)){ROCKY.settings$plots$residualPlots$AY$title <<- Plots.Residual.AY.Title}
  if(!is.null(Plots.Residual.AY.xAxisTitle)){ROCKY.settings$plots$residualPlots$AY$xAxisTitle <<- Plots.Residual.AY.xAxisTitle}
  if(!is.null(Plots.Residual.AY.yAxisTitle)){ROCKY.settings$plots$residualPlots$AY$yAxisTitle <<- Plots.Residual.AY.yAxisTitle}
  if(!is.null(Plots.Residual.CY.Title)){ROCKY.settings$plots$residualPlots$CY$title <<- Plots.Residual.CY.Title}
  if(!is.null(Plots.Residual.CY.xAxisTitle)){ROCKY.settings$plots$residualPlots$CY$xAxisTitle <<- Plots.Residual.CY.xAxisTitle}
  if(!is.null(Plots.Residual.CY.yAxisTitle)){ROCKY.settings$plots$residualPlots$CY$yAxisTitle <<- Plots.Residual.CY.yAxisTitle}
  if(!is.null(Plots.Residual.Fitted.Title)){ROCKY.settings$plots$residualPlots$fitted$title <<- Plots.Residual.Fitted.Title}
  if(!is.null(Plots.Residual.Fitted.xAxisTitle)){ROCKY.settings$plots$residualPlots$fitted$xAxisTitle <<- Plots.Residual.Fitted.xAxisTitle}
  if(!is.null(Plots.Residual.Fitted.yAxisTitle)){ROCKY.settings$plots$residualPlots$fitted$yAxisTitle <<- Plots.Residual.Fitted.yAxisTitle}
  if(!is.null(Plots.Residual.ZeroLine.Color)){ROCKY.settings$plots$residualPlots$zeroLine$color <<- Plots.Residual.ZeroLine.Color}
  if(!is.null(Plots.Residual.ZeroLine.LineType)){ROCKY.settings$plots$residualPlots$zeroLine$lineType <<- Plots.Residual.ZeroLine.LineType}
  if(!is.null(Plots.Residual.MeanLine.Color)){ROCKY.settings$plots$residualPlots$meanLine$color <<- Plots.Residual.MeanLine.Color}
  if(!is.null(Plots.Residual.SE.OneFourthColor)){ROCKY.settings$plots$residualPlots$SE$oneFourth<<- Plots.Residual.SE.OneFourthColor}
  if(!is.null(Plots.Residual.SE.OneColor)){ROCKY.settings$plots$residualPlots$SE$one<<- Plots.Residual.SE.OneColor}
  if(!is.null(Plots.Residual.SE.TwoColor)){ROCKY.settings$plots$residualPlots$SE$two<<- Plots.Residual.SE.TwoColor}
  if(!is.null(Plots.Residual.SE.ThreeColor)){ROCKY.settings$plots$residualPlots$SE$three<<- Plots.Residual.SE.ThreeColor}
  if(!is.null(Plots.Residual.SE.FourColor)){ROCKY.settings$plots$residualPlots$SE$four<<- Plots.Residual.SE.FourColor}
  if(!is.null(Plots.Residual.ParameterLine.Type)){ROCKY.settings$plots$residualPlots$parameterLine$type<<- Plots.Residual.ParameterLine.Type}
  if(!is.null(Plots.Residual.ParameterLine.Width)){ROCKY.settings$plots$residualPlots$parameterLine$width<<- Plots.Residual.ParameterLine.Width}
  if(!is.null(Plots.Residual.ParameterLine.Color)){ROCKY.settings$plots$residualPlots$parameterLine$color<<- Plots.Residual.ParameterLine.Color}
  if(!is.null(Plots.Residual.HeteroLine.Type)){ROCKY.settings$plots$residualPlots$heteroLine$type<<- Plots.Residual.HeteroLine.Type}
  if(!is.null(Plots.Residual.HeteroLine.Width)){ROCKY.settings$plots$residualPlots$heteroLine$width<<- Plots.Residual.HeteroLine.Width}
  if(!is.null(Plots.Residual.HeteroLine.Color)){ROCKY.settings$plots$residualPlots$heteroLine$color<<- Plots.Residual.HeteroLine.Color}
  
}

set.default.settings <- function(){
  load.ROCKY.settings()
}

set.default.settings.OLD <- function(){
  update.settings(Data.StartingAY = 1986, Data.StartingCY = 1995, Data.ClipboardSize = 4096, Hetero.SelectWeights = 'count', Hetero.MaxIter = 10, Hetero.MinWtChange = 0.000001, Forecast.varEst = 'MLE', Forecast.varType = 'param_process', Outliers.yminFactor = 1.1, Outliers.ymaxFactor = 1.1, AYFilter.LogLikCutoff = 0.5, AYFilter.NumberToLag = 3, AYFilter.MaxIterations = 5, Optimizer.tCutoff = 2, Optimizer.maxDist = 3, AutoValidation.NumOfYears = 8, Output.PlotWidth = 1000, Output.PlotQuality = 100)
}

change.loglinear.model <- function(plots = TRUE, comment = TRUE){
  graphics.off()
  
  #change settings
  ROCKY.settings$selected.model <<- "loglinear"
  
  #refit the model using the updated settings
  resObj <<- updateTrends(resObj, plots = plots)
  
  #make a log entry
  myComment <- "Changed model to a loglinear model"
  add.to.log(Comment = myComment)
  
  #messagebox = complete
  shell("./VBScripts/Messages/ChangeModel_Loglinear.vbs", translate = TRUE)
}

change.glm.model <- function(model = "odp", plots = TRUE){
  graphics.off()
  
  #change settings
  ROCKY.settings$selected.model <<- "GLM"
  ROCKY.settings$GLM$selected.model <<- model
  
  #re-fit the model (updated settings will re-fit using the specified model)
  resObj <<- updateTrends(resObj, plots = plots)
  
  #separate comments in the log depending on which model selected
  if(ROCKY.settings$GLM$selected.model=="odp"){
    myComment <- "Changed model to an over-dispersed Poisson GLM"
  }
  else if(ROCKY.settings$GLM$selected.model=="gamma"){
    myComment <- "Changed model to a Gamma GLM"
  }
  else if(ROCKY.settings$GLM$selected.model=="gaussian"){
    myComment <- "Changed model to a Gaussian GLM"
  }
  
  #add a log entry
  add.to.log(Comment = myComment)
  
  #display message box depending on which model selected
  if(ROCKY.settings$GLM$selected.model=="odp"){
    shell("./VBScripts/Messages/ChangeModel_ODPGLM.vbs", translate = TRUE)
  }
  else if(ROCKY.settings$GLM$selected.model=="gamma"){
    shell("./VBScripts/Messages/ChangeModel_GammaGLM.vbs", translate = TRUE)
  }
  else if(ROCKY.settings$GLM$selected.model=="gaussian"){
    shell("./VBScripts/Messages/ChangeModel_GaussianGLM.vbs", translate = TRUE)
  }
}

#update model without doing anything else
refit.model <- function(resObj, updateWeights = TRUE, plots = TRUE, addToLog = TRUE, myComment = "Re-fit model"){
  fitModel(resObj, updateWeights = updateWeights, plots = plots)
  if(addToLog){
    add.to.log(IsNewModel = FALSE, Comment = myComment)
  }
}

##### 3. Functions to Update Trend Parameters ####


#updates DY, CY, and AY trend parameters at once
update.trend.parameters <- function(comment = "Updated all trend group parameters", writeToLog = TRUE, msgbox = TRUE, plots = TRUE){
  graphics.off()
  save.workbook()
  resObj$DYGp <<- data.frame(read_excel(path = selectionsFile, sheet = 'DYGp'))
  resObj$CYGp <<- data.frame(read_excel(path = selectionsFile, sheet = 'CYGp'))
  resObj$AYGp <<- data.frame(read_excel(path = selectionsFile, sheet = 'AYGp'))
  
  #control flow for whether using a custom CY trend/std error or not
  if(resObj$ModelSettings$FutureCYTrend$Source == "Custom"){
    if(resObj$ModelSettings$FutureCYstdError$Source == "Custom"){
      assign(x = 'resObj', value = updateTrends(resObj, plots = plots, customFutureCYTrend = TRUE, customFutureCYStdError = TRUE), inherits = TRUE)
    }
    else{
      assign(x = 'resObj', value = updateTrends(resObj, plots = plots, customFutureCYTrend = TRUE, customFutureCYStdError = FALSE), inherits = TRUE)
    }
  }
  else{
    if(resObj$ModelSettings$FutureCYstdError$Source == "Custom"){
      assign(x = 'resObj', value = updateTrends(resObj, plots = plots, customFutureCYTrend = FALSE, customFutureCYStdError = TRUE), inherits = TRUE)
    }
    else{
      assign(x = 'resObj', value = updateTrends(resObj, plots = plots, customFutureCYTrend = FALSE, customFutureCYStdError = FALSE), inherits = TRUE)
    }
  }
  
  if(writeToLog){
    add.to.log(Comment = comment)
  }
  if(msgbox){
    shell(cmd = "./VBScripts/Messages/Update_AllParameters.vbs", translate = TRUE)
  }
}

update.trends.with.custom.trend.consideration <- function(resObj, plots = TRUE){
  #control flow for whether using a custom CY trend/std error or not
  if(resObj$ModelSettings$FutureCYTrend$Source == "Custom"){
    if(resObj$ModelSettings$FutureCYstdError$Source == "Custom"){
      assign(x = 'resObj', value = updateTrends(resObj, plots = plots, customFutureCYTrend = TRUE, customFutureCYStdError = TRUE), inherits = TRUE)
    }
    else{
      assign(x = 'resObj', value = updateTrends(resObj, plots = plots, customFutureCYTrend = TRUE, customFutureCYStdError = FALSE), inherits = TRUE)
    }
  }
  else{
    if(resObj$ModelSettings$FutureCYstdError$Source == "Custom"){
      assign(x = 'resObj', value = updateTrends(resObj, plots = plots, customFutureCYTrend = FALSE, customFutureCYStdError = TRUE), inherits = TRUE)
    }
    else{
      assign(x = 'resObj', value = updateTrends(resObj, plots = plots, customFutureCYTrend = FALSE, customFutureCYStdError = FALSE), inherits = TRUE)
    }
  }
}

#only updates AY trend parameters
update.AYGp.trend.parameters <- function(comment = "AYGp parameter update", writeToLog = TRUE, msgbox = TRUE){
  graphics.off()
  save.workbook()
  resObj$AYGp <<- data.frame(read_excel(path = selectionsFile, sheet = 'AYGp'))
  resObj <<- updateTrends(resObj)
  if(writeToLog){
    add.to.log(Comment = comment)
  }
  if(msgbox){
    shell(cmd = "./VBScripts/Messages/Update_AYGp.vbs", translate = TRUE)
  }
}

#only updates DY trend parameters
update.DYGp.trend.parameters <- function(comment = "DYGp parameter update", writeToLog = TRUE, msgbox = TRUE){
  graphics.off()
  save.workbook()
  resObj$DYGp <<- data.frame(read_excel(path = selectionsFile, sheet = 'DYGp'))
  resObj <<- updateTrends(resObj)
  if(writeToLog){
    add.to.log(Comment = comment)
  }
  if(msgbox){
    shell(cmd = "./VBScripts/Messages/Update_DYGp.vbs", translate = TRUE)
  }
}

#only updates CY trend parameters
update.CYGp.trend.parameters <- function(comment = "CYGp parameter update", writeToLog = TRUE, msgbox = TRUE){
  graphics.off()
  save.workbook()
  resObj$CYGp <<- data.frame(read_excel(path = selectionsFile, sheet = 'CYGp'))
  resObj <<- updateTrends(resObj)
  if(writeToLog){
    add.to.log(Comment = comment)
  }
  if(msgbox){
    shell(cmd = "./VBScripts/Messages/Update_CYGp.vbs", translate = TRUE)
  }
}


##### 4. Functions to add an entry to the log or to load a model from the log ####


#Add model entry to log
add.to.log <- function(IsNewModel = FALSE, Comment = ""){
  #first, either add a new ID, or start off with 1, if it is a new model
  if(IsNewModel){
    shell(cmd = "./VBScripts/AddToLog/AddToLog_NewModelID.vbs", translate = TRUE)
    myComment <- "Initial Model"
    logID <- 1
  }
  else if(!IsNewModel){
    shell(cmd = "./VBScripts/AddToLog/AddToLog_ID.vbs", translate = TRUE) 
    myComment <- Comment
    logID <- length(resObj$log) + 1
  }
  
  #add number of data points to log
  myNumber <- length(resObj$dat$value[!is.na(resObj$dat$value)]) - length(resObj$outliers$id)
  writeClipboard(str = as.character(myNumber))
  shell(cmd = "./VBScripts/AddToLog/AddToLog_N.vbs", translate = TRUE)
  
  #add number of parameters to log
  numParameters <- length(resObj$trendVars)
  writeClipboard(str = as.character(length(resObj$trendVars)))
  shell(cmd = "./VBScripts/AddToLog/AddToLog_NumberOfParameters.vbs", translate = TRUE)
  
  #find AIC
  myAIC <- logLikCompare(resObj)$logLikModel #just the model loglikelihood
  myAIC <- myAIC * (-2) # -2 * (loglikelihood)
  myAIC <- (2 * numParameters) + myAIC #incorporate the penalty for overfitting
  
  #create loglikelihood table (with AIC)
  logLik.table <- logLikCompare(resObj)
  logLik.table$AIC <- myAIC
  
  #add loglikelihood table to log
  copy.table(obj = logLik.table, col.names = FALSE, row.names = FALSE)
  shell(cmd = "./VBScripts/AddToLog/AddToLog_Loglikelihood.vbs", translate = TRUE)
  
#   #add R-squared to log
#   writeClipboard(str = as.character(find.Rsquared(totalDat = resObj$dat$logvalue, fittedDat = resObj$model$fitted.values)))
#   shell(cmd = "./VBScripts/AddToLog/AddToLog_R2.vbs", translate = TRUE)
  
# browser()
  #final DY parameter
  finalDYGroup <- resObj$DYGp$gp[resObj$DYGp$DY == max(resObj$DYGp$DY)]
  if(finalDYGroup=="0"){
    finalDYParameter <- 0
  }
  else{
    finalDYParameter <- resObj$model$coefficients[names(resObj$model$coefficients)==finalDYGroup]
  }
  writeClipboard(str = as.character(finalDYParameter))
  shell(cmd = "./VBScripts/AddToLog/AddToLog_G_f.vbs", translate = TRUE)

  #final CY parameter
#   finalCYGroup <- resObj$CYGp$gp[resObj$CYGp$CY == max(resObj$CYGp$CY)]
#   finalCYParameter <- resObj$model$coefficients[names(resObj$model$coefficients)==finalCYGroup]
  if(resObj$ModelSettings$FutureCYTrend$Source == 'Custom'){
    finalCY <- max(resObj$CyProjTrendSelect$CY)
    finalCYParameter <- resObj$CyProjTrendSelect$trend[resObj$CyProjTrendSelect$CY == finalCY]
  }
  else{
    finalCY <- max(resObj$CyProjTrendModel$CY)
    finalCYParameter <- resObj$CyProjTrendModel$trend[resObj$CyProjTrendModel$CY == finalCY]    
  }
  writeClipboard(str = as.character(finalCYParameter))
  shell(cmd = "./VBScripts/AddToLog/AddToLog_I_f.vbs", translate = TRUE)
  
  #next, copy total reserve/SE/CV and add to log
  copy.table(resObj$forecast$Total, col.names = FALSE)
  shell(cmd = "./VBScripts/AddToLog/AddToLog_ReserveStats.vbs", translate = TRUE) 
  
  #add the comment to the log
  writeClipboard(str = as.character(myComment))
  shell(cmd = "./VBScripts/AddToLog/AddToLog_Comment.vbs", translate = TRUE) 

#   do all of the above in one step, to speed it up
#   excelLogEntry <- data.frame(c(myNumber, numParameters, logLikCompare(resObj), finalDYParameter, finalCYParameter, resObj$forecast[4][1], 
#                                 resObj$forecast[4][2], resObj$forecast[4][3], myComment))
  
#   excelLogEntry <- data.frame(c(1))
#   excelLogEntry$N <- myNumber
#   excelLogEntry$NumPara <- numParameters
#   excelLogEntry$LL1 <- logLikCompare(resObj)[,1]
#   excelLogEntry$LL2 <- logLikCompare(resObj)[,2]
#   excelLogEntry$LL3 <- logLikCompare(resObj)[,3]
#   excelLogEntry$Gf <- finalDYParameter
#   excelLogEntry$If <- finalCYParameter
#   excelLogEntry$totRes <- resObj$forecast[4][1]
#   excelLogEntry$totSE <- resObj$forecast[4][2]
#   excelLogEntry$totCV <- resObj$forecast[4][3]
#   excelLogEntry$comment <- myComment
# 
#   write.table(x = excelLogEntry, file = "clipboard", sep = , "\t", row.names = FALSE, col.names = FALSE)
#   shell(cmd = "./VBScripts/AddToLog/AddToLog_N.vbs", translate = TRUE)
  
  
  #### add this model to the "log" property of resObj, for recall later
  
  #first, find the log ID ()
  shell(cmd = "./VBScripts/AddToLog/AddToLog_CopyCurrentID.vbs", translate = TRUE) 
  tempID <- as.numeric(readClipboard())

  #create a data frame that contains the log entry being pasted to Excel
    #create entries
  excelLog.ID <- tempID
  excelLog.N <- myNumber
  excelLog.k <- numParameters
  excelLog.ModelLL <- logLik.table[[1,1]]
  excelLog.ActualLL <- logLik.table[[1,2]]
  excelLog.CombinedLL <- logLik.table[[1,3]]
  excelLog.AIC <- logLik.table[[1,4]]
  excelLog.Gf <- as.numeric(finalDYParameter)
  excelLog.If <- as.numeric(finalCYParameter)
  excelLog.Total.Reserve <- resObj$forecast$Total[[1,1]]
  excelLog.Total.SE <- resObj$forecast$Total[[1,2]]
  excelLog.Total.CV <- resObj$forecast$Total[[1,3]]
  excelLog.Comment <- myComment

    #make a list of the entries
  excelLog.data <- c(excelLog.ID, excelLog.N, excelLog.k, excelLog.ModelLL, excelLog.ActualLL, excelLog.CombinedLL, excelLog.AIC, 
                     excelLog.Gf, excelLog.If, excelLog.Total.Reserve, excelLog.Total.SE, excelLog.Total.CV, excelLog.Comment)

    #turn that list into a data frame & add column names
  temp.excelLog <- data.frame(matrix(data = excelLog.data,nrow = 1))
  names(temp.excelLog) <- c("ID", "N", "k", "ModelLL", "ActualLL", "CombinedLL", "AIC", 
                            "Gf", "If", "Total.Reserve", "Total.SE", "Total.CV", "Comment")

  #append this row to the model log
    #if excel model log is empty, then excelLog becomes the new log-otherwise, append to the end
  if(length(resObj$excelLog)==0){
    resObj$excelLog <<- temp.excelLog
  }
  else{
    resObj$excelLog <<- rbind(resObj$excelLog, temp.excelLog)
  }
  

  
  #now, create the log entry (essentially a list of everything that is in the current iteration of the model)
  logEntry <- list(logID = tempID, logDat = resObj$dat, logAYGp = resObj$AYGp, logDYGp = resObj$DYGp, logCYGp = resObj$CYGp, 
                   logDYwGp = resObj$DYwGp, logDYw = resObj$DYw, logtrendVars = resObj$trendVars, 
                   logAYgpFilters = resObj$AYgpFilters, logModel = resObj$model, logOutliers = resObj$outliers, logSettings = resObj$ModelSettings)
  
  #finally, apend the log entry to the resObj$log 
  if(is.null(resObj$log)){
    resObj$log <<- list(logEntry) #if the log is empty, then start a new one
  }
  else if(!is.null(resObj$log)){
    resObj$log <<- append(resObj$log, list(logEntry)) #if the log is not empty, then just add on
  }

  #create reserve table
  resTable <- data.frame(summary(resObj$model)$coefficients)[,c(1,2,3)]

  #copy reserve table
  copy.table(obj = resTable, row.names = TRUE, col.names = FALSE)

  #push reserve table to Excel
  shell(cmd = "./VBScripts/Push/Push_ReserveTable.vbs", translate = TRUE)
}

#Load model from log
load.from.log <- function(log.ID, addToLog = TRUE, msgBox = TRUE, plots = TRUE){
  #clear graphics
  graphics.off()
  
  #change back to previously recorded version of the model
  resObj$dat            <<- resObj$log[[log.ID]]$logDat
  resObj$AYGp           <<- resObj$log[[log.ID]]$logAYGp
  resObj$DYGp           <<- resObj$log[[log.ID]]$logDYGp
  resObj$CYGp           <<- resObj$log[[log.ID]]$logCYGp
  resObj$DYwGp          <<- resObj$log[[log.ID]]$logDYwGp
  resObj$DYw            <<- resObj$log[[log.ID]]$logDYw
  resObj$trendVars      <<- resObj$log[[log.ID]]$logtrendVars
  resObj$AYgpFilters    <<- resObj$log[[log.ID]]$logAYgpFilters
  resObj$model          <<- resObj$log[[log.ID]]$log.ID
  resObj$outliers       <<- resObj$log[[log.ID]]$logOutliers
  resObj$ModelSettings  <<- resObj$log[[log.ID]]$logSettings
  
  #add a note that this model was retrieved from a previous version of the model
  myComment <- paste("This model was retrieved from the model with the ID:", log.ID)
  
  #update the model
  resObj <<- updateTrends(resObj, plots = FALSE)
  
  #ensure that R model parameters are the same as Excel model parameters
  push.ModelParameters()
  
  #update the hetero adjustment
  manual.hetero.adjustment(manualComment = "", writeToLog = FALSE, hetero.msgbox = FALSE, plots = plots)
  
  #add this iteration of the model to the log
  if(addToLog){
    add.to.log(Comment = myComment)
  }
  
  #update outlier list
  update.excel.outlier.list()
  
  if(msgBox){
  #complete message
  shell(cmd = "./VBScripts/Messages/LoadFromLog.vbs", translate = TRUE)
  }
}


##### 5. Functions to adjust for heteroscedasticity, and to clear or reset the adjustment ####


#for when you want to pick the hetero groups, and let ROCKY find the weights
automatic.hetero.adjustment <- function(myComment = "Automatic heteroscedasticity adjustment", reset = FALSE, plots = TRUE){
  graphics.off() #clear graphics
  save.workbook() #save workbook
  resObj <<- DYweightGroups(resObj)
#   resObj$DYwGp <<- data.frame(read_excel(path = selectionsFile, sheet = 'DYwGp')) #read in selected hetero parameters
  resObj <<- iterateWts(resObj,returnIter = TRUE,maxIter = ROCKY.settings$hetero$MaxIter, plots = FALSE) #run iterateWts to find the hetero adjustment
  push.DYwGp() #push groups to Excel
  push.DYw() #push calcluated hetero weights to the Excel worksheet
  updateTrends(resObj, plots = plots) #refit model
  add.to.log(IsNewModel = FALSE, Comment = myComment)
  
#   update.trend.parameters(comment = myComment) #update the model/create a log entry
  
  #if not a hetero reset (see below), display complete message
  if(!reset){
    shell(cmd = "./VBScripts/Messages/Auto_Hetero.vbs", translate = TRUE)
  }
}

#for when you want to pick the hetero groups, and let ROCKY find the weights
custom.group.hetero.adjustment <- function(myComment = "Custom group heteroscedasticity adjustment", reset = FALSE, plots = TRUE, includeMessage = TRUE){
  graphics.off() #clear graphics
  save.workbook() #save workbook
  resObj$DYwGp <<- data.frame(read_excel(path = selectionsFile, sheet = 'DYwGp')) #read in selected hetero parameters
  resObj <<- iterateWts(resObj,returnIter = TRUE,maxIter = ROCKY.settings$hetero$MaxIter, plots = FALSE) #run iterateWts to find the hetero adjustment
  push.DYw() #push calcluated hetero weights to the Excel worksheet
  updateTrends(resObj, plots = plots) #refit model
  add.to.log(IsNewModel = FALSE, Comment = myComment)
  
  #   update.trend.parameters(comment = myComment) #update the model/create a log entry
  
  #if not a hetero reset (see below), display complete message
  if(!reset){
    if(includeMessage){
      shell(cmd = "./VBScripts/Messages/CustomGroup_Hetero.vbs", translate = TRUE)
    }
  }
}

#for when you want to pick the hetero weights manually
manual.hetero.adjustment <- function(manualComment = "Manual heteroscedasticity adjustment", writeToLog = TRUE, hetero.msgbox = TRUE, plots = TRUE){
  #clear graphics
  graphics.off()
  
  #ensure the Excel workbook is saved
  save.workbook()
  
  #read in the hetero weights
  resObj$DYw <<- data.frame(read_excel(path = selectionsFile, sheet = 'DYw'))
  
  #update the model
  resObj <<- fitModel(resObj,updateWeights = TRUE, plots = FALSE)
  
  #update the model/log entry
  update.trend.parameters(comment = manualComment, writeToLog = writeToLog, msgbox = FALSE, plots = plots)
  
  if(hetero.msgbox){
    shell(cmd = "./VBScripts/Messages/Manual_Hetero.vbs", translate = TRUE)
  }
}

#clears the hetero adjustment, makes all weights 1
clear.hetero.adjustment <- function(Comment = "Cleared heteroscedasticity adjustment", write.to.log = TRUE, reset = FALSE, plots = TRUE){
  #divide each weight by itself to reset all to 1
  resObj$DYw$w <<- resObj$DYw$w/resObj$DYw$w
  
  #push these weights to Excel
  push.DYw()
  
  #re-fit model
  manual.hetero.adjustment(manualComment = Comment, writeToLog = write.to.log, hetero.msgbox = FALSE, plots = plots)
  
  #if not a reset, display complete message
  if(!reset){
    shell(cmd = "./VBScripts/Messages/Clear_Hetero.vbs", translate = TRUE)
  }
}

#clears the hetero adjustment, then finds the weights again (to keep adjustment current with any futher updates in the model)
reset.hetero.adjustment <- function(Comment = "Reset heteroscedasticity adjustment"){
  #clear hetero adjustment
  clear.hetero.adjustment(Comment = "", write.to.log = FALSE, reset = TRUE)
  
  #re-do hetero adjustment
  automatic.hetero.adjustment(myComment = Comment, reset = TRUE)
  
  #display complete message
  shell(cmd = "./VBScripts/Messages/Reset_Hetero.vbs", translate = TRUE)
}


##### 6. Functions to optimize the number of parameters ####


#go through a single step of the variable optimizer
one.step.variable.optimizer <- function(standAlone = TRUE){
#   browser()
  #clear graphics
  graphics.off()
  
  #run the variable optimizer (one iteration)
  resObj <<- eliminate_var_diff(resObj,maxDist = ROCKY.settings$optimizer$maxDist)
  
  #flip switch on whether user has selected one-step or automatic variable optimizer
  if(standAlone){
    IsOneStep <<- TRUE
  }
  else if(!standAlone){
    IsOneStep <<- FALSE
  }
  
  #push the new AYGp's, DYGp's, and CYGp's to Excel
  push.AYGp()
  push.DYGp()
  push.CYGp()
  
  #comment
  if(standAlone){
    myComment <<- paste(c('One-Step Variable Optimizer: ', variableOptimizer.optimizeComment), sep = "", collapse = "")
  }
  else{
    myComment <<- paste(c('Automatic Variable Optimizer Step ', variableOptimizer.commentCounter, ': ', variableOptimizer.optimizeComment), sep = "",
                        collapse = "")
  }
  
  #update trend parameters/add entry to log
  update.trend.parameters(comment = myComment, msgbox = FALSE, plots = FALSE)
  
  #clear global variables used in only in this function
  suppressWarnings(rm('variableOptimizer.optimizeComment'))
  suppressWarnings(rm('myComment'))

  #only clear IsOneStep if doing a one-step optimization
  if(standAlone){
    suppressWarnings(rm('IsOneStep'))
  }                    
}

#go throught all steps of the variable optimizer automatically
automatic.variable.optimizer <- function(){
  #This will tell R when we are done (we go until there are no more insignificant variables)
  variableOptimizer.ExistsInsignificantVariables <<- TRUE
  
  #for the comment, create a 2xN list of, for each iteration, what we replace, and what it is replaced by
  variableOptimizer.commentList <<- list()
  variableOptimizer.commentCounter <<- 0
  
  #loop the one.step.variable.optimizer until there are no more insignificant variables
  while(variableOptimizer.ExistsInsignificantVariables){
    one.step.variable.optimizer(standAlone = FALSE)
  }
  
  #at the very end, clear out the variableOptimizer variables used
  suppressWarnings(rm("variableOptimizer.commentList"))
  suppressWarnings(rm("variableOptimizer.commentCounter"))
  suppressWarnings(rm("variableOptimizer.ExistsInsignificantVariables"))
  suppressWarnings(rm("variableOptimizer.optimizeComment"))
  
  #display completion message
  shell(cmd = "./VBScripts/Messages/AllVariablesSignificant.vbs", translate = TRUE)
}


##### 7. Functions to support the validation feature ####


#produce the validation table of results, then push to the validation tab in Excel
automatic.validation.table <- function(){
  #run ROCKY function
  resObj <<- CYstabilityCalc(resObj)
  
  #paste the results in Excel
  write.table(x = resObj$CYstability, file = "clipboard", sep = "\t", row.names = FALSE, col.names = FALSE)
  shell(cmd = "./VBScripts/AutoValidation/Push_AVResults.vbs", translate = TRUE)
}

#produce the validation table of results, then push to the validation tab in Excel
automatic.validation.graph <- function(pushToExcel = TRUE){
  
#   browser()
  
  #run ROCKY function
#   resObj <<- CYstabilityCalc(resObj)
  
  #produce the AV plot
  graphics.off()
  jpeg(filename = './www/AutoValidation/AV.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  plotCYstability(resObj, newWindow = FALSE)
  dev.off()

  #if you want to, push the plot to Excel, and close the plot windows
  if(pushToExcel){
#     save.auto.validation.plot()
    shell(cmd = "./VBScripts/Images/Load_AVPlot.vbs", translate = TRUE)
  }
}

#just does both of the previous functions
automatic.validation <- function(pushToExcel = TRUE){
  automatic.validation.table()
  automatic.validation.graph(pushToExcel = pushToExcel)
}


##### 8. Functions to produce diagnostic/residual plots ####


#----- Normality diagnostic plots --------------------------

#produce all diagnostic plots in a 4x4 grid
diagnostic.plots <- function(pushToExcel = FALSE, newWindow = TRUE){
  if(!pushToExcel) graphics.off()
  normPlots(resObj, newWindow = newWindow)
}

#only produce the normal Q-Q plot
diagnostic.plots.normal.QQ <- function(pushToExcel = FALSE, newWindow = TRUE){
  if(!pushToExcel) graphics.off()
  if(newWindow) x11()
  op <- par(oma = c(0, 1, 0, 0))
  plot(resObj$model,2,
       col=ROCKY.settings$plots$normalDiagnostics$QQPlot$color)
  par(op) 
}

#only produce to box plot
diagnostic.plots.box.plot <- function(pushToExcel = FALSE, newWindow = TRUE){
  if(!pushToExcel) graphics.off()
  dat <- resObj$dat
  dat <- dat[!is.na(dat$logvalue),]
  if(newWindow) x11()
  boxplot(dat$residStd,
          horizontal = ROCKY.settings$plots$normalDiagnostics$boxplot$isHorizontal,
          main = ROCKY.settings$plots$normalDiagnostics$boxplot$title,
          col=ROCKY.settings$plots$normalDiagnostics$boxplot$color)
}

#only produce the histogram
diagnostic.plots.histogram <- function(pushToExcel = FALSE, newWindow = TRUE){
  if(!pushToExcel) graphics.off()
  dat <- resObj$dat
  dat <- dat[!is.na(dat$logvalue),]
  if(newWindow) x11()
  hist(dat$residStd,
       xlab = ROCKY.settings$plots$normalDiagnostics$histogram$xlab,
       ylab = ROCKY.settings$plots$normalDiagnostics$histogram$ylab,
       main = ROCKY.settings$plots$normalDiagnostics$histogram$title,
       freq = ROCKY.settings$plots$normalDiagnostics$histogram$freq,
       nclass = ROCKY.settings$plots$normalDiagnostics$histogram$nclass,
       col = ROCKY.settings$plots$normalDiagnostics$histogram$color)
}

#only produce the kernel density
diagnostic.plots.kernel.density <- function(pushToExcel = FALSE, newWindow = TRUE){
  if(!pushToExcel) graphics.off()
  dat <- resObj$dat
  dat <- dat[!is.na(dat$logvalue),]
  if(newWindow) x11()
  plot(density(dat$residStd),
       col = ROCKY.settings$plots$normalDiagnostics$density$color,
       main = ROCKY.settings$plots$normalDiagnostics$density$title,
       xlab = ROCKY.settings$plots$normalDiagnostics$density$xlab)  
}



#----- Individual Residual Graphs --------------------------

#single plot of residuals vs. DY
DY.residual.graph <- function(newWindow = TRUE){
  graphics.off()
  resPlot(resObj, option = 'DY', newWindow = newWindow)
}

#single plot of residuals vs. CY
CY.residual.graph <- function(newWindow = TRUE){
  graphics.off()
  resPlot(resObj, option = 'CY', newWindow = newWindow)
}

#single plot of residuals vs. AY
AY.residual.graph <- function(newWindow = TRUE){
  graphics.off()
  resPlot(resObj, option = 'AY', newWindow = newWindow)
}

#fitted residual graph
fitted.residual.graph <- function(newWindow = TRUE){
  graphics.off()
  resPlot(resObj, option = 'fitted', newWindow = newWindow)
}

#all 4 residual graphs
all.four.residual.graphs <- function(newWindow = TRUE){
  graphics.off()
  resPlot(resObj, option = 'all', newWindow = newWindow)
}

#----- Model Structure Graphs --------------------------

#single plot of DY parameters
DY.model.structure.plot <- function(newWindow = TRUE){
  graphics.off()
  trendVarPlot(resObj, newWindow = newWindow, option = 'DY')
}

#single plot of AY parameters
AY.model.structure.plot <- function(newWindow = TRUE){
  graphics.off()
  trendVarPlot(resObj, newWindow = newWindow, option = 'AY')
}

#single plot of CY parameters
CY.model.structure.plot <- function(newWindow = TRUE){
  graphics.off()
  trendVarPlot(resObj, newWindow = newWindow, option = 'CY')
}

#single plot of hetero weights
hetero.model.structure.plot <- function(newWindow = TRUE){
  graphics.off()
  trendVarPlot(resObj, newWindow = newWindow, option = 'weights')
}

#all four model structure graphs
model.structure.plot <- function(newWindow = TRUE){
  graphics.off()
  trendVarPlot(resObj, newWindow = newWindow, option = 'all')
}



##### 9. Functions to remove outliers, then add them back in ####


#brings up a plot (by selected "Option") in which outliers can be selected, then removed from the model
remove.outliers <- function(x.axis = 'DY', new.window = TRUE){
  graphics.off() #close other plot windows
  oldNumOutliers <- length(resObj$outliers$id) #before do anything, find current number of outliers
  resObj <<- removeOutliers(resObj,option = x.axis, newWindow = new.window) #run the ROCKY removeOutliers function
  resObj <<- fitModel(resObj) #re-fit now that you have removed outliers
  numOutliers <- length(resObj$outliers$id) #now that outliers have been removed, what is new number of outliers
  outlierChange <- numOutliers - oldNumOutliers #change in the number
  
  #because correct grammar is important to me
  if(outlierChange == 1){
    commentEnd <- " outlier/Total: "
  }
  else {
    commentEnd <- " outliers/Total: "
  }
  
  #comment for log
  myComment <- paste(c("Removed ", as.character(outlierChange), commentEnd, as.character(numOutliers)), sep = "", collapse = "")
  
  #only add entry to log if you actually remove outliers
  if(outlierChange != 0){
    add.to.log(IsNewModel = FALSE, Comment = myComment)
    update.excel.outlier.list()
  }
  
  graphics.off() #ensure outlier windows are closed
  
  #display message
  shell(cmd = "./VBScripts/Messages/Outliers_Removed.vbs", translate = TRUE)
}

#re-includes all previously excluded outliers
add.outliers.back.in <- function(){
#   browser()
  graphics.off()
  oldNumOutliers <- length(resObj$outliers$id)
  
  #only continue if there are actually outliers to re-include
  if(oldNumOutliers != 0){
    assign('resObj', value = addBackOutliers(resObj),inherits = TRUE)
    newNumOutliers <- length(resObj$outliers$id)
    OutlierChange <- oldNumOutliers - newNumOutliers
    
    update.trend.parameters(writeToLog = FALSE, msgbox = FALSE, plots = FALSE)
    
    myComment <- paste(c("Added all ", as.character(OutlierChange), " outliers back to the model"), sep = "", collapse = "")
    
    add.to.log(IsNewModel = FALSE, Comment = myComment)
    
    clear.excel.outlier.list()
    
    #display message
    shell(cmd = "./VBScripts/Messages/Outliers_Restored.vbs", translate = TRUE)
  }
}

#clears Excel outlier list
clear.excel.outlier.list <- function(){
  shell(cmd = "./VBScripts/Clear_Outliers.vbs", translate = TRUE)
}

#update Excel outlier list
update.excel.outlier.list <-function(){
  clear.excel.outlier.list() 
  copy.table(obj = resObj$outliers, col.names = FALSE, row.names = FALSE)
  shell(cmd = "./VBScripts/Push/Push_Outliers.vbs", translate = TRUE)
}


##### 10. Functions to run ROCKY's AY Filtering ####


#let ROCKY do the whole thing for you
automatic.AY.filtering <- function(popUp = TRUE){
  #clear graphics
  graphics.off()
  
  #run the ROCKY process
  resObj <<- AYsmoothLogLik(resObj)
  
  #send new filters to Excel
  push.AYgpFilters()
  
  #create a log entry
  add.to.log(IsNewModel = FALSE, Comment = "Ran the automatic AY filtering process")  
  
  #pop up
  if(popUp){
    shell(cmd = "./VBScripts/Messages/Auto_Filtering.vbs", translate = TRUE)
  }
}

#set your own AY filters, and ROCKY will just use them
manual.AY.filtering <- function(popUp = TRUE){
  #clear graphics
  graphics.off()
  
  #ROCKY process
  resObj$AYgpFilters <<- data.frame(read_excel(path = selectionsFile, sheet = 'AYgpFilters'),stringsAsFactors = FALSE)
  resObj <<- applySelectedAYfilters(resObj)
  
  #create a log entry
  add.to.log(IsNewModel = FALSE, Comment = "Updated manual AY filters")
  
  if(popUp){
    #pop up
    shell(cmd = "./VBScripts/Messages/Manual_Filtering.vbs", translate = TRUE)
  }
}

#remove filters
clear.AY.filtering <- function(popUp = TRUE){
  #clear graphics
  graphics.off()
  
  #ROCKY process
  resObj <<- unfilterDat(resObj)
  
  #push cleared filters to Excel
  push.AYgpFilters()
  
  #create a log entry
  add.to.log(IsNewModel = FALSE, Comment = "Cleared AY filters")
  
  #pop up
  if(popUp){
    shell(cmd = "./VBScripts/Messages/Clear_Filtering.vbs", translate = TRUE)
  }
}


##### 11. Functions to support ROCKY's Custom Future Trend Hetero Fixed Feature ####


# ------ NADA ---------- #


##### 12. Output for Summary Files ####


#create a table with AY's back to 1967, outstanding reserve, ultimate, SD, CV, cond on next period (don't have this)
create.summary.table <- function(){
  
#   refresh.forecast.for.custom.selections()
  
  curAY <- max(resObj$dat$AY)
  
  #create vectors that will become the table
  AY <- c(ROCKY.settings$data$oldestAY:curAY)
  Outstanding <- c()
  Ultimate <- c()
  SE <- c()
  CV <- c()
  
  #table with forecast
  forecast.table <- resObj$forecast$AY
  
  #make Outstanding, SE, CV, and Ultimate vectors
  for(i in 1:(curAY-(ROCKY.settings$data$oldestAY-1))){
    
    #make Outstanding, SE, CV vectors
    if(is.na(as.numeric(forecast.table[,'Reserve'][as.character(i+(ROCKY.settings$data$oldestAY-1))]))){
      Outstanding[i] <- 0
      SE[i] <- 0
      CV[i] <- 0
    }
    else{
      Outstanding[i] <- as.numeric(forecast.table[,'Reserve'][as.character(i+(ROCKY.settings$data$oldestAY-1))])
      SE[i] <- as.numeric(forecast.table[,'se'][as.character(i+(ROCKY.settings$data$oldestAY-1))])
      CV[i] <- as.numeric(forecast.table[,'CV'][as.character(i+(ROCKY.settings$data$oldestAY-1))])
    }
    
    #make Ultimate vector
    Ultimate[i] <- sum(resObj$dat$value[!is.na(resObj$dat$value)][resObj$dat$AY[!is.na(resObj$dat$value)] == (i+(ROCKY.settings$data$oldestAY-1))]) + Outstanding[i]
  }
  
  #get totals
  AY.Total <- "Total"
  Outstanding.Total <- sum(Outstanding)
  Ultimate.Total <- sum(Ultimate)
  SE.Total <- as.numeric(resObj$forecast$Total[,'se'])
  CV.Total <- as.numeric(resObj$forecast$Total[,'CV'])
  
  #append a space
  AY[length(AY)+1] <- ""
  Outstanding[length(Outstanding)+1] <- ""
  Ultimate[length(Ultimate)+1] <- ""
  SE[length(SE)+1] <- ""
  CV[length(CV)+1] <- ""
  
  #append the totals
  AY[length(AY)+1] <- AY.Total
  Outstanding[length(Outstanding)+1] <- Outstanding.Total
  Ultimate[length(Ultimate)+1] <- Ultimate.Total
  SE[length(SE)+1] <- SE.Total
  CV[length(CV)+1] <- CV.Total
  
  #create table 
  AY.Summary <- data.frame(AY)
  
  #add columns
  AY.Summary$Outstanding <- Outstanding
  AY.Summary$Ultimate <- Ultimate
  AY.Summary$SE <- SE
  AY.Summary$CV <- CV
  
  return(AY.Summary)
}

#after you run create.summary.table(), this copies/pastes it to the appropriate spot
push.summary.AY.table <- function(){
  copy.table(obj = create.summary.table(), col.names = FALSE, row.names = FALSE)
  shell(cmd = "./VBScripts/Push/Push_SummaryOutput.vbs", translate = TRUE)
}

#wrapper function just in case we ever want this to do more than just that AY table
push.summary.output <- function(){
  push.summary.AY.table()
  push.risk.free.summary.AY.table()
  save.all.plots() #saves plots so they can be used in Excel
  shell(cmd = "./VBScripts/Images/Load_Images.vbs", translate = TRUE)
  save.ROCKY.settings() #save current settings to load back in next time
}

#refresh forecast to account for custom future trends/std errors
refresh.forecast.for.custom.selections <- function(){
  if(resObj$ModelSettings$FutureCYTrend$Source=='Custom'){    
    if(resObj$ModelSettings$FutureCYstdError$Source=='Custom'){
      update.trend.parameters(comment = "Custom future CY trend/Custom future CY SE", writeToLog = TRUE, msgbox = FALSE, plots = FALSE)
    }   
    else{
      update.trend.parameters(comment = "Custom future CY trend/Modeled future CY SE", writeToLog = TRUE, msgbox = FALSE, plots = FALSE)
    }
  }  
  else{
    if(resObj$ModelSettings$FutureCYstdError$Source=='Custom'){
      update.trend.parameters(comment = "Modeled future CY trend/Custom future CY SE", writeToLog = TRUE, msgbox = FALSE, plots = FALSE)
    }
    else{
      update.trend.parameters(comment = "Modeled future CY trend/Modeled future CY SE", writeToLog = TRUE, msgbox = FALSE, plots = FALSE)
    }    
  }
}

#create a risk-free summary
create.risk.free.summary.table <- function(){
#   browser()
  curAY <- max(resObj$dat$AY)
  
  #create vectors that will become the table
  AY <- c(ROCKY.settings$data$oldestAY:curAY)
  Outstanding <- c()
  Ultimate <- c()
  SE <- c()
  CV <- c()
  resObj.copy <- resObj
  
  #copy current varType 
  myVarType <- ROCKY.settings$forecast$varType
  
  #set varType = 'none'
  ROCKY.settings$forecast$varType <<- 'none'
  
  #run the risk-free reserve forecast
  resObj.copy <- resForecast(resObj = resObj.copy, customCyTrend = FALSE, customCyStdErr = FALSE)
  
  #table with forecast
  forecast.table <- resObj.copy$forecast$AY
  
  #make Outstanding, SE, CV, and Ultimate vectors
  for(i in 1:(curAY-(ROCKY.settings$data$oldestAY-1))){
    
    #make Outstanding, SE, CV vectors
    if(is.na(as.numeric(forecast.table[,'Reserve'][as.character(i+(ROCKY.settings$data$oldestAY-1))]))){
      Outstanding[i] <- 0
      SE[i] <- 0
      CV[i] <- 0
    }
    else{
      Outstanding[i] <- as.numeric(forecast.table[,'Reserve'][as.character(i+(ROCKY.settings$data$oldestAY-1))])
      SE[i] <- as.numeric(forecast.table[,'se'][as.character(i+(ROCKY.settings$data$oldestAY-1))])
      CV[i] <- as.numeric(forecast.table[,'CV'][as.character(i+(ROCKY.settings$data$oldestAY-1))])
    }
    
    #make Ultimate vector
    Ultimate[i] <- sum(resObj.copy$dat$value[!is.na(resObj.copy$dat$value)][resObj.copy$dat$AY[!is.na(resObj.copy$dat$value)] == (i+(ROCKY.settings$data$oldestAY-1))]) + Outstanding[i]
  }
  
  #get totals
  AY.Total <- "Total"
  Outstanding.Total <- sum(Outstanding)
  Ultimate.Total <- sum(Ultimate)
  SE.Total <- as.numeric(resObj.copy$forecast$Total[,'se'])
  CV.Total <- as.numeric(resObj.copy$forecast$Total[,'CV'])
  
  #append a space
  AY[length(AY)+1] <- ""
  Outstanding[length(Outstanding)+1] <- ""
  Ultimate[length(Ultimate)+1] <- ""
  SE[length(SE)+1] <- ""
  CV[length(CV)+1] <- ""
  
  #append the totals
  AY[length(AY)+1] <- AY.Total
  Outstanding[length(Outstanding)+1] <- Outstanding.Total
  Ultimate[length(Ultimate)+1] <- Ultimate.Total
  SE[length(SE)+1] <- SE.Total
  CV[length(CV)+1] <- CV.Total
  
  #create table 
  AY.Summary <- data.frame(AY)
  
  #add columns
  AY.Summary$Outstanding <- Outstanding
  AY.Summary$Ultimate <- Ultimate
  AY.Summary$SE <- SE
  AY.Summary$CV <- CV
  
  #reset the varType to what it was before
  ROCKY.settings$forecast$varType <<- myVarType
#   browser()
  return(AY.Summary)
}

#after you run create.summary.table(), this copies/pastes it to the appropriate spot
push.risk.free.summary.AY.table <- function(){
  copy.table(obj = create.risk.free.summary.table(), col.names = FALSE, row.names = FALSE)
  shell(cmd = "./VBScripts/Push/Push_RiskFreeSummaryOutput.vbs", translate = TRUE)
}

############################
##### 13. Save Function ####
############################

#save settings
#only kind of related, but similar structure to below
save.ROCKY.settings <- function(){
  save.fileName <- "./set/user_settings.ROCKYsettings"
#   saveRDS(object = ROCKY.settings, file = save.fileName)
#   dput(x = ROCKY.settings, file = save.fileName, control = "showAttributes")
  save('ROCKY.settings', file = save.fileName)
}

#save excelLog
save.excelLog <- function(fileName = "temp"){
  save.fileName <- paste(c("./log/", fileName, ".ROCKYlog"), sep = "", collapse = "")
  write.table(x = resObj$excelLog, file = save.fileName, sep = ",", na = "NA", dec = ".", row.names = FALSE)
}

#save modelLog
save.modelLog <- function(fileName = "temp"){
  tempLog <- resObj$log
  save.fileName <- paste(c("./model/", fileName, ".ROCKYmodel"), sep = "", collapse = "")
  saveRDS(object = resObj$log, file = save.fileName)
}

#save model data
save.modelDat <- function(fileName = "temp"){
  temp.dat <- resObj$dat
  temp.fileName <- paste(c("./dat/", fileName, ".ROCKYdata"), sep = "", collapse = "")
  write.table(x = temp.dat, file = temp.fileName, sep = ",", na = "NA", dec = ".", row.names = FALSE)
}

#save all model parts at once
save.model <- function(fileName = "temp"){
  save.modelLog(fileName)
  save.excelLog(fileName)
  save.modelDat(fileName)
  myComment <- paste(c("Model saved under the name: ", fileName), sep = "", collapse = "")
  add.to.log(IsNewModel = FALSE, Comment = myComment)
  shell(cmd = "./VBScripts/Messages/ModelSaved.vbs", translate = TRUE)
}

############################
##### 14. Load Function ####
############################

load.model <- function(fileName, plots = TRUE){
  load.modelLog(fileName)
  load.excelLog(fileName)
  load.modelDat(fileName)
  load.update.excel()
  load.update.ROCKY(fileName)
  shell('./VBScripts/Messages/LoadFromLog.vbs', translate = TRUE)
}

#load the modelLog
load.modelLog <- function(fileName){
  temp.fileName <- paste(c("./model/", fileName, ".ROCKYmodel"), sep = "", collapse = "")
  resObj$log <<- readRDS(file = temp.fileName)
}

#load the excelLog
load.excelLog <- function(fileName){
  temp.fileName <- paste(c("./log/", fileName, ".ROCKYlog"), sep = "", collapse = "")
  resObj$excelLog <<- read.table(file = temp.fileName, header = TRUE, sep = ",", na.strings = "NA", dec = ".")
}

#load the model data
load.modelDat <- function(fileName){
  temp.fileName <- paste(c("./dat/", fileName, ".ROCKYdata"), sep = "", collapse = "")
  temp.dat <- read.table(file = temp.fileName, header = TRUE, sep = ",", na.strings = "NA", dec = ".")
  resObj$dat <<- temp.dat
  resObj$datBackup <<- temp.dat
}

#update log in Excel to reflect loaded model
load.update.excel <- function(){
  shell(cmd = "./VBScripts/Load_ClearLog.vbs", translate = TRUE) #clear log
  copy.table(obj = resObj$excelLog, col.names = FALSE) #copy read in log
  shell(cmd = "./VBScripts/AddToLog/AddToLog_LoadModel.vbs", translate = TRUE) #paste in new log
  push.ModelParameters() #force-push parameters
}

#update ROCKY to reflect loaded model
load.update.ROCKY <- function(fileName){
  currentID <- length(resObj$excelLog$ID)
  load.from.log(log.ID = currentID, addToLog = FALSE, msgBox = FALSE, plots = FALSE)
  myComment <- paste(c("Loaded from model saved under the name: ", fileName), sep = "", collapse = "")
  add.to.log(IsNewModel = FALSE, Comment = myComment)
}

#load settings
load.ROCKY.settings <- function(){
  load.fileName <- "./set/user_settings.ROCKYsettings"
  attach(what = load.fileName)
}

#########################
##### 15. Save Plots ####
#########################

#main function of this section below--saves all plots so they can be imported into Excel
save.all.plots <- function(){
  save.residual.plots()
  save.structure.plots()
  save.diagnostic.plots()
  #push to Excel
}

# RESIDUAL PLOTS ####

#ay plot
save.ay.residual.plot <- function(){
  graphics.off()
  jpeg(filename = './www/Residual/AY.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  resPlot(resObj, option = 'AY', newWindow = FALSE)
  dev.off()
}

#dy plot
save.dy.residual.plot <- function(){
  graphics.off()
  jpeg(filename = './www/Residual/DY.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  resPlot(resObj, option = 'DY', newWindow = FALSE)
  dev.off()
}

#cy plot
save.cy.residual.plot <- function(){
  graphics.off()
  jpeg(filename = './www/Residual/CY.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  resPlot(resObj, option = 'CY', newWindow = FALSE)
  dev.off()
}

#all 4 plots
save.all.residual.plot <- function(){
  graphics.off()
  jpeg(filename = './www/Residual/all.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  resPlot(resObj, option = 'all', newWindow = FALSE)
  dev.off()
}

#do all 4
save.residual.plots <- function(){
  save.ay.residual.plot()
  save.dy.residual.plot()
  save.cy.residual.plot()
  save.all.residual.plot()
}

# MODEL STRUCTURE PLOTS ####

#ay plot
save.ay.structure.plot <- function(){
  graphics.off()
  jpeg(filename = './www/ModelStructure/AY.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  trendVarPlot(resObj, option = 'AY', newWindow = FALSE)
  dev.off()
}

#cy plot
save.cy.structure.plot <- function(){
  graphics.off()
  jpeg(filename = './www/ModelStructure/CY.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  trendVarPlot(resObj, option = 'CY', newWindow = FALSE)
  dev.off()
}

#dy plot
save.dy.structure.plot <- function(){
  graphics.off()
  jpeg(filename = './www/ModelStructure/DY.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  trendVarPlot(resObj, option = 'DY', newWindow = FALSE)
  dev.off()
}

#all 4 plots
save.all.structure.plot <- function(){
  graphics.off()
  jpeg(filename = './www/ModelStructure/all.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  trendVarPlot(resObj, option = 'all', newWindow = FALSE)
  dev.off()
}

# do all 4 at once
save.structure.plots <- function(){
  save.ay.structure.plot()
  save.dy.structure.plot()
  save.cy.structure.plot()
  save.all.structure.plot()
}

# NORMAL DIAGNOSTIC PLOTS ####
# all plots
save.all.normal.plot <- function(){
  graphics.off()
  jpeg(filename = './www/NormalDiagnostic/all.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  normPlots(resObj, newWindow = FALSE)
  dev.off()
}

#QQ plot
save.normal.QQplot <- function(){
  graphics.off()
  jpeg(filename = './www/NormalDiagnostic/QQ.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  diagnostic.plots.normal.QQ(newWindow = FALSE, pushToExcel = TRUE)
  dev.off()
}

#box plot
save.normal.box.plot <- function(){
  graphics.off()
  jpeg(filename = './www/NormalDiagnostic/box.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  diagnostic.plots.box.plot(newWindow = FALSE, pushToExcel = TRUE)
  dev.off()
}

#histogram
save.normal.histogram <- function(){
  graphics.off()
  jpeg(filename = './www/NormalDiagnostic/hist.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  diagnostic.plots.histogram(newWindow = FALSE, pushToExcel = TRUE)
  dev.off()
}

#density
save.normal.density <- function(){
  graphics.off()
  jpeg(filename = './www/NormalDiagnostic/density.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  diagnostic.plots.kernel.density(newWindow = FALSE, pushToExcel = TRUE)
  dev.off()
}

#all 4 at once
save.diagnostic.plots <- function(){
  save.normal.box.plot()
  save.normal.density()
  save.all.normal.plot()
  save.normal.histogram()
  save.normal.QQplot()
}

# AUTOVALIDATION PLOTS ####

#auto validation plot
save.auto.validation.plot <- function(){
  assign(x = 'resObj', value = CYstabilityCalc(resObj, CYholdout = ROCKY.settings$autoValidation$NumOfYears), inherits = TRUE)
  graphics.off()
  jpeg(filename = './www/AutoValidation/AV.jpeg', quality = ROCKY.settings$output$plotQuality, width = ROCKY.settings$output$plotWidth, height = ROCKY.settings$output$plotWidth)
  plotCYstability(resObj, newWindow = FALSE)
  dev.off()
}



#############################################
##### 16. Custom Future CY Functionality ####
#############################################

#toggle custom CY trend on
include.custom.CY.trend <- function(TrueOrFalse = TRUE){
  #toggle custom/modeled CY trend
  if(TrueOrFalse){
    resObj$ModelSettings$FutureCYTrend$Source <<- "Custom"
    read.custom.CY.trend()
    update.trend.parameters(comment = "Updated to include the custom CY trend", writeToLog = TRUE, msgbox = FALSE, plots = FALSE)
    shell(cmd = "./VBScripts/Messages/Update_CY_Trend.vbs", translate = TRUE)
  }
  else{
    resObj$ModelSettings$FutureCYTrend$Source <<- "Model"
    update.trend.parameters(comment = "Updated to exclude the custom CY trend", writeToLog = TRUE, msgbox = FALSE, plots = FALSE)
    shell(cmd = "./VBScripts/Messages/Exclude_CY_Trend.vbs", translate = TRUE)
  }
}

#toggle custom CY trend error on
include.custom.CY.se <- function(TrueOrFalse = TRUE, StdError = 0.1){
  #toggle custom/modeled CY trend
  if(TrueOrFalse){
    resObj$ModelSettings$FutureCYstdError$Source <<- "Custom"
    resObj$ModelSettings$FutureCYstdError$Input <<- StdError
    resObj$CyProjStdErrorSelect$stdError[1] <<- StdError
    update.trend.parameters(comment = "Updated to include the custom CY SE", writeToLog = TRUE, msgbox = FALSE, plots = FALSE)
    shell(cmd = "./VBScripts/Messages/Update_CY_se.vbs", translate = TRUE)
  }
  else{
    resObj$ModelSettings$FutureCYstdError$Source <<- "Model"
    update.trend.parameters(comment = "Updated to exclude the custom CY SE", writeToLog = TRUE, msgbox = FALSE, plots = FALSE)
    shell(cmd = "./VBScripts/Messages/Exclude_CY_se.vbs", translate = TRUE)
  }
}

#toggle custom CY trend off
exclude.custom.CY.trend <- function(){
  include.custom.CY.trend(TrueOrFalse = FALSE)
}

#toggle custom CY trend error off
exclude.custom.CY.se <- function(){
  include.custom.CY.se(TrueOrFalse = FALSE)
}

#read in custom CY trend
read.custom.CY.trend <- function(){
  save.workbook()
  resObj$CyProjTrendSelect <<- data.frame(read_excel(path = selectionsFile, sheet = 'CYpt'))
}

##############################
##### 17. Forecast Extras ####
##############################

#more convenient way to return forecast data
forecastDat <- function(){
  return(resObj$forecast$inc)
}

make.forecast.triangle <- function(){
  dat <- forecastDat()
  numCol <- max(dat$DY)
  numRow <- max(dat$AY) - min(dat$AY) + 1
  
  andy <- data.frame()
  for(i in 1:numCol){
    for(j in 1:numRow){
      myAY <- min(dat$AY)+j-1
      andy$i[j] <- dat$P[dat$AY==myAY && dat$DY==i]
    }
  }
  
  return(andy)
  
}

###############################
##### 18. More Diagnostics ####
###############################

all.model.diagnostics <- function(){
  
  #returns a list with many different diagnostic stats
  
  df$multiple.R.squared <- summary(resObj$model)$r.squared
  df$adjusted.R.squared <- summary(resObj$model)$adj.r.squared
  df$residual.sum.of.squares <- sum((summary(resObj$model)$residuals)^2)
  df$f.statistic <- summary(resObj$model)$fstatistic
  df$residual.std.error <- summary(resObj$model)$sigma
  
  return(df)
  
}


#######################
##### A. Utilities ####
#######################
#####

#save the Reserve Model Selections workbook
save.workbook <- function(){
  shell(cmd = "./VBScripts/Util/SaveWorkbook.vbs", translate = TRUE)
}

model.tag <- function(){
  if(ROCKY.settings$selected.model=="loglinear"){
    return("(LL)")
  }
  else if(ROCKY.settings$selected.model=="GLM"){
    if(ROCKY.settings$GLM$selected.model=="ODP"){
      return("(ODP GLM)")
    }
    else if(ROCKY.settings$GLM$selected.model=="gamma"){
      return("(gamma GLM)")
    }
  }
}

##############
##### END ####
##############