


####################### 
## How To Use ROCKY: ##
#######################  
# 
#   1. Open the Reserve Model Selections.xlsm file in this directory (or can use your own copy of this file)
#   2. Click the button in the upper-right-hand corner of this window that says, "Run App"
# 
































selectionsFile <- 'Reserve Model Selections.xlsm'
new.modelName <- "Model Name?"
load("./set/user_settings.ROCKYsettings")
availableModels <- sub(pattern = ".ROCKYmodel", replacement = "", x = list.files(path = './model/'))
lineTypes <- c('blank', 'solid', 'dashed', 'dotted', 'dotdash', 'longdash', 'twodash')
plotSymbolTypes <- c(0:20)
