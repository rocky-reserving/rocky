###################################################################################################
##### RESERVE MODELING WITH ROCKY:  ANDY'S WORKFLOW UPDATE ########################################
###################################################################################################

# To do: (as of 1/7/2016)
#   4. Pare down file size
#   5. Pull settings from Excel file
#     a. Does this let us mess around with the outlier removal more?
#     b. Maybe add more graph customization?
#     c. Should the user put the model filename to save/load in the Excel file, rather than in R?
#   6. Look at auto-validation graph inside R
#   7. Create custom VBScripts on-the-fly, based on user-selected Excel file name
#   1. Incorporate new hetero method (DONE 1/8/2016)
#   2. Custom future CY trend/variance (DONE 2/9/2016)
#   3. Force a zero trend (Seems to be done 2/11/2016)

##### Instructions/Reminders ######################################################################

# DISCLAIMER:
#   This is still in a pre-production stage, and there are bound to be issues here and 
#   there. If it looks like ROCKY has broken, the best thing I have found is to load from a previous
#   stable version of the model from the log, using the function below, where the number in the 
#   parentheses is the log ID from the Excel log tab. 


# INSTRUCTIONS/NOTES:
# 1. To run any command, put the cursor on that line with that command, and press Ctrl + Enter
# 2. Make sure the Excel file "Reserve Model Selections.xlsx" (located in the same directory
#    as the ROCKY project) is open. 
# 3. DO NOT (and I repeat) DO NOT change the name of this Excel file. I (Andy) am not 
#    sophisticated enough just yet to allow you to do that!
# 4. If you would like to make any notes to yourself in this script file, put a '#' before 
#    whatever you write. This will make your note a comment, and even if you ask, R won't read it
# 5. There are special instructions in the 'Outlier' section below



##### Preamble ####################################################################################

#run all five of these at the start of the session

rm(list = ls()) #clear memory
gc() # clean up garbage
selectionsFile <- 'Reserve Model Selections.xlsm' #Source data/selections file. .xlsx/.xlsm format
suppressPackageStartupMessages(suppressWarnings(source("ROCKY_source_20160329.R"))) #load in ROCKY (and ADRIAN) source
load("./set/user_settings.ROCKYsettings")
new.model() #create a blank loglinear model

change.loglinear.model() #change to fitting a loglinear model & re-fit the model

change.glm.model(model = "odp") #change to fitting a glm, type def'd by model parameter, defaults to odp, can select one of ('odp', 'gamma', 'gaussian')
change.glm.model(model = "gamma")
change.glm.model(model = "gaussian") #as should probably be expected, this produces (apparently) the same results as the loglinear model

# new.model(selectedModel = "GLM", selectedGLMModel = "odp") #create a blank ODP GLM model


update.settings() #tab inside the parentheses to see the possible options


##### Model File Options ##########################################################################

load.model("4Q2014 Paid Loss") #load the model indicated in the quotes

save.model("4Q2014 Paid Loss") #save the model, with model name indicated in the quotes

load.from.log(5) #load the model with logID indicated in the parentheses



##### Parameter Update ############################################################################

#------  Individual Residual Graphs ---------------------------------------------------------------

DY.residual.graph() #see DY residuals on a single graph

CY.residual.graph() #see CY residuals on a single graph

AY.residual.graph() #see AY residuals on a single graph

all.four.residual.graphs() #see the normal 2x2 set of graphs


#------ Individual Trend Parameter Updates --------------------------------------------------------

update.DYGp.trend.parameters() #run after updating DYGp tab in Excel

update.CYGp.trend.parameters() #run after updating CYGp tab in Excel

update.AYGp.trend.parameters() #run after updating AYGp tab in Excel


#------ Update All Trend Parameters at Once -------------------------------------------------------

update.trend.parameters() #update all at once



##### Load Model from Excel Log ###################################################################

load.from.log(log.ID = 5) #load the model with the Excel 'log' ID number in the parentheses



##### Heteroscedasticity adjustment ###############################################################

automatic.hetero.adjustment() #let ROCKY update DYwGp tab, then calculate the weights on DYw tab 

custom.group.hetero.adjustment() #update the DYwGp tab, and ROCKY will calculated the weights (previously named auto hetero adjustment)

manual.hetero.adjustment() #update DYw tab manually, and ROCKY will use your selected weights 

clear.hetero.adjustment() #leave DYwGp tab as it is, but clear the DYw tab, and re-fit the model

reset.hetero.adjustment() #clears hetero adjustment, and resets (for after you update model later)                          



##### Variable Optimizer ##########################################################################

automatic.variable.optimizer() #iteratively remove all non-significant variables

one.step.variable.optimizer() #run a single step of the auto optimizer



##### Outliers ####################################################################################

# NOTES:
#  - in the remove.outliers() function, if there is nothing in the parentheses, it will default to 
#    looking at outliers by DY. If you want to make the x-axis of the graph CY or AY, put CY or AY
#    inside the parentheses
#  - when the window opens, you can just click on the outliers
#  - WARNING: DO NOT try to close this window by X'ing out of the window. Close the outlier 
#    selection process by right-clicking -> Stop

remove.outliers('DY') #click on the points to exclude

add.outliers.back.in() #re-include all outliers




##### AY Filters ##################################################################################

automatic.AY.filtering() #let ROCKY determine the values on the 'AYgpFilters' tab for you

manual.AY.filtering() #read in manually inputed filters from 'AYgpFilters' tab

clear.AY.filtering() #clear AY filters 




##### Automatic Validation ########################################################################

automatic.validation() #remove CY's to see how the model fits--combo of following two functions

#------ Individual Steps --------------------------------------------------------------------------

automatic.validation.table() #only outputs table--no graph

automatic.validation.graph() #only outputs graph--no table




##### Graphs/Plots ################################################################################

#----- Residuals ----------------------------------------------------------------------------------

all.four.residual.graphs() #the 2x2 set of graphs that you normally see 

AY.residual.graph() #zoom in to AY graph

DY.residual.graph() #zoom in to DY graph

CY.residual.graph() #zoom in to CY graph

fitted.residual.graph() #zoom in to fitted log value graph

#----- Model Structure ----------------------------------------------------------------------------

model.structure.plot() #the 2x2 set of graphs you normally see

AY.model.structure.plot() #zoom in to AY graph

DY.model.structure.plot() #zoom in to DY graph

CY.model.structure.plot() #zoom in to CY graph

hetero.model.structure.plot() #zoom in to hetero weights graph




##### Diagnostics #################################################################################

diagnostic.plots() #all 4 normal diagnostic graphs we are used to looking at

#------ Separated Plots ---------------------------------------------------------------------------  

diagnostic.plots.normal.QQ() #just the normal Q-Q graph

diagnostic.plots.box.plot() #only the box plot

diagnostic.plots.histogram() #only the histogram

diagnostic.plots.kernel.density() #only the kernel density




##### Custom Future Trend #########################################################################

# Note that these options should not be used until you have already built the model, as 
# this will often create issues during the model fitting process (leading to negative SE) 

include.custom.CY.trend() #let ROCKY know that you want to use a custom CY trend ('CYpt' tab)

include.custom.CY.se(StdError = 0.0075) #let ROCKY know that you want to use a custom CY se

exclude.custom.CY.trend() #use the modeled future CY trend

exclude.custom.CY.se() #use the modeled future CY trend parameter SE




##### Forecast Output #############################################################################

push.summary.output() #make the same AY table we normally get from ICRFS


