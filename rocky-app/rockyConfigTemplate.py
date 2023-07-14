"""
# rockyConfig
# ===========
The rockyConfig.py file contains all the parameters that are used by rocky. You
can change the values here to suit your needs. This file represents the default/
recommended settings. You can change the values here if needed. 

This file represents a fully-specified rocky model. The file is all that is 
needed to recreate any specific rocky model.
"""

rockyConfig = {
    ## MODEL NAME
    "modelname": "tweediecal", # Change model name here if you want - this is how
                               # python will refer to it
    
    ## LOADING DATA
    "triangle_filename": "triangle.xlsx", # Location of triangle excel file
    "triangle_sheetname": "triangle",     # Name of sheet in triangle excel file
    
    ## MODEL PARAMETERS
    "model": "loglinear", # Model type - current options are "loglinear", "tweedie",
                          # though more can be added

    "model_parameters_filename": "model_parmeters.xlsx", # Location of model parameter
                                                         # excel file
    "model_parameters_sheetname": "param",    # Name of sheet in model parameter excel
                                              # file

    ## HYPERPARAMETER TUNING - controls all the parameters that aren't directly
    ##                         related to describing the triangle data
    "hyperparameter_tuning": "False", # If True, will run hyperparameter tuning. Only
                                      # need to do this once per triangle. If False,
                                      # will use the hyperparameters specified below
                                      # to run the model

    # after hyperparameter tuning, the best hyperparameters will be saved in the

    "hyperparameters": {
        "loglinear": { # Hyperparameters for loglinear model
            
            "l1_ratio": 0.5,  # l1_ratio for elastic net, between 0 and 1
                              # describes the weight of the l1 penalty relative
                              # to the l2 penalty

            "alpha": 0.5,     # alpha for elastic net - controls the strength of the
                              # regularization. Higher values mean more regularization
                              # and simpler models. Lower values mean less regularization
                              # and more complex models
        },

        "tweedie": { # Hyperparameters for tweedie model
            "power": 1.5,     # Power parameter for tweedie model. Must be between
                              # 1 and 3
            "alpha": 0.5,     # alpha for l2 regularization - controls the strength of the
                              # regularization. Higher values mean more regularization
                              # and simpler models. Lower values mean less regularization
                              # and more complex models
        }

    },


    ## FORECAST PARAMETERS
    "forecast_filename": "future_cy.xlsx", # Location of future calendar year excel file
    
    ## OUTPUT
    "output_notebook": "output.html", # Output notebook will be saved here
    "output_filename": "output.xlsx", # Output excel file will be saved here

    ## SESSION PARAMETERS
    "log": "session.json", # Where to save the session log file that can be used to
                           # resume a session or to go back to a previous state

    "overwrite": "False", # If True, will overwrite the output notebook and excel
                          # file without asking for confirmation
}
