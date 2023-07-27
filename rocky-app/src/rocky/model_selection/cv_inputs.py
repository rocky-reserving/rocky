from sklearn.linear_model import TweedieRegressor, ElasticNet, Lasso, Ridge, LinearRegression

param_lookup = {'tweedie': {'alpha':0.5,'power':1},
                'loglinear': {'alpha':0.5,'l1_ratio':0.1}}

model_lookup = {'tweedie': TweedieRegressor}

def _get_blank_loglinear(alpha=None, l1_ratio=None):
    try:
        if alpha==0:
            model = LinearRegression
        elif l1_ratio==0:
            model = Ridge
        elif l1_ratio==1:
            model = Lasso
        else:
            model = ElasticNet
    except TypeError:
        model = ElasticNet

    return model

def cv_inputs(model_type):
    output = {
        'model_type': model_type,
        'params': param_lookup[model_type],
    }
    
    return output

def cv_blank_model(model_type, **kwargs):
    if model_type=='loglinear':
        model = _get_blank_loglinear(**kwargs)
    else:
        model = model_lookup[model_type]
    return model