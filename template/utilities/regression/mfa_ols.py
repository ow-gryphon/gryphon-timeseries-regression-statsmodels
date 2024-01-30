# Import key libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.tools import eval_measures
import scipy as sc
from scipy.special import logsumexp
from collections import OrderedDict
from itertools import combinations
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey, het_white
import traceback

import sys
from . import stats_utilities as functions


def raw_combinations(IVs, num_vars=4, forced_in=None):
        
    # Generate all combinations of independent variables
    var_combinations = []
    
    if forced_in is not None:
        IVs = [var for var in IVs if var not in forced_in]
        var_combinations.append(forced_in)
    
    # Generate models with 1 variable
    for var in IVs:
        var_combinations.append([var])
    
    # Generate models with x variables
    var_counter=1
    for num in range(num_vars+1):
        if num >= var_counter+1:
            var_combinations.extend(list(combinations(IVs, num)))
            var_counter=var_counter+1
            
    # Create a DataFrame from the model combinations
    if forced_in is not None:
        models_df = pd.DataFrame(var_combinations, columns=[f'Var{i+1+len(forced_in)}' for i in range(num_vars)])
        
        for i, forced_var in enumerate(forced_in):
            models_df[f'Var{i+1}'] = forced_var
            
        keep_idx = []
        for idx, row in models_df.iterrows():
            
            var_series = pd.Series(row.values)
            var_series = var_series[var_series!='']
            var_counts = var_series.value_counts()
            
            if any(var_counts > 1):
                pass
            else:
                keep_idx.append(idx) 
            
        models_df = models_df.loc[keep_idx].reset_index(drop=True)
    else:
        models_df = pd.DataFrame(var_combinations, columns=[f'Var{i+1}' for i in range(num_vars)])
    
    return models_df    
    

def find_combinations(IVs, num_vars, mapping=None, max_per_variable=1, max_per_category=2, only_one_transform=True, forced_in=None):

    # Get raw combinations
    combinations = raw_combinations(IVs, num_vars, forced_in)
    
    # if mapping is None:
    #     print("Since mapping is not provided, data type is manually inferred")
    combinations = combinations.fillna("||")
    
    if forced_in is None:
        forced_in = []
    
    for num in range(num_vars + len(forced_in)):
        combinations[f"{num+1} Variable"] = combinations[f'Var{num+1}'].astype('str').str.split("|").str[0]
        combinations[f"{num+1} Transform"] = combinations[f'Var{num+1}'].astype('str').str.split("|").str[1]
        combinations[f"{num+1} Lag"] = combinations[f'Var{num+1}'].astype('str').str.split("|").str[2]
        combinations[f"{num+1} VarTransform"] = combinations[f"{num+1} Variable"] + "|" + combinations[f"{num+1} Transform"]
        
        combinations[f'Var{num+1}'] = np.where(combinations[f'Var{num+1}'] == "||", None, combinations[f'Var{num+1}'])
        combinations[f'{num+1} VarTransform'] = np.where(combinations[f'{num+1} VarTransform'] == "|", None, combinations[f'{num+1} VarTransform'])
    
    # filtering by category is not yet supported
    
    combinations['check'] = True
    
    var_columns = [col for col in combinations.columns if col.endswith(" Variable")]
    transform_columns = [col for col in combinations.columns if col.endswith(" Transform")]
    
    for idx, row in combinations.iterrows():
        
        var_series = pd.Series(row[var_columns])
        var_series = var_series[var_series!='']
        var_counts = var_series.value_counts()
        
        if any(var_counts > max_per_variable):
            combinations.loc[idx, 'check'] = False
        
        var_transforms = pd.DataFrame({"Var": row[var_columns].values, "Transforms": row[transform_columns].values})
        var_transforms = var_transforms[var_transforms != '']
        var_transform_counts = var_transforms.drop_duplicates().groupby("Var").size()
        
        if only_one_transform and any(var_transform_counts > 1):
            combinations.loc[idx, 'check'] = False
    
    combinations = combinations.loc[combinations['check']].reset_index(drop=True)
    
    return combinations[[f'Var{i+1}' for i in range(num_vars+len(forced_in))]]
    

def find_incremental(starting_IVs, candidate_IVs, mapping=None, max_per_variable=2, max_per_category=2, only_one_transform=False):
    
    incremental_variables = pd.DataFrame({"Var1": candidate_IVs})
    starting_IVs.columns = [f'Var{i+2}' for i in range(starting_IVs.shape[1])]
    
    # Get raw combinations
    combinations = incremental_variables.merge(starting_IVs, how='cross')
    
    keep_idx = []
    for idx, row in combinations.iterrows():
        
        var_series = pd.Series(row.values)
        var_series = var_series[var_series!='']
        var_counts = var_series.value_counts()
        
        if any(var_counts > 1):
            pass
        else:
            keep_idx.append(idx) 
            
    combinations = combinations.loc[keep_idx].reset_index(drop=True)
    
    # if mapping is None:
    #     print("Since mapping is not provided, data type is manually inferred")
    combinations = combinations.fillna("||")
    
    for num in range(combinations.shape[1]):
        combinations[f"{num+1} Variable"] = combinations[f'Var{num+1}'].astype('str').str.split("|").str[0]
        combinations[f"{num+1} Transform"] = combinations[f'Var{num+1}'].astype('str').str.split("|").str[1]
        combinations[f"{num+1} Lag"] = combinations[f'Var{num+1}'].astype('str').str.split("|").str[2]
        combinations[f"{num+1} VarTransform"] = combinations[f"{num+1} Variable"] + "|" + combinations[f"{num+1} Transform"]
        
        combinations[f'Var{num+1}'] = np.where(combinations[f'Var{num+1}'] == "||", None, combinations[f'Var{num+1}'])
        combinations[f'{num+1} VarTransform'] = np.where(combinations[f'{num+1} VarTransform'] == "|", None, combinations[f'{num+1} VarTransform'])
    
    # filtering by category is not yet supported
    
    combinations['check'] = True
    
    var_columns = [col for col in combinations.columns if col.endswith(" Variable")]
    transform_columns = [col for col in combinations.columns if col.endswith(" Transform")]
    
    for idx, row in combinations.iterrows():
        
        var_series = pd.Series(row[var_columns])
        var_series = var_series[var_series!='']
        var_counts = var_series.value_counts()
        
        if any(var_counts > max_per_variable):
            combinations.loc[idx, 'check'] = False
        
        var_transforms = pd.DataFrame({"Var": row[var_columns].values, "Transforms": row[transform_columns].values})
        var_transforms = var_transforms[var_transforms != '']
        var_transform_counts = var_transforms.drop_duplicates().groupby("Var").size()
        
        if only_one_transform and any(var_transform_counts > 1):
            combinations.loc[idx, 'check'] = False
    
    
    combinations = combinations.loc[combinations['check']].reset_index(drop=True)
    
    return combinations[[f'Var{i+1}' for i in range(starting_IVs.shape[1]+1)]]





default_options_ols = {"Ljung Box Lags": 4, "Breusch-Godfrey Lags": 4}
def mfa_ols(dataset, oos_dataset, DV, IVs, intercept=True, robust="HC3", get_fitted=True, timeseries=False, detailed=False,
            test_options=default_options_ols):
    '''
    Perform OLS regression on a set of independent variables
    :param dataset: pandas dataframe
    :param DV: name of target variable (dependent variable)
    :param IVs: list of names of independent variables to be evaluated one by one
    :param intercept: Boolean indicating whether of not to include intercept
    :param robust: whether to also get robust p-values. Use None for 'no'. Alternatively, use "HC0", "HC1", "HC2", "HC3"
    :param get_fitted: boolean whether to get fitted values
    :param timeseries: boolean whether the data is in a time-series format (already sorted). Gaps due to missing values will be
    flagged, but will not be accounted for in test statistics
    :param detailed: boolean whether to produce detailed test results and charts
    :param test_options: dictionary with test options
    :return: dictionary with 'results' and 'fitted_values' (if requested)
    '''


    # Check inputs and reformat if necessary
    if DV is None:
        raise ValueError("You must include DV")
    if IVs is None:
        raise ValueError("You must include IVs")

    if isinstance(IVs, str):
        IVs = [IVs]

    model_dataset = dataset[[DV] + IVs].dropna()
    kept_index = model_dataset.index.values

    X = model_dataset[IVs]
    Y = model_dataset[DV]

    if intercept:
        X = sm.add_constant(X)

    model = sm.OLS(Y, X)
    results = model.fit()

    predicted = results.fittedvalues.values
    residuals = results.resid

    oos_dataset = oos_dataset[[DV] + IVs].dropna()
    
    oos_X = oos_dataset[IVs]
    oos_Y = oos_dataset[DV]

    if intercept:
        oos_X = sm.add_constant(oos_X)
    
    # Gain predictions of DV with out-of-sample IVs
    oos_predictions= results.predict(oos_X)
    mean_observed = np.mean(oos_Y)
    tss = np.sum((oos_Y - mean_observed) ** 2)
    rss = np.sum((oos_Y - oos_predictions) ** 2)
    r_squared= 1 - (rss / tss)
    
    # Generate outputs
    results_dict = OrderedDict()
    for i in range(len(IVs)):
        IV = IVs[i]
        results_dict['Var {}'.format(i + 1)] = IV
        results_dict['Var {} Coef'.format(i + 1)] = results.params.iloc[i+1]
        results_dict['Var {} p-val'.format(i + 1)] = results.pvalues.iloc[i+1]

        if robust in ["HC0", "HC1", "HC2", "HC3"]:
            results_dict['Var {} {} p-val'.format(i+1, robust)] = 2 * sc.stats.t.cdf(
                -abs(np.exp(
                    logsumexp(results.params[IV]) - logsumexp(results.__getattribute__(robust + "_se")[IV])
                )), df=results.df_resid)

    if intercept:
        results_dict['Intercept'] = results.params['const']
        results_dict["Intercept p-val"] = results.pvalues['const']

        if robust in ["HC0", "HC1", "HC2", "HC3"]:
            results_dict['Intercept {} p-val'.format(robust)] = 2 * sc.stats.t.cdf(
                -abs(np.exp(
                    logsumexp(results.params['const']) - logsumexp(results.__getattribute__(robust + "_se")['const'])
                )), df=results.df_resid)


    results_dict['# Obs'] = results.nobs
    results_dict['# Miss'] = len(Y) - results.nobs

    results_dict["Rsq"] = results.rsquared
    results_dict["Adj Rsq"] = results.rsquared_adj
    results_dict["MSE"] = eval_measures.rmse(Y, results.fittedvalues.values)**2
    results_dict["RMSE"] = eval_measures.rmse(Y, results.fittedvalues.values)
    results_dict["AIC"] = results.aic
    results_dict["BIC"] = results.bic
    
    results_dict["OOS Rsq"] = r_squared
    
    # Statistical tests
    #VIF = vif(model_dataset[IVs].to_numpy())
    vif_input=sm.add_constant(model_dataset[IVs])
    VIF= [variance_inflation_factor(vif_input.values, i) for i in range(0, vif_input.shape[1])]
    results_dict["Max_VIF"] = max(VIF[1:])

    # Heteroscedasticity tests
    lm, lm_p, f, f_p = het_breuschpagan(residuals, X.to_numpy(), robust = True)
    results_dict["BP p-val"] = lm_p

    # Normality tests
    results_dict["JB p-val"] = sc.stats.jarque_bera(residuals).pvalue
    results_dict["SW p-val"] = sc.stats.shapiro(residuals).pvalue

    # Time-series checks
    if timeseries:
        # Check for gaps within the data
        results_dict["Has Gap"] = "".join([" " if isnan else "a" for isnan in dataset[[DV] + IVs].apply(lambda x: any(np.isnan(x)), axis = 1)]).strip().find(' ') >= 0

        # Serial correlation
        if test_options.get("Ljung Box Lags") is not None:
            ljung_box_results = sms.diagnostic.acorr_ljungbox(residuals, test_options["Ljung Box Lags"])
            results_dict["LjungBox min p-val"] = min(ljung_box_results.lb_pvalue)

        if test_options.get("Breusch-Godfrey Lags") is not None:
            lm, lm_p, fval, fval_p = acorr_breusch_godfrey(results, test_options.get("Breusch-Godfrey Lags"))
            results_dict["BG p-val"] = fval_p

        DW_table = functions.durbin_watson(results)
        results_dict["DW p-val"] = DW_table.loc[DW_table['Item']=='two-sided p-value', 'Value'].values[0]
        
    if detailed:
        detailed_results = OrderedDict()
        detailed_results['VIF'] = pd.DataFrame({"IV": IVs, "VIF": VIF[1:]})
        detailed_results['Cooks_distance'] = functions.cooks_distance(residuals, X.to_numpy())

        if timeseries:
            if test_options.get("Ljung Box Lags") is not None:
                detailed_results["LjungBox"] = ljung_box_results
            
            if test_options.get("Breusch-Godfrey Lags") is not None:
                detailed_results['Breusch-Godfrey'] = functions.breusch_godfrey(reg=results, n_lags = test_options.get("Breusch-Godfrey Lags"))
            
            detailed_results['DW'] = DW_table
            
        regression_object = model

    else:
        detailed_results = None
        regression_object = None

    # Fitted values
    if get_fitted:
        fitted_values = pd.DataFrame({"kept_index": kept_index, "fit": predicted})

    if get_fitted is False:
        fitted_values = None

    return {
        "summary": results_dict,
        "fitted": fitted_values,
        "detailed": detailed_results,
        "model": regression_object
    }


def mfa_ols_wrapper(dataset, oos_dataset, DV, IV_table, intercept=True, robust="HC3", get_fitted=True, timeseries=False,
            test_options=default_options_ols):
    '''
    Perform OLS regression on individual independent variables, with optional forced in variables
    :param dataset: pandas dataframe
    :param DV: name of target variable (dependent variable)
    :param IV_table: pandas table where each row contains the variables in a model
    :param intercept: Boolean indicating whether of not to include intercept
    :param robust: whether to also get robust p-values. Use None for 'no'. Alternatively, use "HC0", "HC1", "HC2", "HC3"
    :param get_fitted: boolean whether to get fitted values
    :param timeseries: whether the data is in a time-series format (already sorted). Gaps due to missing values will be
    flagged, but will not be accounted for in test statistics
    :param test_options: dictionary with test options
    :return: dictionary with 'results' and 'fitted_values' (if requested)
    '''


    # Check inputs and reformat if necessary
    if DV is None:
        raise ValueError("You must include DV")
    if IV_table is None:
        raise ValueError("You must include IV table")

    num_var = IV_table.shape[1]

    # Set up the result table in Pandas
    col_info = OrderedDict()

    # Variable names
    for i in range(num_var):
        col_info['Var {}'.format(i+1)] = pd.Series([], dtype='str')

    col_info['# Obs'] = pd.Series([], dtype='int')
    col_info['# Miss'] = pd.Series([], dtype='int')

    # Coefficients
    if intercept:
        col_info['Intercept'] = pd.Series([], dtype='float')
    for i in range(num_var):
        col_info['Var {} Coef'.format(i+1)] = pd.Series([], dtype='float')

    # P-values
    if intercept:
        col_info["Intercept p-val"] = pd.Series([], dtype='float')
    for i in range(num_var):
        col_info["Var {} p-val".format(i+1)] = pd.Series([], dtype='float')

    if robust in ["HC0", "HC1", "HC2", "HC3"]:
        if intercept:
            col_info['Intercept ' + robust + " p-val"] = pd.Series([], dtype='float')
        for i in range(num_var):
            col_info['Var {} {} p-val'.format(i+1,robust)] = pd.Series([], dtype='float')

    # Model fit
    col_info["Rsq"] = pd.Series([], dtype='float')
    col_info["Adj Rsq"] = pd.Series([], dtype='float')
    col_info["MSE"] = pd.Series([], dtype='float')
    col_info["RMSE"] = pd.Series([], dtype='float')
    col_info["AIC"] = pd.Series([], dtype='float')
    col_info["BIC"] = pd.Series([], dtype='float')
    col_info["OOS Rsq"] = pd.Series([], dtype='float')

    # Statistical tests
    col_info["Max_VIF"] = pd.Series([], dtype='float')
    col_info["BP p-val"] = pd.Series([], dtype='float') # Breusch-Pagan
    col_info["JB p-val"] = pd.Series([], dtype='float') # Jarque-Bera
    col_info["SW p-val"] = pd.Series([], dtype='float') # Shapiro-Wilks

    if timeseries:
        col_info["Has Gap"] = pd.Series([], dtype='bool')

        if test_options.get("Ljung Box Lags") is not None:
            col_info["LjungBox min p-val"] = pd.Series([], dtype='float')

        if test_options.get("Breusch-Godfrey Lags") is not None:
            col_info["BG p-val"] = pd.Series([], dtype='float')

        col_info["DW p-val"] = pd.Series([], dtype='float')


    # Create the pandas
    # output = pd.DataFrame(col_info)
    output = []
    
    # Create Pandas for fitted values
    fitted_values = dataset[[DV]].copy()
    fitted_values.columns = ["Actual"]

    # Loop through model table
    for i in range(IV_table.shape[0]):
    
        if i%10 == 0:
            print(f"Working on Model #{i+1} out of {IV_table.shape[0]}")
    
        IV_list = IV_table.loc[i,:]

        # Remove None and blanks
        IV_list = [x for x in IV_list if x is not None]
        IV_list = [x for x in IV_list if x != 'NA']
        IV_list = [x for x in IV_list if x != ""]

        try:
            # Execute main function
            reg_results = mfa_ols(dataset, oos_dataset, DV, IV_list, intercept, robust, get_fitted, timeseries, False, test_options)

            # Add results to table
            output.append(pd.DataFrame(reg_results['summary'], index=[i]))

            # Fitted values
            if get_fitted:
                predictions = reg_results['fitted']
                fitted_values["Model {}".format(i+1)] = np.nan
                fitted_values.loc[predictions['kept_index'], "Model {}".format(i+1)] = predictions['fit']
        except:
            print("Model {} was not able to execute.".format(i+1))
            traceback.print_exc()

    if get_fitted is False:
        fitted_values = None
    
    output = pd.concat(output)[col_info.keys()]
    #output = pd.concat(output)[[col for col in col_info.keys() if col in output.columns]]
    #Future enhancement to enforce data type for each column

    return {
        "results": output,
        "fitted": fitted_values
    }

