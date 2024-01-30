# Import key libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import eval_measures
import scipy as sc
from scipy.special import logsumexp
from collections import OrderedDict
from warnings import simplefilter 
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from . import stats_utilities as functions


def sfa_ols(dataset, DV, IVs, forced_in=None, intercept=True, robust="HC3", get_fitted=True):
    '''
    Perform OLS regression on individual independent variables, with optional forced in variables
    :param dataset: pandas dataframe
    :param DV: name of target variable (dependent variable)
    :param IVs: list of names of independent variables to be evaluated one by one
    :param forced_in: optional list of forced in variables. All variables named here will be forced in
    :param intercept: Boolean indicating whether of not to include intercept
    :param robust: whether to also get robust p-values. Use None for 'no'. Alternatively, use "HC0", "HC1", "HC2", "HC3"
    :param get_fitted: boolean whether to get fitted values
    :return: dictionary with 'results' and 'fitted_values' (if requested)
    '''


    # Check inputs and reformat if necessary
    if DV is None:
        raise ValueError("You must include DV")
    if IVs is None:
        raise ValueError("You must include IVs")

    if forced_in is not None:
        if isinstance(forced_in, str):
            forced_in = [forced_in]

    # Set up the result table in Pandas
    col_info = OrderedDict()
    col_info['Variable'] = pd.Series([], dtype='str')
    col_info['# Obs'] = pd.Series([], dtype='int')
    col_info['# Miss'] = pd.Series([], dtype='int')
    if intercept:
        col_info['Intercept'] = pd.Series([], dtype='float')
    col_info['Var Coef'] = pd.Series([], dtype='float')

    if forced_in is not None:
        for forced_var in forced_in:
            col_info[forced_var + " Coef"] = pd.Series([], dtype='float')

    if intercept:
        col_info["Intercept p-val"] = pd.Series([], dtype='float')
    col_info["Var p-val"] = pd.Series([], dtype='float')

    if forced_in is not None:
        for forced_var in forced_in:
            col_info[forced_var + " p-val"] = pd.Series([], dtype='float')

    if robust in ["HC0", "HC1", "HC2", "HC3"]:
        if intercept:
            col_info['Int ' + robust + " p-val"] = pd.Series([], dtype='float')
        col_info['Var ' + robust + " p-val"] = pd.Series([], dtype='float')
        if forced_in is not None:
            for forced_var in forced_in:
                col_info[forced_var + " " + robust + " p-val"] = pd.Series([], dtype='float')

    col_info["Rsq"] = pd.Series([], dtype='float')
    col_info["Adj Rsq"] = pd.Series([], dtype='float')
    if forced_in is not None:
        col_info["Rsq vs. forced in"] = pd.Series([], dtype='float')

    col_info["MSE"] = pd.Series([], dtype='float')
    col_info["RMSE"] = pd.Series([], dtype='float')
    col_info["AIC"] = pd.Series([], dtype='float')
    col_info["BIC"] = pd.Series([], dtype='float')
    
    
    
    # Create the pandas
    output = pd.DataFrame(col_info)

    # Create Pandas for fitted values
    fitted_values = dataset[[DV]].copy()
    fitted_values.columns = ["Actual"]

    # If there are forced in variables, we run a regression with just the forced in variables
    if forced_in:

        model_dataset = dataset[[DV] + forced_in]
        model_dataset = model_dataset.dropna()
        kept_index = model_dataset.index.values

        X = model_dataset[forced_in]
        Y = model_dataset[DV]

        if intercept:
            X = sm.add_constant(X)

        model = sm.OLS(Y, X)
        results = model.fit()

        # Generate outputs
        results_dict = OrderedDict()

        results_dict['Variable'] = "None"
        results_dict['# Obs'] = results.nobs
        results_dict['# Miss'] = len(Y) - results.nobs
        if intercept:
            results_dict['Intercept'] = results.params['const']
        results_dict['Var Coef'] = 0

        if forced_in is not None:
            for forced_var in forced_in:
                results_dict[forced_var + " Coef"] = results.params[forced_var]

        if intercept:
            results_dict["Intercept p-val"] = results.pvalues['const']
        results_dict["Var p-val"] = None

        if forced_in is not None:
            for forced_var in forced_in:
                results_dict[forced_var + " p-val"] = results.pvalues[forced_var]

        if robust in ["HC0", "HC1", "HC2", "HC3"]:
            if intercept:
                results_dict['Int ' + robust + " p-val"] = 2 * sc.stats.t.cdf(
                    -abs(np.exp(
                        logsumexp(results.params['const']) - logsumexp(results.__getattribute__(robust + "_se")['const'])
                    )), df=results.df_resid)

            if forced_in is not None:
                for forced_var in forced_in:
                    results_dict[forced_var + " " + robust + " p-val"] = 2 * sc.stats.t.cdf(
                        -abs(np.exp(
                            logsumexp(results.params[forced_var]) - logsumexp(
                                results.__getattribute__(robust + "_se")[forced_var])
                        )), df=results.df_resid)

        results_dict["Rsq"] = results.rsquared
        results_dict["Adj Rsq"] = results.rsquared_adj
        results_dict["MSE"] = eval_measures.rmse(Y, results.fittedvalues.values)**2
        results_dict["RMSE"] = eval_measures.rmse(Y, results.fittedvalues.values)
        results_dict["AIC"] = results.aic
        results_dict["BIC"] = results.bic
        
        DW_table = functions.durbin_watson(results)
        results_dict["Durbin-Watson"] = DW_table.loc[DW_table['Item']=='two-sided p-value', 'Value'].values[0]
        results_dict['Var Std. Coef'] = (results_dict['Var Coef'] * np.std(X,axis=0) / np.std(Y)).iloc[1]
        
        output = output.append(results_dict, ignore_index=True)

        # Fitted values
        if get_fitted:
            fitted_values["ForcedInVars"] = np.nan
            fitted_values.loc[kept_index, "ForcedInVars"] = results.fittedvalues.values

    # Loop through variables
    for i, IV in enumerate(IVs):
        
        if i%10 == 0:
            print("Working on {}, which is #{} out of {}".format(IV, IVs.index(IV)+1,len(IVs)))
            
        if forced_in is not None:
            if IV in forced_in:
                print("Skipping this variable, since it is being forced in already")
                continue

        if forced_in is not None:
            model_dataset = dataset[[DV, IV] + forced_in]
        else:
            model_dataset = dataset[[DV, IV]]

        model_dataset = model_dataset.dropna()
        kept_index = model_dataset.index.values

        if forced_in is not None:
            X = model_dataset[[IV] + forced_in]
        else:
            X = model_dataset[[IV]]
        Y = model_dataset[DV]

        if intercept:
            X = sm.add_constant(X)

        model = sm.OLS(Y, X)
        results = model.fit()

        # Generate outputs
        results_dict = OrderedDict()

        results_dict['Variable'] = IV
        results_dict['# Obs'] = results.nobs
        results_dict['# Miss'] = len(Y) - results.nobs

        if intercept:
            results_dict['Intercept'] = results.params['const']
        results_dict['Var Coef'] = results.params[IV]

        if forced_in is not None:
            for forced_var in forced_in:
                results_dict[forced_var + " Coef"] = results.params[forced_var]

        if intercept:
            results_dict["Intercept p-val"] = results.pvalues['const']
        results_dict["Var p-val"] = results.pvalues[IV]

        if forced_in is not None:
            for forced_var in forced_in:
                results_dict[forced_var + " p-val"] = results.pvalues[forced_var]

        if robust in ["HC0", "HC1", "HC2", "HC3"]:
            if intercept:
                results_dict['Int ' + robust + " p-val"] = 2 * sc.stats.t.cdf(
                    -abs(np.exp(
                        logsumexp(results.params['const']) - logsumexp(results.__getattribute__(robust + "_se")['const'])
                    )), df=results.df_resid)

            results_dict['Var ' + robust + " p-val"] = 2 * sc.stats.t.cdf(
                -abs(np.exp(
                    logsumexp(results.params[IV]) - logsumexp(results.__getattribute__(robust + "_se")[IV])
                )), df=results.df_resid)

            if forced_in is not None:
                for forced_var in forced_in:
                    results_dict[forced_var + " " + robust + " p-val"] = 2 * sc.stats.t.cdf(
                        -abs(np.exp(
                            logsumexp(results.params[forced_var]) - logsumexp(
                                results.__getattribute__(robust + "_se")[forced_var])
                        )), df=results.df_resid)

        results_dict["Rsq"] = results.rsquared
        results_dict["Adj Rsq"] = results.rsquared_adj

        if forced_in:
            X_f = model_dataset[forced_in]
            if intercept:
                X_f = sm.add_constant(X_f)

            forced_in_fitted = sm.OLS(Y, X_f).fit().fittedvalues.values
            results_dict["Rsq vs. forced in"] = 1 - np.nansum((Y - results.fittedvalues.values)**2) / np.nansum((Y - forced_in_fitted)**2)

        results_dict["MSE"] = eval_measures.rmse(Y, results.fittedvalues.values)**2
        results_dict["RMSE"] = eval_measures.rmse(Y, results.fittedvalues.values)
        results_dict["AIC"] = results.aic
        results_dict["BIC"] = results.bic
        results_dict["Durbin-Waston"] = sm.stats.stattools.durbin_watson(results.resid)
        # Get standardized coefficient by using the formula coef(X) * sd(X) / sd(y)
        results_dict['Var Std. Coef'] = (results_dict['Var Coef'] * np.std(X,axis=0) / np.std(Y)).iloc[1]

        if forced_in is not None:
            for forced_var in forced_in:
                results_dict[forced_var + " Std. Coef"] = results_dict[forced_var + " Coef"] * sd(X_f[forced_var]) / sd(Y)
        

        output = pd.concat([output, pd.DataFrame([results_dict])], ignore_index=True)

        # Fitted values
        if get_fitted:
            fitted_values[IV] = np.nan
            fitted_values.loc[kept_index, IV] = results.fittedvalues.values

    if get_fitted is False:
        fitted_values = None

    return {
        "results": output,
        "fitted": fitted_values
    }