import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.tsa.api as sts
from statsmodels.tools import eval_measures
import scipy as sc
from scipy.special import logsumexp
from collections import OrderedDict
from sklearn import metrics
import math
import scipy.stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Any, List, Union, Callable, Tuple, Optional
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from .stats_utilities import *

import sys
sys.path.insert(0,"..")
from data_transformations import timeseries_transforms

def standalone_test(
        dataset,
        DV,
        IVs,
        cut_off_date,
        lags= 4, # The maximum lag included in the test
        test_ADF= True, # Cointegration ADF
        test_BG= True, # Autocorrelation Breusch-Godfrey
        test_LB= True, # Autocorrelation Ljung-Box
        test_DW= True, # Autocorrelation Durbin-Watson
        test_BP= True, # Heteroscedasticity Breusche-Pagan
        test_white= True, # Heteroscedasticity White's
        test_RR= True, # Linearity Ramsey's RESET
        reset_order = 3,
        test_SW= True, # Normality Shapiro-Wilk
        test_JB= True, # Normality Jarque-Bera
        test_vif= True, # Multicollinearity VIF
        test_CD= True, # Cooks Distance
        test_oos= True,
        save_dir='plots'
        ):
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    robust="HC3"
    
    model_dataset = dataset[[DV] + IVs].dropna()
    kept_index = model_dataset.index.values
    
    X = model_dataset[IVs]
    Y = model_dataset[DV]
    
    X = sm.add_constant(X)
    
    model = sm.OLS(Y, X)
    results = model.fit()
    
    predicted = results.fittedvalues.values
    residuals = results.resid
    
    # Generate results
    model_fit = OrderedDict()
    
    for i in range(len(IVs)):
        IV = IVs[i]
        model_fit['Var {}'.format(i + 1)] = IV
        model_fit['Var {} Coef'.format(i + 1)] = results.params.iloc[i]
        model_fit['Var {} p-val'.format(i + 1)] = results.pvalues.iloc[i]
    
        if robust in ["HC0", "HC1", "HC2", "HC3"]:
            model_fit['Var {} {} p-val'.format(i+1, robust)] = 2 * sc.stats.t.cdf(
                -abs(np.exp(
                    logsumexp(results.params[IV]) - logsumexp(results.__getattribute__(robust + "_se")[IV])
                )), df=results.df_resid)
            
    model_fit['# Obs'] = results.nobs
    model_fit['# Miss'] = len(Y) - results.nobs
    
    model_fit["Rsq"] = results.rsquared
    model_fit["Adj Rsq"] = results.rsquared_adj
    model_fit["MSE"] = eval_measures.rmse(Y, predicted)**2
    model_fit["RMSE"] = eval_measures.rmse(Y, predicted)
    model_fit["AIC"] = results.aic
    model_fit["BIC"] = results.bic
    
    # Output the results
    fitted_values = pd.DataFrame({"kept_index": kept_index, "actual":model_dataset[DV], "fit": predicted})
    model_fit= pd.DataFrame.from_dict(model_fit, orient='index', columns=['Value']).transpose()
    
    # Set date as index for the fitted_value
    fitted_values['date'] = dataset.loc[fitted_values['kept_index']].index.values
    fitted_values.set_index('date', inplace=True)
    
    
    # Plot the model fit values
    plt.plot(fitted_values['actual'], label='Actual', color='lightblue')
    plt.plot(fitted_values['fit'], label='Model Fit',color='navy')
    
    # Add label and legend
    plt.xlabel('Time')
    plt.ylabel('origination_ind_count|L|L0')
    plt.legend()
    plt.savefig(f'{save_dir}/Model Fit Plot.png')
    plt.show()

    
    # Plot kernel density estimate
    residuals_df=pd.DataFrame(residuals)
    residuals_df['date'] = residuals_df.index.values
    residuals_df.set_index('date', inplace=True)
    
    plt.figure()
    plt.plot(residuals, 'bo', markersize=3)
    plt.xlabel('Index')
    plt.ylabel('Residuals')
    plt.title('Residual Distribution (Kernel Density Estimate)')
    plt.savefig(f'{save_dir}/Residual Distribution Plot.png')
    plt.show()
    
    # Plot ACF
    plot_acf(residuals)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function (ACF)')
    plt.savefig(f'{save_dir}/ACF Plot.png')
    plt.show()

    # Plot PACF
    plot_pacf(residuals)
    plt.xlabel('Lag')
    plt.ylabel('Partial Autocorrelation')
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.savefig(f'{save_dir}/PACF Plot.png')
    plt.show()

    
    if test_ADF:
    # Generate adf results
        adf_results= adf(actual_Y=residuals,
                lags=lags,
                costant="c",
                criteria= "AIC",
                result=False,
                full=False
                )
    else:
        adf_results = None
    
        
    if test_BG:
        # Generate BG results
        BG= breusch_godfrey(reg=results, # statsmodel regression result
                            n_lags = lags # Int or None specifying the order of the autoregressive scheme
                            )

    else:
        BG=None
        
    if test_LB:
        # Generate ljung_box results
        LB= sms.diagnostic.acorr_ljungbox(residuals, # List: length [n_samples] of the residuals from a regression
                                lags, # Int, list or None for the largest lag that is included 
                                False # Boolean: if True then Box-Pierce test results are also returned
                                )
    else:
        LB=None
    
    
    if test_DW:
        # Generate Durbin-Watson results
        DW= durbin_watson(results)
    else:
        DW=None
        
    if test_BP:
        # Generate Breusch_Pagan results
        BP=breusch_pagan(residuals=residuals, # List: length [n_samples] of the residuals from a regression
                                   explanatory= X # Array: shape [n_samples, n_features] containing all explanatory variable values
                                  )
    else:
        BP=None
    
    if test_white:
        # Generate white's results
        white_results=white(residuals=residuals, # List: length [n_samples] of the residuals from a regression
                                   explanatory= X # Array: shape [n_samples, n_features] containing all explanatory variable values
                                  )
    else:
        white_results = None
        
    if test_RR: 
        # Generate Ramsey's RESET
        RR= ramsey_reset(results, # Statsmodels regression object
                         power = reset_order # an int for the last power of the fitted values to be added as regressors to the unrestricted model [optional, default is 3]
                         )
    else:
        RR = None
        
    if test_SW:
        # Generate Shapiro-Wilk results
        SW=sc.stats.shapiro(residuals # List: Historical values of the residuals
                            )
        SW = pd.DataFrame({'Statistic': [SW.statistic], 'P-value': [SW.pvalue]})
    else:
        SW = None
    
    if test_JB:
        # Generate Shapiro-Wilk results
        JB=sc.stats.jarque_bera(residuals # List: Historical values of the residuals
                                )
        JB = pd.DataFrame({'Statistic': [JB.statistic], 'P-value': [JB.pvalue]})
    else:
        None
    
    if test_vif:
        # Generate vif        
        vif = pd.DataFrame()
        vif["Features"] = X.columns
        vif["VIF"] = [statsmodels.stats.outliers_influence.variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    else:
        vif = None
        
    if test_CD:
        # Generate Cooks D
        CD=cooks_distance(residuals=residuals, # List: length [n_samples] of the residuals from a regression
                          explanatory= X, # Array: shape [n_samples, n_features] containing all explanatory variable values
                          )
        CD = pd.DataFrame(CD,index=['p_value','test_statistic' ])
        
    else:
        CD = None
    
    if test_oos:
        oos_results=oos(dataset,DV,IVs, cut_off_date, save_dir=save_dir)
    else:
        oos_results=None
    
    print("Completed testing for this model")
    
    return {
        'model': results,
        'fitted':fitted_values, 
        'model_stats': model_fit,
        'residuals':residuals,
        'ADF':adf_results,
        'BG':BG,
        'LB':LB,
        'DW':DW,
        'BP':BP,
        'white':white_results,
        'RR':RR,
        'SW':SW,
        'JB':JB,
        'vif':vif,
        'CD':CD,
        'oos':oos_results
        }



def oos(dataset,
        DV,
        IVs,
        cut_off_date,
        save_dir='plots'
        ):
    # Filter the in-sample data that pasts the cut_off_date
    is_data = dataset[dataset.index <= cut_off_date]

    is_model_dataset = is_data[[DV] + IVs].dropna()
    is_kept_index = is_model_dataset.index.values

    is_X = is_model_dataset[IVs]
    is_Y = is_model_dataset[DV]
    
    is_X = sm.add_constant(is_X)

    # Get out-of-sample IVs
    oos_data = dataset[dataset.index > cut_off_date]
    oos_model_dataset = oos_data[[DV] + IVs].dropna()
    oos_X= oos_model_dataset[IVs]
    oos_Y = oos_model_dataset[DV]
    
    oos_X = sm.add_constant(oos_X)

    # Run regression with in-sample data
    is_model = sm.OLS(is_Y, is_X)
    is_results = is_model.fit()
    is_predicted = is_results.fittedvalues.values

    # Gain predictions of DV with out-of-sample IVs
    oos_predictions= is_results.predict(oos_X)

    # Output the results
    is_results= pd.DataFrame({'Actual': is_Y, 'Predicted': is_predicted})
    is_results['date'] = dataset.loc[is_results.index.values].index.values
    is_results.set_index('date', inplace=True)

    oos_results = pd.DataFrame({'Actual': oos_Y, 'Predicted': oos_predictions})
    oos_results['date'] = dataset.loc[oos_results.index.values].index.values
    oos_results.set_index('date', inplace=True)

    # Combine the 2 types of results
    is_results['type']="in-sample"
    oos_results['type']='out-of-sample'

    oos_final=pd.concat([is_results, oos_results])
    
    # Plot the model fit values
    plt.plot(oos_final['Actual'], label='Actual', color='lightblue')
    plt.plot(is_results['Predicted'], label='In-Sample Model Fit', color='navy')
    plt.plot(oos_results['Predicted'], label='Out-of-Sample Model Fit', color='green')
    
    # Add label and legend
    plt.xlabel('Time')
    plt.ylabel('origination_ind_count|L|L0')
    plt.legend()
    plt.savefig(f'{save_dir}/OOS Plot.png')
    plt.show()

    
    return oos_final
    
    
def forecast(filepath,
             DV,
             IVs,
             raw_IVs,
             model,
             transformations_input,
             forecast_start_date,
             data_frequency = 4,
             date_column = "date_quarter", 
             historical_data=None,
             scenario_sheets=None,
             save_dir="plots"):
             
    if scenario_sheets is None:
        raise ValueError("scenario_sheets should be a list of the tab names with scenario data")

    data_overview = {}
    forecasts = []
    
    for sheet in scenario_sheets:
    
        # Input data for different scenarios and filter for only relevant variables
        dataset = pd.read_excel(filepath, sheet_name=sheet)
        
        # https://stackoverflow.com/questions/35339139/what-values-are-valid-in-pandas-freq-tags 
        
        if data_frequency == 4:
            dataset.index = pd.PeriodIndex(dataset[date_column], freq='Q').to_timestamp()
        elif data_frequency == 1:
            dataset.index = pd.PeriodIndex(dataset[date_column], freq='Y').to_timestamp()
        elif data_frequency == 12:
            dataset.index = pd.PeriodIndex(dataset[date_column], freq='M').to_timestamp()
        else:
            raise ValueError("data_frequency must be 1, 4 or 12")
        dataset.index.name = 'date'
        
        # Sort the dataset in descending order of date
        dataset = dataset.sort_values(by='date', ascending=True)
    
        # Transform scenario data
        transformed_data = timeseries_transforms.run_transforms(ts_data=dataset, transformations=transformations_input, frequency=data_frequency)['data']
    
        data_overview[sheet] = pd.concat([dataset[raw_IVs], transformed_data[IVs]],axis=1)
    
    
        # Generate forecasts 
                    
        # Get the forecastes IVs
        X = transformed_data[IVs]
        X = sm.add_constant(X)
        
        # Use regression model to predict the forecast
        forecast = model.predict(X)
    
        # Filter data for future forecast
        forecast = forecast[forecast.index >= forecast_start_date].copy()
    
        # Append information
        forecasts.append(forecast)
    
    if historical_data is not None:
        # Get historical fit
        X = historical_data[IVs]
        X = sm.add_constant(X)
        
        historical_fit = model.predict(X)
    
        # Filter data to remove forecast periods
        historical_fit = historical_fit[historical_fit.index < forecast_start_date].copy()

        # Combine the results
        forecast_results = pd.concat([historical_data[DV]] + [historical_fit] + forecasts, axis=1, keys=["actuals","historical fit"] + scenario_sheets)
        
    else:
        # Combine the results
        forecast_results = pd.concat(forecasts, axis=1, keys=[scenario_sheets])
    
    # Plot the forecast
    ax = plt.figure(figsize=(12,5))
    plt.plot(forecast_results)
    
    # Add label and legend
    plt.xlabel('Time')
    plt.ylabel(DV)
    plt.legend(forecast_results.columns)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(f'{save_dir}/Forecast Plot.png')
    plt.show()
    
    return forecast_results, data_overview, ax
    
   