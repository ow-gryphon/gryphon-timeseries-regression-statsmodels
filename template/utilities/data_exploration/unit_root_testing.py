import sklearn as sk
from scipy import stats
import numpy as np
import pandas as pd
import warnings
import collections
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning


############ statsmodels UNIT TESTS ##########
def unit_root_test_wrapper(data, variables, tests=['ADF', 'KPSS', 'DF-GLS'], maxlag=['Auto'], type=['zero mean','mean'], adf_lag_selection=["BIC"], dfgls_lag_selection=['MAIC2']):

    # Checks
    if len(tests) == 0:
        print("No test requested")
    
    if len(set(tests)-set(['ADF', 'KPSS', 'DF-GLS'])) > 0:
        raise ValueError("tests must be a list containing one or more of ['ADF', 'KPSS', 'DF-GLS'] and nothing else")
    
    if len(set(type)-set(['zero mean', 'mean', 'trend', 'n', 'c', 'ct'])) > 0:
        raise ValueError("type must be a list containing one or more of ['zero mean', 'mean', 'trend', 'n', 'c', 'ct'] and nothing else")
    
    if len(set(adf_lag_selection) - set(['BIC', 'AIC', 'Fixed'])) > 0:
        raise ValueError("adf_lag_selection must be a list containing one or more of ['BIC', 'AIC', 'Fixed'] and nothing else")
    
    if len(set(dfgls_lag_selection) - set(['MAIC', 'MAIC2', 'Fixed'])) > 0:
        raise ValueError("adf_lag_selection must be a list containing one or more of ['MAIC', 'MAIC2', 'Fixed'] and nothing else")
    
    # Parse the maxlag:
    lags = []
    for l in maxlag:
        if isinstance(l, (int, float)):
            pass
        elif l == 'Auto':
            pass
        elif l == "L4":
            pass
        elif l == "L12":
            pass
        else:
            ValueError("maxlag must be a list containing integers or 'Auto', 'L4', 'L12'")
    
    # Loop
    results = []
    
    for i, variable in enumerate(variables):
        if (i%10 == 0) and (len(variables) > 20) :
            print(f"Working on Model #{i+1} out of {len(variables)}")
        try:
            temp_result = unit_root_test(data, variable, tests, maxlag, type, adf_lag_selection, dfgls_lag_selection)
            results.append(temp_result)
        except Exception as e:
            print(f"Failed for variable {variable} with message {e}")

    if len(results)==0:
        raise ValueError("Failed to generate any results")
    
    return pd.concat(results).reset_index(drop=True)
    
    

def unit_root_test(data, variable, tests=['ADF', 'KPSS', 'DF-GLS'], maxlag=['Auto'], type=['zero mean','mean'], adf_lag_selection=["BIC"], dfgls_lag_selection=['MAIC2']):

    if len(tests) == 0:
        print("No test requested")
    
    if len(set(tests)-set(['ADF', 'KPSS', 'DF-GLS'])) > 0:
        raise ValueError("tests must be a list containing one or more of ['ADF', 'KPSS', 'DF-GLS'] and nothing else")
    
    if len(set(type)-set(['zero mean', 'mean', 'trend', 'n', 'c', 'ct'])) > 0:
        raise ValueError("type must be a list containing one or more of ['zero mean', 'mean', 'trend', 'n', 'c', 'ct'] and nothing else")
    
    if len(set(adf_lag_selection) - set(['BIC', 'AIC', 'Fixed'])) > 0:
        raise ValueError("adf_lag_selection must be a list containing one or more of ['BIC', 'AIC', 'Fixed'] and nothing else")
    
    if len(set(dfgls_lag_selection) - set(['MAIC', 'MAIC2', 'Fixed'])) > 0:
        raise ValueError("adf_lag_selection must be a list containing one or more of ['MAIC', 'MAIC2', 'Fixed'] and nothing else")
    
    # variable
    var_data = data[variable].dropna()
    data_length = len(var_data)
    
    # Parse the maxlag:
    lags = []
    for l in maxlag:
        if isinstance(l, (int, float)):
            lags.append(l)
        elif l == 'Auto':
            if (data_length >= 100):
                lags.append(np.floor(12 * np.power(data_length / 100, 1 / 4)))
            else:
                lags.append(np.floor(4 * np.power(data_length / 100, 1 / 4)))
        elif l == "L4":
            lags.append(np.floor(4 * np.power(data_length / 100, 1 / 4)))
        elif l == "L12":
            lags.append(np.floor(12 * np.power(data_length / 100, 1 / 4)))
        else:
            ValueError("maxlag must be a list containing integers or 'Auto', 'L4', 'L12'")
    
    type_mapping = {"zero mean": "n", "mean": "c", "trend": "ct", "n": "n", "c": "c", "ct":"ct"}
    dfgls_mapping = {"zero mean": "skip", "mean": "drift", "trend": "trend", "n": "skip", "c": "drift", "ct":"trend"}
    
    results = []
    # Loop through every combination of tests for this variable
    for test in tests:
        if test == "ADF":
            lag_selection = adf_lag_selection
        elif test == 'DF-GLS':
            lag_selection = dfgls_lag_selection
        else:
            lag_selection = ["NONE"]
        
        for lag in lags:
            for t in type:
                for lag_select in lag_selection:
                    
                    if test == "ADF":
                        try:
                            if lag_select!='Fixed':
                                adf, pvalue, usedlag, nobs, _, _  = adfuller(var_data, maxlag = lag, regression=type_mapping.get(t), autolag=lag_select)
                            else:
                                adf, pvalue, usedlag, nobs, _  = adfuller(var_data, maxlag = lag, regression=type_mapping.get(t), autolag=None)
                        except Exception as e:
                            print(f"Error in running ADF: {e}")
                            adf = np.nan
                            pvalue = np.nan
                            usedlag = np.nan
                            nobs = len(var_data)
                        
                        results.append(pd.DataFrame({
                            "var":[variable],
                            "test":["ADF"],
                            "type": [type_mapping.get(t)],
                            "max lag": [lag],
                            "lag_selection": [lag_select],
                            "lag":[usedlag],
                            "statistic":[adf],
                            "p-value":[pvalue]
                        }))    
                    
                    elif test == "KPSS":
                        if type_mapping.get(t) == "n":
                            continue
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', InterpolationWarning)
                                kpss_stat, pvalue, usedlag, _  = kpss(var_data, nlags = lag, regression=type_mapping.get(t))
                            nobs = len(var_data)
                            
                            if pvalue == 0.01:
                                pvalue = "<1%"
                            elif pvalue == 0.1:
                                pvalue = ">10%"
                            
                        except Exception as e:
                            print(f"Error in running KPSS: {e}")
                            kpss_stat = np.nan
                            pvalue = np.nan
                            nobs = len(var_data)
                            usedlag = lag
                        
                        results.append(pd.DataFrame({
                            "var":[variable],
                            "test":["KPSS"],
                            "type": [type_mapping.get(t)],
                            "max lag": [np.nan],
                            "lag_selection": ["N/A"],
                            "lag":[usedlag],
                            "statistic":[kpss_stat],
                            "p-value":[pvalue]
                        }))
                        
                    elif test == "DF-GLS":
                        if dfgls_mapping.get(t) == 'skip':
                            continue
                        try:
                            DF_results = DF_GLS_test(data, var=variable, specification=dfgls_mapping.get(t), max_lag=lag, lag_selection=lag_select)
                            dfgls = DF_results.t_stat
                            nobs = len(var_data)
                            usedlag = DF_results.lag
                            
                            # p-value determination
                            p_category = sum(DF_results.crit_vals >= DF_results.t_stat)
                            if p_category == 3:
                                pvalue = "<1%"
                            elif p_category == 2:
                                pvalue = "1%-5%"
                            elif p_category == 1:
                                pvalue = "5%-10%"
                            elif p_category == 0:
                                pvalue = ">10%"
                            else:
                                pvalue = ""
                            
                        except Exception as e:
                            print(f"Error in running DF-GLS: {e}")
                            dfgls = np.nan
                            pvalue = np.nan
                            nobs = len(var_data)
                            usedlag = lag
                            
                        results.append(pd.DataFrame({
                            "var":[variable],
                            "test":["DF-GLS"],
                            "type": [type_mapping.get(t)],
                            "max lag": [lag],
                            "lag_selection": [lag_select],
                            "lag":[usedlag],
                            "statistic":[dfgls],
                            "p-value":[pvalue]
                        }))    
                        
    return pd.concat(results)


############ CUSTOM UNIT TESTS ###############
from sklearn import linear_model

class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculates standard errors,
    t-statistics and p-values for model coefficients (betas).
    Additional attributes available after .fit() are `se`, `t` and `p` which are of the shape (y.shape[1], X.shape[1]),
    which is (n_features, n_coefs) This class sets the intercept to 0 by default, since usually we include it in X.

    This code was taken, with some minor refinements, from
    "https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression"

    NOTE: the additional statistics (se, t, and p) are only accurate if fit_intercept = False.
    So if intercept is needed, it should be provided as part of the X variable
    """

    def __init__(self, fit_intercept=False):

        if fit_intercept:
            warnings.warn("Note: se, t, and p-values are not accurate when using fit_intercept." +
                  "Instead, run with fit_intercept = False and add an intercept column to X")

        super(LinearRegression, self).__init__(fit_intercept=fit_intercept)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        inverse_covariance = np.linalg.inv(np.dot(X.T, X))
        self.se = np.array([
            np.sqrt(np.diagonal(sse[i] * inverse_covariance)) for i in range(sse.shape[0])
        ])
        self.t = self.coef_ / self.se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self


def ADF_test(used_data, var, specification, max_lag, lag_selection):
    """
        Performs Augmented-Dickey Fuller unit root test for a single column in pandas Dataframe, with lag selection

        Args:
            :param used_data: a single pandas DataFrame containing variable to be tested
            :param var: a single string with the name of the variable to be tested
            :param specification: a single string with the value 'none', 'drift', or 'trend', representing the ADF regression specification
            :param max_lag: an integer representing the maximum lag to consider for the test
            :param lag_selection: a single string with the value 'Fixed', "AIC', or 'BIC' representing the criteria for selecting the optimal lag (Fixed uses the max_lag as optimal lag)
        Returns:
            :return: A dictionary with the keys:value pairs -- regression:summary of the ADF regression output -- crit_vals:the relevant critical values for the t-statistic -- t_stat:the t-statistic -- lag:the selected lag -- parameters:key input parameters to the function -- lag_selection:trace of the criteria by lag order
    """

    adf_data = used_data[[var]].dropna()

    # Check for internal NAs
    adf_index = adf_data.index.values
    assert np.all(np.diff(adf_index) == 1), "There may be NAs in {}".format(var)

    data_length = len(adf_data[var])

    # Check length of dataset
    if (data_length == 0):
        raise Exception('No observations available')

    # Set maximum lag
    orig_max_lag = max_lag  # Just for reference
    if (str(max_lag) == "Auto"):
        if (data_length >= 100):
            max_lag = np.floor(12 * np.power(data_length / 100, 1 / 4))
        else:
            max_lag = np.floor(4 * np.power(data_length / 100, 1 / 4))

    adf_data = adf_data.assign(**{'l1_{}'.format(var): adf_data[var].shift(1)})

    # Perform lag selection
    if lag_selection != "Fixed":
        track_criteria = {}
        for lag in range(0, int(max_lag) + 1):
            # Generate necessary lags
            adf_data = adf_data.assign(**{'d_l{}_{}'.format(lag, var): adf_data[var].diff(1).shift(lag)})
            truncated_adf_data = adf_data.shift(-1-int(max_lag)).dropna()  # This ensures that all regressions run on the same number of observations

            # Run regressions
            adf_reg = linear_model.LinearRegression(fit_intercept=(specification != "none"))
            truncated_y = truncated_adf_data[['d_l0_{}'.format(var)]].values
            if (specification == "trend"):
                truncated_adf_data = truncated_adf_data.assign(trend=truncated_adf_data.reset_index().index.values)
            if (lag == 0):
                if (specification == "trend"):
                    truncated_x = truncated_adf_data[['l1_{}'.format(var)] + ["trend"]].values
                else:
                    truncated_x = truncated_adf_data[['l1_{}'.format(var)]].values
            else:
                if (specification == "trend"):
                    truncated_x = truncated_adf_data[['l1_{}'.format(var)] + ['d_l{}_{}'.format(temp_lag, var) for temp_lag in range(1, int(lag) + 1)] + ["trend"]].values
                else:
                    truncated_x = truncated_adf_data[['l1_{}'.format(var)] + ['d_l{}_{}'.format(temp_lag, var) for temp_lag in range(1, int(lag) + 1)]].values

            adf_reg = adf_reg.fit(X=truncated_x, y=truncated_y)

            predicted = adf_reg.predict(truncated_x)
            residuals = truncated_y - predicted
            reg_length = len(predicted)

            num_params = int(specification != "none")+len(adf_reg.coef_[0])
            # Calculate metrics
            if (lag_selection == "AIC"):
                track_criteria[lag] = np.log(1 / (reg_length) * sum(np.power(residuals, 2))) + num_params * 2 / (reg_length)

            elif (lag_selection == "BIC"):
                track_criteria[lag] = np.log(1 / (reg_length) * sum(np.power(residuals, 2))) + num_params * np.log(reg_length) / (reg_length)

        # Get optimal lag
        selected_lag = int(min(track_criteria, key=track_criteria.get))
        lag_selection_tracker = "{} by lag {}".format(lag_selection, track_criteria)

    else:
        selected_lag = int(max_lag)
        lag_selection_tracker = None
        for lag in range(0, int(max_lag) + 1):
            adf_data = adf_data.assign(**{'d_l{}_{}'.format(lag, var): adf_data[var].diff(1).shift(lag)})

    # Run ADF with the selected lag
    adf_y_var = ['d_l0_{}'.format(var)]
    if (specification == "trend"):
        adf_data = adf_data.assign(trend=adf_data.reset_index().index.values)
    if (selected_lag == 0):
        if (specification == "trend"):
            adf_x_var = ['l1_{}'.format(var), "trend"]
        else:
            adf_x_var = ['l1_{}'.format(var)]
    else:
        if (specification == "trend"):
            adf_x_var = ['l1_{}'.format(var)] + ['d_l{}_{}'.format(temp_lag, var) for temp_lag in range(1, int(selected_lag) + 1)] + ["trend"]
        else:
            adf_x_var = ['l1_{}'.format(var)] + ['d_l{}_{}'.format(temp_lag, var) for temp_lag in range(1, int(selected_lag) + 1)]

    adf_reg = LinearRegression(fit_intercept=False)
    slim_adf_data = adf_data[adf_y_var + adf_x_var].dropna()

    if (specification != "none"):
        slim_adf_data = slim_adf_data.assign(intercept = np.ones(len(slim_adf_data)))
        adf_x_var = adf_x_var + ["intercept"]
        adf_reg = adf_reg.fit(X=slim_adf_data[adf_x_var].values, y=slim_adf_data[adf_y_var].values)
    else:
        adf_reg = adf_reg.fit(X=slim_adf_data[adf_x_var].values,y=slim_adf_data[adf_y_var].values)

    # Find the approximate critical values
    use_n = adf_data.shape[0] - 1  # Length of diff(var)

    bins = np.array([25, 50, 100, 500])
    rowselect = np.digitize(use_n, bins, right = False)

    if (specification == "none"):
        tau_crit_none = pd.DataFrame({'01%': [-2.66, -2.62, -2.6, -2.58, -2.58, -2.58],
                                      '05%': [-1.95, -1.95, -1.95, -1.95, -1.95, -1.95],
                                      '10%': [-1.6, -1.61, -1.61, -1.62, -1.62, -1.62]})

        relevant_crit = tau_crit_none.loc[int(rowselect), :]

    elif (specification == "drift"):
        tau_crit_drift = pd.DataFrame({'01%': [-3.75, -3.58, -5.51, -3.46, -3.44, -3.43],
                                       '05%': [-3.0, -2.93, -2.89, -2.88, -2.87, -2.86],
                                       '10%': [-2.63, -2.6, -2.58, -2.57, -2.57, -2.57]})

        relevant_crit = tau_crit_drift.loc[int(rowselect), :]

    elif (specification == "trend"):
        tau_crit_trend = pd.DataFrame({'01%': [-4.38, -4.15, -4.04, -3.99, -3.98, -3.96],
                                       '05%': [-3.6, -3.5, -3.45, -3.43, -3.42, -3.41],
                                       '10%': [-3.24, -3.18, -3.15, -3.13, -3.13, -3.12]})

        relevant_crit = tau_crit_trend.loc[int(rowselect), :]

    else:
        raise ValueError("specification must be either none, drift, or trend")

    # Create the coefficient table
    ADF_regression = pd.DataFrame({"Vars": adf_x_var,
                                   "coef": adf_reg.coef_[0],
                                   "std-err": adf_reg.se[0],
                                   "t-stat": adf_reg.t[0],
                                   "p-val": adf_reg.p[0]})

    return DF_results(ADF_regression, relevant_crit, adf_reg.t[0][0], selected_lag,
                      {"specification": specification, "max_lag": max_lag, "lag_selection": lag_selection},
                      lag_selection_tracker)



def basic_DF_GLS_test(dfgls_data, var, specification, used_lag, max_lag):
    """
        This is a utility function that performs Augmented-Dickey Fuller unit root test on Generalized Least Square de-trended data from a single column in pandas Dataframe

        Args:
            :param dfgls_data: a single pandas DataFrame containing variable to be tested, with no missing values for the key variable
            :param var: a single string with the name of the variable to be tested
            :param specification: a single string with the value 'drift', or 'trend', representing the GLS detrending approach
            :param used_lag: an integer representing the lag to use in the ADF regression
            :param max_lag: an integer representing the maximum lag to consider in the ADF regression
        Returns:
            :return: An tuple with the following MAIC value, t_statistic, and a summary of the DF-GLS regression output
    """

    # Decision to shorten data here, as that is how the ur.ers function in R does it when de-trending the data
    y_var = dfgls_data.shift(-(int(max_lag)-int(used_lag)))[var].dropna().values
    nobs = len(y_var)
    # Perform GLS de-trending
    if (specification == "drift"):
        ahat = 1 - 7 / nobs
        ya = np.append(y_var[0], y_var[1:nobs] - ahat * y_var[0:(nobs - 1)])
        za1 = np.append(1.0, (1-ahat)*np.ones(nobs - 1))

        yd_reg = LinearRegression(fit_intercept=False)
        yd_reg = yd_reg.fit(X=za1.reshape(-1, 1), y=ya.reshape(-1, 1))
        yd = y_var - yd_reg.coef_[0][0]
    elif (specification == "trend"):
        ahat = 1 - 13.5 / nobs
        ya = np.append(y_var[0], y_var[1:nobs] - ahat * y_var[0:(nobs - 1)])
        za1 = np.append(1.0, (1-ahat)*np.ones(nobs - 1))

        trd1 = np.arange(1, 1 + nobs)
        za2 = np.append(1.0, trd1[1:nobs] - ahat * trd1[0:(nobs - 1)])
        yd_reg = LinearRegression(fit_intercept=False)
        yd_reg = yd_reg.fit(X=np.transpose(np.array([za1, za2])), y=ya.reshape(-1, 1))

        yd = y_var - yd_reg.coef_[0][0] - yd_reg.coef_[0][1] * trd1

    # Prepare dataset for ADF test on GLS-detrended series
    adf_data = pd.DataFrame({"yd": yd})
    adf_data = adf_data.assign(**{'l1_{}'.format("yd"): adf_data["yd"].shift(1)})
    for lag in range(0, int(used_lag) + 1):
        adf_data = adf_data.assign(**{'d_l{}_{}'.format(lag, "yd"): adf_data["yd"].diff(1).shift(lag)})

    if (used_lag == 0):
        adf_x_var = ['l1_{}'.format("yd")]
    else:
        adf_x_var = ['l1_{}'.format("yd")] + ['d_l{}_{}'.format(temp_lag, "yd") for temp_lag in
                                               range(1, int(used_lag) + 1)]
    adf_y_var = ['d_l0_{}'.format("yd")]

    # Run ADF on the GLS-detrended series
    adf_reg = LinearRegression(fit_intercept=False)
    slim_adf_data = adf_data[adf_y_var + adf_x_var].dropna()
    adf_reg = adf_reg.fit(X=slim_adf_data[adf_x_var].values, y=slim_adf_data[adf_y_var].values)
    predicted = adf_reg.predict(slim_adf_data[adf_x_var].values)
    residuals = slim_adf_data[adf_y_var].values - predicted
    res_len = len(residuals)

    # Get the MAIC from this equation
    yd_trunc = yd[np.arange(len(yd) - res_len - 1, len(yd) - 1)]
    betaco = adf_reg.coef_[0][0]
    MAIC = np.log(np.mean(np.power(residuals, 2))) + 2 * (
    1 / np.mean(np.power(residuals, 2)) * (np.power(betaco, 2)) * (np.sum(np.power(yd_trunc, 2))) + used_lag) / res_len

    DFGLS_regression = pd.DataFrame({"Vars": adf_x_var,
                                   "coef": adf_reg.coef_[0],
                                   "std-err": adf_reg.se[0],
                                   "t-stat": adf_reg.t[0],
                                   "p-val": adf_reg.p[0]})

    return MAIC, adf_reg.t[0][0], DFGLS_regression


def basic_DF_OLS_test(dfgls_data, var, specification, used_lag, max_lag):
    """
        This is a utility function that performs Augmented-Dickey Fuller unit root test on Ordinary Least Square de-trended data from a single column in pandas Dataframe, which is used purely for lag selection purposes

        Args:
            :param dfgls_data: a single pandas DataFrame containing variable to be tested, with no missing values for the key variable
            :param var: a single string with the name of the variable to be tested
            :param specification: a single string with the value 'drift', or 'trend', representing the OLS detrending approach
            :param used_lag: an integer representing the lag to use in the ADF regression
            :param max_lag: an integer representing the maximum lag to consider in the ADF regression
        Returns:
            :return: MAIC2 value as a double
    """
    # Decision to shorten data here (see rationale for basic_DF_GLS_test)
    y_var = dfgls_data.shift(-(int(max_lag)-int(used_lag)))[var].dropna().values
    nobs = len(y_var)
    intercept = np.ones(nobs)
    # Perform OLS de-trending
    if (specification == "drift"):
        yd_reg = LinearRegression(fit_intercept=False)
        yd_reg = yd_reg.fit(X=intercept.reshape(-1, 1), y=y_var.reshape(-1, 1))
        yd = y_var - yd_reg.coef_[0][0]

    elif (specification == "trend"):
        trd1 = np.arange(1, 1 + nobs)
        yd_reg = LinearRegression(fit_intercept=False)
        yd_reg = yd_reg.fit(X=np.transpose(np.array([intercept, trd1])), y=y_var.reshape(-1, 1))
        yd = y_var - yd_reg.coef_[0][0] - yd_reg.coef_[0][1] * trd1

    # Prepare dataset for ADF test on OLS-detrended series
    adf_data = pd.DataFrame({"yd": yd})
    adf_data = adf_data.assign(**{'l1_{}'.format("yd"): adf_data["yd"].shift(1)})
    for lag in range(0, int(used_lag) + 1):
        adf_data = adf_data.assign(**{'d_l{}_{}'.format(lag, "yd"): adf_data["yd"].diff(1).shift(lag)})

    if (used_lag == 0):
        adf_x_var = ['l1_{}'.format("yd")]
    else:
        adf_x_var = ['l1_{}'.format("yd")] + ['d_l{}_{}'.format(temp_lag, "yd") for temp_lag in
                                               range(1, int(used_lag) + 1)]
    adf_y_var = ['d_l0_{}'.format("yd")]

    # Run ADF on the OLS-detrended series
    adf_reg = LinearRegression(fit_intercept=False)
    slim_adf_data = adf_data[adf_y_var + adf_x_var].dropna()
    adf_reg = adf_reg.fit(X=slim_adf_data[adf_x_var].values, y=slim_adf_data[adf_y_var].values)
    predicted = adf_reg.predict(slim_adf_data[adf_x_var].values)
    residuals = slim_adf_data[adf_y_var].values - predicted
    res_len = len(residuals)

    # Get the MAIC2 from this equation
    yd_trunc = yd[np.arange(len(yd) - res_len - 1, len(yd) - 1)]
    betaco = adf_reg.coef_[0][0]
    MAIC2 = np.log(np.mean(np.power(residuals, 2))) + 2 * (1 / np.mean(np.power(residuals, 2)) * (np.power(betaco, 2)) * (np.sum(np.power(yd_trunc, 2))) + used_lag) / res_len

    return MAIC2


def DF_GLS_test(used_data, var, specification, max_lag, lag_selection):
    """
        Performs Augmented-Dickey Fuller unit root test on Generalized Least Square de-trended data from a single column in pandas Dataframe, with lag selection

        Args:
            :param used_data: a single pandas DataFrame containing variable to be tested
            :param var: a single string with the name of the variable to be tested
            :param specification: a single string with the value 'drift', or 'trend', representing the detrending approach
            :param max_lag: an integer representing the maximum lag to consider for the test
            :param lag_selection: a single string with the value 'Fixed', "MAIC', or 'MAIC2' representing the criteria for selecting the optimal lag (Fixed uses the max_lag as optimal lag). Note that MAIC2 is essentially lag-selection based on ADF on OLS-detrended series.
        Returns:
            :return: A dictionary with the keys:value pairs -- regression:summary of the final DF-GLS regression output -- crit_vals:the relevant critical values for the t-statistic -- t_stat:the t-statistic -- lag:the selected lag -- parameters:key input parameters to the function -- lag_selection:trace of the criteria by lag order
    """
    dfgls_data = used_data[[var]].dropna()

    # Check for internal NAs -- only works for integer indices. Need to be updated. 
    # dfgls_index = dfgls_data.index.values
    # assert np.all(np.diff(dfgls_index) == 1), "There may be NAs in {}".format(var)

    data_length = len(dfgls_data[var])

    # Check length of dataset
    if (data_length == 0):
        raise Exception('No observations available')

    # Set maximum lag
    orig_max_lag = max_lag  # Just for reference
    if (str(max_lag) == "Auto"):
        if (data_length >= 100):
            max_lag = np.floor(12 * np.power(data_length / 100, 1 / 4))
        else:
            max_lag = np.floor(4 * np.power(data_length / 100, 1 / 4))

    dfgls_data = dfgls_data.assign(**{'l1_{}'.format(var): dfgls_data[var].shift(1)})

    if lag_selection != "Fixed":
        track_criteria = {}
        for lag in range(0, int(max_lag) + 1):
            dfgls_data = dfgls_data.assign(**{'d_l{}_{}'.format(lag, var): dfgls_data[var].diff(1).shift(lag)})

            # Run regressions to get criteria
            if (lag_selection == "MAIC"):
                track_criteria[lag] = basic_DF_GLS_test(dfgls_data, var, specification, lag, max_lag)

            elif (lag_selection == "MAIC2"):
                track_criteria[lag] = basic_DF_OLS_test(dfgls_data, var, specification, lag, max_lag)
        
        # Get optimal lag
        selected_lag = int(min(track_criteria, key=track_criteria.get))
        lag_selection_tracker = "{} by lag {}".format(lag_selection, track_criteria)

    else:
        selected_lag = int(max_lag)
        lag_selection_tracker = None
        for lag in range(0, int(max_lag) + 1):
            dfgls_data = dfgls_data.assign(**{'d_l{}_{}'.format(lag, var): dfgls_data[var].diff(1).shift(lag)})

    # Run DFGLS with the selected lag
    MAIC, t_stat, regression = basic_DF_GLS_test(dfgls_data, var, specification, selected_lag, selected_lag)

    # Find the approximate critical values
    use_n = data_length # Length of var

    if (specification == "drift"):

        relevant_crit = pd.DataFrame({'01%': [-2.5658 - 1.96 / use_n - 10.04 / (np.power(use_n ,2))], '05%': [-1.9393 - 0.398 / use_n], '10%': [-1.6156 - 0.181 / use_n]}).loc[0, :]

    elif (specification == "trend"):

        bins = np.array([50, 100, 200])
        rowselect = np.digitize(use_n, bins, right=False)

        crit_trend = pd.DataFrame({'01%': [-3.77, -3.58, -3.46, -3.48],
                                       '05%': [-3.19, -3.03, -2.93, -2.89],
                                       '10%': [-2.89, -2.74, -2.64, -2.57]})

        relevant_crit = crit_trend.loc[int(rowselect), :]


    else:
        raise ValueError("specification must be either drift, or trend")

    return DF_results(regression, relevant_crit,t_stat, selected_lag,
                      {"specification": specification,"max_lag": max_lag,"lag_selection": lag_selection},
                      lag_selection_tracker)


class DF_results():

    def __init__(self, regression, crit, stat, selected_lag, test_settings, lag_selection_tracker):
        self.regression = regression
        self.crit_vals = crit
        self.t_stat = stat
        self.lag = selected_lag
        self.test_settings = test_settings
        self.lag_selection = lag_selection_tracker
