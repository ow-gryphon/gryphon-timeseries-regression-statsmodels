import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.tsa.api as sts
from sklearn import metrics
import math
import scipy.stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey, het_white, linear_reset

from typing import Any, List, Union, Callable, Tuple, Optional
from scipy.stats import norm
from scipy.linalg import qr, inv


    


def adf(actual_Y,
        lags=4,
        costant="c",
        criteria= "AIC",
        result=False,
        full=False):
    adf_results= sts.stattools.adfuller(actual_Y, # List: Historical values of the dependent variables 
                               lags, # Int: the maximum lag included in the test, None is 12 * (n/100)^0.25
                               costant, #Str: the constant and trend order to include in regression-  
                                                # 'c': constant only i.e. drift
                                                # 'ct': constant and the trend
                                                # 'ctt': constant and a linear and quadratic trend
                                                # 'nc': no constant or trend i.e. just unit root test
                               criteria, # Str: how to automatically assign the number of lags used 
                                            # None refers to the maxlag lags
                                            # 'AIC' or 'BIC' results in the number of lags chosen correspond to the information criteria
                                            # 't-stat' based choice of maxlag. Stats with maxlag and drios a lag until the t-statistic on the last lag length is significant using a 5%-sized test
                               result, # Boolean: if True a result instance is returned additionally to the adf statistic
                               full # Boolean: if True returns the full regression results
                              )

    # Create a dictionary for adf_results
    adf_dict = {
        'ADF Statistic': adf_results[0],
        'p-value': adf_results[1],
        'Number of Lags': adf_results[2],
        'Number of Observations': adf_results[3],
        'Max Information Criterion': adf_results[5]
    }

    # Create DataFrame from the dictionary
    adf_df = pd.DataFrame.from_dict(adf_dict, orient='index', columns=['Value'])

    # Create a separate DataFrame for the Critical Values
    critical_values_df = pd.DataFrame.from_dict(adf_results[4], orient='index', columns=['Value'])

    # Concatenate the DataFrames vertically
    adf_df = pd.concat([adf_df, critical_values_df])

    # Reset the index
    adf_df.reset_index(inplace=True)

    # Rename the columns
    adf_df.columns = ['Variable', 'Value']

    # Display the DataFrame
    return adf_df


def gini(
    y_true: List,
    y_score: Union[List[float], List[List[float]]],
    average: str = "macro",
    sample_weight: Optional[List] = None,
    invert: bool = False,
    confidence_interval: Optional[Callable] = None,
    level: float = 0.95,
    **kwargs: Any,
) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Calculates the Gini coefficient and optional confidence intervals from the ROC curve

    Uses metrics.roc_auc_curve

    :param y_true: an array of true values,
    :param y_score: an array of shape [n_samples] or [n_samples, n_classes]
        These scores are the target scores; either are the probability estimates of the positive
        class, confidence values or non-threshold measure of decisions
    :param average: a string to determine the type of averaging performed on the data
        [optional, default is 'macro']
        None: the scores for each class are returned. If none is not assigned, this parameter
        determines the type of averaging to be performed

        micro:
            calculates the metrics globally by considering each element of the label indicator
            matrix as a label i.e. total true positives and total false positives

        macro:
            calculates the metrics for each label and finds the unweighted mean. Does not take
            into account label imbalance

        weighted:
            calculates the metrics for each label, then finds the average where each metric
            is labeled by the support (number of true instances for each label)

        samples:
            calculates metrics for each instance, and finds the average

    :param sample_weight: an array of shape [n_samples] to determine how each sample is weighted in
        calculations, [optional, default is None]
    :param invert: a boolean value which determines which form of the Gini from ROC calculation to use.
        if True, :math:`G = 1 - (2 AUC)`. if False, :math:`G = (2 AUC) - 1`
        [optional, default is False]
    :param confidence_interval: a callable to indicate which confidence interval function
        is to be used. [optional, default is None]
        Any method which has API following (y_true, y_score, gini, auc, level))
    :param level: a float indicating the confidence level used in the confidence_interval.
        [optional, default is 0.95]
    :param kwargs: a dictionary containing any optional parameters that may be required for the
        confidence_interval.

    :return: 
        Gini coefficient, as a float

        lower and upper bound confidence interval if confidence interval function is selected. These
        are returned with the gini coefficient as a tuple of the format (gini, (lower CI, upper CI)).
    """
    auc = metrics.roc_auc_score(y_true, y_score, average=average, sample_weight=sample_weight)

    result = (2 * auc) - 1

    if invert == True:
        result = -1 * result

    if confidence_interval:
        try:
            result = (
                result,
                confidence_interval(y_true, y_score, result, auc, level, **kwargs),
            )

        except Exception as e:
            raise Exception(
                f"Error in confidence interval used in gini function. The error in the confidence interval function is '{e}'"
            )

    return result


def vif(explanatory: List[List[float]]) -> Union[float, List[float]]:
    """
    VIF score for explanatory variable(s)

    Uses statsmodels.stats.outliers_influence.variance_inflation_factor

    :param explanatory: an array of shape [n_samples, n_features] containing all explanatory
        variable values. Note that statsmodels explicitly requires a constant term to fit
        y=mx+c (vs y=mx) and will assume that this is absent from the input.

    :return: VIF for each coefficient included in the model, as an array of floats.
        Note, the first float corresponds to the first coefficient data inputted as a row in
        explanatory
    """
    # Add constant 1 to all variables
    explanatory = sm.add_constant(explanatory)

    # Calculate VIF & discard first column of added constants
    return [
        variance_inflation_factor(explanatory, i)
        for i in range(0, explanatory.shape[1])
    ]


def r_squared_mcfadden(likelihood_fitted: float, likelihood_null: float) -> float:
    """
    Computes the McFadden r squared statistic

    :param likelihood_fitted: a float of the (maximised) likelihood value from the current
        fitted model
    :param likelihood_null: a float of the corresponding value of the likelihood from the
        null model

    return: McFadden r squared statistic, as a float
    """
    if likelihood_fitted <= 0 or likelihood_null <= 0:
        raise ValueError("parameters need to be positive")

    return 1 - (math.log(likelihood_fitted) / math.log(likelihood_null))


def breusch_pagan(
    residuals: List[float], explanatory: List[List[float]], robust = True
) -> pd.DataFrame:
    """
    Breusch-Pagan test statistic and corresponding p-value

    :param residuals: a list of length [n_samples] of the residuals from a regression,
    :param explanatory: an array of shape [n_samples, n_feautres] containing all explanatory
        variable values
    :param robust: Boolean whether the robust (studentized) statistic should be usd

    :return:
        test statistic and p-values in a dataframe
    """

    lm, lm_p, f, f_p = het_breuschpagan(residuals, explanatory, robust)
    return pd.DataFrame({"LM Statistic": lm, "LM p-val": lm_p, "F Statistic": f, "F p-val": f_p},index=[0])
    
    
    
def white(residuals: List[float], explanatory: List[List[float]]) -> pd.DataFrame:
    """
    White test for homoskedasticity

    Uses scipy.stats.chi2.sf

    :param explanatory: an  array of shape [n_samples, n_features] containing all explanatory
        variable values, parameters are the columns, samples are the rows i.e.
        designmatrix = [[x_1],...,[x_k]].T,
    :param y_true: an array of true values,

    :return: 
        White test statistic, as a float
        
        approximate p-value, as a float
    """
    
    lm, lm_p, f, f_p = het_white(residuals, explanatory)
    return pd.DataFrame({"LM Statistic": lm, "LM p-val": lm_p},index=[0])


def breusch_godfrey(
    reg, 
    n_lags: int = 1
) -> pd.DataFrame:
    """
    Breusch Godfrey test statistic and p-value

    :param reg: a statsmodels object with the fitted regression
    :param n_lags: an int specifying the maximum order of the autoregressive scheme\

    :return: 
        Breusch Godfrey test statistic, as a float
        p-value, as a float
    """

    results = []
    for lag in range(n_lags):
        
        lm, lm_p, fval, fval_p = acorr_breusch_godfrey(reg, lag + 1)
        results.append(pd.DataFrame({"lag": lag+1, "F statistic": fval, "F p-val": fval_p, "LM statistic": lm, "LM p-val": lm_p}, index=[0]))
        
    return pd.concat(results).reset_index(drop=True)


def durbin_watson(reg) -> float:
    """
    Durbin-Watson test statistic

    :param reg: Statsmodel regression,
    
    :return: Durbin-Watson test statistic, as a float  
    """
    
    res = reg.resid
    dw = np.sum(np.diff(res) ** 2) / np.sum(res ** 2)

    y = reg.model.endog
    X = reg.model.exog

    Q, R = qr(X, mode='economic')
    Q1 = inv(R.T @ R)

    k = X.shape[1]
    n = X.shape[0]

    AX = -np.concatenate((np.full([1,X.shape[1]], np.nan), X[:-1]), axis=0) + 2*X - np.concatenate((X[1:],np.full([1,X.shape[1]], np.nan)), axis=0)

    # Adjust the first and last rows according to the R code
    AX[0, :] = X[0, :] - X[1, :]
    AX[-1, :] = X[-1, :] - X[-2, :]
    
    XAXQ = X.T @ AX @ Q1
    P = 2 * (n - 1) - np.sum(np.diag(XAXQ))
    Q = 2 * (3 * n - 4) - 2 * np.sum(np.diag((AX.T @ AX) @ Q1)) + np.sum(np.diag(XAXQ @ XAXQ))
    dmean = P / (n - k)
    dvar = 2 / ((n - k) * (n - k + 2)) * (Q - P * dmean)
    
    
    return pd.DataFrame({"Item": ["Statistic", "two-sided p-value", "positive autocorrelation p-value", "negative autocorrelation p-value"],
                  "Value": [dw, 2 * norm.sf(abs(dw - dmean) / np.sqrt(dvar)), norm.sf(dw, loc=dmean, scale=np.sqrt(dvar)), norm.cdf(dw, loc=dmean, scale=np.sqrt(dvar))]})



def ramsey_reset(
    reg, 
    power: int = 3,
    cov_type = 'HC3'
) -> Tuple[float, float]:
    """
    Ramsey RESET test on linear model specification for omitted variable bias. Refer to https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.linear_reset.html

    :return: 
        F-test statistic, as a float 
        approximate p-value, as a float
    """
    
    
    # https://github.com/statsmodels/statsmodels/issues/8879
    # Re running the regression without Pandas. This can eventually be removed (as upcoming version of statsmodels won't have this issue)
    reg = sm.OLS(reg.model.endog, reg.model.exog).fit()
    
    results = []
    
    F1 = linear_reset(reg, power, use_f=True, cov_type=cov_type, test_type='fitted')
    Chi1 = linear_reset(reg, power, use_f=False, cov_type=cov_type, test_type='fitted')
    F2 = linear_reset(reg, power, use_f=True, cov_type=cov_type, test_type='exog')
    Chi2 = linear_reset(reg, power, use_f=False, cov_type=cov_type, test_type='exog')
    
    return pd.DataFrame({"Type": ["Fitted", "Exogenous"], "ChiSq statistic": [Chi1.statistic, Chi2.statistic], "ChiSq p-val": [Chi1.pvalue, Chi2.pvalue],  "F statistic": [F1.statistic, F2.statistic], "F p-val": [F1.pvalue, F2.pvalue]})
    

def restricted_f_test(
    restricted_r_squared: float,
    unrestricted_r_squared: float,
    n_obs: int,
    n_params_unrestricted: int,
    n_params_restricted: int,
) -> Tuple[float, float]:
    """
    Restricted f test to test for marginal paramater signficance

    Uses scipy.stats.f.sf

    :param restricted_r_squared: a float for the value of the r squared coefficient of
        determination for the restricted model
    :param unrestricted_r_squared: a float for the value of the r squared coefficient of
        determination for the unrestricted model
    :param n_restrictions: an int for the number of restrictions placed on the model
    :param n_obs: an int for the number of observations in the model
    :param n_params: an int for the number of parameters in the unrestricted model 

    :return: 
        test statistic, as a float 
        
        approximate p-value, as a float
    """
    if (
        isinstance(restricted_r_squared + unrestricted_r_squared, float) == False
        or isinstance(n_obs + n_params_unrestricted + n_params_restricted, int) == False
    ):
        raise ValueError(
            "r squared parameters should be floats, count parameters \
            should be integers"
        )
    if any(x <= 0 for x in (restricted_r_squared, unrestricted_r_squared)) or any(
        x >= 1 for x in (restricted_r_squared, unrestricted_r_squared)
    ):
        raise ValueError(
            "r squared parameters should lie in the interval 0 <= r_squared < 1"
        )
    if (
        any(x < 1 for x in (n_obs, n_params_restricted))
        or n_params_unrestricted < n_params_restricted
    ):
        raise ValueError(
            "count parameters should be positive integers, where the restricted \
            model has fewer parameters than the unrestricted"
        )

    n_restrictions = n_params_unrestricted - n_params_restricted
    test_statistic = (
        (unrestricted_r_squared - restricted_r_squared) / n_restrictions
    ) / ((1 - unrestricted_r_squared) / (n_obs - n_params_unrestricted))
    p_value = scipy.stats.f.sf(
        test_statistic, n_restrictions, n_obs - n_params_unrestricted
    )
    return (test_statistic, p_value)


def cooks_distance(
    residuals: List[float], explanatory: List[List[float]]
) -> Tuple[Union[float, List[float]], Union[float, List[float]]]:
    """
    Cook's distance and corresponding p-value

    :param residuals: a list of length [n_samples] of the residuals from a regression,
    :param explanatory: an array of shape [n_samples, n_features] containing all explanatory
        variable values

    :return:
        Cook's distances, as an array
        
        p-values, as an array
    """
    if len(np.shape(residuals)) != 1:
        raise ValueError("Residuals should be a list of dimension 1")

    if np.shape(residuals)[0] != np.shape(explanatory)[0]:
        raise ValueError("Residuals and explanatory should be same length")

    residuals = np.array([residuals]).T
    regression = sm.OLS(residuals, explanatory).fit()
    influence = regression.get_influence()

    return influence.cooks_distance


def precision_recall_plot(y_true: List, y_pred: List):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    no_skill = len(y_true[y_true==1]) / len(y_true)
    plt.figure()
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='Random')
    plt.plot(recall, precision, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    return plt.gcf()


def roc_plot(y_true: List, y_pred: List):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot([0,1], [0,1], linestyle='--', label='Random')
    plt.plot(fpr, tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    return plt.gcf()

