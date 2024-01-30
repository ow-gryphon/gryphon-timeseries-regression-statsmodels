import pandas as pd
import numpy as np


transformation_names = pd.DataFrame([{"Transformation":"No Transform","Mnemonics":"N"},
                                     {"Transformation":"Log Transform","Mnemonics":"L"},
                                     {"Transformation":"QoQ Diff","Mnemonics":"QD"},
                                     {"Transformation":"QoQ Log Diff","Mnemonics":"QLD"},
                                     {"Transformation":"QoQ % Change","Mnemonics":"QPD"},
                                     {"Transformation":"QoQoQ Diff","Mnemonics":"QQD"},
                                     {"Transformation":"QoQ Abs Diff","Mnemonics":"QAbsD"},
                                     {"Transformation":"QoQ Ratio","Mnemonics":"QR"},
                                     {"Transformation":"1Y Average","Mnemonics":"YA"},
                                     {"Transformation":"YoY Diff","Mnemonics":"YD"},
                                     {"Transformation":"YoY Log Diff","Mnemonics":"YLD"},
                                     {"Transformation":"YoY % Change","Mnemonics":"YPD"},
                                     {"Transformation":"YoY Abs Diff","Mnemonics":"YAbsD"},
                                     {"Transformation":"YoY Ratio","Mnemonics":"YR"},
                                     {"Transformation":"YoYoY Diff","Mnemonics":"YYD"},
                                     {"Transformation":"MoM Diff","Mnemonics":"MD"},
                                     {"Transformation":"MoM Log Diff","Mnemonics":"MLD"},
                                     {"Transformation":"MoM % Change","Mnemonics":"MPD"},
                                     {"Transformation":"MoMoM Diff","Mnemonics":"MMD"},
                                     {"Transformation":"MoM Abs Diff","Mnemonics":"MAbsD"},
                                     {"Transformation":"MoM Ratio","Mnemonics":"MR"},
                                     {"Transformation":"1Q Average","Mnemonics":"QA"},
                                     {"Transformation":"Demean","Mnemonics":"Cent"},
                                     {"Transformation":"Normalize","Mnemonics":"Norm"},
                                     {"Transformation":"Demean by ID","Mnemonics":"Cent_ID"},
                                     {"Transformation":"Normalize by ID","Mnemonics":"Norm_ID"},
                                     {"Transformation":"Demean by Time","Mnemonics":"Cent_T"},
                                     {"Transformation":"Normalize by Time","Mnemonics":"Norm_ID"},
                                     {"Transformation":"Detrend","Mnemonics":"DeTrend"},
                                     {"Transformation":"Deseasonalize","Mnemonics":"DeSeas"}])

def create_var_mapping_table(variables, standardized_variables, data_frequency=4):
    var=['Main Category','DV or IV or Dummy?','# lags']
    monthly_data_transformations=["No Transform","Log Transform","QoQ Diff","QoQ Log Diff",
                                  "QoQ % Change","QoQoQ Diff","QoQ Abs Diff","QoQ Ratio","1Y Average","YoY Diff",
                                  "YoY Log Diff","YoY % Change","YoY Abs Diff","YoY Ratio",
                                  "MoM Diff","MoM Log Diff","MoM % Change","MoMoM Diff","MoM Abs Diff",
                                  "MoM Ratio","1Q Average","Normalize","Normalize by ID",
                                  "Normalize by Time","Demean by Time","Detrend","Deseasonalize"]
    
    quarterly_data_transformations=["No Transform","Log Transform","QoQ Diff","QoQ Log Diff",
                                  "QoQ % Change","QoQoQ Diff","QoQ Abs Diff","QoQ Ratio","1Y Average","YoY Diff",
                                  "YoY Log Diff","YoY % Change","YoY Abs Diff","YoY Ratio",
                                  "Normalize","Normalize by ID",
                                  "Normalize by Time","Demean by Time","Detrend","Deseasonalize"]
    
    yearly_data_transformations=["No Transform","Log Transform","YoY Diff",
                                  "YoY Log Diff","YoY % Change","YoY Abs Diff","YoY Ratio",
                                  "Normalize","Normalize by ID",
                                  "Normalize by Time","Demean by Time","Detrend","Deseasonalize"]
    

    table = pd.DataFrame({'Variable Name':variables, 'Standardized Name': standardized_variables})
    table[var]= None
    
    if data_frequency==12:
        for column in monthly_data_transformations:
            table[column] = False
    elif data_frequency==4:
        for column in quarterly_data_transformations:
            table[column] = False
    elif data_frequency==1:
        for column in yearly_data_transformations:
            table[column] = False
    else:
        return ("Invalid data_frequency")
    
    

    return table
    

def run_transforms(ts_data=None, transformations=None, transformation_names=transformation_names,frequency=None):
    # Check for null
    if ts_data is None or transformations is None:
        return None
    
    # Check for time-series nature
    if pd.api.types.is_datetime64_any_dtype(ts_data.index)== False:
        return None
    
    ts_data = ts_data.copy()

    rename_dict = dict(zip(transformations['Variable Name'], transformations['Standardized Name']))
    ts_data.rename(columns=rename_dict)

    # Check the lags input. If empty set lags = 2 as default
    transformations['# lags'] = transformations['# lags'].fillna(2)

    function_calls = 0  # Count number of times the transformation function has been called
    output_matrix = None
    output_name = None

    lag_column_index = list(transformations.columns).index("# lags")

    for i, temp_name in enumerate(ts_data.columns):

        # Obtain the transformation information
        temp_transformation = transformations[transformations['Standardized Name'] == temp_name].reset_index(drop=True)

        if len(temp_transformation) == 0:
            continue
        elif len(temp_transformation) > 1:
            print(f"More than 1 row of transformations found for {temp_name}")
            continue

        # For each variable, check if there are any transformations for it
        transformation_choices = temp_transformation.iloc[0,(lag_column_index+1):]
        count_transformations = transformation_choices.sum()

         # If no transformation, then skip to the next one
        if count_transformations == 0:
            continue

        # Get variable time-series
        temp_var = ts_data[temp_name]

        # Obtain the lag
        temp_lags = temp_transformation.loc[0, "# lags"]

        # Obtain the transformation
        temp_transformation_chosen = [colname for colname in transformation_choices.index if transformation_choices[colname]] 
        if len(temp_transformation_chosen) == 0:
            continue # No transformation

        temp_results = time_series_transformations(variable=temp_var, 
                                                   var_name=temp_name, 
                                                   transformations=temp_transformation_chosen, 
                                                   lags=temp_lags, 
                                                   transformation_names=transformation_names,
                                                   frequency=frequency)
    
        if temp_results is None:
            continue
        
        function_calls += 1
        
        if function_calls == 1:
            output_matrix = temp_results['matrix']
            output_name = temp_results['variable_name']
        else:
            output_matrix = pd.concat([output_matrix, temp_results['matrix']], axis=1)
            output_name = pd.concat([output_name, temp_results['variable_name']], axis=0)
            
    if function_calls == 0:
        return None    
    
    if output_matrix is None:
        output = {"data": None, "var_mapping": None}
        return output
    else:
        output_matrix.columns = output_name["FullName"]
        output = {"data": output_matrix, "var_mapping": output_name}
        return output
    
    

def time_series_transformations(variable=None, var_name=None, transformations=None, lags=2, transformation_names=None, frequency=None):
    # If any input is empty, simply return NULL
    if variable is None or var_name is None or transformations is None:
        return None
    
    # Check for time-series nature
    if pd.api.types.is_datetime64_any_dtype(variable.index)== False:
        return None
    
    variable = variable.copy()    
    
    # Frequency of the data
    my_freq = frequency
    # Count the number of columns used
    col_counter = 0
    
    # Loop through the set of transformations
    for j in range(len(transformations)):
        temp_transformation = transformations[j]
        
        try:
            trans_name = transformation_names.loc[transformation_names['Transformation'] == temp_transformation, 'Mnemonics'].values[0]
        except Exception as E:
            raise ValueError(f"Failed to find transformation for {temp_transformation}")
    
        # Transformations will be different based on frequency of the data
        if my_freq == 4:
            temp = None  # Variable storing the transformation
            if temp_transformation == "No Transform":
                temp = variable
            elif temp_transformation == "Log Transform":
                temp = np.log(variable)
            elif temp_transformation == "QoQ Diff":
                temp = variable-variable.shift(1)
            elif temp_transformation == "QoQ Log Diff":
                temp = np.log(variable)-np.log(variable).shift(1)
            elif temp_transformation == "QoQ % Change":
                temp = (variable/variable.shift(1)) - 1
            elif temp_transformation == "QoQoQ Diff":
                temp = variable - variable.shift(1) - (variable.shift(1) - variable.shift(2))
            elif temp_transformation == "QoQ Abs Diff":
                temp = np.abs(variable-variable.shift(1))
            elif temp_transformation == "QoQ Ratio":
                temp = variable / variable.shift(1)
            elif temp_transformation == "1Y Average":
                temp = (variable + variable.shift(1) +variable.shift(2)+variable.shift(3)) / 4
            elif temp_transformation == "YoY Diff":
                temp = variable-variable.shift(4)
            elif temp_transformation == "YoY Log Diff":
                temp = np.log(variable)-np.log(variable).shift(4)
            elif temp_transformation == "YoY % Change":
                temp = (variable/variable.shift(4)) - 1
            elif temp_transformation == "YoY Abs Diff":
                temp = np.abs(variable-variable.shift(4))
            elif temp_transformation == "YoY Ratio":
                temp = variable / variable.shift(4)
            elif temp_transformation == "Demean":
                temp = variable - np.mean(variable, axis=0, keepdims=True)
            elif temp_transformation == "Normalize":
                temp = (variable - np.mean(variable, axis=0, keepdims=True)) / np.std(variable, axis=0, keepdims=True)
            elif temp_transformation == "Detrend":
                temp = variable - np.polyval(np.polyfit(np.arange(len(variable)), variable, 1), np.arange(len(variable)))
            elif temp_transformation == "Deseasonalize":
                temp = variable - np.polyval(np.polyfit(np.arange(len(variable)) % 1, variable, 1), np.arange(len(variable)) % 1)
            else:
                continue
        elif my_freq == 12:
            temp = None  # Variable storing the transformation
            if temp_transformation == "No Transform":
                temp = variable
            elif temp_transformation == "Log Transform":
                temp = np.log(variable)
            elif temp_transformation == "MoM Diff":
                temp = variable-variable.shift(1)
            elif temp_transformation == "MoM Log Diff":
                temp = np.log(variable)-np.log(variable).shift(1)
            elif temp_transformation == "MoM % Change":
                temp = (variable/variable.shift(1)) - 1
            elif temp_transformation == "MoMoM Diff":
                temp = variable - variable.shift(1) - (variable.shift(1) - variable.shift(2))
            elif temp_transformation == "MoM Abs Diff":
                temp = np.abs(variable-variable.shift(1))
            elif temp_transformation == "MoM Ratio":
                temp = variable / variable.shift(1)
            elif temp_transformation == "1Q Average":
                temp = (variable + np.roll(variable, -1) + np.roll(variable, -2)) / 3
            elif temp_transformation == "1Y Average":
                temp = (variable + np.roll(variable, -1) + np.roll(variable, -2) + np.roll(variable, -3) + np.roll(variable, -4) + np.roll(variable, -5) + np.roll(variable, -6) + np.roll(variable, -7) + np.roll(variable, -8) + np.roll(variable, -9) + np.roll(variable, -10) + np.roll(variable, -11)) / 12
            elif temp_transformation == "QoQ Diff":
                temp = variable-variable.shift(3)
            elif temp_transformation == "QoQ Log Diff":
                temp = np.log(variable)-np.log(variable).shift(3)
            elif temp_transformation == "QoQ % Change":
                temp = (variable/variable.shift(3)) - 1
            elif temp_transformation == "QoQ Abs Diff":
                temp = np.abs(variable-variable.shift(3))
            elif temp_transformation == "QoQ Ratio":
                temp = variable / variable.shift(3)
            elif temp_transformation == "YoY Diff":
                temp = variable-variable.shift(12)
            elif temp_transformation == "YoY Log Diff":
                temp = np.log(variable)-np.log(variable).shift(12)
            elif temp_transformation == "YoY % Change":
                temp = (variable/variable.shift(12)) - 1
            elif temp_transformation == "YoY Abs Diff":
                temp = np.abs(variable-variable.shift(12))
            elif temp_transformation == "YoY Ratio":
                temp = variable / variable.shift(12)
            elif temp_transformation == "Demean":
                temp = variable - np.mean(variable, axis=0, keepdims=True)
            elif temp_transformation == "Normalize":
                temp = (variable - np.mean(variable, axis=0, keepdims=True)) / np.std(variable, axis=0, keepdims=True)
            elif temp_transformation == "Detrend":
                temp = variable - np.polyval(np.polyfit(np.arange(len(variable)), variable, 1), np.arange(len(variable)))
            elif temp_transformation == "Deseasonalize":
                temp = variable - np.polyval(np.polyfit(np.arange(len(variable)) % 1, variable, 1), np.arange(len(variable)) % 1)
            else:
                continue
        elif my_freq == 1:
            temp = None  # Variable storing the transformation
            if temp_transformation == "No Transform":
                temp = variable
            elif temp_transformation == "Log Transform":
                temp = np.log(variable)
            elif temp_transformation == "YoY Diff":
                temp = variable-variable.shift(1)
            elif temp_transformation == "YoY Log Diff":
                temp = np.log(variable)-np.log(variable).shift(1)
            elif temp_transformation == "YoY % Change":
                temp = (variable/variable.shift(1)) - 1
            elif temp_transformation == "YoYoY Diff":
                temp = variable - variable.shift(1) - (variable.shift(1) - variable.shift(2))
            elif temp_transformation == "YoY Abs Diff":
                temp = np.abs(variable-variable.shift(1))
            elif temp_transformation == "YoY Ratio":
                temp = variable / variable.shift(1)
            elif temp_transformation == "Demean":
                temp = variable - np.mean(variable, axis=0, keepdims=True)
            elif temp_transformation == "Normalize":
                temp = (variable - np.mean(variable, axis=0, keepdims=True)) / np.std(variable, axis=0, keepdims=True)
            elif temp_transformation == "Detrend":
                temp = variable - np.polyval(np.polyfit(np.arange(len(variable)), variable, 1), np.arange(len(variable)))
            elif temp_transformation == "Deseasonalize":
                temp = variable - np.polyval(np.polyfit(np.arange(len(variable)) % 1, variable, 1), np.arange(len(variable)) % 1)
            else:
                continue
        else:
            return None
        

        # Lag the variable
        temp2 = None
        for k in range(lags + 1):
            
            temp2 = pd.Series(temp).shift(k)

            col_counter += 1

            if col_counter == 1:
                output_matrix = temp2
                output_name = pd.DataFrame({"Variable": [var_name], 
                                            "Transform": [trans_name], 
                                            "Lag": [f"L{k}"], 
                                            "FullName": [f"{var_name}|{trans_name}|L{k}"]})
            else:
                output_matrix = pd.concat([output_matrix, temp2], axis=1)
                output_name = pd.concat([output_name, pd.DataFrame({"Variable": [var_name], 
                                            "Transform": [trans_name], 
                                            "Lag": [f"L{k}"], 
                                            "FullName": [f"{var_name}|{trans_name}|L{k}"]})])
            temp2 = None

    if col_counter == 0:
        output = None
    else:
        output = {}
        output['matrix'] = output_matrix
        output['variable_name'] = output_name
    return output



def train_test_split(data, last_insample):
    
    train_dataset = data[data.index <= last_insample].copy()
    test_dataset = data[data.index > last_insample].copy()
    
    return train_dataset, test_dataset