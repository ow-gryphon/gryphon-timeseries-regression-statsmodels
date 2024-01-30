import os
import re
import matplotlib.pyplot as plt


def clean_filename(filename):
    '''This function takes a string and returns a clean version that can be used as a valid filename in Windows.
    
    Parameters:
    filename (str): The string to be cleaned.
    
    Returns:
    str: The cleaned string.
    '''
    return re.sub(r'[\\/:*?"<>|]', '_', filename)


def plot_ts(data, timestamp=None, variables=None, plot_title = 'Timeseries plot', use_grid = False, save_dir='plots', **kwargs):
    '''
    This function plots a timeseries from a pandas DataFrame using matplotlib. 

    Parameters:
    data (pandas.DataFrame): The DataFrame containing the data to be plotted.
    timestamp (str, optional): The column name in the DataFrame that contains the timestamp. If None, the DataFrame's index is used. Default is None.
    variables (str or list of str): The column name(s) in the DataFrame that contain the variables to be plotted. If a string is provided, it is converted to a list.
    plot_title (str, optional): The title of the plot. Default is 'Timeseries plot'.
    use_grid (bool, optional): If True, a grid is added to the plot. Default is False.
    **kwargs: Additional keyword arguments to be passed to the pandas.DataFrame.plot() function.

    Raises:
    ValueError: If the 'variables' argument is not provided or is neither a string nor a list.

    Returns:
    ax (matplotlib.axes.Axes): The Axes object with the plot.

    Example:
    plot_ts(df, timestamp='Date', variables=['Variable1', 'Variable2'], plot_title='My Plot', use_grid=True)
    '''
    
    if variables is None:
        raise ValueError("Argument variables was not provided")
        
    if isinstance(variables, list):
        pass
    elif isinstance(variables, str):
        variables = [variables]
    else:
        raise ValueError("Argument variables is neither a list or a string")
        
    if timestamp is None:
        plot_data = data[variables].copy()
    else:
        plot_data = data[[timestamp] + variables].copy()
        plot_data.set_index(timestamp, inplace=True)
    
    ax = plot_data.plot(**kwargs)
    ax.set_title(plot_title) 
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')  
    ax.grid(use_grid)  
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(f'{save_dir}/{clean_filename(variables[0])}.png')
    plt.show()
    

    return ax
    
    
def plot_all_ts(data, timestamp=None, variables=None, plot_title = 'Timeseries plot', use_grid = False, save_dir='plots', **kwargs):
    '''
    This function generates a timeseries plot for each variable in a pandas DataFrame and saves them as .png files.

    Parameters:
    data (pandas.DataFrame): The DataFrame containing the data to be plotted.
    timestamp (str): The column name in the DataFrame that contains the timestamp. If None, the row index will be used. Default is None
    variables (list of str): The column names in the DataFrame that contain the variables to be plotted. If None, all numeric variables will be used. Default is None
    plot_title (str, optional): The base title of the plots. The variable name is appended to this base title. Default is 'Timeseries plot'.
    use_grid (bool, optional): If True, a grid is added to the plot. Default is False.
    save_dir (str, optional): The directory where the plot images will be saved. Default is 'plots'.
    **kwargs: Additional keyword arguments to be passed to the pandas.DataFrame.plot() function.

    Returns:
    Message stating number of files created in the output folder
    '''
    
    data = data.select_dtypes(include=['number']).copy()
    
    if variables is None:
        variables = list(set(data.columns)-set([timestamp]))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    counter = 0
    for var in variables:
        
        ax = plot_ts(data, timestamp, var, plot_title=f'{plot_title} - {var}', use_grid=use_grid,save_dir=save_dir, **kwargs)
        plt.show()
        plt.close()
        
        counter += 1
    
    return f"{counter} .png files have been created in the {save_dir} folder"