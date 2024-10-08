def SetDatetimeCols(dataframe):
    """
    Converts specified columns in a dataframe to datetime format.
    
    Args:
        dataframe (pandas.DataFrame): The input dataframe.
        
    Returns:
        pandas.DataFrame: The dataframe with specified columns converted to datetime format.
    """
    import pandas as pd
    dataframe = dataframe.copy()
    
    datetime_cols = ['release',
                     'published_store',
                     'published_meta',
                     'published_stsp',
                     'published_hltb',
                     'published_igdb',
                     'all_time_peak_date']
    
    for col in datetime_cols:
        if col in dataframe.columns:
            dataframe[col] = pd.to_datetime(dataframe[col])

    return dataframe

def URLMerge(df1, df1_url_column: str, df2, df2_url_column: str, how='inner'):
    '''
    Merge two DataFrames based on URL columns.
    
    Parameters:
        df1 (pd.DataFrame): The first DataFrame to be merged.
        df1_url_column (str): The name of the URL column in the first DataFrame.
        df2 (pd.DataFrame): The second DataFrame to be merged.
        df2_url_column (str): The name of the URL column in the second DataFrame.
        how (str, optional): The type of merge to be performed. Defaults to 'inner'.
    
    Returns:
        pd.DataFrame: The merged DataFrame.
    '''
    
    import pandas as pd

    merge_col = 'merge_col'
    
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    df1_copy[merge_col] = df1_copy[df1_url_column].str.extract(r'app/(\d+)')
    df2_copy[merge_col] = df2_copy[df2_url_column].str.extract(r'app/(\d+)')
    
    return pd.merge(df1_copy, df2_copy, on=merge_col, how=how)

def StringCompare(str1: str, str2: str, thratio: float=None):
    """
    Calculates the similarity ratio between two strings.

    Parameters:
    str1 (str): The first string to compare.
    str2 (str): The second string to compare.
    thratio (float, optional): The similarity ratio threshold required for the strings to be considered similar.
                                If None, the similarity ratio is returned without comparison to the specified ratio.

    Returns:
    bool: True if the similarity ratio between the strings is greater than or equal to the specified ratio,
          False otherwise.
    ratio (float): The similarity ratio between the strings if ratio is None.
    """
    from difflib import SequenceMatcher

    if thratio == None:
        return SequenceMatcher(None, str1, str2).ratio()
    else:
        return SequenceMatcher(None, str1, str2).ratio() >= thratio