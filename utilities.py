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