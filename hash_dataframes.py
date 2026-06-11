import hashlib
import pandas as pd

def hash_row(row):
    # Fix float precision for consistency
    row_str = "_".join(f"{x:.6f}" for x in row.values)
    
    # Deterministic hash
    return hashlib.sha256(row_str.encode()).hexdigest()


def add_row_hash(df, precision=6, column_name="row_hash"):
    """
    Adds a deterministic hash column to a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        precision (int): Number of decimal places for floats
        column_name (str): Name of the output hash column

    Returns:
        pd.DataFrame: Copy of DataFrame with added hash column
    """
    df_copy = df.copy()

    def hash_row(row):
        row_str = "_".join(f"{x:.{precision}f}" for x in row.values)
        return hashlib.sha256(row_str.encode()).hexdigest()

    df_copy[column_name] = df_copy.apply(hash_row, axis=1)

    return df_copy



if __name__ == '__main__':

    df_current = pd.read_csv('csv_files/samples_current.csv')
    df_current = df_current[df_current.columns[:-1]]
    df_waves = pd.read_csv('csv_files/samples_swh.csv')
    df_waves = df_waves[df_waves.columns[:-1]]

    df_waves_hashed = add_row_hash(df_waves)
    df_current_hashed = add_row_hash(df_current)

    assert df_waves_hashed['row_hash'].is_unique
    assert df_current_hashed['row_hash'].is_unique

    df_waves_hashed.to_csv('./csv_files/waves_hashed.csv')
    df_current_hashed.to_csv('./csv_files/current_hashed.csv')


