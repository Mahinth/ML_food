import pandas as pd

def validate_data(df):
    """
    Validates if the uploaded dataframe has the required columns.
    """
    required_columns = ['timestamp', 'food_demand']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    return True, "Validation successful"

def initial_preprocess(df):
    """
    Basic preprocessing such as converting timestamp to datetime.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df
