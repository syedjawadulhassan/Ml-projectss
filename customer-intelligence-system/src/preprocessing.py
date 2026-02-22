import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df