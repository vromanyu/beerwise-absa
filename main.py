import pandas as pd

from modules.dataframe_creator.df_creator import generate_processed_dataframe_chunks

if __name__ == "__main__":
    df = pd.DataFrame()
    dataframes = generate_processed_dataframe_chunks()
    for dataframe in dataframes:
        df = pd.concat([df, dataframe], ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    print(df)
