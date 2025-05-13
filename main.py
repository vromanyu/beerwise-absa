from modules.dataframe_creator.df_creator import generate_processed_dataframe
import pandas as pd
import modules.processing.processor as p
import modules.algorithms.multinomial_nb as m


def main():
    return generate_processed_dataframe()


if __name__ == "__main__":
    final_df = pd.DataFrame()
    frames = main()
    for frame in frames:
        final_df = pd.concat([final_df, frame], ignore_index=True)
    final_df.reset_index(inplace=True, drop=True)
    final_df.to_excel("./dataset/sample_dataset_as_excel.xlsx",
                      index=False, engine="openpyxl")
    df: pd.DataFrame = m.load_dataframe(
        f"dataset/sample_dataset_as_excel.xlsx")
