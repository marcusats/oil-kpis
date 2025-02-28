import pandas as pd

class FilterHandler:
    def __init__(self, df):
        """
        Initialize with a dataset.
        :param df: Pandas DataFrame containing standardized data.
        """
        self.df = df

    def get_available_filters(self):
        """
        Automatically detect possible filters based on dataset columns.
        :return: Dictionary of available filters (categorical & numerical).
        """
        available_filters = {"categorical": [], "numerical": []}

        for column in self.df.columns:
            unique_values = self.df[column].nunique()

            # If column has few unique values, it's a categorical filter
            if unique_values < 20 and self.df[column].dtype == "object":
                available_filters["categorical"].append(column)

            # If column contains continuous numerical values, it's a range filter
            elif self.df[column].dtype in ["int64", "float64"]:
                available_filters["numerical"].append(column)

        return available_filters

    def apply_filters(self, filters):
        """
        Apply user-defined filters dynamically.
        :param filters: Dictionary containing filter parameters.
        :return: Filtered DataFrame.
        """
        filtered_df = self.df.copy()

        # Apply categorical filters
        for column, value in filters.get("categorical", {}).items():
            if column in filtered_df.columns and value != "All":
                filtered_df = filtered_df[filtered_df[column] == value]

        # Apply numerical range filters
        for column, range_values in filters.get("numerical", {}).items():
            if column in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[column] >= range_values[0]) & 
                    (filtered_df[column] <= range_values[1])
                ]

        return filtered_df
