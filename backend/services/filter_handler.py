import pandas as pd
import json

class FilterHandler:
    def __init__(self, df, mapping_path):
        self.df = df
        
        # Load column mapping from JSON file
        try:
            with open(mapping_path, "r") as f:
                self.mapping = json.load(f)
        except Exception as e:
            self.mapping = {}
            print(f"Error loading mapping file: {str(e)}")
            
        # Initialize column types
        self.categorical_cols = []
        self.numerical_cols = {}
        self.date_cols = []
        
        self._initialize_column_types()
    
    def _initialize_column_types(self):
        """Identify column types from mapping and data"""
        # From mapping
        if self.mapping:
            for col_type, cols in self.mapping.get("columns", {}).items():
                if col_type == "categorical":
                    self.categorical_cols.extend(cols)
                elif col_type in ["numerical", "metric"]:
                    self.numerical_cols[col_type] = cols
                elif col_type == "date":
                    self.date_cols.extend(cols)
        
        # Auto-detect from dataframe
        for col in self.df.columns:
            if col not in self.categorical_cols and col not in self.date_cols and not any(col in cols for cols in self.numerical_cols.values()):
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    if "metric" not in self.numerical_cols:
                        self.numerical_cols["metric"] = []
                    self.numerical_cols["metric"].append(col)
                elif pd.api.types.is_datetime64_dtype(self.df[col]):
                    self.date_cols.append(col)
                else:
                    self.categorical_cols.append(col)
    
    def get_available_filters(self):
        """Return available filter columns by type"""
        return {
            "categorical": self.categorical_cols,
            "numerical": self.numerical_cols,
            "date": self.date_cols
        }
    
    def apply_filters(self, filters):
        """Apply filters to dataframe"""
        filtered_df = self.df.copy()
        
        # Apply categorical filters
        for col, value in filters.get("categorical", {}).items():
            if col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col] == value]
        
        # Apply numerical filters
        for col, range_vals in filters.get("numerical", {}).items():
            if col in filtered_df.columns and len(range_vals) == 2:
                filtered_df = filtered_df[(filtered_df[col] >= range_vals[0]) & 
                                          (filtered_df[col] <= range_vals[1])]
        
        # Apply date filters
        for col, date_range in filters.get("date", {}).items():
            if col in filtered_df.columns and len(date_range) == 2:
                # Convert string dates to datetime
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                
                # Convert column to datetime if it's not already
                if not pd.api.types.is_datetime64_dtype(filtered_df[col]):
                    filtered_df[col] = pd.to_datetime(filtered_df[col])
                
                filtered_df = filtered_df[(filtered_df[col] >= start_date) & 
                                          (filtered_df[col] <= end_date)]
        
        return filtered_df 