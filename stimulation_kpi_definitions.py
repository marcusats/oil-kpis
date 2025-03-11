import pandas as pd
import json

def get_kpis(mapping_path):
    """
    Returns all predefined KPI functions as a dictionary.
    """
    # Load column mapping dynamically from the provided JSON file
    with open(mapping_path, "r", encoding="utf-8") as f:
        COLUMN_MAPPING = json.load(f)

    # Extract column groups based on "group" instead of "category"
    COLUMN_GROUPS = {}

    for column_name, properties in COLUMN_MAPPING.items():
        group = properties.get("group")  # ✅ Use "group" instead of "category"
        if group:
            if group not in COLUMN_GROUPS:
                COLUMN_GROUPS[group] = []
            COLUMN_GROUPS[group].append(column_name)

    ### Predefined KPI Functions
    def total_chemical_percentage(df):
        """
        Calculates the total volume of chemicals used and their percentage contribution.
        """
        chemical_columns = COLUMN_GROUPS["Chemical Usage"]  # Uses dynamically loaded groups

        total_volume = df[chemical_columns].sum().sum()  # Total chemical usage
        chemical_percentages = (df[chemical_columns].sum() / total_volume) * 100  

        return {"Percentage Breakdown": chemical_percentages.to_dict()}

    def total_chemical_usage_per_lease(df):
        """
        Calculates the total volume of chemicals used per lease.
        """
        chemical_columns = COLUMN_GROUPS["Chemical Usage"]
        return df.groupby("Lease")[chemical_columns].sum().to_dict()

    def total_number_of_activities(df):
        """
        Calculates the total number of stimulation activities performed.
        """
        return {"Total Activities": len(df)}

    def total_revenue_per_intervention(df):
        """
        Calculates total revenue generated per intervention type.
        """
        return df.groupby("Intervention Type")[COLUMN_GROUPS["Costs"][1]].sum().to_dict()

    return {
        "Total Chemical Percentage": total_chemical_percentage,
        "Total Chemical Usage per Lease": total_chemical_usage_per_lease,
        "Total Number of Activities": total_number_of_activities,
        "Total Revenue per Intervention": total_revenue_per_intervention
    }
