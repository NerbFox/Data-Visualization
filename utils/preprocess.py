import streamlit as st
import pandas as pd
from pathlib import Path

@st.cache_data
def preprocess_data(
    dfs: list = [], 
    columns: list = [
        'Government expenditure on education, total (% of government expenditure)',
        'Average years of schooling',
        'GDP per capita, PPP (constant 2021 international $)',
        'Productivity: output per hour worked',
        'Unemployment, total (% of total labor force) (modeled ILO estimate)',
        'Literacy rate',
        'Combined - average years of education for 15-64 years male and female youth and adults',
        '$3.65 a day - Share of population in poverty',
        ],
    ):
    
    # cleaning missing values
    # We need to modify the code to only check for columns that exist in each dataframe
    for df in dfs:
        # Get the intersection of important columns and existing columns in the dataframe
        columns_to_check = [col for col in columns if col in df.columns]
        if columns_to_check:  # Only drop if there are columns to check
            df.dropna(subset=columns_to_check, inplace=True)

    # cleaning duplicates
    for df in dfs:
        df.drop_duplicates(inplace=True)
        
@st.cache_data
def get_gdp_data():
    """
    Load and transform GDP data from CSV.

    Returns:
        pd.DataFrame: DataFrame with columns ['Country Name', 'Country Code', 'Year', 'GDP']
    """
    DATA_FILENAME = Path(__file__).parent / '../data/gdp.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2023

    # Pivot year columns into 'Year' and 'GDP'
    gdp_df = raw_gdp_df.melt(
        id_vars=['Country Code', 'Country Name'],
        value_vars=[str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        var_name='Year',
        value_name='GDP',
    )

    # Convert 'Year' to integer
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

@st.cache_data
def get_education_expenditure_data():
    """
    Load and transform education expenditure data from CSV.

    Returns:
        pd.DataFrame: DataFrame with columns ['Entity', 'Code', 'Year', 'Education Expenditure Share']
    """
    DATA_FILENAME = Path(__file__).parent / '../data/share-of-education-in-government-expenditure.csv'
    raw_df = pd.read_csv(DATA_FILENAME)
    
    raw_df['Year'] = pd.to_numeric(raw_df['Year'], errors='coerce')
    
    raw_df['Government expenditure on education, total (% of government expenditure)'] = pd.to_numeric(
        raw_df['Government expenditure on education, total (% of government expenditure)'], errors='coerce'
    )

    return raw_df[['Entity', 'Code', 'Year', 'Government expenditure on education, total (% of government expenditure)']]

@st.cache_data
def get_avg_years_school_gdp():
    """
    Load average years of schooling vs GDP data.

    Returns:
        pd.DataFrame: DataFrame with columns ['Country Code', 'Year', 'Average Years of Schooling', 'GDP']
    """
    DATA_FILENAME = 'data/average-years-of-schooling-vs-gdp-per-capita.csv'
    df = pd.read_csv(DATA_FILENAME)
    df['Year'] = pd.to_numeric(df['Year'])
    return df

@st.cache_data
def get_first_last_value(df, column_name, code, from_year, to_year):
    """
    Safely extracts the first and last value for a specific column in a DataFrame,
    filtered by country code and year. Returns NaN if the row is missing.
    """
    first_row = df[(df['Year'] == from_year) & (df['Code'] == code)]
    last_row = df[(df['Year'] == to_year) & (df['Code'] == code)]

    first_value = first_row[column_name].iat[0] if not first_row.empty else float('nan')
    last_value = last_row[column_name].iat[0] if not last_row.empty else float('nan')

    return first_value, last_value

@st.cache_data
def get_common_year_range(country_code, df_column_pairs):
    """
    Given a country code and a list of (DataFrame, column_name) pairs,
    return the overlapping available year range across all datasets based on
    non-NaN values in their respective value columns.
    """
    year_ranges = []

    for idx, (df, value_column) in enumerate(df_column_pairs):
        if not {'Code', 'Year', value_column}.issubset(df.columns):
            print(f"Dataset {idx} skipped: missing 'Code', 'Year', or '{value_column}' column.")
            continue

        # Clean and filter
        df = df.copy()
        df['Code'] = df['Code'].astype(str).str.strip().str.upper()

        filtered = df[
            (df['Code'] == country_code.strip().upper()) &
            (df['Year'].notna()) &
            (pd.to_numeric(df['Year'], errors='coerce').notnull()) &
            (df[value_column].notna())
        ]
        filtered['Year'] = filtered['Year'].astype(int)

        if not filtered.empty:
            min_year = filtered['Year'].min()
            max_year = filtered['Year'].max()
            year_ranges.append((min_year, max_year))
        else:
            print(f"Dataset {idx} has no valid data for {country_code} in '{value_column}'.")

    if not year_ranges:
        return None

    print(f"Year ranges for {country_code}: {year_ranges}")
    # Find the overlapping year range
    overall_start = max(start for start, _ in year_ranges)
    overall_end = min(end for _, end in year_ranges)

    if overall_start > overall_end:
        return None  # No overlap

    return overall_start, overall_end

@st.cache_data
def get_country_name(code, df, code_column='Code', name_column='Entity'):
    """
    Given a country code and a DataFrame, return the corresponding country name.
    """
    code = str(code).strip().upper()

    if code_column not in df.columns or name_column not in df.columns:
        raise ValueError(f"Missing '{code_column}' or '{name_column}' column in DataFrame.")

    match = df[df[code_column].astype(str).str.strip().str.upper() == code]

    if not match.empty:
        return match[name_column].iloc[0]
    return None