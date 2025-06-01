import streamlit as st
import pandas as pd
import os
from config import Config

@st.cache_data
def load_all_datasets():
    """Load all datasets and return as a dictionary."""
    datasets = {}
    
    # GDP data
    datasets['gdp'] = get_gdp_data()
    datasets['avg_years_school_gdp'] = get_avg_years_school_gdp()
    datasets['education_expenditure'] = get_education_expenditure_data()
    
    # Direct CSV loads with consistent naming
    csv_files = {
        'education_gdp': "average-years-of-schooling-vs-gdp-per-capita.csv",
        'education_productivity': "productivity-vs-educational-attainment.csv", 
        'unemployment': "unemployment-rate.csv",
        'education_literacy': "literacy-rates-vs-average-years-of-schooling.csv",
        'education_poverty': "poverty-vs-mean-schooling.csv",
        'pisa_reading': "pisa-reading-scores.csv",
        'pisa_math': "pisa-math-scores.csv",
        'pisa_science': "pisa-science-scores.csv"
    }
    
    for key, filename in csv_files.items():
        datasets[key] = pd.read_csv(os.path.join(Config.DATA_DIRPATH, filename))

    # Process PISA average
    datasets['pisa_avg'] = create_pisa_average(
        datasets['pisa_reading'], 
        datasets['pisa_math'], 
        datasets['pisa_science']
    )
    
    return datasets

def create_pisa_average(pisa_reading, pisa_math, pisa_science):
    """Create averaged PISA scores."""
    pisa_avg = pisa_reading[['Countries', 'PISA reading scores, 2022']].merge(
        pisa_math[['Countries', 'PISA math scores, 2022']], on='Countries', how='outer'
    ).merge(
        pisa_science[['Countries', 'PISA science scores, 2022']], on='Countries', how='outer'
    )
    
    pisa_avg['PISA average scores, 2022'] = pisa_avg[
        ['PISA reading scores, 2022', 'PISA math scores, 2022', 'PISA science scores, 2022']
    ].mean(axis=1, skipna=True)
    
    return pisa_avg

# Move existing functions here from preprocess.py
@st.cache_data
def get_gdp_data():
    """
    Load and transform GDP data from CSV.

    Returns:
        pd.DataFrame: DataFrame with columns ['Country Name', 'Country Code', 'Year', 'GDP']
    """
    DATA_FILENAME = os.path.join(Config.DATA_DIRPATH, 'gdp.csv')
    MIN_YEAR = Config.MIN_YEAR
    MAX_YEAR = Config.MAX_YEAR
    
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

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
    DATA_FILENAME = os.path.join(Config.DATA_DIRPATH, 'share-of-education-in-government-expenditure.csv')
    
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
    DATA_FILENAME = os.path.join(Config.DATA_DIRPATH, 'average-years-of-schooling-vs-gdp-per-capita.csv')
    
    df = pd.read_csv(DATA_FILENAME)
    df['Year'] = pd.to_numeric(df['Year'])
    
    return df