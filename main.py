from pathlib import Path
from utils import test, create_bubble, create_plot

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px

# test()

# Kalau mau tema gelap ini
# sns.set_theme(style="dark")

# -----------------------------------------------------------------------------
# Streamlit page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title='Punya Nigel Nerb Dashboard',
    page_icon=':earth_americas:',
    # layout='centered',
    layout='wide',
)

# -----------------------------------------------------------------------------
# Data loading and transformation
# -----------------------------------------------------------------------------


# First Five Rows:
#         Entity Code  Year  Average years of schooling  GDP per capita, PPP (constant 2021 international $) Population (historical) World regions according to OWID 
# 0  Afghanistan  AFG  1990                    0.871962                                                NaN                12045622.0                             NaN 
# 1  Afghanistan  AFG  1991                    0.915267                                                NaN                12238831.0                             NaN 

@st.cache_data
def get_gdp_data():
    """
    Load and transform GDP data from CSV.

    Returns:
        pd.DataFrame: DataFrame with columns ['Country Name', 'Country Code', 'Year', 'GDP']
    """
    DATA_FILENAME = Path(__file__).parent / 'data/gdp.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2023

    # Pivot year columns into 'Year' and 'GDP'
    gdp_df = raw_gdp_df.melt(
        id_vars=['Country Code'],
        value_vars=[str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        var_name='Year',
        value_name='GDP',
    )

    # Convert 'Year' to integer
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

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
    for df in df_list_cleaned:
        # Get the intersection of important columns and existing columns in the dataframe
        columns_to_check = [col for col in columns if col in df.columns]
        if columns_to_check:  # Only drop if there are columns to check
            df.dropna(subset=columns_to_check, inplace=True)

    # cleaning duplicates
    for df in df_list_cleaned:
        df.drop_duplicates(inplace=True)

import os
DATA_DIR =  os.path.join(os.path.dirname(__file__), 'data/')

gdp_df = get_gdp_data()
df_avg_years_school_gdp = get_avg_years_school_gdp()
education_gdp = pd.read_csv(DATA_DIR + "average-years-of-schooling-vs-gdp-per-capita.csv")
education_expenditure = pd.read_csv(DATA_DIR + "share-of-education-in-government-expenditure.csv")
education_productivity = pd.read_csv(DATA_DIR + "productivity-vs-educational-attainment.csv")
unemployment = pd.read_csv(DATA_DIR + "unemployment-rate.csv")
education_literacy = pd.read_csv(DATA_DIR + "literacy-rates-vs-average-years-of-schooling.csv")
education_poverty = pd.read_csv(DATA_DIR + "poverty-vs-mean-schooling.csv")

df_list_cleaned = [
    gdp_df,
    df_avg_years_school_gdp,
    education_gdp,
    education_expenditure,
    education_productivity,
    unemployment,
    education_literacy,
    education_poverty,
]

# Preprocess the data
preprocess_data(dfs=df_list_cleaned)

# -------------------------------------------------------------
# Page content

st.markdown("""
# :earth_americas: Education Dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. The data currently goes up to 2022, and some years may have missing data points.
""")

# -----------------------------------------------------------------------------
# User controls

min_year = int(gdp_df['Year'].min())
max_year = int(gdp_df['Year'].max())

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_year,
    max_value=max_year,
    value=[min_year, max_year]
)

countries = gdp_df['Country Code'].unique()
# Indonesia, Singapore, USA, China, Japan, Korea (3 letters)
# There is n/a for certain years in some countries, 
default_countries = ['SGP', 'USA', 'CHN', 'JPN', 'KOR']

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    options=sorted(countries),
    default=[c for c in default_countries if c in countries]
)

if not selected_countries:
    st.warning("Select at least one country to display data.")

# -----------------------------------------------------------------------------
# Data filtering

filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries)) &
    (gdp_df['Year'] >= from_year) &
    (gdp_df['Year'] <= to_year)
]

# -----------------------------------------------------------------------------
# Metrics for selected years

st.header(f'GDP in {to_year}', divider='gray')

# Set up columns for displaying metrics
cols = st.columns(6)

first_year_df = gdp_df[gdp_df['Year'] == from_year]
last_year_df = gdp_df[gdp_df['Year'] == to_year]

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]
    with col:
        # Get GDP values for the first and last year
        first_gdp_row = first_year_df[first_year_df['Country Code'] == country]['GDP']
        last_gdp_row = last_year_df[last_year_df['Country Code'] == country]['GDP']

        first_gdp = first_gdp_row.iat[0] / 1e9 if not first_gdp_row.empty else float('nan')
        last_gdp = last_gdp_row.iat[0] / 1e9 if not last_gdp_row.empty else float('nan')

        if math.isnan(first_gdp) or first_gdp == 0:
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )
        
# -----------------------------------------------------------------------------
# Visualization

st.header('GDP over time', divider='gray')

if not filtered_gdp_df.empty:
    # Add GDP in billions of USD
    filtered_gdp_df['GDP (Billions USD)'] = filtered_gdp_df['GDP'] / 1e9

    # Show table with year (no comma), GDP (USD), and GDP (Billions USD)
    # display_df = filtered_gdp_df[['Country Code', 'Year', 'GDP', 'GDP (Billions USD)']].copy()
    # display_df['Year'] = display_df['Year'].astype(int).astype(str)  # Ensure no comma
    # st.dataframe(display_df, use_container_width=True)

    df_avg_years_school_gdp.rename(
        columns={
            'Entity': 'Country',
            'Code': 'Country Code',
            'Average years of schooling': 'Average Years of Schooling',
            'GDP per capita, PPP (constant 2021 international $)': 'GDP (PPP)',
            'Population (historical)': 'Population',
            'World regions according to OWID': 'Region (OWID)'
        },
        inplace=True
    )

    # if df_avg_years_school_gdp['Country Code'].isin(selected_countries):
    
    
    display_df = df_avg_years_school_gdp[
        df_avg_years_school_gdp['Country Code'].isin(selected_countries) &
        (df_avg_years_school_gdp['Year'] == to_year)
    ]
        
    display_df['Year'] = display_df['Year'].astype(int).astype(str)  # Ensure no comma
    st.dataframe(display_df, use_container_width=True)

    bubble_plot = create_bubble(display_df, 
                                #  x_label='Year', 
                                 x_label='Average Years of Schooling', 
                                 y_label='GDP (PPP)', 
                                 size_label='Population', 
                                 text_label='Country')
    
    st.plotly_chart(bubble_plot, use_container_width=True)
    
    # scatter_plot with min/max size for bubbles
    st.scatter_chart(
        data=display_df,
        x='GDP (PPP)',
        y='Average Years of Schooling',
        color='Country Code',
        use_container_width=True,
        size='Population',
        # size_min=10,   # minimal bubble size
        # size_max=60,   # maximal bubble size
    )
    
    # Create a line plot
    display_df = df_avg_years_school_gdp[
        df_avg_years_school_gdp['Country Code'].isin(selected_countries) &
        (df_avg_years_school_gdp['Year'] >= from_year) &
        (df_avg_years_school_gdp['Year'] <= to_year) 
    ]
    
    st.line_chart(
        data=display_df,
        x='GDP (PPP)',
        y='Average Years of Schooling', 
        color='Country Code',
        use_container_width=True,
    )
    
    st.line_chart(
        data=filtered_gdp_df,
        x='Year', 
        y='GDP (Billions USD)',
        color='Country Code',
        use_container_width=True,
    )
else:
    st.info("No data available for the selected countries and years.")
 
