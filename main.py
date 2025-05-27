from pathlib import Path
from utils import test

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# test()

# Kalau mau tema gelap ini
# sns.set_theme(style="dark")

# -----------------------------------------------------------------------------
# Streamlit page configuration

st.set_page_config(
    page_title='Nerb Dashboard',
    page_icon=':earth_americas:',
    # layout='centered',
    layout='wide',
)

# -----------------------------------------------------------------------------
# Data loading and transformation

@st.cache_data
def get_gdp_data():
    """
    Load and transform GDP data from CSV.

    Returns:
        pd.DataFrame: DataFrame with columns ['Country Name', 'Country Code', 'Year', 'GDP']
    """
    DATA_FILENAME = Path(__file__).parent / 'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

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

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Page content

st.markdown("""
# :earth_americas: GDP Dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website.
The data currently goes up to 2022, and some years may have missing data points.
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
default_countries = ['IND', 'USA', 'CHN', 'JPN', 'KOR', 'SGP']

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
    st.line_chart(
        filtered_gdp_df,
        x='Year',
        y='GDP',
        color='Country Code',
    )
else:
    st.info("No data available for the selected countries and years.")