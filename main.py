from pathlib import Path
from utils import (
    create_bubble, 
    create_plot, 
    preprocess_data, 
    get_gdp_data, 
    get_avg_years_school_gdp, 
    get_first_last_value, 
    get_country_name,
    get_education_expenditure_data
)

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------------------------------------------------------
# Streamlit page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title='Education Dashboard',
    page_icon=':earth_americas:',
    # layout='centered',
    layout='wide',
)

# -----------------------------------------------------------------------------
# Data loading and transformation
# -----------------------------------------------------------------------------

import os
DATA_DIR =  os.path.join(os.path.dirname(__file__), 'data/')

gdp_df = get_gdp_data()
df_avg_years_school_gdp = get_avg_years_school_gdp()

education_gdp = pd.read_csv(DATA_DIR + "average-years-of-schooling-vs-gdp-per-capita.csv")
education_expenditure = get_education_expenditure_data()
education_productivity = pd.read_csv(DATA_DIR + "productivity-vs-educational-attainment.csv")
unemployment = pd.read_csv(DATA_DIR + "unemployment-rate.csv")
education_literacy = pd.read_csv(DATA_DIR + "literacy-rates-vs-average-years-of-schooling.csv")
education_poverty = pd.read_csv(DATA_DIR + "poverty-vs-mean-schooling.csv")
pisa_reading_scores = pd.read_csv(DATA_DIR + "pisa-reading-scores.csv")
pisa_math_scores = pd.read_csv(DATA_DIR + "pisa-math-scores.csv")
pisa_science_scores = pd.read_csv(DATA_DIR + "pisa-science-scores.csv")

# Merge the three PISA DataFrames on 'Countries' and calculate the average row-wise
pisa_avg = pisa_reading_scores[['Countries', 'PISA reading scores, 2022']].merge(
    pisa_math_scores[['Countries', 'PISA math scores, 2022']],
    on='Countries',
    how='outer'
).merge(
    pisa_science_scores[['Countries', 'PISA science scores, 2022']],
    on='Countries',
    how='outer'
)
pisa_avg['PISA average scores, 2022'] = pisa_avg[
    ['PISA reading scores, 2022', 'PISA math scores, 2022', 'PISA science scores, 2022']
].mean(axis=1, skipna=True)

df_list_cleaned = [
    gdp_df,
    df_avg_years_school_gdp,
    education_gdp,
    education_expenditure,
    education_productivity,
    unemployment,
    education_literacy,
    education_poverty,
    pisa_reading_scores,
    pisa_math_scores,
    pisa_science_scores
]

# Preprocess the data
preprocess_data(dfs=df_list_cleaned)

# -------------------------------------------------------------
# Page content
# :earth_americas:   world icon
st.markdown("""
# Education Dashboard 
This dashboard provides insights into the relationship between education and economic indicators across various countries.
""")

# -----------------------------------------------------------------------------
# User controls

min_year = int(gdp_df['Year'].min())
max_year = int(gdp_df['Year'].max())

# choose dropdown for main country
st.sidebar.header('User Controls')
st.sidebar.markdown("""
Select the year range and countries to analyze the data.
""")




# dropdown for countries
# highlight_country = st.selectbox(
#     'Main Country',
#     options=sorted(education_gdp['Entity'].unique()),
#     index=education_gdp['Entity'].unique().tolist().index('Indonesia') if 'Indonesia' in education_gdp['Entity'].unique() else 0
# )
sorted_country_names = sorted(gdp_df['Country Name'].unique())
highlight_country = st.sidebar.selectbox(
    'Main Country',
    options=sorted_country_names,
    index = sorted_country_names.index('Indonesia') if 'Indonesia' in sorted_country_names else 0
)

entity_to_code = gdp_df.dropna(subset=['Country Code']).drop_duplicates("Country Name")[["Country Name", "Country Code"]].set_index("Country Name")["Country Code"].to_dict()
code_to_entity = gdp_df.dropna(subset=['Country Code']).drop_duplicates("Country Code")[["Country Code", "Country Name"]].set_index("Country Code")["Country Name"].to_dict()

highlight_country = entity_to_code.get(highlight_country, None)

from_year, to_year = st.sidebar.slider(
    'Select Year Range',
    min_value=min_year,
    max_value=max_year,
    value=[min_year, max_year]
)


countries = gdp_df['Country Code'].unique()

# Indonesia, Singapore, USA, China, Japan, Korea (3 letters)
# There is n/a for certain years in some countries, 
default_countries = ['SGP', 'USA', 'CHN', 'JPN', 'KOR']

    
selected_countries = st.sidebar.multiselect(
    'Which countries would you like to view?',
    options=sorted(countries),
    default=[c for c in [highlight_country if highlight_country not in default_countries else None] + default_countries if c in countries ],
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


filtered_education_expenditure = education_expenditure[
    (education_expenditure['Entity'].isin(selected_countries)) &
    (education_expenditure['Year'] >= from_year) &
    (education_expenditure['Year'] <= to_year)
]


# -----------------------------------------------------------------------------
# Metrics for selected years

# Filter and rename columns beforehand
metric_education_expenditure = education_expenditure[
    (education_expenditure['Code'] == highlight_country) &
    (education_expenditure['Government expenditure on education, total (% of government expenditure)'].notna())
][['Entity', 'Code', 'Year', 'Government expenditure on education, total (% of government expenditure)']].rename(
    columns={'Government expenditure on education, total (% of government expenditure)': 'Expenditure'}
)

metric_education_literacy = education_literacy[
    (education_literacy['Code'] == highlight_country) &
    (education_literacy['Literacy rate'].notna())
][['Entity', 'Code', 'Year', 'Literacy rate']].rename(columns={'Literacy rate': 'Literacy'})

metric_avg_years_school_gdp = df_avg_years_school_gdp[
    (df_avg_years_school_gdp['Code'] == highlight_country) &
    (df_avg_years_school_gdp['Average years of schooling'].notna())
][['Entity', 'Code', 'Year', 'Average years of schooling']].rename(columns={'Average years of schooling': 'Schooling'})

merged_df = metric_education_expenditure.merge(metric_education_literacy, on=['Entity', 'Code', 'Year'], how='inner').merge(metric_avg_years_school_gdp, on=['Entity', 'Code', 'Year'], how='inner')

first_row = merged_df.iloc[0]
last_row = merged_df.iloc[-1]
metric_final_year = last_row['Year']

st.subheader(f'Highlights Up to {last_row["Year"]}', divider='gray')

first_expenditure = metric_education_expenditure[metric_education_expenditure['Code'] == 'IDN']['Expenditure'].iat[0] if not metric_education_expenditure.empty else "No Data"
start_expenditure_year = metric_education_expenditure[metric_education_expenditure['Code'] == 'IDN']['Year'].iat[0]
last_expenditure = last_row['Expenditure']

first_literacy = metric_education_literacy[metric_education_literacy['Code'] == 'IDN']['Literacy'].iat[0] if not metric_education_literacy.empty else "No Data"
start_literacy_year = metric_education_literacy[metric_education_literacy['Code'] == 'IDN']['Year'].iat[0]
last_literacy = last_row['Literacy']

first_average_years_school = metric_avg_years_school_gdp[metric_avg_years_school_gdp['Code'] == 'IDN']['Schooling'].iat[0] if not metric_avg_years_school_gdp.empty else "No Data"
start_average_school_gdp_year = metric_avg_years_school_gdp[metric_avg_years_school_gdp['Code'] == 'IDN']['Year'].iat[0]
last_average_years_school = last_row['Schooling']

highlight_country_name = get_country_name(highlight_country, df_avg_years_school_gdp)

pisa_reading = pisa_reading_scores.loc[
    (pisa_reading_scores['Countries'] == highlight_country_name) & 
    pisa_reading_scores['PISA reading scores, 2022'].notna(),
    'PISA reading scores, 2022'
]

pisa_math = pisa_math_scores.loc[
    (pisa_math_scores['Countries'] == highlight_country_name) & 
    pisa_math_scores['PISA math scores, 2022'].notna(),
    'PISA math scores, 2022'
]

pisa_science = pisa_science_scores.loc[
    (pisa_science_scores['Countries'] == highlight_country_name) & 
    pisa_science_scores['PISA science scores, 2022'].notna(),
    'PISA science scores, 2022'
]

highlight_pisa_reading_score = pisa_reading.iat[0] if not pisa_reading.empty else "No Data"
highlight_pisa_math_score = pisa_math.iat[0] if not pisa_math.empty else "No Data"
highlight_pisa_science_score = pisa_science.iat[0] if not pisa_science.empty else "No Data"

if "No Data" in (highlight_pisa_reading_score, highlight_pisa_math_score, highlight_pisa_science_score):
    average_pisa_score = "No Data"
else:
    average_pisa_score = (highlight_pisa_reading_score + highlight_pisa_math_score + highlight_pisa_science_score) / 3

indicators = [
    {
        "label": "Avg. PISA Scores 2022",
        "first": average_pisa_score,
        "last": average_pisa_score,
        "help": f''' **PISA Reading Score**  
        {highlight_pisa_reading_score if highlight_pisa_reading_score != "No Data" else "No Data"}  
        **PISA Math Score**  
        {highlight_pisa_math_score if highlight_pisa_math_score != "No Data" else "No Data"}  
        **PISA Science Score**  
        {highlight_pisa_science_score if highlight_pisa_science_score != "No Data" else "No Data"}  
        ''',
        "unit": ""
    },
    {
        "label": "Education Expenditure",
        "first": first_expenditure,
        "last": last_expenditure,
        "help": f"Data shown is from {start_expenditure_year} - {metric_final_year}.",
        "unit": "%"
    },
    {
        "label": "Avg. Years of Schooling",
        "first": first_average_years_school,
        "last": last_average_years_school,
        "help": f"Data shown is from {start_average_school_gdp_year} - {metric_final_year}.",
        "unit": ""
    },
    {
        "label": "Literacy Rate",
        "first": first_literacy,
        "last": last_literacy,
        "help": f"Data shown is from {start_literacy_year} - {metric_final_year}.",
        "unit": "%"
    },
]

cols = st.columns(len(indicators))


for i, indicator in enumerate(indicators):
    col = cols[i % len(cols)]
    with col:
        first = indicator["first"]
        last = indicator["last"]
        label = indicator["label"]
        unit = indicator["unit"]

        # If either value is a string (e.g., "No Data"), display as-is
        if isinstance(first, str) or isinstance(last, str):
            value = last if isinstance(last, str) else first
            growth = ""
            delta_color = "off"
        elif math.isnan(first) or math.isnan(last) or first == 0 or first - last == 0:
            growth = ""
            delta_color = "off"
            value = f"{last:,.2f}{unit}" if unit else f"{last:,.2f}"
        else:
            growth = f"{last / first:,.2f}x"
            delta_color = "normal"
            value = f"{last:,.2f}{unit}" if unit else f"{last:,.2f}"

        st.metric(
            label=f"{label}",
            value=value,
            delta=growth,
            delta_color=delta_color,
            help=indicator['help'] if 'help' in indicator else ""
        )

        
# -----------------------------------------------------------------------------
# Visualization

st.subheader('Education Relationships', divider='gray')

if filtered_education_expenditure.empty:
    st.info("No education expenditure data available for the selected countries and years.")
else:
    pass

# if not filtered_gdp_df.empty:
#     # Add GDP in billions of USD
#     filtered_gdp_df = filtered_gdp_df.copy()
#     filtered_gdp_df['GDP (Billions USD)'] = filtered_gdp_df['GDP'] / 1e9
# else:
#     st.info("No data available for the selected countries and years.")

# Prepare data for bubble and line plots
avg_schooling_df = df_avg_years_school_gdp.rename(
    columns={
        'Entity': 'Country',
        'Code': 'Country Code',
        'Average years of schooling': 'Average Years of Schooling',
        'GDP per capita, PPP (constant 2021 international $)': 'GDP (PPP)',
        'Population (historical)': 'Population',
        'World regions according to OWID': 'Region (OWID)'
    }
)

# Bubble plot: use latest year in selected range
bubble_year = to_year
bubble_data = avg_schooling_df[
    (avg_schooling_df['Country Code'].isin(selected_countries)) &
    (avg_schooling_df['Year'] == bubble_year)
].copy()

bubble_data = bubble_data.merge(
    pisa_avg,
    left_on='Country',
    right_on='Countries',
    how='left'
)

if bubble_data.empty:
    st.info("No average years of schooling data available for the selected countries and years.")
else:
    # Fill missing PISA scores for clear "No Data" indication
    min_pisa = bubble_data['PISA average scores, 2022'].min()
    fillna_val = int(min_pisa - 165) if pd.notnull(min_pisa) else -999
    bubble_data['PISA average scores, 2022'] = bubble_data['PISA average scores, 2022'].fillna(fillna_val)
    bubble_data['PISA average score 2022'] = bubble_data['PISA average scores, 2022'].apply(
        lambda x: "No Pisa Data" if x == fillna_val else round(x, 2)
    )

    # Layout: Bubble plot and line plot side by side
    col1, col2 = st.columns(2)
    with col1:
        bubble_plot = create_bubble(
            bubble_data,
            x='GDP (PPP)',
            y='Average Years of Schooling',
            size='PISA average scores, 2022',
            size_label='PISA average score 2022',
            x_label='GDP (US$)',
            y_label='Average Years of Schooling',
            text_label='Country',
            custom_tooltip=True,
            tooltip_columns=[
                'Country',
                'Country Code',
                'Region (OWID)',
                'Year',
                'GDP (PPP)',
                'Average Years of Schooling',
                'PISA average score 2022'
            ],
        )
        st.plotly_chart(bubble_plot, use_container_width=True, height=500)

    with col2:
        # Line plot: all years in selected range
        line_data = avg_schooling_df[
            (avg_schooling_df['Country Code'].isin(selected_countries)) &
            (avg_schooling_df['Year'] >= from_year) &
            (avg_schooling_df['Year'] <= to_year)
        ].copy()
        if line_data.empty:
            st.info("No average years of schooling data available for the selected countries and years.")
        else:
            st.line_chart(
                data=line_data,
                x='GDP (PPP)',
                y='Average Years of Schooling',
                color='Country Code',
                use_container_width=True,
                height=500,
            )
