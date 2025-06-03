from pathlib import Path

from utils import (
    create_bubble, 
    create_plot, 
    create_line_chart,
    create_bar_chart,
    display_two_vis,
    
    preprocess_data, 
    get_gdp_data, 
    get_avg_years_school_gdp, 
    get_pisa_score,
    get_country_name,
    get_education_expenditure_data
)

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from config import Config

# -----------------------------------------------------------------------------
# Streamlit page configuration
# -----------------------------------------------------------------------------

st.set_page_config(**Config.PAGE_CONFIG)


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
pisa_reading = pisa_reading_scores[['Countries', 'PISA reading scores, 2022', 'Global rank']].rename(
    columns={'Global rank': 'Reading global rank'}
)

pisa_math = pisa_math_scores[['Countries', 'PISA math scores, 2022', 'Global rank']].rename(
    columns={'Global rank': 'Math global rank'}
)

pisa_science = pisa_science_scores[['Countries', 'PISA science scores, 2022', 'Global rank']].rename(
    columns={'Global rank': 'Science global rank'}
)

pisa_avg = pisa_reading.merge(pisa_math, on='Countries', how='outer').merge(pisa_science, on='Countries', how='outer')
pisa_avg['PISA average scores, 2022'] = pisa_avg[
    ['PISA reading scores, 2022', 'PISA math scores, 2022', 'PISA science scores, 2022']
].mean(axis=1, skipna=True)

pisa_avg = pisa_avg.sort_values(by='PISA average scores, 2022', ascending=False).reset_index(drop=True)
pisa_avg['Overall rank'] = pisa_avg.index + 1

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

# Inner join df_avg_years_school_gdp with gdp_df and education_expenditure on 'Country Code' and 'Entity
# gdp_df_copy = gdp_df.rename(columns={'Country Code': 'Code', 'Country Name': 'Entity'})
# df_merge_country = df_avg_years_school_gdp.merge(
#     gdp_df_copy[[]],
#     on=['Code', 'Entity'],
#     how='inner'
# ).merge(
#     education_expenditure,
#     on=['Code', 'Entity'],
#     how='inner'
# )

# Exclude countries that have no single data point in average years of schooling and expenditure
# df_merge_country = df_merge_country[
#     df_merge_country['Average years of schooling'].notna() &
#     df_merge_country['Government expenditure on education, total (% of government expenditure)'].notna()
# ]

# # Save to csv in folder data
# df_merge_country.to_csv(DATA_DIR + 'df_merge_country.csv', index=False)

# Read data from CSV
# df_merge_country = pd.read_csv(DATA_DIR + 'df_merge_country.csv')

# Take unique entity and code and save it to csv
# df_unique_countries = df_merge_country[['Entity', 'Code']].drop_duplicates().reset_index(drop=True)
# df_unique_countries.to_csv(DATA_DIR + 'df_unique_countries.csv', index=False)

df_unique_countries = pd.read_csv(DATA_DIR + 'df_unique_countries.csv')
df_unique_countries = df_unique_countries.rename(columns={'Entity': 'Country Name', 'Code': 'Country Code'})

# Preprocess the data
preprocess_data(dfs=df_list_cleaned)

# -------------------------------------------------------------
# Page content
# :earth_americas:   world icon
st.header('Education Dashboard :earth_americas:')
# st.markdown("""
# This dashboard provides insights into the relationship between education and economic indicators across various countries.
# """)

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
sorted_country_names = sorted(df_unique_countries['Country Name'].unique())
highlight_country = st.sidebar.selectbox(
    'Main Country',
    options=sorted_country_names,
    index = sorted_country_names.index('Indonesia') if 'Indonesia' in sorted_country_names else 0
)

entity_to_code = df_unique_countries.dropna(subset=['Country Code']).drop_duplicates("Country Name")[["Country Name", "Country Code"]].set_index("Country Name")["Country Code"].to_dict()
code_to_entity = df_unique_countries.dropna(subset=['Country Code']).drop_duplicates("Country Code")[["Country Code", "Country Name"]].set_index("Country Code")["Country Name"].to_dict()

highlight_country = entity_to_code.get(highlight_country, None)

from_year, to_year = st.sidebar.slider(
    'Select Year Range',
    min_value=min_year,
    max_value=max_year,
    value=[min_year, max_year]
)

countries = df_unique_countries['Country Code'].unique()

default_countries = ['SGP', 'USA', 'CHN', 'CHE', 'BRA', 'CAN', 'CHL', 'COL', 'HRV']
    
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
    (education_expenditure['Code'].isin(selected_countries)) &
    (education_expenditure['Year'] >= from_year) &
    (education_expenditure['Year'] <= to_year)
]

# Round expenditure values to 3 decimal places
filtered_education_expenditure['Government expenditure on education, total (% of government expenditure)'] = filtered_education_expenditure[
    'Government expenditure on education, total (% of government expenditure)'
].round(3)

filtered_avg_schooling = df_avg_years_school_gdp[
    (df_avg_years_school_gdp['Code'].isin(selected_countries)) &
    (df_avg_years_school_gdp['Year'] >= from_year) &
    (df_avg_years_school_gdp['Year'] <= to_year)
].dropna(subset=['Average years of schooling', 'GDP per capita, PPP (constant 2021 international $)'])

# rename 
filtered_education_literacy = education_literacy.rename(
    columns={
        'Literacy rate': 'Literacy Rate',
        'Code': 'Country Code',
        'Entity': 'Country'
    }
)

# merge unemployment and average years of schooling based on year and code
filtered_unemployment = unemployment.merge(
    df_avg_years_school_gdp,
    on=['Entity', 'Code', 'Year'],
    how='inner'
)

filtered_unemployment = filtered_unemployment.rename(
    columns={
        'Code': 'Country Code',
        'Entity': 'Country',
        'Average years of schooling' :'Average years of schooling',
        'Unemployment, total (% of total labor force) (modeled ILO estimate)':'Unemployment Rate'
    }
)


# filtered_unemployment = filtered_unemployment.dropna(subset=['Unemployment Rate'])
# st.markdown(f'{filtered_unemployment[:1]}')
# st.markdown(f'{df_avg_years_school_gdp[:1]}')
# rename

# st markdown df
# st.markdown(f'{filtered_education_expenditure.head(5)}')
# -----------------------------------------------------------------------------
# Metrics for selected years

metric_dfs = [{'title': 'Education Expenditure', 'data': None}, 
              {'title': 'Literacy Rate', 'data': None}, 
              {'title': 'Avg. Years of Schooling', 'data': None}]

metric_education_expenditure = education_expenditure[
    (education_expenditure['Code'] == highlight_country) &
    (education_expenditure['Government expenditure on education, total (% of government expenditure)'].notna())
][['Entity', 'Code', 'Year', 'Government expenditure on education, total (% of government expenditure)']].rename(
    columns={'Government expenditure on education, total (% of government expenditure)': 'Value'}
)

metric_education_literacy = education_literacy[
    (education_literacy['Code'] == highlight_country) &
    (education_literacy['Literacy rate'].notna())
][['Entity', 'Code', 'Year', 'Literacy rate']].rename(columns={'Literacy rate': 'Value'})

metric_avg_years_school_gdp = df_avg_years_school_gdp[
    (df_avg_years_school_gdp['Code'] == highlight_country) &
    (df_avg_years_school_gdp['Average years of schooling'].notna())
][['Entity', 'Code', 'Year', 'Average years of schooling']].rename(columns={'Average years of schooling': 'Value'})

data_map = {
    'Education Expenditure': metric_education_expenditure,
    'Literacy Rate': metric_education_literacy,
    'Avg. Years of Schooling': metric_avg_years_school_gdp
}

for metric in metric_dfs:
    metric['data'] = data_map.get(metric['title'])

metric_sets = [set(df['data']['Year']) for df in metric_dfs if df['data'] is not None and not df['data'].empty]
common_years = set.intersection(*metric_sets) if metric_sets else None

metric_final_year = None
final_metric_dfs = []

for df in metric_dfs:
    if df['data'] is not None and not df['data'].empty:
        if common_years:
            metric_final_year = max(common_years)
            filtered_data = df['data'][df['data']['Year'] == metric_final_year]
        else:
            filtered_data = df['data'].sort_values('Year').iloc[[-1]]
        final_metric_dfs.append({'title': df['title'], 'data': filtered_data})
    else:
        final_metric_dfs.append({'title': df['title'], 'data': pd.DataFrame(columns=['Entity', 'Code', 'Year', 'Value'])})

if not metric_final_year and final_metric_dfs:
    metric_final_year = max(
        (df['data']['Year'].iat[0] for df in final_metric_dfs if not df['data'].empty),
        default=None
    )

def get_first_value(df):
    return (df['Value'].iat[0], df['Year'].iat[0]) if not df.empty else ("No Data", None)

first_expenditure, start_expenditure_year = get_first_value(metric_education_expenditure)
first_literacy, start_literacy_year = get_first_value(metric_education_literacy)
first_avg_school, start_average_school_gdp_year = get_first_value(metric_avg_years_school_gdp)

first_metrics = [get_first_value(df['data'])[0] if df['data'] is not None else "No Data" for df in metric_dfs]

final_metric_years = []
last_metrics = []
for df in final_metric_dfs:
    if not df['data'].empty:
        final_metric_years.append(df['data']['Year'].iat[0])
        last_metrics.append(df['data']['Value'].iat[0])
    else:
        final_metric_years.append(None)
        last_metrics.append("No Data")

# --- PISA Scores ---
highlight_country_name = get_country_name(highlight_country, df_avg_years_school_gdp)

hightlight_pisa_df = pisa_avg[pisa_avg['Countries'] == highlight_country_name]

if not hightlight_pisa_df.empty:
    row = hightlight_pisa_df.iloc[0]

    average_pisa_score = row['PISA average scores, 2022']
    average_pisa_rank = int(row['Overall rank'])

    highlight_pisa_math_score = row['PISA math scores, 2022']
    math_pisa_rank = f"Rank {int(row['Math global rank'])}"

    highlight_pisa_reading_score = row['PISA reading scores, 2022']
    reading_pisa_rank = f"Rank {int(row['Reading global rank'])}"

    highlight_pisa_science_score = row['PISA science scores, 2022']
    science_pisa_rank = f"Rank {int(row['Science global rank'])}"

else:
    average_pisa_score = "No Data"
    average_pisa_rank = ""
    highlight_pisa_math_score = "No Data"
    math_pisa_rank = ""
    highlight_pisa_reading_score = "No Data"
    reading_pisa_rank = ""
    highlight_pisa_science_score = "No Data"
    science_pisa_rank = ""

# --- Display ---
st.subheader(f'{highlight_country_name} Highlights Up to {metric_final_year}', divider='gray')

indicators = [
    {
        "label": "Avg. PISA Scores (2022)",
        "first": (
            f"{average_pisa_score:.2f} (Rank {average_pisa_rank})"
            if average_pisa_score != "No Data" else "No Data"
        ),
        "last": (
            f"{average_pisa_score:.2f} (Rank {average_pisa_rank})"
            if average_pisa_score != "No Data" else "No Data"
        ),
        "help": 
        (
            f"**Description**: "
            f"The Programme for International Student Assessment (PISA) is a worldwide study by the Organisation for Economic Co-operation and Development (OECD) intended to evaluate educational systems by measuring 15-year-old school pupils' performance on math, science, and reading. \n\n"
        ) + (
            f"**PISA Reading Score**: "
            f"{highlight_pisa_reading_score:.3f} ({reading_pisa_rank})  \n"
            if highlight_pisa_reading_score != "No Data" else
            "**PISA Reading Score**: No Data\n"
        ) + (
            f"**PISA Math Score**: "
            f"{highlight_pisa_math_score:.3f} ({math_pisa_rank})  \n"
            if highlight_pisa_math_score != "No Data" else
            "**PISA Math Score**: No Data\n"
        ) + (
            f"**PISA Science Score**: "
            f"{highlight_pisa_science_score:.3f} ({science_pisa_rank})"
            if highlight_pisa_science_score != "No Data" else
            "**PISA Science Score**: No Data"
        ),
        "unit": ""
    }
]

title_to_start_year = {
    "Education Expenditure": start_expenditure_year,
    "Literacy Rate": start_literacy_year,
    "Avg. Years of Schooling": start_average_school_gdp_year
}

help_map = {
    "Literacy Rate": f"**Description**: Literacy rates measure the share of people who can read and write, typically assessed through self-reports, literacy tests, or estimates based on education levels. Countries use various methods, including surveys, censuses, and indirect data.\n\n",
    "Education Expenditure": f"**Description**: Government spending on education is shown as a percentage of total government spending across all sectors. It includes local, regional, and national budgets, plus international funding given to the government.\n\n",
    "Avg. Years of Schooling": f"**Description**: Average number of years (excluding years spent repeating individual grades) adults over 25 years participated in formal education.\n\n"
}

unit_map = {
    "Education Expenditure": "%",
    "Literacy Rate": "%",
    "Avg. Years of Schooling": ""
}

for i, df in enumerate(final_metric_dfs):
    title = df['title']
    last_value = df['data']['Value'].iat[0] if not df['data'].empty else "No Data"
    last_year = df['data']['Year'].iat[0] if not df['data'].empty else None
    first_value = first_metrics[i]
    start_year = title_to_start_year.get(title)
    unit = unit_map.get(title, "")
    help_text = help_map.get(title, "")

    if start_year and last_year:
        help_text += f"Data shown is from {start_year}"
        if start_year != last_year:
            help_text += f" to {last_year}."
    else:
        help_text += "No available data exists for the selected country"

    indicators.append({
        "label": title,
        "first": first_value,
        "last": last_value,
        "help": help_text,
        "unit": unit
    })

# --- Streamlit Metric Display ---
cols = st.columns(len(indicators))

for i, indicator in enumerate(indicators):
    col = cols[i % len(cols)]
    with col:
        first = indicator["first"]
        last = indicator["last"]
        label = indicator["label"]
        unit = indicator["unit"]

        if isinstance(first, str) or isinstance(last, str):
            value = last if isinstance(last, str) else first
            growth = ""
            delta_color = "off"
        elif math.isnan(first) or math.isnan(last) or first == 0 or first - last == 0:
            value = f"{last:,.2f}{unit}"
            growth = ""
            delta_color = "off"
        else:
            growth = f"{last / first:,.2f}x"
            value = f"{last:,.2f}{unit}"
            delta_color = "normal"

        st.metric(label=label, value=value, delta=growth, delta_color=delta_color, help=indicator["help"], border=False, label_visibility="visible")
        
# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

# Visualizations of Average Years of Schooling vs GDP
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

# st.markdown('### :school: Average Years of Schooling vs GDP')
# st.subheader(':school: Average Years of Schooling vs GDP', divider='gray',
#                 help="**Average Years of Schooling**: Average number of years (excluding years spent repeating individual grades) adults over 25 years participated in formal education.\n\n" +
#                 "**GDP**: Average economic output per person in a country or region per year. This data is adjusted for inflation and for differences in living costs between countries.\n\n"
#             )
# Check if df from selected years is empty
if filtered_avg_schooling.empty:
    st.info("No average years of schooling data available for the selected countries and years.")
else:
    display_two_vis(
        avg_schooling_df=avg_schooling_df,
        unemployment_df=filtered_unemployment,
        pisa_avg=pisa_avg,
        selected_countries=selected_countries,
        from_year=from_year,
        to_year=to_year,
        height=400,
    )

# -----------------------------------------------------------------------------

# Education Expenditure Visualization
if filtered_education_expenditure.empty:
    st.info("No education expenditure data available for the selected countries and years.")
else:
    # Chart type selection
    # st.markdown('---')
    st.subheader("🪙 Education Expenditure Trends", divider='gray', 
                 help="**Education Expenditure**: Government spending on education as a percentage of total government expenditure. This includes local, regional, and national budgets, plus international funding given to the government.\n\n"
            )
    option = st.radio(
        'Education Expenditure Chart Type:',
        ['Line Chart', 'Bar Chart'],
        horizontal=True,
        help="Line chart: See trends over time. Bar chart: Compare values."
    )
    
    # Create visualizations based on user selection
    if option == 'Line Chart':
        st.markdown(f"#### 📈 Education Expenditure Trends ({from_year} - {to_year})")
                
                     
        # st.markdown("*Click on the legend to hide/highlight it*")
        
        chart = create_line_chart(
            df=filtered_education_expenditure,
            x='Year',
            y='Government expenditure on education, total (% of government expenditure)',
            color='Code',
            x_label='Year',
            y_label='Gov. Education Expenditure (%)',
            height=500,
            show_markers=True,
        )
        st.plotly_chart(chart, use_container_width=True)
        
        # Add some insights text
        
    elif option == 'Bar Chart':
        # Bar chart options
        col1, col2 = st.columns([2, 1])
        
        with col2:
            bar_year = st.selectbox(
                'Select Year:',
                options=sorted(filtered_education_expenditure['Year'].unique(), reverse=True),
                help="Choose which year to compare countries"
            )
        with col1:
            st.markdown(f"#### 📊 Education Expenditure Comparison ({bar_year})")
        
        
        # Filter data for selected year
        bar_data = filtered_education_expenditure[
            filtered_education_expenditure['Year'] == bar_year
        ].copy()
        
        if not bar_data.empty:
            chart = create_bar_chart(
                df=bar_data,
                x='Code',
                y='Government expenditure on education, total (% of government expenditure)',
                color='Code',
                x_label='Country Code',
                y_label='Gov. Education Expenditure (%)',
                height=500,
                sort_values=True
            )
            st.plotly_chart(chart, use_container_width=True)
            
        else:
            st.warning(f"No data available for {bar_year}.")