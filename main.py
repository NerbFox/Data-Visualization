from pathlib import Path

from utils import (
    create_bubble, 
    create_plot, 
    create_line_chart,
    create_bar_chart,
    
    preprocess_data, 
    get_gdp_data, 
    get_avg_years_school_gdp, 
    get_first_last_value, 
    get_common_year_range, 
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
    (education_expenditure['Code'].isin(selected_countries)) &
    (education_expenditure['Year'] >= from_year) &
    (education_expenditure['Year'] <= to_year)
]

# -----------------------------------------------------------------------------
# Metrics for selected years

# st.header(f'GDP in {to_year}', divider='gray')
# highlight_country = 'IDN'
highlight_country_name = get_country_name(highlight_country, df_avg_years_school_gdp)

df_column_pairs = [
    (education_expenditure, 'Government expenditure on education, total (% of government expenditure)'),
    (education_literacy, 'Literacy rate'),
    (df_avg_years_school_gdp, 'Average years of schooling')
]
overall_start_year, overall_end_year = get_common_year_range(highlight_country, df_column_pairs)
st.subheader(f'Highlights During {overall_start_year}-{overall_end_year}', divider='gray')

# print(f"Overall Start Year: {overall_start_year}, Overall End Year: {overall_end_year}")

first_year_df = gdp_df[gdp_df['Year'] == overall_start_year]
last_year_df = gdp_df[gdp_df['Year'] == overall_end_year]

first_expenditure, last_expenditure = get_first_last_value(
    education_expenditure,
    'Government expenditure on education, total (% of government expenditure)',
    highlight_country,
    overall_start_year,
    overall_end_year
)

first_literacy, last_literacy = get_first_last_value(
    education_literacy,
    'Literacy rate',
    highlight_country,
    overall_start_year,
    overall_end_year
)

first_average_years_school, last_average_years_school = get_first_last_value(
    df_avg_years_school_gdp,
    'Average years of schooling',
    highlight_country,
    overall_start_year,
    overall_end_year
)

# print(f"Literacy Rate: {first_literacy} -> {last_literacy}")

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
        "unit": ""
    },
    {
        "label": "Education Expenditure",
        "first": first_expenditure,
        "last": last_expenditure,
        "unit": "%"
    },
    {
        "label": "Avg. Years of Schooling",
        "first": first_average_years_school,
        "last": last_average_years_school,
        "unit": ""
    },
    {
        "label": "Literacy Rate",
        "first": first_literacy,
        "last": last_literacy,
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
            delta_color=delta_color
        )
        
# -----------------------------------------------------------------------------
# Visualization

st.subheader('Education Relationships', divider='gray')

if filtered_education_expenditure.empty:
    st.info("No education expenditure data available for the selected countries and years.")
else:
    # Chart type selection
    option = st.radio(
        'Education Expenditure Chart Type:',
        ['Line Chart', 'Bar Chart'],
        horizontal=True,
        help="Line chart: See trends over time. Bar chart: Compare values."
    )
    
    # Create visualizations based on user selection
    if option == 'Line Chart':
        st.markdown("### 📈 Education Expenditure Trends")
        st.markdown("*Click on a line in the legend or chart to highlight it*")
        
        chart = create_line_chart(
            df=filtered_education_expenditure,
            x='Year',
            y='Government expenditure on education, total (% of government expenditure)',
            color='Code',
            x_label='Year',
            y_label='Education Expenditure (% of Gov. Expenditure)',
            height=500,
            show_markers=True,
        )
        st.plotly_chart(chart, use_container_width=True)
        
        # Add some insights text
        
    elif option == 'Bar Chart':
        # Bar chart options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Education Expenditure Comparison")
        
        with col2:
            bar_year = st.selectbox(
                'Select Year:',
                options=sorted(filtered_education_expenditure['Year'].unique(), reverse=True),
                help="Choose which year to compare countries"
            )
        
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
                y_label='Education Expenditure (% of Gov. Expenditure)',
                height=500,
                sort_values=True
            )
            st.plotly_chart(chart, use_container_width=True)
            
        else:
            st.warning(f"No data available for {bar_year}.")



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
