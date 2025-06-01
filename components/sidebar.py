import streamlit as st
from config import Config

def create_sidebar_controls(gdp_df):
    """Create sidebar with user controls."""
    min_year = int(gdp_df['Year'].min())
    max_year = int(gdp_df['Year'].max())
    default_countries = Config.DEFAULT_COUNTRIES or []
    
    st.sidebar.header('User Controls')
    st.sidebar.markdown("Select the year range and countries to analyze the data.")
    
    # Country selection
    sorted_country_names = sorted(gdp_df['Country Name'].unique())
    highlight_country = st.sidebar.selectbox(
        'Main Country',
        options=sorted_country_names,
        index=sorted_country_names.index('Indonesia') if 'Indonesia' in sorted_country_names else 0
    )
    
    # Year range
    from_year, to_year = st.sidebar.slider(
        'Select Year Range',
        min_value=min_year,
        max_value=max_year,
        value=[min_year, max_year]
    )
    
    # Country codes mapping
    entity_to_code = gdp_df.dropna(subset=['Country Code']).drop_duplicates("Country Name")[
        ["Country Name", "Country Code"]
    ].set_index("Country Name")["Country Code"].to_dict()
    
    highlight_country_code = entity_to_code.get(highlight_country, None)
    countries = gdp_df['Country Code'].unique()
    
    selected_countries = st.sidebar.multiselect(
        'Which countries would you like to view?',
        options=sorted(countries),
        default=[c for c in [highlight_country_code if highlight_country_code not in default_countries else None] + default_countries if c in countries],
    )
    
    if not selected_countries:
        st.warning("Select at least one country to display data.")
    
    return {
        'highlight_country': highlight_country_code,
        'from_year': from_year,
        'to_year': to_year,
        'selected_countries': selected_countries
    }