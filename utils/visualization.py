import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import streamlit as st

def create_bubble(df, x, y, size, x_label, y_label, size_label, text_label, custom_tooltip=False, tooltip_columns=None, height=500, title=None):
    """
    Create a bubble chart with color and legend support.
    """
    color = 'Country Code' if 'Country Code' in df.columns else None
    hover_data = []
    if tooltip_columns and custom_tooltip:
        hover_data = tooltip_columns
        # {col: True for col in tooltip_columns}
    fig = px.scatter(
        data_frame=df,
        title=title,
        x=x,
        y=y,
        size=size,
        # text=text_label,
        color=color,
        size_max=40,
        labels={
            x: x_label,
            y: y_label,
            size: size_label,
            'text': text_label
        },
        hover_data=hover_data
    )
    fig.update_traces(textposition='bottom center')
    # remove the text from the bubbles
    # fig.update_traces(text=None)
    fig.update_layout(legend_title_text=color if color else "Legend", 
                        height=height)
    return fig

def create_plot(df, type, x_label, y_label, size_label, text_label=None, color_label=None):
    """
    Create a plot based on the specified type and labels.
    
    Parameters:
    - df: DataFrame containing the data to plot.
    - type: Type of plot ('line', 'scatter', etc.).
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - size_label: Label for the size of the bubbles (if applicable).
    - text_label: Label for the text displayed on the bubbles (if applicable).
    - color_label: Optional color parameter for the plot.
    
    Returns:
    - fig: The created plot figure.
     """

    if type == 'line':
        st.line_chart(
            data=df,
            x=x_label, 
            y=y_label,
            color=color_label,
            use_container_width=True,
        )
    elif type == 'bubble':
        fig = px.scatter(
            data_frame=df,
            x=x_label,
            y=y_label,
            size=size_label,
            text=text_label,
            size_max=60
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        raise ValueError("Unsupported plot type. Supported types: 'line', 'bubble'.")

def remove_nans(df):
    """
    Remove rows with NaN values from the DataFrame.
    """
    important_column = [
        'Government expenditure on education, total (% of government expenditure)',
        'Average years of schooling',
        'GDP per capita, PPP (constant 2021 international $)',
        'Productivity: output per hour worked',
        'Unemployment, total (% of total labor force) (modeled ILO estimate)',
        'Literacy rate',
        'Combined - average years of education for 15-64 years male and female youth and adults',
        '$3.65 a day - Share of population in poverty',
    ]
    df = df.dropna()
    return df

def display_two_vis(
    avg_schooling_df, pisa_avg, selected_countries, from_year, to_year, height=500
):
    """
    Display a bubble plot (latest year) and a line plot (all years in range)
    for average years of schooling vs GDP, with PISA scores as bubble size.
    """
    # Bubble plot: use latest year in selected range
    # Round numeric values to 3 decimal places
    # Convert GDP (PPP) to millions for better readability
    avg_schooling_df['GDP (PPP)'] = avg_schooling_df['GDP (PPP)'].apply(
        lambda x: round(x / 1e3, 3) if pd.notnull(x) else x
    )
    avg_schooling_df['Average Years of Schooling'] = avg_schooling_df['Average Years of Schooling'].round(3)
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
    
    col1, col2 = st.columns(2)

    with col1:
        if bubble_data.empty:
            st.info("No average years of schooling data available for the selected countries and years.")
        else:
            # Fill missing PISA scores for clear "No Data" indication
            st.subheader(f"📖 Average Years of Schooling vs GDP ({bubble_year})", divider="gray",
                help="**Average Years of Schooling**: Average number of years (excluding years spent repeating individual grades) adults over 25 years participated in formal education.\n\n" +
                "**GDP**: Average economic output per person in a country or region per year. This data is adjusted for inflation and for differences in living costs between countries.\n\n"
                )

            min_pisa = bubble_data['PISA average scores, 2022'].min()
            fillna_val = int(min_pisa - 165) if pd.notnull(min_pisa) else -999
            bubble_data['PISA average scores, 2022'] = bubble_data['PISA average scores, 2022'].fillna(fillna_val)
            bubble_data['PISA average score 2022'] = bubble_data['PISA average scores, 2022'].apply(
                lambda x: "No Pisa Data" if x == fillna_val else round(x, 2)
            )
            bubble_plot = create_bubble(
                bubble_data,
                x='GDP (PPP)',
                y='Average Years of Schooling',
                size='PISA average scores, 2022',
                size_label='PISA average score 2022',
                x_label='GDP (Thousand USD)',
                y_label='Average Years of Schooling',
                text_label='Country',
                custom_tooltip=True,
                tooltip_columns=[
                    'Country',
                    'Country Code',
                    # 'Region (OWID)',
                    'Year',
                    'GDP (PPP)',
                    'Average Years of Schooling',
                    'PISA average score 2022'
                ], height=height
            )
            st.plotly_chart(bubble_plot, use_container_width=True)

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
            st.subheader(f"🏫 Average Years of Schooling vs GDP ({from_year} to {to_year})", divider="gray",
                help="**Average Years of Schooling**: Average number of years (excluding years spent repeating individual grades) adults over 25 years participated in formal education.\n\n" +
                "**GDP**: Average economic output per person in a country or region per year. This data is adjusted for inflation and for differences in living costs between countries.\n\n"
                )
            line = create_line_chart(
                df=line_data,
                x='GDP (PPP)',
                y='Average Years of Schooling',
                x_label='GDP (Thousand USD)',
                color='Country Code',
                height=height,
            )
            st.plotly_chart(line, use_container_width=True)
            # st.line_chart(
            #     data=line_data,
            #     x='GDP (PPP)',
            #     y='Average Years of Schooling',
            #     color='Country Code',
            #     use_container_width=True,
            # )

def create_line_chart(df, x, y, color=None, x_label=None, y_label=None, 
                        custom_tooltip=False, tooltip_columns=None, 
                        height=500, show_markers=True, line_shape='linear'):
    """
    Create an advanced line chart using Plotly.
    
    Parameters:
    - df: DataFrame containing the data
    - x: Column name for x-axis
    - y: Column name for y-axis  
    - color: Column name for grouping lines by color
    - title: Chart title
    - x_label: Custom x-axis label
    - y_label: Custom y-axis label
    
    - custom_tooltip: Whether to use custom tooltip
    - tooltip_columns: List of columns to show in tooltip
    - height: Chart height in pixels
    
    - line_shape: Line shape ('linear', 'spline', 'hv', 'vh', 'hvh', 'vhv')
    
    Returns:
    - Plotly figure object
    """
    
    # Set default labels
    x_label = x_label or x
    y_label = y_label or y
    
    # Prepare hover data
    hover_data = {}
    if custom_tooltip and tooltip_columns:
        hover_data = {col: True for col in tooltip_columns if col in df.columns}
    
    # Create the line chart
    fig = px.line(
        df, 
        x=x, 
        y=y, 
        color=color,
        labels={
            x: x_label,
            y: y_label,
            color: color if color else "Series"
        },
        hover_data=hover_data,
        markers=show_markers,
        line_shape=line_shape
    )
    
    # Customize layout
    fig.update_layout(
        height=height,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_bar_chart(df, x, y, color=None, x_label=None, y_label=None, 
                    height=500, orientation='v', sort_values=False):
    """
    Create an interactive bar chart.
    
    Parameters:
    - df: DataFrame containing the data
    - x: Column name for x-axis
    - y: Column name for y-axis
    - color: Column name for coloring bars
    - x_label: Custom x-axis label
    - y_label: Custom y-axis label
    - height: Chart height in pixels
    - orientation: 'v' for vertical, 'h' for horizontal
    - sort_values: Whether to sort bars by value
    
    Returns:
    - Plotly figure object
    """
    
    # Set default labels
    x_label = x_label or x
    y_label = y_label or y
    
    # Sort data if requested
    if sort_values:
        df = df.sort_values(y, ascending=False if orientation == 'v' else True)
    
    # Create the bar chart
    fig = px.bar(
        df,
        x=x if orientation == 'v' else y,
        y=y if orientation == 'v' else x,
        color=color,
        orientation=orientation,
        labels={
            x: x_label,
            y: y_label,
            color: color if color else "Series"
        }
    )
    
    # Update layout
    fig.update_layout(
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        # Enable hover and click interactions
        hovermode='closest'
    )
    
    # Update hover template
    if orientation == 'v':
        hovertemplate = f'<b>%{{x}}</b><br>{y_label}: %{{y:.2f}}<extra></extra>'
    else:
        hovertemplate = f'<b>%{{y}}</b><br>{x_label}: %{{x:.2f}}<extra></extra>'
    
    fig.update_traces(hovertemplate=hovertemplate)
    
    return fig