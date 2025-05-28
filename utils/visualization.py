import plotly.express as px
import pandas as pd
import streamlit as st

def test():
    """
    This is a test function to check if the visualization module is working correctly.
    """
    print("Visualization module is working correctly.")
    return True

def create_bubble(df, x_label, y_label, size_label, text_label):
    """
    Create a bubble chart showing GDP vs Average Years of Schooling.
    """
    fig = px.scatter(
        data_frame=df,
        x=x_label,
        y=y_label,
        size=size_label,
        text=text_label,
        size_max=60
    )

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