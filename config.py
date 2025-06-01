from typing import List, Dict, Optional, Any
import streamlit as st
from streamlit.commands.page_config import Layout, PageIcon, MenuItems, InitialSideBarState
import os

class Config:
    """Application configuration"""
    # ----------------------------
    # Streamlit page configuration
    # -----------------------------
    __PAGE_TITLE: str = 'Education Dashboard'
    __PAGE_ICON: PageIcon = ':earth_americas:'
    __LAYOUT: Layout = 'wide'
    __INIT_SIDEBAR_STATE: InitialSideBarState = 'auto'
    __MENU_ITEMS: MenuItems =  {
        'About': 'This is a sample education dashboard built with Streamlit.',        
    }

    PAGE_CONFIG = {
        'page_title': __PAGE_TITLE,
        'page_icon': __PAGE_ICON,
        'layout': __LAYOUT,
        'initial_sidebar_state': __INIT_SIDEBAR_STATE,
        'menu_items': __MENU_ITEMS
    }
    """Streamlit page configuration"""


    # ----------------------------
    # Global Sidebar configuration
    # -----------------------------
    DEFAULT_MAIN_COUNTRY: str = 'Indonesia'
    DEFAULT_COUNTRIES: Optional[List[str]] = ['SGP', 'USA', 'CHN', 'JPN', 'KOR']
    MIN_YEAR: int = 1960
    MAX_YEAR: int = 2023
    
    
    # ----------------------------
    # Data Loading Configuration
    # -----------------------------
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIRPATH = os.path.join(PROJECT_ROOT, 'data')

    IMPORTANT_COLUMNS = [
                'Government expenditure on education, total (% of government expenditure)',
                'Average years of schooling',
                'GDP per capita, PPP (constant 2021 international $)',
                'Productivity: output per hour worked',
                'Unemployment, total (% of total labor force) (modeled ILO estimate)',
                'Literacy rate',
                'Combined - average years of education for 15-64 years male and female youth and adults',
                '$3.65 a day - Share of population in poverty',
            ]
    """Important data columns for preprocessing"""
    

# Global config instances
APP_CONFIG = Config()