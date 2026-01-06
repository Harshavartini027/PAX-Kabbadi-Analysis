import streamlit as st
import pandas as pd
from pymongo import MongoClient

# 1. Initialize connection using st.secrets
@st.cache_resource
def init_connection():
    # This pulls the URI from your secrets file automatically
    return MongoClient(st.secrets["mongo"]["uri"])

client = init_connection()

# 2. Function to fetch data and convert to DataFrame
@st.cache_data(ttl=600) # Caches data for 10 minutes
def get_data(collection_name):
    db = client.KabaddiDB
    items = list(db[collection_name].find())
    # Remove the MongoDB internal ID before returning
    for item in items:
        item.pop('_id', None)
    return pd.DataFrame(items)

# 3. Load your dataframes
matches = get_data("match_scores")
attendance = get_data("attendance")

# Now your existing dashboard code (charts, tables, etc.) 
# will use these 'matches' and 'attendance' DataFrames!