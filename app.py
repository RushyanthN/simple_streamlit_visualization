import os
import io
import pickle
import altair as alt
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from utils.b2 import B2
from utils.modeling import *
#fgh

# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
REMOTE_DATA = 'Apple-Twitter-Sentiment-DFE_encoded11.csv'


# ------------------------------------------------------
#                        CONFIG
# ------------------------------------------------------
#load_dotenv()simple_streamlit_visualization

# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_KEYID'],
        secret_key=os.environ['B2_APPKEY'])



# ------------------------------------------------------
#                        CACHING
# ------------------------------------------------------
@st.cache_data
def get_data():

    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df_apple = b2.get_df(REMOTE_DATA)
    df_apple['date'] = pd.to_datetime(df_apple['date'], format='%a %b %d %H:%M:%S %z %Y')
    df_apple['day_month_year'] = df_apple['date'].dt.strftime('%d/%m/%Y')  
    return df_apple


@st.cache_resource
def get_model():
    with open('./model.pickle', 'rb') as f:
        analyzer = pickle.load(f)
    
    return analyzer

st.title("Sentiment Confidence by Day")

df_apple = get_data()
analyzer = get_model()

# ------------------------------
# PART 1 : Filter Data
# ------------------------------

 
df_apple = df_apple.rename(columns={'sentiment:confidence': 'sentiment_confidence'})
df_filtered = df_apple.groupby('day_month_year')['sentiment_confidence'].mean().reset_index()


st.dataframe(df_filtered)

# ------------------------------
# PART 2 : Plots
# ------------------------------



chart = alt.Chart(df_filtered).mark_bar().encode(
            x='day_month_year',
            y='sentiment_confidence',
            tooltip=['day_month_year', 'sentiment_confidence']
        ).properties(
            width=800,
            height=500
        ).interactive()
st.altair_chart(chart, use_container_width=True)

