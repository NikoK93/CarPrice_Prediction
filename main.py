from numpy.lib.shape_base import tile
import streamlit as st
from multiapp import MultiApp
from apps import regression, EDA, methods

app = MultiApp()

st.title('''Car Price Prediction ''')
st.title('''Based on Slovenian Car Distribution''')

app.add_app('Regression model', regression.app)
app.add_app('EDA', EDA.app)
app.add_app('Methods', methods.app)
app.run()