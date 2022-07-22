#import std libraries

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px

# Write a title
st.title('Welcome to the first penguin data explorer')
st.write('**Starting** the *build* of `penguin` :penguin: :mag:')
# Write data taken from https://allisonhorst.github.io/palmerpenguins/
st.write('Data is taken from [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/)')
# Put image https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/man/figures/lter_penguins.png
st.image('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/man/figures/lter_penguins.png')
# Write heading for Data
st.header('Data')
# Read csv file and output a sample of 20 data points
df_penguins = pd.read_csv('penguins_extra.csv')
st.write('Displaying a sample of 20 points in our dataset', df_penguins.sample(20))
# Add a selectbox for species
species = st.selectbox("Select which Species you want", df_penguins.species.unique())
# Display a sample of 20 data points according to the species selected with corresponding title
st.write('Displaying data points of {species}', df_penguins[df_penguins['species']==species])
# Plotting seaborn
fig, ax=plt.subplots()
ax = sns.scatterplot(data=df_penguins, x='bill_length_mm',y='flipper_length_mm', hue='species')
## to plot only data of the selected species:
# ax = sns.scatterplot(df_penguins[df_penguins['species']==species], x='bill_length_mm',y='flipper_length_mm', hue='species')
st.pyplot(fig)
# Plotting plotly
fig = px.scatter(data_frame=df_penguins, x='bill_length_mm',y='flipper_length_mm', color='species', animation_frame='species', range_x=(0,100), range_y=(170,250))
st.plotly_chart(fig)
# Bar chart count of species per('island')['species'].count()
st.bar_chart(df_penguins.groupby('island')['species'].count())
# Maps
st.map(df_penguins)
st.write('For mapping reference check out [deckgl](https://deckgl.readthedocs.io/en/latest/) or [streamlit.io](https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart)')
# Reference https://deckgl.readthedocs.io/en/latest/
# Reference https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart
# sidebar comment
slider_choice = st.sidebar.selectbox('You can have various options',['yes','no'])

if slider_choice=='yes':
    st.write('yes selected')
else:
    st.write('no selected')

img_variable = st.sidebar.file_uploader('Upload an image',type=['png','jpg','svg'])
from PIL import Image
if img_variable is not None:
    st.image(Image.open(img_variable))

csv_variable = st.sidebar.file_uploader('Upload a csv file',type=['csv'])

if csv_variable is not None:
    df = pd.read_csv(csv_variable)
    st.write(df)

st.markdown(f"""
<style>
.stApp{{
    background-image: url(https://images.unsplash.com/photo-1596544701302-2a61b8bbca35?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1742&q=80);
    background-size: cover;
}}
</style>
""",unsafe_allow_html=True)

