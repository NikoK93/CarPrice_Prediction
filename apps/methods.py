import streamlit as st
import pandas as pd



def app():
    st.markdown("""
    # Methods
    ## Data collection

    The data was gathered though web-scraping a Slovenian flee market car website - www.avto.net
    In order to collect the data a spider was utilised through the library Scrapy. The spider searched through each car brand and extracted links to specific suburls of cars.
    The suburls were then parsed and scraped for relevant data. 


    """)
    st.write('Figure 1 - dataset from scraping')
    df_showcase = pd.read_csv('./demo_data/data_demo_1.csv')
    st.dataframe(df_showcase[:20])
    st.write(" ** An example of scraped index 0 features ** ")
    st.write(df_showcase['features'][0])
    st.markdown(""" 
    
    The main variables are:
    * features - a collection of relevant table data from the website. This requires heacy cleaning in the next phase.
    * price - the listed price of the car
    * prodajalec - the lister seller/s of the car
    * title - includes the brand and model of the car
    * url - the url from which the data came from. 

    """)
    st.markdown(""" 
    
    ## Data cleaning
   
    After much hacking and slashin, many new columns have been extracted from the scraped features.
    """)
  
    df_clean_data = pd.read_csv('./demo_data/data_demo_2.csv')
    st.dataframe(df_clean_data[:20])
    st.markdown(""" 
    
   
    Since new scraping session is initied daily, we wan't to update the dataframe with new values.
    The extracted ID from the **URL** feature allows us to drop duplicate entries.
    """)
    st.markdown(""" 
    
    ## Further Data cleaning
   
    Based on the EDA a few more tweaks are required, like removing outliers.
    

    """)

    NAME = 'joinedData.csv'
    df = pd.read_csv(NAME)
    df_subset = df[['letnik', 'prevozeni_km', 'Gorivo',
       'Menjalnik', 'Oblika', 'price', 'Motor KM','brands']]

    st.markdown(""" 
    
    ## Data transformation - Pipeline
   
    Missing values from the selected features were droped to preserve the original data distribution. Categorical variables **[Menjalnik, Oblika, brands, Gorivo]** were transformed with Sklearn OneHotEndoer. 
    To acces the transformations later. In the same manner numerical features **[prevozeni_km, Motor KM, letnik]** were scaled with a Sklearn RobustScaler
    

    """)