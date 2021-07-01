import streamlit as st
import pandas as pd



def app():
   st.markdown("""
   # Methods
   ## Data collection

   The data was gathered by web-scraping a Slovenian website that lists new and used vehicles. In order to collect the data, a spider was utilised through the library Scrapy. The spider searched through each car brand and extracted links to specific suburbs of cars. The sub-URLs were then parsed and scraped for relevant data.  


   """)
   st.write('Figure 1 - dataset from scraping')
   df_showcase = pd.read_csv('./demo_data/data_demo_1.csv')
   st.dataframe(df_showcase[:20])
   st.write(" ** An example of scraped index 0 features ** ")
   st.write(df_showcase['features'][0])
   st.markdown(""" 

   The main variables are:
   * features - a collection of relevant table data from the website. This requires heavy cleaning in the next phase.
   * price - the listed price of the car
   * prodajalec - the lister seller/s of the car
   * title - includes the brand and model of the car
   * URL - the URL from which the data came.  

   """)
   st.markdown(""" 

   ## Data cleaning

   After much hacking and slashig, many new columns have been extracted from the scraped features.
   """)
   st.write('Figure 2 - mostly clean dataset')
   df_clean_data = pd.read_csv('./demo_data/data_demo_2.csv')
   st.dataframe(df_clean_data[:20])
   st.markdown(""" 


   Since a new scraping session is initiated daily, we want to update the data frame with new values.
   The extracted ID from the **URL** feature allows us to drop duplicate entries. 
   """)
   st.markdown(""" 

   ## Further Data cleaning

   Based on the EDA a few more tweaks were required, for example, removing outliers that could negatively affect the model. 


   """)

   NAME = 'joinedData.csv'
   df = pd.read_csv(NAME)
   df_subset = df[['letnik', 'prevozeni_km', 'Gorivo',
      'Menjalnik', 'Oblika', 'price', 'Motor KM','brands']]

   st.markdown(""" 

   ## Data transformation - Pipeline

   Missing values from the selected features were dropped to preserve the original data distribution. Categorical variables **[Menjalnik, Oblika, brands, Gorivo]** were transformed with Sklearn OneHotEndoer. 
   To access the transformations later. In the same manner numerical features **[prevozeni_km, Motor KM, letnik]** were scaled with a Sklearn RobustScaler.

   ## Modelling
   The data was split into a training test set with a 90/10 ratio. The Presented XGBoost model was trained and optimised with a Random Grid Search to improve the baseline model.   

   ## Results
   The model received an R2 score of 95% and an RMSE score of 2041. The most important features of the model were **letnik and Motor Km**. With an average error of around 2000€, the model is not yet suitable for real-world applications. Future work on the projects aims to improve the performance of the model/s. 
   """)