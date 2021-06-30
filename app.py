import streamlit as st
from datetime import date, timedelta
import pandas as pd
import numpy as np
import scipy.stats
import joblib
from matplotlib.backends.backend_agg import RendererAgg
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, BaggingRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff

plt.style.use('fivethirtyeight')


NAME = 'joinedData.csv'

df = pd.read_csv(NAME)
df_subset = df[['letnik', 'Starost', 'prevozeni_km', 'Gorivo',
       'Menjalnik', 'Oblika', 'Barva', 'Notranjost', 'price', 'Motor KM','model','brands']]

with st.echo(code_location='below'):

    group_labels = ['Prices in Euro - €']

    colors = ['slategray']

    # Create distplot with curve_type set to 'normal'
    #fig = ff.create_distplot([df_subset['price']], group_labels,curve_type='normal',
                         #   colors=colors)

    # Add title
    #fig.update_layout(title_text='Distplot with Normal Distribution')
    #fig = px.histogram(df_subset['price'],  histnorm='probability')
    #st.write(fig)
# different imputers

def impute_numerical_mean(data):
    
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    d = imp_mean.fit_transform(data)
    return d

def impute_cat_mode(data):
    
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    d = imp_mean.fit_transform(data)
    return d

def get_missing_cols(df):
    data = df.columns[df.isna().any()].tolist()
    return data

# imputing values

def impute_columns(df):
    
    missing_col = get_missing_cols(df)
    
    for i in missing_col:
        if df[i].dtype == 'O':
            df[i] = impute_cat_mode(df[i].values.reshape(-1, 1))
        else:
            df[i] = impute_numerical_mean(df[i].values.reshape(-1, 1))
            
def get_drop_and_impute(df):
    # returns two dataframes, one with imputed and another with droped values
    df_temp_imputed = df.copy()
    df_temp_drop_na = df.copy()
    
    impute_columns(df_temp_imputed)
    df_temp_drop_na.dropna(inplace=True)
    
    return df_temp_imputed, df_temp_drop_na
    

def format_data(df):
    typ_i, X_imp, y_imp= return_impute_drop(df)
    typ, X_drop, y_drop = return_impute_drop(df, return_imp=False)
    
    return [typ_i, typ], [(X_imp, y_imp), (X_drop, y_drop)]

def regression_difference(typ, x_y_pairs):
    
    results = []
    
    models = [RandomForestRegressor(), LinearRegression(), BaggingRegressor(), BayesianRidge()]
    
    for name, (X, y) in zip(typ, x_y_pairs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r_sq = model.score(X_train, y_train)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred, squared=False)

            results.append((model, name, r_sq, r2, mse))
    return results

#typ, X_imp, y_imp= return_impute_drop(df_subset)
#typ, X_drop, y_drop = return_impute_drop(df_subset, return_imp=False)

def return_impute_drop_test(df_subset, return_imp = True):
    
    #assuming missing value, returning two dataframes one with imputed values and another with removed rows where nan are present
    df_imp, df_drop = get_drop_and_impute(df_subset)
    df_drop.reset_index(inplace=True)
    
    obj = df_imp.select_dtypes(include=np.object)
    num = df_imp.select_dtypes(exclude=np.object)
    
    df_imp[obj.columns] = df_imp[obj.columns].astype('category')
    df_drop[obj.columns] = df_drop[obj.columns].astype('category')
    
    for i in obj.columns:
        encoder = joblib.load(f'enc_1_{i}.pkl')
        df_imp[i] = pd.DataFrame(encoder.transform(df_imp[i]))
        
    y_imp = df_imp['price'] 
    X_imp = df_imp.drop('price',axis=1)

    y_drop = df_drop['price'] 
    X_drop = df_drop.drop('price',axis=1)
    
    if return_imp:
        return 'imputed', X_imp, y_imp
    else:
        return 'removed', X_drop, y_drop
     

from scipy import stats
from sklearn.preprocessing import RobustScaler
def data_pipeline(df):
    
    #assuming missing value, returning two dataframes one with imputed values and another with removed rows where nan are present
    
    #cutting outliers
    
    #df = df[(df['letnik'] > 1901) & (df['letnik'] < 2026)]
    
    # discretizing
    '''
    bins = pd.IntervalIndex.from_tuples([(1900, 1990), (1991, 2000), (2001, 2006), (2007, 2012), (2013, 2018), (2019, 2025)])
    #df["letnik"] = pd.qcut(df['letnik'], q=10)
    df["letnik"] = pd.cut(df['letnik'], bins=[1900, 1990, 2000, 2005, 2011, 2017, 2025])
    df.letnik = df["letnik"].astype('object')
    '''
    # Cutting the extremes 
    #p_10 = df.price.quantile(0.01)
    #p_90 = df.price.quantile(0.9)
    #df = df[df.price.gt(p_10) & df.price.lt(p_90)]
    
    
   
    obj = df.select_dtypes(include=np.object)
    num = df.select_dtypes(exclude=np.object)
    
    for i in obj.columns:
        encoder = joblib.load(f'enc_1_{i}.pkl')
        df[i] = pd.DataFrame(encoder.transform(df[i]))
    
    
    #df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    X = df.drop('price',axis=1)

    #scale_columns = ['letnik', 'prevozeni_km']
    # scale continious variables
    scale_columns = ['prevozeni_km', 'Motor KM', 'letnik']

    for i in scale_columns:
        scaler = joblib.load(f'scaler_{i}.pkl')
        #print(pd.DataFrame(scaler.transform(X[i].values.ravel())))
        X[i] = pd.DataFrame(scaler.transform(X[i].values.reshape(-1, 1)))
    
    return X


st.write("""

# Car price prediction based on Slovenian car distribution

## Methods

1. Data collection
2. Data cleaning
3. Modelling

""")

#typ, X_imp, y_imp = return_impute_drop_test(df_subset)

#st.dataframe(X_imp)

encoder_model= joblib.load(f'enc_1_model.pkl')
encoder_brands = joblib.load(f'enc_1_brands.pkl')
encoder_gorivo = joblib.load(f'enc_1_Gorivo.pkl')
encoder_oblika = joblib.load(f'enc_1_Oblika.pkl')
encoder_menjalnik = joblib.load(f'enc_1_Menjalnik.pkl')
encoder_notranjost = joblib.load(f'enc_1_Notranjost.pkl')
encoder_starost = joblib.load(f'enc_1_Starost.pkl')
encoder_barva = joblib.load(f'enc_1_Barva.pkl')


model = st.sidebar.selectbox('Model:', encoder_model.classes_ )
brands = st.sidebar.selectbox('Znamka:', encoder_brands.classes_ )
gorivo = st.sidebar.selectbox('Gorivo:', encoder_gorivo.classes_ )  
oblika = st.sidebar.selectbox('Oblika:', encoder_oblika.classes_ )
menjalnik = st.sidebar.selectbox('Menjalnik:', encoder_menjalnik.classes_ )
notranjost = st.sidebar.selectbox('Notranjost:', encoder_notranjost.classes_ )
starost = st.sidebar.selectbox('Starost:', encoder_starost.classes_ )
barva = st.sidebar.selectbox('Barva:', encoder_barva.classes_ )

motor = st.sidebar.number_input('Moč motorja:')
prevozeni_km = st.sidebar.number_input('Prevoženi kilometri:')
letnik = st.sidebar.number_input('Letnik:')

col_names = df_subset.columns
df_test = pd.DataFrame(columns = col_names)

data = dict.fromkeys(col_names, 0)


data['model'] = model
data['brands'] = brands
data['Motor KM'] = motor
data['Notranjost'] = notranjost
data["Barva"] = barva
data["Oblika"] = oblika
data["Menjalnik"] = menjalnik
data["Gorivo"] = gorivo
data["prevozeni_km"] = prevozeni_km
data["Starost"] = starost
data["letnik"] = letnik

#print(data)

output = pd.DataFrame()
output = output.append(data, ignore_index=True)
#print(output)

X = data_pipeline(output)

st.dataframe(X)



print(X)

random_forests = joblib.load('model_test.joblib')
print(random_forests.predict(X))
st.header(str(int(random_forests.predict(X))) + ' €')


#import tensorflow as tf 

#from keras.models import load_model
#model = tf.keras.models.load_model('model_NN.h5')
#print(model.predict(X_imp))
#st.header(str(int(model.predict(X_imp))) + ' €')