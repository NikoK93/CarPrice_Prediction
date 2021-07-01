import streamlit as st

import pandas as pd
import numpy as np
import scipy
from scipy import stats
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder,LabelEncoder



def get_one_hot_enc(feature_col, enc):
    """
    maps an unseen column feature using one-hot-encoding previously fit against training data 
    returns: a pd.DataFrame of newly one-hot-encoded feature
    """
    assert isinstance(feature_col, pd.Series)
    assert isinstance(enc, OneHotEncoder)
    unseen_vec = feature_col.values.reshape(-1, 1)
    encoded_vec = enc.transform(unseen_vec).toarray()
    encoded_df = pd.DataFrame(encoded_vec)
    return encoded_df

def data_pipeline(df, return_one_hot = False):
    
    
    obj = df.select_dtypes(include=np.object)
    
    if return_one_hot:
        for i in obj.columns:
            enc = joblib.load(f'./encoders/one_hot_encoder_{i}.pkl')
            
            train_X_encoded = get_one_hot_enc(df[i], enc)
            
            column_name = enc.get_feature_names([i])
            one_hot_encoded_frame =  pd.DataFrame(train_X_encoded.values, columns= column_name)
            
            df.drop(i, axis=1, inplace=True)
            df[column_name] = one_hot_encoded_frame
     # Saves Lable encoders
    else:    
        for i in obj.columns:    
            encoder = joblib.load(f'./encoders/label_encoder_{i}.pkl')
            df[i] = pd.DataFrame(encoder.transform(df[i]))
    

    # scale continous variables
    scale_columns = ['prevozeni_km', 'Motor KM', 'letnik']

    for i in scale_columns:
        scaler = joblib.load(f'scaler_{i}.pkl')
        df[i] = pd.DataFrame(scaler.transform(df[i].values.reshape(-1, 1)))
    
    return df



def app():
    encoder_brands = joblib.load(f'./encoders/one_hot_encoder_brands.pkl')
    encoder_gorivo = joblib.load(f'./encoders/one_hot_encoder_Gorivo.pkl')
    encoder_oblika = joblib.load(f'./encoders/one_hot_encoder_Oblika.pkl')
    encoder_menjalnik = joblib.load(f'./encoders/one_hot_encoder_Menjalnik.pkl')

    # Categorical input from encoder classes
    brands = st.sidebar.selectbox('Znamka:', encoder_brands.get_feature_names() )
    gorivo = st.sidebar.selectbox('Gorivo:', encoder_gorivo.get_feature_names() )  
    oblika = st.sidebar.selectbox('Oblika:', encoder_oblika.get_feature_names() )
    menjalnik = st.sidebar.selectbox('Menjalnik:', encoder_menjalnik.get_feature_names() )

    # Numerical input
    motor = st.sidebar.number_input('Moč motorja:')
    prevozeni_km = st.sidebar.number_input('Prevoženi kilometri:')
    letnik = st.sidebar.number_input('Letnik:')


    # Feature names
    col_names = ['letnik', 'prevozeni_km', 'Gorivo', 'Menjalnik', 'Oblika', 'Motor KM','brands']

    # Create dataframe and insert user selected values
    data = dict.fromkeys(col_names, 0)

    data['brands'] = brands
    data['Motor KM'] = motor
    data["Oblika"] = oblika
    data["Menjalnik"] = menjalnik
    data["Gorivo"] = gorivo
    data["prevozeni_km"] = prevozeni_km
    data["letnik"] = letnik

    # Dictionary to pandas dataframe
    output = pd.DataFrame()
    output = output.append(data, ignore_index=True)

    # Insert the data though the data pipiline
    X = data_pipeline(output, return_one_hot=True)

    model_names = ['XGBoost']

    # Caching the model import for better performance
    @st.cache(hash_funcs={"builtins.dict": lambda _: None})
    def get_models():
        models = dict.fromkeys(model_names, 0)
        #models['Random Forests'] = joblib.load('./models/model_one_hot_RF.joblib')
        models['XGBoost'] = joblib.load('./models/model_one_hot_XGB.joblib')
        return models

    # Loading the models in a dictionary with model_name:model pairs
    models = get_models()

    # User model selection
    model_selections = st.selectbox('Select model:', model_names)

    # Model prediction output
    st.markdown(f"<h1 style='text-align: center; color: black;'>{(str(int(models[model_selections].predict(X.values))) + ' €')}</h1>", unsafe_allow_html=True)
    print(models[model_selections].predict(X.values))
    
    sns.set_style("white")
    NAME = 'joinedData.csv'
    df = pd.read_csv(NAME)

    fig, ax = plt.subplots()
    sns.set_style("white")

    xx = np.linspace(df['price'].min(), df['price'].max(), 100)

    ax.plot(xx, stats.lomax.pdf(xx, *stats.lomax.fit(df['price'])), color="r", lw=2.5)
    plt.axvline(int(models[model_selections].predict(X.values)))

    percentile = scipy.stats.percentileofscore(df['price'], int(models[model_selections].predict(X.values)))
    st.markdown(f"<h1 style='text-align: center; color: black;'>{'Percentile rank: ' + str(round(percentile, 2))}</h1>", unsafe_allow_html=True)
    
    ax.set(xlim=(0,200000))


    st.pyplot(fig)

