from operator import index
import streamlit as st
import plotly.express as px
#from pycaret.regression import setup, compare_models, pull, save_model, load_model
from pandas_profiling import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os

import pandas as pd
import numpy as np


# Data Viz Pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns


# ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import io
from streamlit_option_menu import option_menu
from  PIL import Image

from memory_profiler import profile

st.set_page_config(layout="wide", page_title=" 💻📊 Analyse de Données UVS ")
st.markdown('<style>' + open('./style/style.css').read() + '</style>', unsafe_allow_html=True)
git, linkedin = st.columns(2)
git.markdown("[![Foo](https://img.icons8.com/material-outlined/48/000000/github.png)](https://github.com/lbfacto)")
git.info(" 👆 Récupérez l'intégralité du code ici")


st.title("  💻 Analyse automatisée de vos données 📊 ")


    # Then, drop the column as usual.
@profile
def main():
    with st.sidebar:
        st.image("https://www.campus-teranga.com/site/images/actualite/20210804-610aa19bbdf57.jpg")
        st.title("AutoML_Palu_UVS")
        st.image("./style/iiidata.png")
        choice = option_menu("Application AUTOML UVS SENEGAL", ["Upload","Profiling","Plots","Model Building", "Analyser","Exporter"],
                    icons =['upload file','task','download'],
                    menu_icon="cast", default_index=1,
                        styles={
        "container": {"padding": "5!important", "background-color": "#2ec147 "},
        "icon": {"color": "blue", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},

    }

)
    df = pd.read_csv('Data.SoucheO7O3_final.csv', index_col=0)
    if choice == "Upload":

        data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xls"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())


    elif choice == "Profiling":

        st.title("Exploratory Data Analysis")
        profile_df = df.profile_report() # faireun profiling sur le dataset
        st_profile_report(profile_df)


    elif choice == 'Model Building':
        st.subheader("Building ML Models")
        data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xls"])

        if data is not None:


                df = pd.read_csv(data)
                st.dataframe(df.head())
                df=df.drop(['ID','Type_Hb', 'Gpe_ABO', 'Rhesus','Sexe', 'TEST'], axis=1)
                conditionlist=[(df['ratio']>2),# ratio > 2 antigenepositif au palu
                            (df['ratio']<=2)] # ratio<=2 antigene nagtif au palu
                diagnostic = ['0', '1']
                df['Resultat'] = np.select(conditionlist, diagnostic)

                features = ['Age',	'ratio',	'G6PD',	'EP_6M_AVT',	'AcPf_6M_AVT',	'EP_1AN_AVT','AcPf_1AN_AVT',	'EP_6M_APR'	,'AcPf_6M_APR',	'EP_1AN_APR',	'AcPf_1AN_APR']
                target = ['Resultat']

                for attr in ['mean', 'ste', 'largest']:
                    for feature in features:
                        target.append(feature + "_" + attr)

                df['Resultat'] = df['Resultat'].astype(str).astype(int)
                X = df.drop(columns='Resultat', axis=1)
                Y = df['Resultat']
                seed =None
        # prepare models
                models = []
                models.append(('LR', LogisticRegression()))
                models.append(('LDA', LinearDiscriminantAnalysis()))
                models.append(('KNN', KNeighborsClassifier()))
                models.append(('CART', DecisionTreeClassifier()))
                models.append(('NB', GaussianNB()))
                models.append(('SVM', SVC(kernel='linear')))
        # evaluate each model in turn

                model_names = []
                model_mean = []
                model_std = []
                all_models = []
                scoring = 'accuracy'
                for name, model in models:

                    kfold = model_selection.KFold(n_splits=10, random_state=seed)
                    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                    model_names.append(name)
                    model_mean.append(cv_results.mean())
                    model_std.append(cv_results.std())

                    accuracy_results = {"model name":name,"model_accuracy":cv_results.mean(),"standard deviation":cv_results.std()}
                    all_models.append(accuracy_results)


                if st.checkbox("Metrics As Table"):
                    st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Algo","Mean of Accuracy","Std"]))
                if st.checkbox("Metrics As JSON"):
                    st.json(all_models)


    elif choice == 'Plots':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Data Visualization")

        data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xls"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())


            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
                st.pyplot()

            # Customizable Plot

            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
            selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)


            if st.button("Generate Plot"):
                st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

                # Plot By Streamlit
            if type_of_plot == 'area':
                    cust_data = df[selected_columns_names]
                    st.area_chart(cust_data)

            if type_of_plot == 'bar':
                    cust_data = df[selected_columns_names]
                    st.bar_chart(cust_data)

            if type_of_plot == 'line':
                    cust_data = df[selected_columns_names]
                    st.line_chart(cust_data)

                # Custom Plot
            if type_of_plot:
                    cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
                    st.write(cust_plot)
                    st.pyplot()

    elif choice == 'Analyser':
        st.header("Analyse de la qualité et exploration des données")
        profile_df = df.profile_report()


        st_profile_report(profile_df)
        profile_df.to_file("output.html")

        st.success("Rapport genéré correctement, rendez-vous dans l'onglet 'EXPORTER' pour télécharger votre rapport 💾 ")
    elif choice == 'Exporter':
        with open("output.html", 'rb') as f:
            dw = st.download_button("Télécharger le rapport 💾 ", f, "rapport_analyse_data.html")
            if dw :
                st.balloons()

                st.success("Rapport correctement téléchargé.")
    #else:
    # comment
        #add_page_visited_detail('A Propos', datetime.now())
        #st.subheader("A Propos")

if __name__ == '__main__':
        main()






