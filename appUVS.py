import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import io
import webbrowser
import yaml
import streamlit_menu as menu
#import streamlit_authenticator as stauth
import yaml
import streamlit as st
from yaml.loader import SafeLoader
import streamlit.components.v1 as components


url = 'https://www.gmail.com/'



logo = Image.open(r'C:/Users/dell/Desktop/logoUvs/uvs.JPEG')


with st.sidebar:
    st.image("https://www.campus-teranga.com/site/images/actualite/20210804-610aa19bbdf57.jpg")
    choose = option_menu("Application de detection Paludisme", ["About", "Prediction Paludisme","Enregistrer Patient", "Contact"],
                    icons=['house', 'bi bi-graph-down-arrow', 'bi bi-droplet-fill', 'bi bi-file-person-fill'],
                    menu_icon="app-indicator", default_index=0,
                    styles={
    "container": {"padding": "5!important", "background-color": "#fafafa"},
    "icon": {"color": "orange", "font-size": "25px"},
    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected": {"background-color": "#02ab21"},
}
)

logo = Image.open(r'C:/Users/dell/Desktop/logoUvs/uvs.JPEG')
#profile = Image.open(r'C:\Users\13525\Desktop\medium_profile.png')
if (choose == "About"):
    col1, col2 = st.columns( [0.8, 0.2])

    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
    </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">A propos du createur</p>',
                    unsafe_allow_html=True)
    with col2:               # To display brand log
        st.image(logo, width=150)

    st.write("Abdoulaye BA etudiant en MASTER 2 BIG DATA UNIVERSITE DU SENEGAL, Aussi ingenieur des traveaux informatiques à l'hopaital aristide le dantec et Administrateur Reseaux et sytemes d'information et gestionnaire de parc informatique le lien du repos sur github est disponibles sur ce lien: https://github.com/lbfacto")
#st.image(profile, width=700 )rue




#analyse des donnes

elif (choose =="Prediction Paludisme"):
    st.title("Predictioon Paludisme sur des sujet au senegal") #titre de l
    palu_pedict = pickle.load(open('trained_model.pkl','rb'))

# change the input_data to a numpy array
#Les colones
    col1,col2, col3,col4=st.columns(4)
    Age = st.text_input('Age')

    ratio =st.text_input('ratio')

    G6PD = st.text_input('G6PD')

    EP_6M_AVT = st.text_input('EP_6M_AVT')

    AcPf_6M_AVT = st.text_input('AcPf_6M_AVT')

    EP_1AN_AVT = st.text_input('EP_1AN_AVT')

    AcPf_1AN_AVT =st.text_input('AcPf_1AN_AVT')

    EP_6M_APR = st.text_input('EP_6M_APR')

    AcPf_6M_APR = st.text_input('AcPf_6M_APR')

    EP_1AN_APR = st.text_input('EP_1AN_APR')

    AcPf_1AN_APR = st.text_input('AcPf_1AN_APR')

    palu_diagnosis =''


    if st.button('Resultat Paludisme'):
        palu_pediction = palu_pedict.predict([[Age,ratio,G6PD,EP_6M_AVT,AcPf_6M_AVT,EP_1AN_AVT,AcPf_1AN_AVT,EP_6M_APR	,AcPf_6M_APR,EP_1AN_APR	,AcPf_1AN_APR]])
        if(palu_pediction[0]==1):

            palu_diagnosis = 'Antigene positif Personne atteint du paludisme'
        else:
            palu_diagnosis= 'Antigene negatif personne n_est pas atteint du paludisme'

        st.success(palu_diagnosis)



elif choose == "Enregistrer Patient":
    st.markdown(""" <style> .font {

    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
    </style> """, unsafe_allow_html=True)

    st.markdown('<p class="font">Enregister Resultat</p>', unsafe_allow_html=True)

    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    #st.write('Please help us improve!')

        Nom=st.text_input(label='Entrer Name') #Collect user feedback
        Age=st.text_input(label='Entrer Age') #Collect user feedback

        Prenom=st.text_input(label=' Prenom') #Collect user feedback
        Email=st.text_input(label='Entrer Email') #Collect user feedback
        Telephone=st.text_input(label='Entrer Telephone') #Collect user feedback
        Adresse=st.text_input(label='Entrer Adresse') #Collect user feedback
        Resultat=st.text_input(label='Entrer Resultat') #Collect user feedback
        Avis_Du_Medecin=st.text_input(label='Avis') #Collect user feedback
        submitted = st.form_submit_button('Submit')

        if submitted:
            st.write('Donner Patient enregistrer')



elif choose == "Contact":
    if st.button('Open browser'):
        webbrowser.open_new_tab(url)


    st.markdown(""" <style> .font {

    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
    </style> """, unsafe_allow_html=True)

    st.markdown('<p class="font">Votre Avis</p>', unsafe_allow_html=True)

    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
#st.write('Please help us improve!')

        Nom=st.text_input(label='Entrer votre Nom') #Collect user feedback
        Prenom=st.text_input(label='Entrer votre Prenom')
        Email=st.text_input(label='Entrer votre Email') #Collect user feedback
        Message=st.text_input(label='Votre Message') #Collect user feedback
        submitted = st.form_submit_button('Submit')

        if submitted:
            st.write('Message recu merci du feedback')










