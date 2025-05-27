import streamlit as st
import pandas as pd
import pickle
import time
from streamlit_option_menu import option_menu

# --- New Chatbot Imports ---
from groq import Groq
import tempfile
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import wave
import requests
import pyttsx3
# --- End New Chatbot Imports ---

# --- Function to Inject Custom CSS ---
def inject_css(file_path):
    """
    Injects custom CSS from a local file into the Streamlit app.
    """
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Erreur: Le fichier CSS '{file_path}' est introuvable. Assurez-vous qu'il est dans le même répertoire.")
        st.stop()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="ClientInsight Pro - DashSphere",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Inject Custom CSS from style.css ---
inject_css('style.css')

# --- Load Machine Learning Model (Cached for Performance) ---
@st.cache_resource
def load_model():
    """
    Loads the pre-trained machine learning model from 'model.pkl'.
    """
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Erreur : Le fichier 'model.pkl' est introuvable. Assurez-vous qu'il est dans le même répertoire que 'app.py'.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}. Vérifiez que 'model.pkl' est un fichier pickle valide.")
        st.stop()

model = load_model()

# --- Preprocessing Function for Model Input ---
def preprocess_input(credit_score, geography_display, gender_display, age, tenure, balance,
                     num_products, has_cr_card, is_active_member, estimated_salary):
    """
    Transforms raw user input into the format expected by the machine learning model.
    """
    try:
        gender_encoded = 0 if gender_display == 'Homme' else 1
        if geography_display == 'France':
            geography_encoded = 0.5014
        elif geography_display == 'Allemagne':
            geography_encoded = 0.2509
        else: # 'Espagne'
            geography_encoded = 0.2477

        has_cr_card_encoded = 1 if has_cr_card == 'Oui' else 0
        is_active_member_encoded = 1 if is_active_member == 'Oui' else 0

        input_data = pd.DataFrame([[
            credit_score, geography_encoded, gender_encoded, age, tenure, balance,
            num_products, has_cr_card_encoded, is_active_member_encoded, estimated_salary
        ]], columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                     'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
        return input_data
    except Exception as e:
        st.error(f"Erreur lors du prétraitement des données : {e}. Veuillez vérifier les types de données des entrées.")
        return None

# --- Main Application Header ---
st.markdown("<h1>Client<span style='color: var(--primary-color);'>Insight Pro</span></h1>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; font-size: 1.2rem; color: var(--secondary-color); margin-bottom: 2.5rem;'>
    Votre portail intelligent pour l'analyse prédictive et l'assistance client.
    </p>
    """, unsafe_allow_html=True)

# --- Horizontal Navigation with streamlit_option_menu ---
selected = option_menu(
    menu_title=None,
    options=["Prédiction de Désabonnement", "Chatbot d'Assistance"],
    icons=["graph-up", "chat-text"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "var(--background-dark)", "margin-bottom": "2rem"},
        "icon": {"color": "var(--text-light)", "font-size": "18px"},
        "nav-link": {
            "font-size": "18px",
            "text-align": "center",
            "margin": "0px 0.75rem",
            "background-color": "transparent",
            "color": "#00bcd4",
            "border": "2px solid #8A2BE2",
            "border-radius": "2rem",
            "transition": "all 0.3s ease",
            "--hover-color": "#00bcd4"
        },
        "nav-link-selected": {
            "background-color": "var(--primary-color)",
            "color": "var(--background-dark)",
            "border-radius": "2rem",
            "border": "2px solid var(--primary-color)",
            "box-shadow": "0 5px 20px rgba(0, 188, 212, 0.6)",
            "transform": "translateY(-2px)"
        },
    }
)

if selected == "Prédiction de Désabonnement":
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "prediction_message" not in st.session_state:
        st.session_state.prediction_message = None

    st.markdown("<h3><svg viewBox='0 0 24 24' width='30' height='30' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M22 12h-4l-3 9L9 3l-3 9H2'></path></svg>Outil de Prédiction Client</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--secondary-color); margin-bottom: 2rem; text-align: center;'>Entrez les détails du client ci-dessous pour analyser le risque de désabonnement.</p>", unsafe_allow_html=True)

    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        message = st.session_state.prediction_message

        st.markdown("<div class='results-display'>", unsafe_allow_html=True)
        st.markdown("<h4>Résultat de la Prédiction</h4>", unsafe_allow_html=True)

        if "error" in result and result["error"]:
            st.error(message)
        elif result["prediction"] == 1:
            st.markdown(f"<p class='churn-risk'>💔 Risque ÉLEVÉ de Désabonnement !</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='probability'>Probabilité estimée : <strong>{(result['probability'] * 100):.2f}%</strong></p>", unsafe_allow_html=True)
            st.warning(message)
        else:
            st.markdown(f"<p class='no-churn-risk'>✅ Faible Risque de Désabonnement</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='probability'>Probabilité estimée : <strong>{(result['probability'] * 100):.2f}%</strong></p>", unsafe_allow_html=True)
            st.success(message)

        st.markdown("<div style='text-align: center; margin-top: 1.5rem;'>", unsafe_allow_html=True)
        if st.button("Nouvelle Prédiction", key="new_prediction_btn"):
            st.session_state.prediction_result = None
            st.session_state.prediction_message = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.form("prediction_form_main_content", clear_on_submit=False):
        if st.session_state.prediction_result is not None:
             st.markdown("<div style='margin-top: 2rem; border-top: 1px solid var(--border-dark); padding-top: 2rem;'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            credit_score = st.slider("Score de Crédit", 350, 850, 650, key="main_credit_score")
            geography = st.selectbox("Géographie", ['France', 'Allemagne', 'Espagne'], key="main_geography")
            gender = st.selectbox("Sexe", ['Homme', 'Femme'], key="main_gender")
            age = st.number_input("Âge", min_value=18, max_value=92, value=35, key="main_age")
            tenure = st.number_input("Ancienneté (années)", min_value=0, max_value=10, value=5, key="main_tenure")

        with col2:
            balance = st.number_input("Solde du Compte (€)", min_value=0.0, value=0.0, step=0.01, key="main_balance")
            num_products = st.number_input("Nombre de Produits", min_value=1, max_value=4, value=1, key="main_num_products")
            has_cr_card = st.selectbox("Possède une Carte de Crédit ?", ['Oui', 'Non'], key="main_has_cr_card")
            is_active_member = st.selectbox("Est un Membre Actif ?", ['Oui', 'Non'], key="main_is_active_member")
            estimated_salary = st.number_input("Salaire Estimé Annuel (€)", min_value=0.0, value=50000.0, step=0.01, key="main_estimated_salary")

        st.markdown("<div style='text-align: center; margin-top: 2rem;'>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Prédire le Désabonnement", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        with st.spinner("Analyse en cours..."):
            time.sleep(2)

            input_data = preprocess_input(
                credit_score, geography, gender, age, tenure, balance,
                num_products, has_cr_card, is_active_member, estimated_salary
            )

            if input_data is not None:
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]

                st.session_state.prediction_result = {
                    "prediction": int(prediction),
                    "probability": float(prediction_proba[1]) if prediction == 1 else float(prediction_proba[0])
                }
                if prediction == 1:
                    st.session_state.prediction_message = "Action Requise : Ce client présente un risque significatif de désabonnement. Une intervention rapide (offre personnalisée, contact proactif) est cruciale pour la rétention."
                else:
                    st.session_state.prediction_message = "Bonne nouvelle : Ce client est stable. Continuez à maintenir une relation positive pour assurer sa satisfaction et sa fidélité."
            else:
                st.session_state.prediction_result = {"error": True}
                st.session_state.prediction_message = "Impossible de traiter les données. Vérifiez les entrées."
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Chatbot d'Assistance":
    load_dotenv()
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    st.markdown("<h3><svg viewBox='0 0 24 24' width='30' height='30' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M21 15a2 2 0 0 1-2 2H7l-4 4V3a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'></path></svg>Assistant Virtuel</h3>", unsafe_allow_html=True)

    st.sidebar.title("⚙️ Paramètres du Chatbot")
    if st.sidebar.button("Effacer l'historique du Chatbot"):
        st.session_state.messages = []
        st.session_state.messages.append({"role": "bot", "content": "Bonjour ! Je suis votre assistant virtuel. Comment puis-je vous aider aujourd'hui ?"})
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "bot", "content": "Bonjour ! Je suis votre assistant virtuel. Comment puis-je vous aider aujourd'hui ?"})

    # --- Display chat history with icons ---
    for message_item in st.session_state.messages:
        if message_item["role"] == "user":
            st.markdown(f"""
                <div class='message-bubble user-message'>
                    <span class='message-icon'>👤</span> {message_item['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='message-bubble bot-message'>
                    <span class='message-icon'>🤖</span> {message_item['content']}
                </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # --- End chat history display with icons ---


    with st.form("chat_input_form", clear_on_submit=True):
        message_input = st.text_input("💬 Entrez votre message ici :", "", key="chat_text_input")
        audio_file_uploader = st.file_uploader("📢 Téléversez un message audio (format m4a, mp3, wav)", type=["m4a", "mp3", "wav"], key="chat_audio_uploader")

        col_voice, col_send = st.columns([0.5, 0.5])
        with col_voice:
            speak_button = st.form_submit_button("🎤 Parler", type="secondary", help="Enregistre votre voix pendant 5 secondes")
        with col_send:
            send_button = st.form_submit_button("✉️ Envoyer Message", type="primary")

    if audio_file_uploader:
        st.audio(audio_file_uploader, format="audio/mp3")

    def record_audio():
        duration = 5
        samplerate = 16000
        with st.spinner("🎙 Enregistrement en cours... Parlez maintenant !"):
            audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
            sd.wait()

            temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with wave.open(temp_audio_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(samplerate)
                wf.writeframes(audio_data.tobytes())

            return temp_audio_path

    processed_message = ""

    if speak_button:
        audio_path = record_audio()
        if audio_path:
            with open(audio_path, "rb") as file:
                with st.spinner("Transcription audio en cours..."):
                    transcription = client.audio.transcriptions.create(
                        file=("audio.wav", file.read()),
                        model="whisper-large-v3",
                        response_format="json",
                        language="fr",
                        temperature=0.0
                    )
            processed_message = transcription.text
            st.write(f"**📝 Texte détecté :** {processed_message}")
            os.remove(audio_path)

    elif audio_file_uploader:
        with st.spinner("🎙 Transcription en cours..."):
            filename = tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file_uploader.name.split('.')[-1]}").name
            with open(filename, "wb") as f:
                f.write(audio_file_uploader.getvalue())

            with open(filename, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(audio_file_uploader.name, file.read()),
                    model="whisper-large-v3",
                    response_format="json",
                    language="fr",
                    temperature=0.0
                )
            processed_message = transcription.text
            st.write(f"**📝 Texte détecté :** {processed_message}")
            os.remove(filename)

    elif send_button and message_input:
        processed_message = message_input

    if processed_message:
        st.session_state.messages.append({"role": "user", "content": processed_message})
        with st.spinner("Le chatbot réfléchit..."):
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": processed_message}],
                model="llama-3.3-70b-versatile",
            )
            response_text = chat_completion.choices[0].message.content

            st.session_state.messages.append({"role": "bot", "content": response_text})

        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# --- Global Footer ---
st.markdown("---")
st.markdown("""
    <p class='app-footer'>
        Conçu avec passion pour l'excellence de l'expérience client | © 2024 VotreEntreprise<br>
        Optimisé pour une interaction intuitive.
    </p>
    """, unsafe_allow_html=True)
