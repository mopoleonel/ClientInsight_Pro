import streamlit as st
import pandas as pd
import pickle
import time
import os
from streamlit_option_menu import option_menu
import base64
import tempfile
import json

# --- New Chatbot Imports ---
from groq import Groq
# load_dotenv() # Retiré car nous utilisons st.secrets pour la clé GROQ
import pyttsx3 # For text-to-speech, keep this if you want it
# --- End New Chatbot Imports ---

# --- Custom Streamlit Component for Audio Recording ---
def audio_recorder_component():
    """
    Streamlit component to record audio in the browser and return it as base64.
    """
    # Define the component name used in JavaScript's postMessage
    component_name = "audio_recorder_custom_component"

    st.components.v1.html(
        f"""
        <style>
        .audio-recorder-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 10px;
            border-radius: 8px;
            background-color: #262730;
            margin-top: 10px;
        }}
        .audio-recorder-button {{
            background-color: #FF4B4B;
            color: white;
            border: none;
            border-radius: 50%;
            width: 45px;
            height: 45px;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.1s ease;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }}
        .audio-recorder-button:hover {{
            background-color: #E63B3B;
            transform: scale(1.05);
        }}
        .audio-recorder-button:active {{
            transform: scale(0.95);
        }}
        .audio-recorder-button.recording {{
            background-color: #28a745;
            animation: pulse 1.5s infinite;
        }}
        .audio-recorder-status {{
            margin-top: 8px;
            font-size: 0.8em;
            color: #ccc;
        }}
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); }}
            70% {{ box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }}
        }}
        </style>
        <div class="audio-recorder-container">
            <button id="recordButton" class="audio-recorder-button">🎤</button>
            <div id="status" class="audio-recorder-status">Appuyez pour enregistrer</div>
        </div>

        <script>
            const recordButton = document.getElementById('recordButton');
            const statusDiv = document.getElementById('status');
            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;

            function sendAudioToStreamlit(data) {{ // Doubled {{
                if (window.parent.streamlitReportReady) {{ // Doubled {{
                    window.parent.streamlitReportReady();
                }} // Doubled }}
                console.log("Sending data to Streamlit. Data length:", data ? data.length : 0); // DEBUG JS
                window.parent.postMessage({{ // Doubled {{
                    type: 'streamlit:setComponentValue',
                    componentName: '{component_name}',
                    value: data,
                }}, '*'); // Doubled }}
            }} // Doubled }}

            recordButton.onclick = async () => {{ // Doubled {{
                if (!isRecording) {{ // Doubled {{
                    isRecording = true;
                    recordButton.classList.add('recording');
                    recordButton.innerHTML = '🛑';
                    statusDiv.innerText = 'Enregistrement... Appuyez pour arrêter.';
                    audioChunks = [];

                    try {{ // Doubled {{
                        const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }}); // Doubled {{
                        // Ensure audio/webm is supported, fallback if needed
                        const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' :
                                       MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' :
                                       'audio/ogg'; // Fallback to a common type
                        
                        console.log("Using MIME type for recording:", mimeType); // DEBUG JS
                        mediaRecorder = new MediaRecorder(stream, {{ mimeType: mimeType }}); // Doubled {{

                        mediaRecorder.ondataavailable = event => {{ // Doubled {{
                            audioChunks.push(event.data);
                        }}; // Doubled }}

                        mediaRecorder.onstop = async () => {{ // Doubled {{
                            const audioBlob = new Blob(audioChunks, {{ type: mimeType }}); // Doubled {{
                            const reader = new FileReader();
                            reader.readAsDataURL(audioBlob);
                            reader.onloadend = () => {{ // Doubled {{
                                const base64data = reader.result.split(',')[1];
                                console.log("Audio recorded. Base64 length:", base64data ? base64data.length : 0); // DEBUG JS
                                sendAudioToStreamlit(base64data);
                            }}; // Doubled }}
                            stream.getTracks().forEach(track => track.stop());
                        }}; // Doubled }}

                        mediaRecorder.start();
                    }} catch (err) {{ // Doubled {{
                        console.error('Error accessing microphone or media recording:', err); // DEBUG JS
                        statusDiv.innerText = 'Erreur: Accès micro refusé ou impossible. Vérifiez les permissions de votre navigateur.';
                        isRecording = false;
                        recordButton.classList.remove('recording');
                        recordButton.innerHTML = '🎤';
                    }} // Doubled }}

                }} else {{ // Doubled {{
                    isRecording = false;
                    recordButton.classList.remove('recording');
                    recordButton.innerHTML = '🎤';
                    statusDiv.innerText = 'Appuyez pour enregistrer';
                    if (mediaRecorder && mediaRecorder.state === 'recording') {{ // Doubled {{
                        mediaRecorder.stop();
                    }} // Doubled }}
                }} // Doubled }}
            }}; // Doubled }}
        </script>
        """,
        height=100,
        scrolling=False
    )
    # Return the value from session_state using the defined component_name
    return st.session_state.get(component_name, None)

# --- End Custom Streamlit Component ---

# --- Function to Inject Custom CSS ---
def inject_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Erreur CSS: Le fichier '{file_path}' est introuvable. Assurez-vous qu'il est dans le même répertoire.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur CSS: Impossible d'injecter le CSS depuis '{file_path}'. Erreur: {e}")
        st.stop()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="ClientInsight Pro - DashSphere",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_css('style.css')

# --- Load Machine Learning Model (Cached for Performance) ---
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Erreur Modèle: Le fichier 'model.pkl' est introuvable. Assurez-vous qu'il est dans le même répertoire que 'app.py'.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur Modèle: Impossible de charger le modèle. Vérifiez que 'model.pkl' est un fichier pickle valide. Erreur: {e}")
        st.stop()

model = load_model()

# --- Preprocessing Function for Model Input ---
def preprocess_input(credit_score, geography_display, gender_display, age, tenure, balance,
                     num_products, has_cr_card, is_active_member, estimated_salary):
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
        st.error(f"Erreur Prétraitement: Impossible de prétraiter les données d'entrée. Erreur: {e}")
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
            time.sleep(2) # Simule un délai de traitement

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

    st.markdown("</div>", unsafe_allow_html=True) # Clôture hypothétique d'un div parent pour la section prédiction

elif selected == "Chatbot d'Assistance":
    # load_dotenv() # Retiré
    client = None # Initialisation
    try:
        # Modification pour utiliser st.secrets
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except KeyError: # Spécifiquement pour une clé manquante dans st.secrets
        st.error("Erreur d'API Groq: La clé 'GROQ_API_KEY' n'est pas configurée dans les secrets de Streamlit.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur d'initialisation Groq: Impossible de se connecter à l'API Groq. Erreur: {e}. Vérifiez votre clé et votre connexion.")
        st.stop()

    # Le reste de votre code pour le Chatbot d'Assistance reste identique
    st.markdown("<h3><svg viewBox='0 0 24 24' width='30' height='30' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M21 15a2 2 0 0 1-2 2H7l-4 4V3a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'></path></svg>Assistant Virtuel</h3>", unsafe_allow_html=True)

    st.sidebar.title("⚙️ Paramètres du Chatbot")
    if st.sidebar.button("Effacer l'historique du Chatbot"):
        st.session_state.messages = []
        st.session_state.messages.append({"role": "bot", "content": "Bonjour ! Je suis votre assistant virtuel. Comment puis-je vous aider aujourd'hui ?"})
        if 'last_processed_recorded_audio' in st.session_state: # Aussi effacer ce cache
            del st.session_state['last_processed_recorded_audio']
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
            if 'audio_data' in message_item and message_item['audio_data']: 
                try:
                    st.audio(base64.b64decode(message_item['audio_data']), format='audio/webm', start_time=0)
                except Exception as e:
                    st.warning(f"Impossible de lire l'audio enregistré : {e}")
        else:
            st.markdown(f"""
                <div class='message-bubble bot-message'>
                    <span class='message-icon'>🤖</span> {message_item['content']}
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("<h4>Ou enregistrez votre voix :</h4>", unsafe_allow_html=True)
    recorded_audio_base64 = audio_recorder_component()

    with st.form("chat_input_form", clear_on_submit=True):
        message_input = st.text_input("💬 Entrez votre message ici :", "", key="chat_text_input")
        audio_file_uploader = st.file_uploader("📢 Téléversez un message audio (format m4a, mp3, wav)", type=["m4a", "mp3", "wav"], key="chat_audio_uploader")
        send_button = st.form_submit_button("✉️ Envoyer Message", type="primary")

    processed_message_content = ""
    audio_to_display = None 

    component_name = "audio_recorder_custom_component" 

    if recorded_audio_base64 and recorded_audio_base64 != st.session_state.get('last_processed_recorded_audio', None):
        st.info("DEBUG (Python): Détection d'un nouvel audio enregistré. Traitement en cours...")
        st.session_state['last_processed_recorded_audio'] = recorded_audio_base64 

        try:
            st.info(f"DEBUG (Python): Base64 audio reçu. Longueur : {len(recorded_audio_base64) if recorded_audio_base64 else 0}")
            
            audio_bytes = base64.b64decode(recorded_audio_base64)
            audio_to_display = recorded_audio_base64 

            st.info(f"DEBUG (Python): Audio décodé en octets. Longueur : {len(audio_bytes)}")

            if not audio_bytes:
                st.error("Erreur Enregistrement: L'audio enregistré est vide après décodage Base64.")
                if component_name in st.session_state:
                    del st.session_state[component_name]
                st.rerun() 
            
            temp_audio_path = "" # Initialisation pour éviter UnboundLocalError
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio_file:
                temp_audio_file.write(audio_bytes)
            temp_audio_path = temp_audio_file.name

            st.info(f"DEBUG (Python): Fichier temporaire créé à : {temp_audio_path}")
            if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                st.error(f"Erreur Enregistrement: Le fichier audio temporaire est vide ou n'a pas été créé correctement à {temp_audio_path}.")
                if os.path.exists(temp_audio_path): 
                    os.remove(temp_audio_path)
                if component_name in st.session_state:
                    del st.session_state[component_name]
                st.rerun()

            with open(temp_audio_path, "rb") as file:
                with st.spinner("Transcription audio en cours..."):
                    try: 
                        transcription = client.audio.transcriptions.create(
                            file=("recorded_audio.webm", file.read()), 
                            model="whisper-large-v3",
                            response_format="json",
                            language="fr",
                            temperature=0.0
                        )
                        processed_message_content = transcription.text
                        st.info(f"DEBUG (Python): Transcription réussie: '{processed_message_content}'")
                    except Exception as groq_transcription_error:
                        st.error(f"Erreur Transcription (Groq API) : {groq_transcription_error}")
                        processed_message_content = "" 

            if os.path.exists(temp_audio_path): 
                os.remove(temp_audio_path) 

            if processed_message_content:
                st.session_state.messages.append({"role": "user", "content": processed_message_content, "audio_data": audio_to_display})
                with st.spinner("Le chatbot réfléchit..."):
                    try:
                        chat_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": processed_message_content}],
                            model="llama3-70b-8192", # Modèle exemple, ajustez si besoin
                        )
                        response_text = chat_completion.choices[0].message.content
                        st.session_state.messages.append({"role": "bot", "content": response_text})
                        st.info(f"DEBUG (Python): Réponse du chatbot obtenue.")
                    except Exception as e_chat: # Renommé pour éviter conflit avec e extérieur
                        st.error(f"Erreur Chatbot (API): Impossible d'obtenir une réponse du chatbot Groq. Erreur: {e_chat}")
                        st.session_state.messages.append({"role": "bot", "content": "Désolé, je n'ai pas pu traiter votre demande. Une erreur est survenue lors de la communication avec le service de chatbot."})
            else:
                st.warning("Aucune transcription obtenue, le chatbot ne sera pas interrogé.")

            if component_name in st.session_state:
                del st.session_state[component_name]
            st.rerun()

        except Exception as e_outer: # Renommé pour éviter conflit
            st.error(f"Erreur Générale (Audio Enregistré): Une erreur inattendue est survenue lors du traitement de l'audio enregistré. Erreur: {e_outer}")
            if component_name in st.session_state:
                del st.session_state[component_name]
            if 'last_processed_recorded_audio' in st.session_state:
                del st.session_state['last_processed_recorded_audio']
            st.rerun() 

    elif send_button:
        user_input_for_chatbot = "" # Pour stocker le texte à envoyer au chatbot
        if audio_file_uploader:
            st.info("Traitement du fichier audio téléversé...")
            try:
                original_filename = audio_file_uploader.name
                temp_file_path_upload = tempfile.NamedTemporaryFile(delete=False, suffix=f".{original_filename.split('.')[-1]}").name
                with open(temp_file_path_upload, "wb") as f:
                    f.write(audio_file_uploader.getvalue())

                with open(temp_file_path_upload, "rb") as file_to_transcribe:
                    with st.spinner("Transcription audio (téléversé) en cours..."):
                        transcription = client.audio.transcriptions.create(
                            file=(original_filename, file_to_transcribe.read()),
                            model="whisper-large-v3",
                            response_format="json",
                            language="fr",
                            temperature=0.0
                        )
                user_input_for_chatbot = transcription.text
                # Pas besoin d'ajouter audio_data ici car c'est un fichier uploadé, pas un enregistrement direct pour relecture simple
                st.session_state.messages.append({"role": "user", "content": user_input_for_chatbot}) 
                os.remove(temp_file_path_upload) 
            except Exception as e_upload_form: # Renommé
                st.error(f"Erreur Transcription (Fichier Téléversé): {e_upload_form}")
                user_input_for_chatbot = ""
        elif message_input:
            user_input_for_chatbot = message_input
            st.session_state.messages.append({"role": "user", "content": user_input_for_chatbot})

        if user_input_for_chatbot: # S'il y a eu du contenu (texte ou transcription d'upload)
            st.info("DEBUG (Python): Message (formulaire) envoyé au Chatbot.") # Message de debug plus clair
            with st.spinner("Le chatbot réfléchit..."):
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": user_input_for_chatbot}],
                        model="llama3-70b-8192", # Modèle exemple
                    )
                    response_text = chat_completion.choices[0].message.content
                    st.session_state.messages.append({"role": "bot", "content": response_text})
                except Exception as e_chat_form: # Renommé
                    st.error(f"Erreur Chatbot (API Formulaire): {e_chat_form}")
                    st.session_state.messages.append({"role": "bot", "content": "Désolé, je n'ai pas pu traiter votre demande (formulaire)."})
        
        if 'last_processed_recorded_audio' in st.session_state:
            del st.session_state['last_processed_recorded_audio']
        
        st.rerun() 

    st.markdown("</div>", unsafe_allow_html=True) # Fermeture hypothétique du div.content-card

# --- Global Footer ---
st.markdown("---")
st.markdown("""
    <p class='app-footer'>
        Conçu avec passion pour l'excellence de l'expérience client | © 2024 VotreEntreprise<br>
        Optimisé pour une interaction intuitive.
    </p>
    """, unsafe_allow_html=True)
