import streamlit as st
import requests
import json
import time

# FastAPI backend URL
BACKEND_URL = "http://localhost:8000"

# --- Helper Functions ---
def get_admin_settings():
    try:
        response = requests.get(f"{BACKEND_URL}/admin/settings/")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching admin settings: {e}")
        # Return a default structure on error to prevent crashes
        return {
            "selected_llm_provider": "openai",
            "selected_llm_model": "gpt-3.5-turbo",
            "selected_embedding_provider": "openai",
            "selected_embedding_model": "text-embedding-ada-002",
            "custom_prompt_template": "Error: Could not load prompt.",
            "available_openai_llms": ["gpt-3.5-turbo"],
            "available_gemini_llms": ["gemini-pro", "gemini-1.5-flash-latest", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
            "available_openai_embedding_models": ["text-embedding-ada-002"],
            "available_gemini_embedding_models": ["models/embedding-001"],
            "pinecone_index_name": "N/A",
            "pinecone_index_dimension": 0
        }

def save_admin_settings(settings_payload: dict):
    try:
        # Ensure all required fields are present in the payload for the backend
        required_fields = [
            "selected_llm_provider", "selected_llm_model",
            "selected_embedding_provider", "selected_embedding_model",
            "custom_prompt_template"
        ]
        for field in required_fields:
            if field not in settings_payload:
                st.error(f"Missing field in settings payload: {field}")
                return None
        
        response = requests.post(f"{BACKEND_URL}/admin/settings/", data=settings_payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error updating admin settings: {e}")
        if response is not None:
            st.error(f"Backend response: {response.text}")
        return None

def upload_document_to_backend(file_bytes, filename):
    try:
        files = {'file': (filename, file_bytes, "application/octet-stream")}
        # llm_provider_for_embedding is now handled by backend global settings
        response = requests.post(f"{BACKEND_URL}/upload-document/", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading document: {e}")
        return None

def query_backend(question: str, llm_model: str):
    try:
        # llm_provider is now handled by backend global settings
        payload = {"question": question, "llm_model": llm_model}
        response = requests.post(f"{BACKEND_URL}/query/", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying backend: {e}")
        return None

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="RAG Chatbot")

# Theme selection
st.session_state.theme = st.sidebar.selectbox("Theme", ["light", "dark"], index=0 if "theme" not in st.session_state else (0 if st.session_state.theme == "light" else 1))

# Apply theme
if st.session_state.theme == "dark":
    st.markdown(
        """
        <style>
        body {
            color: white;
            background-color: #262730;
        }
        .streamlit-expanderHeader {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Add custom CSS for fonts and general styles
st.markdown(
    """
    <style>
    /* Import custom fonts here if needed */
    /* @import url('https://fonts.googleapis.com/css2?family=Your+Font+Here&display=swap'); */

    body {
        font-family: 'Arial', sans-serif; /* Replace with your custom font */
        line-height: 1.6;
        color: #333;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a; /* Example heading color */
    }

    /* Add more custom styles below */
    /*
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    */

    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "admin_settings" not in st.session_state:
    st.session_state.admin_settings = get_admin_settings()

if "welcome_note" not in st.session_state:
    st.session_state.welcome_note = "Question"

if "example_questions" not in st.session_state:
    st.session_state.example_questions = "What are the key features of IndiVillage?\nWhat is the company's mission?"

chat_tab = st.tabs(["Chat"])[0]

with chat_tab:
    st.header("Welcome to IndiVillage")

    # Initialize show_sources flag
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = False

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"] and st.session_state.show_sources:
                with st.expander("View Sources", expanded=False):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source Chunk {i+1}** (Metadata: `{source.get('metadata', {})}`):")
                        st.text_area(f"source_{message.get('timestamp', 'old')}_{i}", source.get('page_content', 'N/A'), height=100, disabled=True, key=f"src_exp_{message.get('timestamp', 'old')}_{i}")
            if "latency" in message and "llm_used" in message:
                st.caption(f"LLM: {message['llm_used']} | Embedding: {message.get('embedding_model_used', 'N/A')} | Time: {message['latency']:.2f}s | Tokens: {message.get('token_usage', 'N/A')}")
                # Display token usage and cost in a table
                col1, col2, col3 = st.columns(3)
                col1.metric("Latency", f"{message['latency']:.2f}s")

                token_usage_display = message.get('token_usage', 'N/A')
                cost_value = message.get('cost')
                cost_display = f"${cost_value:.5f}" if cost_value is not None else "N/A"

                col2.metric("Tokens Used", token_usage_display)
                col3.metric("Cost", cost_display)

            elif "latency" in message:
                st.caption(f"Response time: {message['latency']:.2f}s")
    # LLM Model Selection with Save Button
    admin_settings = st.session_state.admin_settings
    all_available_llms = admin_settings.get('available_openai_llms', []) + \
                         admin_settings.get('available_gemini_llms', [])
    current_selected_llm_from_settings = admin_settings.get('selected_llm_model', '')

    llm_model_index = 0
    if not all_available_llms: # No models available from settings
        all_available_llms = ["No models configured"]
        current_selected_llm_from_settings = all_available_llms[0]
    elif current_selected_llm_from_settings and current_selected_llm_from_settings in all_available_llms:
        llm_model_index = all_available_llms.index(current_selected_llm_from_settings)
    elif all_available_llms: # Default to first if current not found or not set
        current_selected_llm_from_settings = all_available_llms[0]
        llm_model_index = 0


    col1, col2 = st.columns([4, 1]) # Adjust ratio as needed

    with col1:
        selected_llm_for_query = st.selectbox(
            "Select LLM Model:",
            options=all_available_llms,
            index=llm_model_index,
            key="chat_llm_model_selection"
        )

    with col2:
        # This empty div helps align the button vertically if selectbox is taller.
        # A more robust solution might involve custom CSS if perfect alignment is critical.
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Save LLM", key="save_chat_llm_selection", disabled=(selected_llm_for_query == "No models configured")):
            if selected_llm_for_query and selected_llm_for_query != "No models configured":
                # Determine provider
                new_provider = ""
                if selected_llm_for_query in admin_settings.get('available_openai_llms', []):
                    new_provider = "openai"
                elif selected_llm_for_query in admin_settings.get('available_gemini_llms', []):
                    new_provider = "gemini"

                if new_provider:
                    payload = {
                        "selected_llm_provider": new_provider,
                        "selected_llm_model": selected_llm_for_query,
                        "selected_embedding_provider": admin_settings.get('selected_embedding_provider'),
                        "selected_embedding_model": admin_settings.get('selected_embedding_model'),
                        "custom_prompt_template": admin_settings.get('custom_prompt_template')
                    }
                    with st.spinner("Saving LLM choice..."):
                        update_response = save_admin_settings(payload)
                        if update_response:
                            st.success(update_response.get("message", "LLM choice saved successfully!"))
                            st.session_state.admin_settings = get_admin_settings() # Refresh settings
                            st.rerun()
                        else:
                            st.error("Failed to save LLM choice. Check backend logs.")
                else:
                    st.error(f"Could not determine provider for model: {selected_llm_for_query}. Ensure it's listed in admin settings.")
            else:
                st.warning("No valid LLM model selected to save.")
    # Accept user input
    if prompt := st.chat_input(st.session_state.welcome_note):
        user_message_ts = time.time()
        user_message = {"role": "user", "content": prompt, "timestamp": user_message_ts}
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_content = ""
            with st.spinner("Thinking..."):
                query_start_time = time.time()
                backend_response = query_backend(prompt, selected_llm_for_query)
                query_end_time = time.time()

                if backend_response:
                    answer = backend_response.get("answer", "Sorry, I couldn't find an answer.")
                    sources = backend_response.get("source_documents", [])
                    latency = backend_response.get("latency", (query_end_time - query_start_time))
                    llm_used = backend_response.get("llm_used", "N/A")
                    embedding_model_used = backend_response.get("embedding_model_used", "N/A")

                    for chunk in answer.split():
                        full_response_content += chunk + " "
                        message_placeholder.markdown(full_response_content + "▌")
                        time.sleep(0.05)
                    message_placeholder.markdown(full_response_content)
                    
                    st.caption(f"LLM: {llm_used} | Embedding: {embedding_model_used} | Time: {latency:.2f}s")
                    
                    # Display token usage and cost in a table
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Latency", f"{latency:.2f}s")

                    if backend_response.get("token_usage") is not None and backend_response.get("cost") is not None:
                        col2.metric("Tokens Used", backend_response.get("token_usage", "N/A"))
                        col3.metric("Cost", f"${backend_response.get('cost'):.5f}")
                    else:
                        col2.metric("Tokens Used", backend_response.get("token_usage", "N/A"))
                        col3.metric("Cost", "N/A")

                    if st.session_state.show_sources and sources:
                        with st.expander("View Sources", expanded=True):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source Chunk {i+1}** (Metadata: `{source.get('metadata', {})}`):")
                                st.text_area(f"source_curr_{i}", source.get('page_content', 'N/A'), height=100, disabled=True, key=f"src_curr_exp_{i}")
                    
                    assistant_message = {
                        "role": "assistant",
                        "content": full_response_content,
                        "sources": sources if st.session_state.show_sources else [],
                        "latency": latency,
                        "llm_used": llm_used,
                        "embedding_model_used": embedding_model_used,
                        "token_usage": backend_response.get("token_usage"),
                        "cost": backend_response.get("cost"),
                        "timestamp": time.time()
                    }
                else:
                    full_response_content = "Failed to get a response from the backend."
                    message_placeholder.markdown(full_response_content)
                    assistant_message = {
                        "role": "assistant",
                        "content": full_response_content,
                        "latency": query_end_time - query_start_time,
                        "timestamp": time.time()
                    }

                st.session_state.messages.append(assistant_message)

if False:  # hide admin panel
    with st.sidebar:
        st.header("🛠️ Admin Panel")

        if st.session_state.admin_settings:
            settings = st.session_state.admin_settings

            st.subheader("Current Configuration:")
            st.write(f"**LLM:** `{settings.get('selected_llm_provider', 'N/A')} / {settings.get('selected_llm_model', 'N/A')}`")
            st.write(f"**Embedding:** `{settings.get('selected_embedding_provider', 'N/A')} / {settings.get('selected_embedding_model', 'N/A')}`")
            st.write(f"**Pinecone Index:** `{settings.get('pinecone_index_name', 'N/A')}` (Dim: `{settings.get('pinecone_index_dimension', 'N/A')}`)")
            
            st.divider()
            st.subheader("Update Settings:")

            # LLM Provider and Model Selection
            selected_llm_provider = st.selectbox(
                "LLM Provider:",
                options=["openai", "gemini"],
                index=["openai", "gemini"].index(settings.get('selected_llm_provider', 'openai')),
                key="admin_llm_provider"
            )
            
            llm_model_options = []
            if selected_llm_provider == "openai":
                llm_model_options = settings.get('available_openai_llms', [])
            elif selected_llm_provider == "gemini":
                llm_model_options = settings.get('available_gemini_llms', [])
            
            current_llm_model = settings.get('selected_llm_model', '')
            llm_model_index = 0
            if current_llm_model in llm_model_options:
                llm_model_index = llm_model_options.index(current_llm_model)

            selected_llm_model = st.selectbox(
                "LLM Model:",
                options=llm_model_options,
                index=llm_model_index,
                key="admin_llm_model"
            )

            # Embedding Provider and Model Selection
            selected_embedding_provider = st.selectbox(
                "Embedding Provider:",
                options=["openai", "gemini"],
                index=["openai", "gemini"].index(settings.get('selected_embedding_provider', 'openai')),
                key="admin_embedding_provider"
            )

            embedding_model_options = []
            if selected_embedding_provider == "openai":
                embedding_model_options = settings.get('available_openai_embedding_models', [])
            elif selected_embedding_provider == "gemini":
                embedding_model_options = settings.get('available_gemini_embedding_models', [])

            current_embedding_model = settings.get('selected_embedding_model', '')
            embedding_model_index = 0
            if current_embedding_model in embedding_model_options:
                embedding_model_index = embedding_model_options.index(current_embedding_model)
            
            selected_embedding_model = st.selectbox(
                "Embedding Model:",
                options=embedding_model_options,
                index=embedding_model_index,
                key="admin_embedding_model"
            )
            st.caption(f"Note: Pinecone index dimension is {settings.get('pinecone_index_dimension', 'N/A')}. Ensure selected embedding model is compatible.")


            # Custom Prompt Template
            custom_prompt_template = st.text_area(
                "Custom Prompt Template:",
                value=settings.get('custom_prompt_template', "Error loading prompt."),
                height=200,
                key="admin_prompt_template"
            )

            if st.button("Save Admin Settings"):
                payload = {
                    "selected_llm_provider": selected_llm_provider,
                    "selected_llm_model": selected_llm_model,
                    "selected_embedding_provider": selected_embedding_provider,
                    "selected_embedding_model": selected_embedding_model,
                    "custom_prompt_template": custom_prompt_template
                }
                with st.spinner("Saving settings..."):
                    update_response = save_admin_settings(payload)
                    if update_response:
                        st.success(update_response.get("message", "Settings updated successfully!"))
                        st.session_state.admin_settings = get_admin_settings() # Refresh settings
                        st.rerun()
                    else:
                        st.error("Failed to save settings. Check backend logs.")
        else:
            st.warning("Could not load admin settings from backend. Please ensure the backend is running and accessible.")

        st.divider()
        # Document Upload
        st.subheader("📄 Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a .txt, .pdf, or .docx file",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=False
        )

        if uploaded_file is not None:
            # The embedding model used for upload is now determined by the global admin settings
            if st.button(f"Process and Embed '{uploaded_file.name}'"):
                file_bytes = uploaded_file.getvalue()
                with st.spinner(f"Processing {uploaded_file.name}... (using current admin embedding settings)"):
                    upload_response = upload_document_to_backend(file_bytes, uploaded_file.name)
                    if upload_response:
                        st.success(upload_response.get("message", "File processed."))
                    else:
                        st.error("File processing failed.")
        
        st.divider()
        st.subheader("⚙️ API Key Status (Server-side)")
        st.info("API keys are managed via the `.env` file on the server.")
        
        st.divider()
        st.subheader("💬 Chatbot Welcome Note")
        st.text_area("Welcome Note:", key="welcome_note", value=st.session_state.welcome_note)

        st.divider()
        st.subheader("❓ Example Questions")
        st.text_area("Example Questions:", key="example_questions", value=st.session_state.example_questions)

        st.session_state.show_sources = st.toggle("Show Source Document Chunks", value=False)

# --- Footer ---
st.markdown("---")
st.markdown("RAG Chatbot v0.2.0") # Version bump