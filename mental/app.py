import os
import random
import tempfile
from gtts import gTTS
from transformers import pipeline
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr

# Initialize components
def initialize_components():
    # Speech-to-text pipeline
    stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    
    # LLM with Groq
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )
    
    # Create or load vector database
    if not os.path.exists("./chroma_db"):
        vector_db = create_vector_db()
    else:
        embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma(
            persist_directory='./chroma_db',
            embedding_function=embeddings
        )
    
    return stt_pipe, llm, vector_db

def create_vector_db():
    # Load PDF documents
    loader = DirectoryLoader("TrainingDoc", glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()

    prompt_template = """
You are a friendly and empathetic **mental health companion** who provides warm and supportive conversations.
You create a **safe and welcoming space** for users to express themselves, where their feelings are respected and understood.
You offer guidance that is caring and thoughtful, helping users feel heard and supported. Your responses are simple, clear, and focus on offering emotional support and encouragement.
Always maintain a calm, friendly tone that promotes well-being and encourages self-care without using formal research references or citations.

## Tone & Style:
- Warm, non-judgmental, and professional.
- Responses are **short and concise** (2-3 lines).
- Encouraging, uplifting, and supportive.
- Uses a mix of **empathy, storytelling, humor (when appropriate), and life insights**.
- The response should be comforting and sympathetic to the user's emotions.
- The response should offer support and validation for the user's emotions.
- The response should be tailored to the user's specific situation.

Context: {context}
User Question: {question}
Your Response: A professional, concise, empathetic answer with research-backed insights."""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

# Audio processing functions
def audio_to_text(audio_path, stt_pipe):
    if audio_path is None:
        return None
    try:
        text = stt_pipe(audio_path)["text"]
        return text
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return None

def text_to_audio(text):
    if not text.strip():
        return None
    try:
        tts = gTTS(text=text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None

# Chat processing
def process_user_input(user_input, chat_history, qa_chain):
    if not user_input.strip():
        return chat_history, None

    response = qa_chain({"query": user_input})
    answer = response["result"]
    
    # Add crisis support if needed
    crisis_keywords = [
        'suicide', 'self-harm', 'kill myself', 'cutting', 'hurt myself', 'end my life',
        'no reason to live', 'want to die', 'jump off', 'overdose', 'slit my wrists',
        'hang myself', 'drown myself', 'hopeless', 'worthless', 'useless', 'no one cares',
        'empty', 'tired of life', 'can\'t go on', 'nothing matters', 'lost all hope',
        'abused', 'molested', 'assaulted', 'raped', 'harassed', 'beaten', 'domestic violence',
        'forced', 'threatened', 'bullied', 'panic attack', 'can\'t breathe', 'chest pain',
        'heart racing', 'shaking', 'dizzy', 'faint', 'feel like dying'
    ]
    
    if any(keyword in user_input.lower() for keyword in crisis_keywords):
        crisis_resources = """\n\n💙 *You are valued and important.* Please reach out for help:
        \n- 🇮🇳 India: Vandrevala Foundation - 1860 266 2345
        \n- 🇺🇸 USA: National Suicide Prevention Lifeline - 988
        \n- 🇬🇧 UK: Samaritans - 116 123
        \n- 🌍 International: https://findahelpline.com/
        \n\nIf in immediate danger, please call emergency services."""
        answer += crisis_resources

    # Generate varied responses
    response_variations = [
        f"💬 {answer}",
        f"🤔 {answer} What are your thoughts?",
        f"🌟 {answer}",
        f"🌿 {answer} How does that resonate with you?"
    ]
    final_answer = random.choice(response_variations)
    
    audio_path = text_to_audio(final_answer)
    chat_history.append((user_input, final_answer))
    return chat_history, audio_path

# Initialize components
stt_pipe, llm, vector_db = initialize_components()
qa_chain = setup_qa_chain(vector_db, llm)

# Gradio interface
def gradio_chat(user_input, chat_history):
    chat_history, audio_path = process_user_input(user_input, chat_history, qa_chain)
    return "", chat_history, audio_path

def voice_chat(audio_path, chat_history):
    if audio_path is None:
        return chat_history, None

    user_input = audio_to_text(audio_path, stt_pipe)
    if not user_input or not user_input.strip():
        return chat_history + [("", "Could not understand the audio. Please try again.")], None

    chat_history, audio_path = process_user_input(user_input, chat_history, qa_chain)
    return chat_history, audio_path

# Custom CSS
custom_css = """
.gradio-container {
    background: linear-gradient(to right, #f5f7fa, #c3cfe2);
    font-family: 'Arial', sans-serif;
}
.chatbot {
    min-height: 400px;
}
.chatbot .message {
    padding: 12px;
    border-radius: 15px;
    margin: 5px 0;
    max-width: 80%;
}
.chatbot .user-message {
    background: #4b5563;
    color: white;
    margin-left: auto;
}
.chatbot .bot-message {
    background: #155e75;
    color: white;
    margin-right: auto;
}
button {
    border-radius: 10px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
}
button.primary {
    background: #155e75 !important;
    color: white !important;
}
button.secondary {
    background: #4b5563 !important;
    color: white !important;
}
textarea {
    border-radius: 10px !important;
    padding: 12px !important;
}
"""

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""
    # 🌿 CalmMe: Your Mental Health Companion
    A safe space to share your thoughts and feelings. How are you doing today?
    """)
    
    with gr.Row():
        chatbot = gr.Chatbot(
            elem_classes="chatbot",
            bubble_full_width=False,
            show_label=False
        )
        audio_output = gr.Audio(visible=False, autoplay=True)
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message here...",
            container=False,
            scale=7
        )
        text_submit = gr.Button("Send", variant="primary", scale=1)
    
    with gr.Row():
        voice_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Or record a voice message"
        )
        voice_submit = gr.Button("Send Voice", variant="primary")
        clear = gr.Button("Clear Chat", variant="secondary")
    
    # Event handlers
    text_submit.click(
        gradio_chat,
        [msg, chatbot],
        [msg, chatbot, audio_output]
    )
    
    voice_submit.click(
        voice_chat,
        [voice_input, chatbot],
        [chatbot, audio_output]
    )
    
    msg.submit(
        gradio_chat,
        [msg, chatbot],
        [msg, chatbot, audio_output]
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)


demo.launch(debug = False, share = True)