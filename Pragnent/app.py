# app.py
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import gradio as gr
import os

# Initialize Groq with environment variable
llm = ChatGroq(
    temperature=0.7,
    groq_api_key=os.environ.get("Groq_API_Key"),  # Set in HF Secrets
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

# Configure paths for Hugging Face Space
VECTOR_DB_PATH = "./chroma_db"
PDF_DIR = "./Pregnancy"

# Initialize or load vector database
if not os.path.exists(VECTOR_DB_PATH):
    # Create new vector database
    loader = DirectoryLoader(PDF_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=VECTOR_DB_PATH)
else:
    # Load existing database
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

retriever = vector_db.as_retriever()

# Load food classification model
food_classifier = pipeline(
    "image-classification", 
    model="./models/indian_food_finetuned_model",
    device_map="auto"
)

def classify_food(image):
    """Classify food images with confidence thresholding"""
    if image is None:
        return None, 0.0
    results = food_classifier(image)
    if not results:
        return None, 0.0
    top_result = results[0]
    label = top_result["label"]
    score = top_result["score"]
    if score < 0.3 or "non-food" in label.lower():
        return None, score
    return label, score

def format_history(chat_history, max_exchanges=5):
    """Format conversation history for context"""
    recent_history = chat_history[-max_exchanges:]
    return "\n".join(
        f"User: {user}\nAssistant: {assistant}" 
        for user, assistant in recent_history
    )

def calculate_metrics(status, pre_weight, current_weight, height, 
                     gest_age=None, time_since_delivery=None, breastfeeding=None):
    """Calculate pregnancy/postpartum metrics"""
    if None in [pre_weight, current_weight, height]:
        return "Missing required fields: weight and height"
    
    height_m = height / 100
    pre_bmi = pre_weight / (height_m ** 2)
    
    if status == "Pregnant":
        if not gest_age or not (0 <= gest_age <= 40):
            return "Invalid gestational age (0-40 weeks)"
        
        # BMI-based recommendations
        bmi_ranges = [
            (18.5, 12.5, 18),
            (25, 11.5, 16),
            (30, 7, 11.5),
            (float('inf'), 5, 9)
        ]
        for max_bmi, min_gain, max_gain in bmi_ranges:
            if pre_bmi < max_bmi:
                break
                
        current_gain = current_weight - pre_weight
        expected_min = (min_gain / 40) * gest_age
        expected_max = (max_gain / 40) * gest_age
        
        if current_gain < expected_min:
            advice = "Consider nutritional counseling"
        elif current_gain > expected_max:
            advice = "Consult your healthcare provider"
        else:
            advice = "Good progress! Maintain balanced diet"
            
        return (f"Pre-BMI: {pre_bmi:.1f}\nWeek {gest_age} recommendation: "
                f"{expected_min:.1f}-{expected_max:.1f} kg\n"
                f"Your gain: {current_gain:.1f} kg\n{advice}")
    
    elif status == "Postpartum":
        if None in [time_since_delivery, breastfeeding]:
            return "Missing postpartum details"
        
        current_bmi = current_weight / (height_m ** 2)
        if breastfeeding == "Yes":
            advice = ("Aim for 0.5-1 kg/month loss while breastfeeding\n"
                     "Focus on nutrient-dense foods")
        else:
            advice = "Gradual weight loss through diet and exercise"
            
        return (f"Current BMI: {current_bmi:.1f}\n"
               f"{time_since_delivery} weeks postpartum\n{advice}")
    
    return "Select pregnancy status"

def chat_function(user_input, image, chat_history):
    """Generate responses based on user input and chat history."""
    history_str = format_history(chat_history)
    crisis_keywords = [
        "suicide", "self-harm", "kill myself", "cutting", "hurt myself", "end my life",
        "hopeless", "worthless", "can’t go on", "panic attack", "feel like dying"
    ]
    newborn_keywords = ["newborn", "baby", "infant", "feeding", "sleep", "colic"]

    if image:
        food_name, confidence = classify_food(image)
        if food_name:
            if user_input:
                prompt = f"""
Previous conversation:
{history_str}
The user uploaded an image of {food_name} and asked: '{user_input}'.
Provide a response tailored to pregnancy or postpartum needs.
"""
            else:
                prompt = f"""
Previous conversation:
{history_str}
The user uploaded an image of {food_name}.
Provide pregnancy-specific nutritional advice.
"""
            response = llm.invoke(prompt).content
        else:
            response = "I couldn’t identify a food item in the image. Please upload a clearer picture."
    else:
        if not user_input.strip():
            response = "Please type a message or upload an image."
        elif any(keyword in user_input.lower() for keyword in crisis_keywords):
            response = """
I'm really sorry you're feeling this way. You’re not alone, and help is available.
Please reach out to someone you trust or contact a helpline:
- 🇮🇳 India: Vandrevala Foundation - 1860 266 2345
- 🇺🇸 USA: National Suicide Prevention Lifeline - 988
- 🇬🇧 UK: Samaritans - 116 123
- 🌍 International: https://findahelpline.com/
If you’re in immediate danger, call emergency services (911/112).
"""
        elif any(keyword in user_input.lower() for keyword in newborn_keywords):
            prompt = f"""
Previous conversation:
{history_str}
The user asked: '{user_input}'.
Provide basic guidance on newborn care.
"""
            response = llm.invoke(prompt).content
        else:
            docs = retriever.get_relevant_documents(user_input)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"""
Previous conversation:
{history_str}
Context: {context}
Current question: {user_input}
Assistant:
"""
            response = llm.invoke(prompt).content

    chat_history.append((user_input or "[Image Uploaded]", response))
    return chat_history

# Custom CSS with specified colors
custom_css = """
/* General layout */
.gradio-container {
    background: radial-gradient(ellipse at top left, #111827, #155e75, #0e7490);
    font-family: 'Arial', sans-serif;
}

/* Chatbot bubble styling */
.chatbot .bubble {
    border-radius: 15px;
    padding: 10px 15px;
    margin: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.3); /* Darker shadow for depth */
}
.chatbot .bubble:nth-child(odd) {
    background: #2a6a7f; /* Muted teal for bot messages */
    color: #e0e7ff; /* Light text for readability */
}
.chatbot .bubble:nth-child(even) {
    background: #374151; /* Dark gray for user messages */
    color: #e0e7ff;
}

/* Buttons */
button {
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
}
button.primary {
    background: #155e75 !important; /* Deep teal for primary actions */
    color: #e0e7ff !important;
}
button.primary:hover {
    background: #0e7490 !important;
    transform: scale(1.05);
}
button.secondary {
    background: #4b5563 !important; /* Muted gray for secondary */
    color: #e0e7ff !important;
}
button.secondary:hover {
    background: #991b1b !important; /* Fixed hover color */
    transform: scale(1.05);
}

/* Textbox */
textarea {
    border-radius: 10px !important;
    border: 1px solid #4b5563 !important; /* Gray border */
    padding: 10px !important;
    background: #1f2937 !important; /* Dark background for input */
    color: #e0e7ff !important;
}
"""

# Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🌸FirstSteps-Maternal Wellness Companion 🌸")
    gr.Markdown("""Welcome! I'm here to support you through pregnancy and postpartum with advice on mental health, nutrition, fitness, and newborn care. Ask me anything or upload a food image!""")

    chatbot = gr.Chatbot(
        height=600,
        label="Conversation",
        value=[[None, "Welcome! I'm here to support you through pregnancy and postpartum. Ask me anything or upload a food image for nutritional advice."]]
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Chat with Me")
            msg = gr.Textbox(label="Your Message", placeholder="Type your question here...")
            img = gr.Image(label="Upload Food Image", type="pil")
            send_btn = gr.Button("Send")

        with gr.Column(scale=1):
            gr.Markdown("## Pregnancy Metrics")
            status = gr.Radio(["Pregnant", "Postpartum"], label="Your Status")
            pre_weight = gr.Number(label="Pre-pregnancy Weight (kg)")
            current_weight = gr.Number(label="Current Weight (kg)")
            height = gr.Number(label="Height (cm)")
            gest_age = gr.Number(label="Gestational Age (weeks)", visible=False)
            time_since_delivery = gr.Number(label="Time Since Delivery (weeks)", visible=False)
            breastfeeding = gr.Radio(["Yes", "No"], label="Breastfeeding?", visible=False)
            calc_btn = gr.Button("Calculate Metrics")

    with gr.Row():
        clear_btn = gr.Button("Clear Chat")

    def update_visibility(status):
        if status == "Pregnant":
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        elif status == "Postpartum":
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    status.change(update_visibility, inputs=status, outputs=[gest_age, time_since_delivery, breastfeeding])

    def handle_send(msg, img, chat_history):
        chat_history = chat_function(msg, img, chat_history)
        return "", None, chat_history

    send_btn.click(handle_send, inputs=[msg, img, chatbot], outputs=[msg, img, chatbot])

    def handle_calc(status, pre_weight, current_weight, height, gest_age, time_since_delivery, breastfeeding, chat_history):
        metrics_response = calculate_metrics(status, pre_weight, current_weight, height, gest_age, time_since_delivery, breastfeeding)
        chat_history.append(("Pregnancy Metrics Calculation", metrics_response))
        return chat_history

    calc_btn.click(handle_calc,
                   inputs=[status, pre_weight, current_weight, height, gest_age, time_since_delivery, breastfeeding, chatbot],
                   outputs=chatbot)

    clear_btn.click(lambda: [], outputs=chatbot)

    gr.HTML('<div class="disclaimer">**Disclaimer**: This app offers general guidance and is not a substitute for professional medical advice. Consult your healthcare provider for personalized recommendations.</div>')

demo.launch(debug=False, share  = True)