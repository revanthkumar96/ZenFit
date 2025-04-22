import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline
import gradio as gr
import shutil

# Initialize environment variables from Hugging Face secrets
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Constants
DATA_DIR = "diet"
VECTOR_DB_DIR = "chroma_db"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
CSV_PATH = os.path.join(DATA_DIR, "indian_food.csv")

def initialize_llm():
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=GROQ_API_KEY,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )
    return llm

def create_vector_db():
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    
    documents = []
    
    # Load PDFs if any exist
    if os.path.exists(PDF_DIR) and len(os.listdir(PDF_DIR)) > 0:
        pdf_loader = DirectoryLoader(PDF_DIR, glob='*.pdf', loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
    
    # Load CSV if it exists
    if os.path.exists(CSV_PATH):
        csv_loader = CSVLoader(CSV_PATH)
        documents.extend(csv_loader.load())
    
    if not documents:
        raise ValueError("No documents found to create vector database. Please add PDFs or CSV files.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Use smaller embedding model for Spaces compatibility
    embeddings = HuggingFaceBgeEmbeddings(
        model_name='BAAI/bge-small-en-v1.5',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Clear existing vector DB if it exists
    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR)
    
    vector_db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=VECTOR_DB_DIR
    )
    vector_db.persist()
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are a knowledgeable assistant focused on Indian cuisine, Fitness, and health. Use the provided context to answer accurately.

    Context: {context}
    User: {question}
    Assistant:"""

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

# Initialize food classifier with a smaller model
try:
    food_classifier = pipeline(
        "image-classification", 
        model="./indian_food_finetuned_model",
        device="cpu"
    )
except Exception as e:
    print(f"Could not load food classifier: {e}")
    food_classifier = None

def classify_food(image):
    if image is None or food_classifier is None:
        return None, 0.0
    try:
        results = food_classifier(image)
        top_result = results[0]
        label = top_result['label']
        score = top_result['score']
        non_food_labels = ["non-food", "unknown", "object", "item"]
        if any(non_food in label.lower() for non_food in non_food_labels) or score < 0.3:
            return None, score
        return label, score
    except Exception as e:
        print(f"Error classifying food: {e}")
        return None, 0.0

def calculate_bmi(weight, height):
    try:
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 24.9:
            category = "Normal weight"
        elif 25 <= bmi < 29.9:
            category = "Overweight"
        else:
            category = "Obese"
        return round(bmi, 2), category
    except:
        return 0.0, "Unknown"

def get_diet_plan(bmi_category):
    diet_plans = {
        "Underweight": "Increase calorie intake with protein-rich Indian foods like paneer, lentils, nuts, and ghee. Eat frequent small meals.",
        "Normal weight": "Maintain a balanced diet with dals, whole grains like brown rice, vegetables, and healthy fats like coconut oil.",
        "Overweight": "Focus on portion control, high-fiber foods like millets, lean proteins like chicken tikka, and reduce fried items. Increase activity.",
        "Obese": "Adopt a calorie-deficit diet with sabzi, lean meats, healthy fats like mustard oil, and avoid sweets. Consult a nutritionist."
    }
    return diet_plans.get(bmi_category, "Maintain a healthy balanced Indian diet.")

def generate_meal_suggestions(bmi_category, llm):
    prompt_template = """You are a diet assistant specializing in Indian cuisine. Based on the BMI category "{bmi_category}", suggest a one-day meal plan with breakfast, lunch, and dinner.

    Example Format:
    - *Breakfast*: (Meal suggestion)
    - *Lunch*: (Meal suggestion)
    - *Dinner*: (Meal suggestion)

    Use simple, healthy Indian dishes."""
    prompt = prompt_template.format(bmi_category=bmi_category)
    response = llm.invoke(prompt).content
    return response

def get_fitness_recommendations(goal, activity_level):
    if "weight loss" in goal.lower():
        if "sedentary" in activity_level.lower():
            return "Start with light exercises like walking or yoga for 30 minutes daily. Gradually increase intensity."
        elif "active" in activity_level.lower():
            return "Incorporate cardio exercises like running or cycling, and strength training 3-4 times a week."
    elif "muscle gain" in goal.lower():
        return "Focus on strength training with weights, and include protein-rich foods in your diet."
    elif "general fitness" in goal.lower():
        return "Maintain a balanced routine with cardio, strength, and flexibility exercises."
    return "Please specify a clear fitness goal (e.g., weight loss, muscle gain) for tailored advice."

first_aid_db = {
    "cut": "Clean the wound with water, apply antiseptic, and cover with a bandage. Seek medical help if deep.",
    "burn": "Cool the burn under running water for 10 minutes. Do not apply ice. Cover with a sterile dressing.",
    "sprain": "Rest, ice the area, compress with a bandage, and elevate the limb. Avoid weight-bearing activities.",
    "fever": "Rest, stay hydrated, and take paracetamol if needed. Consult a doctor if fever persists.",
    "headache": "Rest in a quiet, dark room. Stay hydrated and consider over-the-counter pain relief if necessary."
}

def get_first_aid_advice(symptom):
    return first_aid_db.get(symptom.lower(), "Describe your symptom clearly for first aid advice.")

def chat_function(user_input, image, chat_history, state):
    fitness_keywords = ["fitness", "exercise", "workout", "physical activity", "gym", "training", "muscle", "weight loss", "strength"]
    first_aid_keywords = ["first aid", "injury", "wound", "cut", "burn", "sprain", "fever", "headache", "pain", "symptom"]

    if image is not None:
        food_name, confidence = classify_food(image)
        if food_name is None:
            if confidence < 0.3:
                response = "The image does not appear to be a food item or the classification confidence is too low (below 30%). Please upload a clear image of a food item."
            else:
                response = "The image is irrelevant or not recognized as a food item. Please upload an image of a food item."
            user_message = user_input if user_input else "[Image attached]"
            chat_history.append((user_message, response))
        else:
            if user_input:
                prompt = f"""The user shared an image of {food_name}, an Indian dish, and said: '{user_input}'.
                Provide:
                1. Name of the dish: {food_name}
                2. Average calories per serving (estimate based on typical Indian recipes).
                3. Pros (health benefits, nutritional value).
                4. Cons (potential drawbacks).
                Address the user's message with relevant advice."""
            else:
                prompt = f"""The user shared an image of {food_name}, an Indian dish.
                Provide:
                1. Name of the dish: {food_name}
                2. Average calories per serving (estimate based on typical Indian recipes).
                3. Pros (health benefits, nutritional value).
                4. Cons (potential drawbacks)."""
            response = llm.invoke(prompt).content
            user_message = user_input if user_input else "[Image attached]"
            chat_history.append((user_message, response))
    else:
        input_lower = user_input.lower()
        if state["context"] is None:
            if any(keyword in input_lower for keyword in fitness_keywords):
                state["context"] = "fitness"
                response = "Sure! What is your fitness goal? (e.g., weight loss, muscle gain, general fitness)"
            elif any(keyword in input_lower for keyword in first_aid_keywords):
                for symptom in first_aid_db:
                    if symptom in input_lower:
                        response = get_first_aid_advice(symptom)
                        break
                else:
                    response = "Please specify your symptom for first aid advice (e.g., cut, burn)."
            else:
                response_dict = qa_chain({"query": user_input})
                response = response_dict["result"]
            chat_history.append((user_input, response))
        elif state["context"] == "fitness":
            if "goal" not in state["data"]:
                state["data"]["goal"] = user_input
                response = "Got it. What is your current activity level? (e.g., sedentary, active)"
                chat_history.append((user_input, response))
            elif "activity_level" not in state["data"]:
                state["data"]["activity_level"] = user_input
                goal = state["data"]["goal"]
                activity_level = user_input
                response = get_fitness_recommendations(goal, activity_level)
                chat_history.append((user_input, response))
                state["context"] = None
                state["data"] = {}
    return chat_history, state

def calculate_bmi_and_display(name, age, height, weight, gender, chat_history):
    bmi, category = calculate_bmi(weight, height)
    diet_plan = get_diet_plan(category)
    meal_suggestions = generate_meal_suggestions(category, llm)
    message = f"Hello {name}! Based on your details:\n- Age: {age}\n- Height: {height} cm\n- Weight: {weight} kg\n- Gender: {gender}\n- BMI: {bmi} ({category})\n\n**Recommended Diet Plan**: {diet_plan}\n\n**Suggested Meals**:\n{meal_suggestions}"
    chat_history.append(("BMI Calculation", message))
    return chat_history

# Initialize components
try:
    llm = initialize_llm()
    vector_db = create_vector_db()
    qa_chain = setup_qa_chain(vector_db, llm)
except Exception as e:
    print(f"Initialization error: {e}")
    # Fallback to LLM-only mode if vector DB fails
    llm = initialize_llm()
    qa_chain = None

# Gradio interface
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
    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
}
.chatbot .bubble:nth-child(odd) {
    background: #2a6a7f; /* Muted teal for bot messages */
    color: #e0e7ff;
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
    background: radial-gradient(ellipse at top left, #111827, #155e75, #0e7490);
    color: #e0e7ff !important;
}
button.primary:hover {
    background: radial-gradient(ellipse at top left, #155e75, #0e7490);
    transform: scale(1.05);
}
button.secondary {
    background:radial-gradient(ellipse at top left, #155e75, #0e7490);
    color: #e0e7ff !important;
}
button.secondary:hover {
    background:#FF0000;
    transform: scale(1.05);
}

/* Textbox and inputs */
textarea, input[type="text"], input[type="number"] {
    border-radius: 10px !important;
    border: 1px solid #4b5563 !important;
    padding: 10px !important;
    background: #1f2937 !important;
    color: #e0e7ff !important;
}

/* Image upload */
.gr-file-upload, .gr-image {
    border-radius: 10px !important;
    border: 1px solid #4b5563 !important;
    background: #1f2937 !important;
}

/* Radio buttons */
input[type="radio"] + label {
    color: #e0e7ff !important;
}
"""

with gr.Blocks(theme=gr.themes.Glass(primary_hue="teal", secondary_hue="sky", neutral_hue="slate"), css=custom_css) as demo:
    gr.Markdown(
        """
        # 🌟 NutraFit: Your Health Companion 🌟
        Welcome to NutraFit! I'm here to help with Indian cuisine tips, fitness advice, BMI calculations, and first aid guidance. Upload a food image, ask a question, or calculate your BMI to get started!
        """
    )
    chatbot = gr.Chatbot(
        height=400,
        bubble_full_width=False,
        show_label=False,
        placeholder="Know about your food, diet and workout"
    )
    conversation_state = gr.State({"context": None, "data": {}})

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Chat")
            msg = gr.Textbox(label="Your Message", placeholder="Ask about food, fitness, or first aid...")
            img = gr.Image(label="Upload Food Image", type="pil")
            send_btn = gr.Button("Send 📤", variant="primary")
        with gr.Column():
            gr.Markdown("## BMI Calculator")
            name = gr.Textbox(label="Name", placeholder="Enter your name")
            age = gr.Number(label="Age")
            height = gr.Number(label="Height (cm)")
            weight = gr.Number(label="Weight (kg)")
            gender = gr.Radio(["Male", "Female", "Other"], label="Gender")
            calc_btn = gr.Button("Calculate BMI 📊", variant="primary")

    send_btn.click(
        chat_function,
        inputs=[msg, img, chatbot, conversation_state],
        outputs=[chatbot, conversation_state]
    )
    calc_btn.click(
        calculate_bmi_and_display,
        inputs=[name, age, height, weight, gender, chatbot],
        outputs=chatbot
    )
    clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
    clear_btn.click(
        lambda: ([], {"context": None, "data": {}}),
        None,
        [chatbot, conversation_state]
    )

# For Hugging Face Spaces deployment
demo.launch()