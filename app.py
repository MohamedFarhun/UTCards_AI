import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import datetime
import re
from prompts import generate_fraud_prompt,generate_chatbot_prompt
import openai
from PIL import Image
import pytesseract
from sklearn.ensemble import RandomForestClassifier
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import cv2


# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app_ui_enhancements():
    # Custom CSS for titles, headers, and model detail boxes
    st.markdown("""
    <style>
    /* Titles and Headers - Adjusted font size and added color */
    .stApp .stMarkdown h1 {
        font-size: 1.75rem; /* Reduced font size */
        color: var(--primary);
        animation: fadeIn 1s ease-in-out;
    }

    .stApp .stMarkdown h2, .stApp .stMarkdown h3 {
        font-size: 1.5rem; /* Reduced font size */
        text-align:center;
        color: var(--primary);
        animation: fadeIn 1s ease-in-out;
    }

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Model Detail Boxes */
    .model-detail-box {
        padding: 10px;
        border-radius: 15px;
        border: 2px solid var(--secondary);
        background-color: var(--background-secondary);
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        animation: fadeIn 1s ease-in-out;
    }

    .model-detail-box:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }

    /* Responsive */
    @media (max-width: 640px) {
        .stApp .stMarkdown h1, .stApp .stMarkdown h2, .stApp .stMarkdown h3 {
            font-size: calc(1.375rem + 1.5vw);
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Set page configuration
st.set_page_config(
    page_title="UTCards",  # Page title displayed in the browser's title bar
    page_icon="🎴",         # A relevant emoji or image icon URL
    layout="wide",         # Use the full screen width
    initial_sidebar_state="expanded",  # If using a sidebar, set to 'expanded' or 'collapsed'
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/mohamed-farhun-m-098b68227/',  # Custom help URL
        'Report a bug': "https://github.com/MohamedFarhun/UTCards_AI/issues",  # Custom bug report URL
        'About': "# This is UTCards, your personal card assistant!"  # Custom about text
    }
)

def load_data():
    data = pd.read_csv('files/extended_synthetic_credit_card_transactions.csv')
    # Convert TransactionDate to datetime
    data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
    return data

def extract_card_features(df):
    # Extract the first digit (MII)
    df['MII'] = df['CardNumber'].apply(lambda x: int(str(x)[0]))

    # Map MII to industry
    mii_to_industry = {
        1: 'Airlines', 2: 'Airlines_and_Financial', 3: 'Travel_and_Entertainment',
        4: 'Banking_and_Financial', 5: 'Banking_and_Financial', 6: 'Merchandising_and_Banking',
        7: 'Petroleum', 8: 'Healthcare_and_Communications', 9: 'Government'
    }
    df['Industry'] = df['MII'].map(mii_to_industry)

    # Determine card length to differentiate between Amex and others
    df['CardLength'] = df['CardNumber'].apply(lambda x: len(str(x)))

    # One-hot encoding for categorical variables 'Industry' and 'CardLength'
    df = pd.get_dummies(df, columns=['Industry', 'CardLength'])

    return df

def validate_card(row):
    card_num = str(row['CardNumber'])
    valid_length = len(card_num) in [15, 16]
    
    if not valid_length:
        return False  # Invalid length

    valid_start = False
    if len(card_num) == 15 and card_num.startswith('3'):
        valid_start = True  # Valid Amex
    elif len(card_num) == 16:
        if card_num.startswith('4'):
            valid_start = int(card_num[1:6]) in range(200000, 700000)  # Valid Visa
        elif card_num.startswith('5'):
            valid_start = int(card_num[1:6]) in range(200000, 700000)  # Valid Mastercard
        elif card_num.startswith('6'):
            valid_start = int(card_num[1:6]) in range(200000, 700000)  # Valid Discover

    return valid_start

def feature_engineering(df):
    # Interaction Features
    df['Amount_Hour_Interaction'] = df['Amount'] * df['Hour']
    
    # Aggregated Features
    # Use 'CardNumber' as a unique identifier if 'CardID' is not available
    df['Average_Amount_Per_Card'] = df.groupby('CardNumber')['Amount'].transform('mean')
    
    return df

def train_model(X_train, y_train):
    # Training a RandomForestClassifier without hyperparameter tuning
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Analysis Page
def analysis_page(model=None):
    st.title('💱 Transaction Analysis')

    # Load data
    data = load_data()

    # Create the 'Hour' column based on 'TransactionDate'
    data['Hour'] = data['TransactionDate'].dt.hour

    # Extract features from CardNumber
    data = extract_card_features(data)

    # Validate card numbers based on the given conditions
    data['ValidCard'] = data.apply(validate_card, axis=1)

    # Perform feature engineering including interaction features
    data = feature_engineering(data)

    # Transaction Amount Analysis
    st.header('Transaction Amount Analysis')
    fig = px.histogram(data, x='Amount', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # Category-wise Analysis
    st.header('Category-wise Analysis')
    fig = px.bar(data['Category'].value_counts(), template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # Temporal Analysis
    st.header('Temporal Analysis')
    transactions_by_month = data.groupby('Hour').size()
    fig = px.line(transactions_by_month, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # Predictive Modeling with XGBoost after Hyperparameter Tuning
    features = ['Amount', 'Hour', 'ValidCard', 'Amount_Hour_Interaction', 'Average_Amount_Per_Card'] + \
               list(data.columns[data.columns.str.startswith('Industry_') | data.columns.str.startswith('CardLength_')])
    X = data[features]
    y = (data['Status'] == 'Completed').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)  # Using the RandomForest model

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Display model details and accuracy
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='model-detail-box' style='border-color: #ff4b4b;'><h4 style='color: #ff4b4b;'>Model Used</h4><h3 style='color:#ccc90e'>Random Forest Classifier</h3></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='model-detail-box' style='border-color: #ff4b4b;'><h4 style='color: #ff4b4b;'>Accuracy</h4><h3 style='color:#ccc90e'>{accuracy:.2%}</h3></div>", unsafe_allow_html=True)
    
    # Feature Engineering for Fraud Detection
    # Create features that might be indicative of fraud
    # Predictive Modeling
    st.header('Anomalies Detected')
    data['Hour'] = data['TransactionDate'].dt.hour
    data['DayOfWeek'] = data['TransactionDate'].dt.dayofweek

    # Convert 'ExpirationDate' to datetime format
    data['ExpirationDate'] = pd.to_datetime(data['ExpirationDate'], format='%m/%y', errors='coerce')

    # Extract year and month from 'ExpirationDate'
    data['ExpirationYear'] = data['ExpirationDate'].dt.year
    data['ExpirationMonth'] = data['ExpirationDate'].dt.month

    # Assuming 'Amount' and 'Hour' could be important features for fraud detection
    features = data[['Amount', 'Hour','CardNumber','ExpirationYear', 'ExpirationMonth']]
    
    # Data Scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Anomaly Detection Model
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)  # contamination is an estimate of the % of anomalies
    model.fit(features_scaled)
    data['anomaly'] = model.predict(features_scaled)
    data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})  # Convert to 0 for normal, 1 for anomaly

    # Assuming 'model' is your fitted IsolationForest model
    anomaly_scores = model.decision_function(features_scaled)

    # Normalize scores to a positive scale for visualization
    data['anomaly_score'] = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())

    # Visualize the results
    fig = px.scatter(data, x='TransactionDate', y='Amount', color='anomaly',template='plotly_dark',
                     size='anomaly_score',  # Add size dimension based on anomaly score
                     color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)

    # Display anomaly detection results
    anomalies = data[data['anomaly'] == 1]
    # Display model and accuracy in styled boxes
    anomaly_percentage = (len(anomalies) / len(data)) * 100  # Percentage of transactions identified as anomalies
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='model-detail-box' style='border-color: #ff4b4b;'><h4 style='color: #ff4b4b;'>Detected Anomalies</h4><h3 style='color:#ccc90e'>{len(anomalies)}</h3></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='model-detail-box' style='border-color: #ff4b4b;'><h4 style='color: #ff4b4b;'>Anomalies Percentage</h4><h3 style='color:#ccc90e'>{anomaly_percentage:.2f}%</h3></div>", unsafe_allow_html=True)

    # After displaying anomaly detection results
    st.header('Check Your Credit Card for Potential Fraud')

    with st.form("credit_card_form"):
        card_number = st.text_input("Card Number",placeholder="Enter a valid card number (16-19 digits)")
        card_holder = st.text_input("Card Holder Name")
        # Assuming you want to allow expiration years within a certain range
        current_year = datetime.datetime.now().year
        year_range = list(range(current_year, current_year + 20))
        expiration_year = st.selectbox("Expiration Year", options=year_range)
        # For months, since they are always 1-12, you can directly list them
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        expiration_month = st.selectbox("Expiration Month", options=months)
        cvv = st.text_input("CVV",placeholder="Enter a valid CVV (3-4 digits)")
        amount = st.number_input("Transaction Amount")
        submit_button = st.form_submit_button("Check for Fraud")
    
    # Validation for Card Number
    card_number_valid = re.match(r"^\d{16,19}$", card_number) is not None
    if not card_number_valid and card_number:
        st.error("Enter a valid card number (16-19 digits).")

    # Validation for CVV
    cvv_valid = re.match(r"^\d{3,4}$", cvv) is not None
    if not cvv_valid and cvv:
        st.error("Enter a valid CVV (3-4 digits).")

    if submit_button and card_number_valid and cvv_valid:
        # For simplicity, assuming the current hour represents the transaction hour
        current_hour = datetime.datetime.now().hour

        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'CardNumber': [card_number],
            'ExpirationYear': [expiration_year],
            'ExpirationMonth': [expiration_month],
            'Amount': [amount],
            'Hour': [current_hour]  # Adjust this part based on your model's requirements
        })

        # Data Scaling
        input_data_scaled = scaler.transform(input_data[['Amount', 'Hour', 'CardNumber', 'ExpirationYear', 'ExpirationMonth']])

        # Predict fraud and get anomaly score
        is_fraudulent = model.predict(input_data_scaled)[0]  # Adjust based on your model
        anomaly_score = model.decision_function(input_data_scaled)[0]  # Get the anomaly score for the transaction

        openai.api_key = st.secrets["API_KEY"]

        # Check if the anomaly score exceeds the threshold
        is_fraudulent = anomaly_score >= 0.11

        # Generate prompt with anomaly score
        prompt = generate_fraud_prompt(card_number, amount, current_hour, anomaly_score)
        response = openai.Completion.create(engine="gpt-3.5-turbo-instruct", prompt=prompt, max_tokens=500)
        explanation = response.choices[0].text.strip()

        # Display the response in a styled box
        response_color = "#ff4b4b" if is_fraudulent else "#54bf22"  # Red for fraud, green for no fraud
        st.markdown(f"<div style='padding: 10px; border-radius: 10px; border: 2px solid {response_color}; color: {response_color}; margin-bottom: 10px;'>{explanation}</div>", unsafe_allow_html=True)

        # Optionally, display the anomaly score for additional context
        st.write(f"Anomaly Score: {anomaly_score:.2f}")

    # Optionally, you can provide more context or action items based on the fraud status
    if is_fraudulent:
        # If anomaly score indicates potential fraud
        st.markdown("### ⚠️ Potential Fraud Detected")
        st.markdown("Action may be required. Please review the transaction carefully.")
    else:
        st.markdown("### ✅ Transaction Verified")

# Initialize the EasyOCR Reader
reader = easyocr.Reader(['en'])

def preprocess_image_for_ocr(image_path):
    image = Image.open(image_path)
    # Convert to grayscale
    image = image.convert('L')
    # Enhance contrast and sharpness
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    sharpener = ImageEnhance.Sharpness(image)
    image = sharpener.enhance(2)
    return np.array(image)

def ocr_credit_card(image_path):
    # Function to extract credit card information using OCR with EasyOCR
    image = preprocess_image_for_ocr(image_path)
    
    # Adjust reader.readtext parameters to improve OCR results
    results = reader.readtext(image, contrast_ths=0.05, adjust_contrast=0.7, add_margin=0.1, width_ths=0.7, decoder='beamsearch')
    
    # Initialize variables to store extracted information
    card_number = "Not found"
    expiry_date = "Not found"
    card_holder = "Not found"
    card_type = "Unknown"

    # Regular expressions to find credit card information
    card_number_pattern = r'(\d{4}[-\s]?){3}\d{4}'
    expiry_date_pattern = r'\d{2}/\d{2}'
    card_holder_pattern = r'[A-Z]{2,}(?: [A-Z]{2,})+'
    
    # Process OCR results
    for (bbox, text, prob) in results:
        if re.search(card_number_pattern, text):
            card_number = text
            if text.startswith('4'):
                card_type = "Visa"
            elif text.startswith('5'):
                card_type = "MasterCard"
            elif text.startswith('34') or text.startswith('37'):
                card_type = "American Express"
        elif re.search(expiry_date_pattern, text):
            expiry_date = text
        elif re.search(card_holder_pattern, text):
            card_holder = text

    return card_number, expiry_date, card_holder, card_type

def chatbot():
    st.title('👨‍💼 Personalised UTCards Assistant')

    st.subheader("Extract your credit card details")

    # Initialize or retrieve previous chat history from the session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

        
    # Upload credit card image
    uploaded_file = st.file_uploader("Upload your credit card image", type=["jpg", "png", "jpeg"])
        
    # When file is uploaded, process it with OCR
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Credit Card', use_column_width=True)

        # Save the uploaded image to the file system
        with open("files/credit_card.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Call the OCR function
        card_number, expiry_date, card_holder, card_type = ocr_credit_card("files/credit_card.jpg")
        
        # Display the extracted information in a styled and animated box
        st.markdown(f"""
        <div style="padding: 10px; margin-top: 10px; border-radius: 10px;
            background: var(--card-background-color);
            box-shadow: 0 2px 12px rgba(0,0,0,0.2);
            animation: fadeIn 1s ease-out;">
            <h3 style="color: var(--font-color);">Extracted Credit Card Details:</h3>
            <p style="color: var(--font-color);"><b>Card Number:</b> {card_number}</p>
            <p style="color: var(--font-color);"><b>Expiry Date:</b> {expiry_date}</p>
            <p style="color: var(--font-color);"><b>Card Holder:</b> {card_holder}</p>
            <p style="color: var(--font-color);"><b>Card Type:</b> {card_type}</p>
        </div>
        """, unsafe_allow_html=True)

        # Chat input for user queries
        user_query = st.chat_input("Ask me anything about credit card transactions or financial advice")

        # Process the user query
        if user_query:
            st.subheader('UTCards chatbot 🤖')
            # Append user query to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            # Generate a prompt for the OpenAI API based on the user query and previous chat history
            prompt = generate_chatbot_prompt(st.session_state.chat_history)

            # Retrieve the OpenAI API key from Streamlit's secrets
            openai.api_key = st.secrets["API_KEY"]

            try:
                # Send the prompt to the OpenAI API
                response = openai.Completion.create(
                    engine="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    max_tokens=500,
                    stop=None
                )

                # Extract the AI's response
                ai_response = response.choices[0].text.strip()

                # Append the AI's response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

            except Exception as e:
                # Handle exceptions
                st.error(f"An error occurred: {str(e)}")

        # Display the chat history using Streamlit's chat_message component
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message(name="user"):
                    st.write(message["content"])
            else:  # assistant's message
                with st.chat_message(name="assistant"):
                    st.write(message["content"])


def about():
    st.title('🔒 About UTCards AI')

    # Custom CSS for styling compatible with both light and dark themes
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
    <style>
    .highlight { color: #2E86C1; font-weight: bold; }
    .about-text, .contact-info {
        padding: 20px;
        border-radius: 10px;
        background-color: var(--secondary-background-color);
        margin-bottom: 20px;
        color: var(--text-color);
        font-size: 20px;
        transition: transform 0.3s;
        transform: perspective(1000px) rotateY(0deg);
    }
    .about-text:hover, .contact-info:hover {
        transform: perspective(1000px) rotateY(20deg);
    }
    </style>
    """, unsafe_allow_html=True)

    # Introduction and vision
    st.markdown("""
    <div class="about-text">
    Welcome to <span class="highlight">UTCards AI</span>, an innovative platform at the forefront of reshaping the financial landscape with our <span class="highlight">Instant Card Issuance (ICI)</span> solution. In a world where speed and security are paramount, UTCards AI harnesses the unparalleled potential of <span class="highlight">blockchain technology</span> and sophisticated <span class="highlight">AI-driven security</span> to deliver a seamless, secure, and user-friendly experience for instant virtual card issuance.
    At UTCards AI, we recognize the growing need for agile <span class="highlight">financial solutions </span>that keep pace with the dynamic demands of consumers and businesses alike. Our platform is designed not just to meet these demands but to anticipate and evolve with them, ensuring that every transaction is not only swift but fortified with the highest standards of <span class="highlight">security and privacy.</span>
    Our vision extends beyond the immediate convenience of instant card issuance. We are committed to pioneering a future where financial transactions are not just transactions but secure, intelligent interactions that empower users with <span class="highlight">control, flexibility, and peace of mind</span>. Join us on this journey as we redefine what's possible in the financial ecosystem, one transaction at a time.
    </div>
    """, unsafe_allow_html=True)
        
    # Contact information
    st.markdown("""
    <div class="contact-info">
        <p>For more information or inquiries, reach out to us:</p>
        <p>Email: <a href="mailto:contact@utcards.ai" style="color: var(--primary-color);">contact@utcards.ai</a></p>
        <p><i class="fab fa-github icon"></i> <a href="https://github.com/akashjana2123/UTCards">Github</a></p>
    </div>
    """, unsafe_allow_html=True)

def home():
    st.title("🦾 UTCards AI: Revolutionizing Instant Card Issuance")

    # Custom CSS for styling and spacing
    st.markdown("""
    <style>
    .highlight {
        color: #e75480;  /* Adjust color for light theme */
    }
    .content {
        font-size: 20px;  /* Increase font size */
        margin-bottom: 20px;  /* Add spacing at the bottom */
    }
    .container {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);  /* Simple box shadow for a subtle 3D effect */
        transition: transform 0.3s;  /* Smooth transformation for hover effect */
    }
    .container:hover {
        transform: scale(1.02);  /* Slightly increase size on hover for a dynamic effect */
    }
    @media (prefers-color-scheme: dark) {
        .highlight {
            color: #ffcccb;  /* Adjust color for dark theme */
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Using containers for better alignment, structure, and hover effects
    with st.container():
        col1, col2 = st.columns([2, 3])
        with col1:
            st.image("files/UTcards.png", use_column_width=True)
        with col2:
            st.markdown("""
            <div class="container content">
            Welcome to UTCards, where we're transforming the financial ecosystem with our 
            Instant Card Issuance (ICI) solution. Our dashboard facilitates the rapid issuance of 
            virtual cards, incorporating <span class="highlight">cutting-edge blockchain technology</span> for unparalleled security 
            and privacy.
            </div>
            """, unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
            <div class="container content">
            Our <span class="highlight">use-n-throw card feature</span> ensures each card is invalidated post-use, drastically 
            reducing scam risks. The integration of <span class="highlight">AI and machine learning</span> enables real-time fraud 
            detection, keeping your transactions safe.
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.image("files/UTcards3.png", use_column_width=True)

    with st.container():
        col1, col2 = st.columns([2, 3])
        with col1:
            st.image("files/UTcards2.jfif", use_column_width=True)
        with col2:
            st.markdown("""
            <div class="container content">
            The UTCards AI Assistant is at the heart of our user experience, offering <span class="highlight">24/7 support</span> 
            and personalized financial advice. Our platform is not just a tool but a financial 
            companion, adapting and evolving to meet your needs.
            </div>
            """, unsafe_allow_html=True)

        st.warning("Join us in redefining secure and efficient financial transactions, making financial management seamless, and setting new standards in the ICI space.")

# Main function to run the app
def main():
    app_ui_enhancements() 
    st.sidebar.title("💳 UTCards AI Assistant")
    page = st.sidebar.radio("Navigate through pages", ["🏘️ Home","❓About","📊Analysis","🤖UTCards Chatbot"])
    st.sidebar.title("User Guide") 
    with st.sidebar.expander('🧠 Enhancing User Experience with AI'):
        st.markdown("""
        - **Personalized Assistance**: Utilizing machine learning to analyze user transactions and preferences, enabling the chatbot to provide personalized financial advice.
        - **Continuous Learning**: Implementing feedback loops so the chatbot learns from each interaction, improving its responses over time.
        - **Natural Language Understanding**: Employing NLP techniques to understand user queries better and provide more accurate and human-like responses.
        - **Predictive Assistance**: Analyzing transaction patterns to predict users' needs and offer proactive assistance.
        - **Security and Privacy**: Ensure user data is handled securely, using AI to enhance privacy protections and fraud detection.
        """)
    
    # Collecting User Feedback
    st.sidebar.title("Feedback")
    rating = st.sidebar.slider("Rate your experience", 1, 5, 3)
    if st.sidebar.button("Submit Rating"):
        st.sidebar.success(f"Thanks for rating us {rating} stars!")
        st.sidebar.markdown(
            "Do visit our [Github Repository](https://github.com/akashjana2123/UTCards)"
        )

    if page == "🏘️ Home":
       home()
    elif page == "❓About":
        about()
    elif page == "📊Analysis":
        analysis_page()
    elif page == "🤖UTCards Chatbot":
        chatbot()

if __name__ == "__main__":
    main()
