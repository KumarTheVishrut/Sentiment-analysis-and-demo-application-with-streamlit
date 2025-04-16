import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load models
lr_model = joblib.load('logistic_regression_model.joblib')
svm_model = joblib.load('svm_model.joblib')

# Class labels
class_names = ['Negative', 'Neutral', 'Positive']

# Configure Streamlit app
st.set_page_config(page_title="ML News Sentiment Analysis", layout="wide")

# Sidebar controls
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Page", 
                               ["Performance Comparison", "Real-time Analysis"])

# Main content
if app_mode == "Performance Comparison":
    st.title("ML News Sentiment Model Comparison")
    
    # Create columns for side-by-side comparison
    col1, col2 = st.columns(2)
    
    # Logistic Regression metrics
    with col1:
        st.header("Logistic Regression")
        st.image('logistic_regression_cm.png')
        
    # SVM metrics    
    with col2:
        st.header("Support Vector Machine")
        st.image('svm_cm.png')
        
    # Additional metrics
    st.subheader("Classification Reports")
    with st.expander("Show/Hide Reports"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Logistic Regression Report**")
            st.code("""              precision    recall  f1-score   support

           0       0.60      0.72      0.66       120
           1       0.82      0.83      0.83       536
           2       0.71      0.63      0.67       255

    accuracy                           0.76       911
   macro avg       0.71      0.73      0.72       911
weighted avg       0.76      0.76      0.76       911""")
        
        with col2:
            st.write("**SVM Report**")
            st.code("""              precision    recall  f1-score   support

           
           0       0.55      0.67      0.60       120
           1       0.83      0.82      0.83       536
           2       0.70      0.64      0.67       255

    accuracy                           0.75       911
   macro avg       0.69      0.71      0.70       911
weighted avg       0.76      0.75      0.75       911
""")

else:
    st.title("Real-time Sentiment Analysis")
    
    # User input
    user_input = st.text_area("Enter ML news text for analysis:", 
                             height=150)
    model_choice = st.selectbox("Select Model:", 
                               ("Logistic Regression", "Support Vector Machine"))
    
    if st.button("Analyze"):
        if user_input:
            # Get selected model
            model = lr_model if model_choice == "Logistic Regression" else svm_model
            
            # Make prediction
            prediction = model.predict([user_input])[0]
            probabilities = model.predict_proba([user_input])[0]
            
            # Display results
            st.subheader("Analysis Results")
            
            # Confidence visualization
            st.write("### Prediction Confidence")
            fig, ax = plt.subplots()
            sns.barplot(x=probabilities, y=class_names, palette="viridis")
            ax.set_xlabel("Probability")
            ax.set_xlim(0, 1)
            st.pyplot(fig)
            
            # Final prediction
            st.success(f"Predicted Sentiment: {class_names[prediction]}")
        else:
            st.warning("Please enter some text to analyze")
