import streamlit as st
from utils.emotion_recognition import detect_emotion
from utils.voice_interface import listen, speak
from utils.reminders import set_reminder
from transformers import pipeline

# Load GPT-2 model
generator = pipeline('text-generation', model='gpt-2')

# App title and description
st.title("Healthcare LLM App")
st.write("Welcome to your personal healthcare assistant!")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose a feature:", [
    "Drug Overcome", "Mental Health Tips", "Pill Reminder", 
    "Emotion Recognition", "Voice Interface", "Fitness Integration", 
    "Progress Tracking", "Emergency Contact", "Symptom Checker"  # Added new option
])

# Drug Overcome
if options == "Drug Overcome":
    st.header("Drug Overcome")
    user_input = st.text_input("Ask me about overcoming drug addiction:")
    if user_input:
        response = generator(user_input, max_length=100, num_return_sequences=1)
        st.write(response[0]['generated_text'])

# Mental Health Tips
if options == "Mental Health Tips":
    st.header("Mental Health Tips")
    user_input = st.text_input("Ask me for mental health tips:")
    if user_input:
        response = generator(user_input, max_length=100, num_return_sequences=1)
        st.write(response[0]['generated_text'])

# Pill Reminder
if options == "Pill Reminder":
    st.header("Pill Reminder")
    pill_time = st.time_input("Set a time for your pill reminder:")
    if st.button("Set Reminder"):
        set_reminder(pill_time)
        st.write(f"Reminder set for {pill_time}")

# Emotion Recognition
if options == "Emotion Recognition":
    st.header("Emotion Recognition")
    user_text = st.text_input("How are you feeling today?")
    if user_text:
        emotion = detect_emotion(user_text)
        st.write(f"Detected Emotion: {emotion}")

# Voice Interface
if options == "Voice Interface":
    st.header("Voice Interface")
    if st.button("Start Voice Command"):
        command = listen()
        if command:
            response = generator(command, max_length=100, num_return_sequences=1)
            speak(response[0]['generated_text'])

# Fitness Integration
if options == "Fitness Integration":
    st.header("Fitness Integration")
    api_key = st.text_input("Enter your fitness app API key:")
    if api_key:
        # Add code to fetch fitness data
        st.write("Fitness data will be displayed here.")

# Progress Tracking
if options == "Progress Tracking":
    st.header("Progress Tracking")
    progress = st.slider("How are you feeling today?", 0, 100)
    st.write(f"Your progress: {progress}%")

# Emergency Contact
if options == "Emergency Contact":
    st.header("Emergency Contact")
    contact = st.text_input("Enter emergency contact number:")
    if st.button("Save Contact"):
        st.write(f"Emergency contact saved: {contact}")

# Symptom Checker
if options == "Symptom Checker":
    st.header("Symptom Checker")
    symptoms = st.text_input("Describe your symptoms:")
    if symptoms:
        response = generator(symptoms, max_length=100, num_return_sequences=1)
        st.write(response[0]['generated_text'])