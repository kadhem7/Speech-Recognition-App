import nltk
import streamlit as st
import speech_recognition as sr
import string
import re

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw')
nltk.download('stopwords')
nltk.download('wordnet')

# Load text data
with open(r'C:\Users\K\Desktop\newspeechrecognition\temperature.txt', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

# Tokenize the text into sentences
sentences = sent_tokenize(data)

# Preprocess each sentence
def preprocess(sentence):
    words = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    #words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_words = []
    for word in words:
        lemmatized_word = lemmatizer.lemmatize(word)
        lemmatized_words.append(lemmatized_word)
    return words

# Preprocess each sentence in the text
#corpus = [preprocess(sentence) for sentence in sentences]

corpus = []
for sentence in sentences:
    preprocessed_sentence = preprocess(sentence)
    corpus.append(preprocessed_sentence)

# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    query = preprocess(query)
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence

# Define a function to transcribe speech into text
def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio_text = recognizer.listen(source)
        st.info("Transcribing...")

        try:
            text = recognizer.recognize_google(audio_text)
            return text
        except sr.UnknownValueError:
            return "Sorry, I did not get that."
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service: {e}"

# Define chatbot function
def chatbot(question):
    most_relevant_sentence = get_most_relevant_sentence(question)
    return most_relevant_sentence

# Streamlit app
def main():
    st.title("Speech Recognition App")
    st.write("Click on the microphone to start speaking:")
    st.title("Chatbot")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")

    # Add a button to trigger speech recognition
    if st.button("Start Recording"):
        text = transcribe_speech()
        st.write("Transcription: ", text)

    # Get the user's question
    question = st.text_input("You:")
    # Create a button to submit the question
    if st.button("Submit"):
        # Call the chatbot function with the question and display the response
        response = chatbot(question)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
