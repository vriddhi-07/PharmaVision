import streamlit as st
from transformers import pipeline
import os
import openai
from openai import OpenAI
from keys import OPENAI_API_KEY

# Initialize the model
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# load css file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("styles.css")

# Function to fetch drug info
def get_drug_info(question):

    prompt = f'''If the following question is medicine related, answer it, otherwise say that you can only answer medicinal questions because you're a pharmaceutical assistant.
    The question is: {question}'''
    print("")
    # Generate response
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": prompt
        }])
    return response.choices[0].message.content

# Streamlit UI
st.subheader("Familiarize yourself with your medication")

# Input for questions
medications_input = st.text_area("Ask anything about your medicine!")

if st.button("Submit"):
    if medications_input:

        # Generate response from the model
        with st.spinner('Generating response...'):
            result = get_drug_info(medications_input)
        
        # Display the result
        st.subheader("Results")
        st.write(result)
    else:
        st.error("Please type your query.")
