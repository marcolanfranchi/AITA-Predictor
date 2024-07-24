import streamlit as st
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd 
import torch
from openai.embeddings_utils import get_embedding, cosine_similarity

load_dotenv()

# Load OpenAI API key
openai_key = os.getenv('OPENAI_KEY')
client = OpenAI(api_key=openai_key)

# Load trained model (logarithmic regression)
with open('logregress_model.pkl', 'rb') as file:
    model = pickle.load(file)

def get_embedding(text, model="text-embedding-3-large"): # Code snippet from https://platform.openai.com/docs/guides/embeddings/use-cases
   """
   Get the embedding of a text using the OpenAI API text-embedding-3-large model
   """
   text = text.replace("\n", " ")
   embedding = torch.tensor(client.embeddings.create(input = [text], model=model).data[0].embedding)
   embedding = np.array(embedding).reshape(1, -1)
   return embedding

# Define a function to make predictions
def predict(embedding):
    """
    Given some input text, convert the text to a vector and use the model to make a prediction
    """
    prediction = model.predict(embedding)[0]
    return 'Asshole' if prediction == 0 else 'Not the Asshole'

# initialize the streamlit app
st.set_page_config(page_title='Am I the Asshole?', page_icon='🤔', layout='centered', initial_sidebar_state='auto')
st.title('Am I the Asshole?')

story = st.text_area("Enter your story (max 250 characters):", "")

if st.button('Ask'):
    if story:
        if len(story) <= 250:
            embedding = get_embedding(story)
            result = predict(embedding)
            st.write("Story: ", story)
            # st.dataframe(pd.DataFrame(embedding))
            st.write(f'You are: {result}')
        else:
            st.write("Please limit your story to 250 characters.")
    else:
        st.write("Please enter a story to get a prediction.")
