import streamlit as st
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_extras.let_it_rain import rain
import streamlit_scrollable_textbox as stx
import altair as alt

load_dotenv()

# Load OpenAI API key
openai_key = os.getenv('OPENAI_KEY')
client = OpenAI(api_key=openai_key)
submissions = pd.read_pickle('output/openai_embedded_large_all.pkl')

# Load trained model (logistic regression)
with open('logregress_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load t-SNE data
df_tsne = pd.read_pickle('output/tsne.pkl')


@st.cache_data
def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    embedding = torch.tensor(client.embeddings.create(input=[text], model=model).data[0].embedding)
    embedding = np.array(embedding).reshape(1, -1)
    return embedding


@st.cache_data
def predict(embedding):
    prediction = model.predict(embedding.reshape(1, -1))[0]
    return 'Asshole' if prediction == 0 else 'Not the Asshole'


@st.cache_data
def get_similar_posts(df, embedding, n=3):
    df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(np.array(x).reshape(1, -1), embedding).flatten()[0])
    res = df.sort_values('similarities', ascending=False).head(n)
    return res

@st.cache_data
def plot_tsne_with_annotations(similar_posts):
    colors = {'Asshole': 'red', 'Not the A-hole': 'blue'}
    color_scale = alt.Scale(domain=list(colors.keys()), range=list(colors.values()))

    base_chart = alt.Chart(df_tsne).mark_circle(size=60, opacity=0.5).encode(
        x='tsne1',
        y='tsne2',
        color=alt.Color('label', scale=color_scale),
        tooltip=['label', 'selftext']
    )

    annotations = []
    idx = 1
    for index, row in similar_posts.iterrows():
        embedding_data = df_tsne[df_tsne['selftext'] == row['selftext']]
        annotations.append(alt.Chart(pd.DataFrame({
            'title': '⬆ '+f'Similar Story {idx}',
            'label': row['link_flair_text'],
            'selftext': row['selftext'],
            'tsne1': embedding_data['tsne1'],
            'tsne2': embedding_data['tsne2']
        })).mark_text(
            align='left',
            baseline='middle',
            dx=7,
            dy=-7,
            fontWeight='bold',
            fontSize=12,
            color='white'
        ).encode(
            text='title',
            tooltip=['label', 'selftext'],
            x='tsne1',
            y='tsne2'
        ))
        idx += 1

    chart = base_chart + alt.layer(*annotations)

    st.altair_chart(chart, use_container_width=True)


st.set_page_config(page_title='Am I the Asshole?', page_icon='🤔', layout='centered', initial_sidebar_state='auto')
st.title('Am I ?')

def emoji_animation(model_result):
    if model_result == 'Asshole':
        emoji = "😡"
    elif model_result == 'Not the Asshole':
        emoji = "😊"
    rain(
        emoji=emoji,
        font_size=54,
        falling_speed=3,
        animation_length="1s",
    )


story = st.text_area("Enter your story (max 500 characters):", "")

if st.button('ask'):
    if story:
        if len(story) <= 500:
            embedding = get_embedding(story)
            result = predict(embedding)
            with st.container():
                st.markdown(f'- **story:** {story}')
                st.markdown(f'- **the model says you are:** {result}')
                emoji_animation(result)
                res = get_similar_posts(submissions, embedding, n=3)
                st.markdown("these stories are most similar to yours:")
                story_num = 1
                for index, row in res.iterrows():
                    with st.container():
                        st.write('---')
                        st.markdown(f'**{story_num})**')
                        story_title = row['title']
                        story = row['selftext']
                        reddit_classification = row['link_flair_text']
                        st.markdown(f'*{story_title}*')
                        st.markdown(f'(this story was classified as **{reddit_classification}** by redditors)')
                        stx.scrollableTextbox(story, height=250, fontFamily='system-ui')
                        story_num += 1

                # plotting t-SNE chart with annotations for similar stories
                st.write("---")
                st.write("This is where the most similar stories are located in the t-SNE plot:")
                plot_tsne_with_annotations(res)

        else:
            st.write("Please limit your story to 500 characters.")
    else:
        st.write("Please enter a story to get a prediction.")

