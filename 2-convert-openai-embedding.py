from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd 
import torch

def main():
    """
    This program loads the filtered and balanced data and adds the OpenAI embeddings to the DataFrame.
    """
    def get_embedding(text, model="text-embedding-3-small"): # Code snippet from https://platform.openai.com/docs/guides/embeddings/use-cases
        text = text.replace("\n", " ")
        return torch.tensor(client.embeddings.create(input = [text], model=model).data[0].embedding)


    load_dotenv()
    openai_key = os.getenv('OPENAI_KEY')
    client = OpenAI(api_key=openai_key)

    submissions = pd.read_json('output/filtered_and_balanced.json.gz' , compression='gzip')

    submissions['embedding'] =  submissions['selftext'].apply(lambda x: get_embedding(x, model='text-embedding-3-large'))

    submissions.to_pickle('output/openai_embedded_large.pkl')


if __name__ == '__main__':
    main()
    
