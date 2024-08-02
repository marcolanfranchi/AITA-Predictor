from sentence_transformers import SentenceTransformer , util
import torch
import numpy as np
import pandas as pd
import spacy


def main():
    """
    This program loads the filtered and balanced data and adds embeddings to the DataFrame.
    """

    def process_text(text):
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents] # break text into its sentences
        # Move sentences to device
        chunk_embeddings = model.encode(sentences, convert_to_tensor=True, device=device) # move to gpu for quicker generation 
        story_embedding = torch.mean(chunk_embeddings, dim=0) # change this to sum if you want to sum them instead of average 
        return story_embedding.cpu()  # Move tensor back to CPU before returning | we need to do this in case we wanna use numpy arrays in the future (we do)


    model = SentenceTransformer('paraphrase-MiniLM-L6-v2') # sentence transormer model
    nlp = spacy.load('en_core_web_sm') # Spacy model that we use to partition sentences 

    # Set device to gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    submissions = pd.read_json('output/filtered_and_balanced.json.gz' , compression='gzip')
    submissions['embedding'] = submissions['selftext'].apply(process_text)

    submissions.to_pickle('output/paraphrase_mini_l6_embedded_averaged.pkl')


if __name__ == '__main__':
    main()