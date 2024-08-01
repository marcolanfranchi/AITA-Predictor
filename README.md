# README


# AITA Predictor
This project is a part of SFU's [CMPT 353 Summer 2024](https://www.sfu.ca/outlines.html?2024/summer/cmpt/353/d100), Computational Data Science. 

<!-- #### -- Project Status: [Active, On-Hold, Completed] -->

## Project Intro/Objective
In this project, we aimed to develop a machine learning model capable of categorizing posts from [r/AmiTheAsshole](https://www.reddit.com/r/AmItheAsshole/). We collected data from 2022-2023 and narrowed it down to the two post flair categories that were most common: those being “YTA”(You’re The Asshole) or “NTA” (Not The Asshole). Our goal is to reliably predict the consensus of the community based on the content of each submission by categorizing it into one of those 2 categories. 

### Methods Used
* Inferential Statistics
* Text Embeddings
* Machine Learning
* Data Visualization
* Predictive Modeling

### Technologies
* Python
* scikit-learn
* Hadoop
* Torch
* Spark
* Pandas
* pickle
* Jupyter
* Streamlit

## Before Starting
- Our dataset of r/AmItheAsshole posts was collected directly from the Hadoop cluster provided as part of the course material in CMPT 353. We began our exploration by using data from 2023, but later expanded and included posts from 2022. This process is in ([`0. get_reddit_data.py`](0.get_reddit_data.py)). Since this process has a long run-time and did not need to be ran again, we saved the data to a directory of [zipped json files](). 
    - If you wish to run the next couple of steps, the second of which will require an OpenAI api key for feature generation, you can download this data [here]() and place it in the ([**FILL THIS IN**]()) directory. Otherwise, you do not need this data.
- After 
- For the feature generation part of the project, [`2.Convert_OpenAI_Embedding.ipynb`](2.Convert_OpenAI_Embedding), we generated vector embeddings for the text of each post using OpenAI's [`text-embedding-3-large`](https://platform.openai.com/docs/guides/embeddings) embedding model. This file also has a long run-time, and requires an OpenAI Api key.
    - If you wish to run `2.Convert_OpenAI_Embedding.ipynb`, download the data mentioned above and create a `.env` file in the root directory and add your OpenAI Api key to the file:
    ```OPENAI_KEY='your_openai_api_key'```
    - Otherwise, the dataset including the vector embedding for each row created from this file is located at [xxx](), so you can proceed without executing this file.

## Getting Started

1. Clone this repo and open a terminal in the root directory.

    ```
    git clone https://github.com/Localbrownboy/AITA_Predictor.git

    cd AITA_predictor
    ```

2. If you wish to run [`1. unload_data.py`](1.unload_data.py):
    - SSH 
    
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. etc...

*If your project is well underway and setup is fairly complicated (ie. requires installation of many packages) create another "setup.md" file and link to it here*  

5. Follow setup [instructions](Link to file)

## Featured Notebooks/Analysis/Deliverables
* [Notebook/Markdown/Slide Deck Title](link)
* [Notebook/Markdown/Slide DeckTitle](link)
* [Blog Post](link)


## Contributing DSWG Members

**Team Leads (Contacts) : [Full Name](https://github.com/[github handle])(@slackHandle)**

#### Other Members:

|Name     |  Slack Handle   | 
|---------|-----------------|
|[Full Name](https://github.com/[github handle])| @johnDoe        |
|[Full Name](https://github.com/[github handle]) |     @janeDoe    |

## Contact
* If you haven't joined the SF Brigade Slack, [you can do that here](http://c4sf.me/slack).  
* Our slack channel is `#datasci-projectname`
* Feel free to contact team leads with any questions or if you are interested in contributing!