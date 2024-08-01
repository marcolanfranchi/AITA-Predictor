
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
* Hadoop
* Python
* Torch
* Spark
* Pandas
* scikit-learn
* Jupyter
* Streamlit

## Before Starting
- The initial dataset of r/AmItheAsshole posts was collected directly from the Hadoop cluster provided as part of the course material in CMPT 353. We began our exploration by using data from 2023, but later expanded and included posts from 2022. Since, this process ([`1. unload_data.py`](1.unload_data.py)) has a long run-time, we saved the data to a directory of [zipped json files]()
- For the feature generation part of the project, [`2.Convert_OpenAI_Embedding`](2.Convert_OpenAI_Embedding), we used 

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](Repo folder containing raw data) within this repo.

    *If using offline data mention that and how they may obtain the data from the froup)*
    
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