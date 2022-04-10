# CZ4034 Information Retrieval Project
Group 9 (Jun Rui, Ming Da, Jun Xiang, Jun Long)

## Background
The aim of this project is to build an information retrieval system for sentiment analysis. Its system will be powered by a search engine to query over the corpus crawled and perform sentiment analysis over it. The data that we want to work on will be the 2022 Russian invasion of Ukraine.

## Tools Required
- Anaconda (Jupyter Notebook)
- PySolr
- Streamlit

## Project Setup

### Requirements
Install the requirement packages
```
pip install -r requirements.txt
```

### PySolr Environment
1. To start running the PySolr Environment, enter the following command in Powershell:
```
cd solr
solr/bin start
```

2. Open any web browser (Eg. Chrome, Edge) and type the following url:
```
localhost:8983
```

### Web Application

1. Unarchive `sklearn-pkl.zip`

2. Install the requirement packages in order to run this application.

```
pip install -r requirements.txt
```

3. To access the streamlit web application, enter the following command:

```
streamlit run app.py
```
