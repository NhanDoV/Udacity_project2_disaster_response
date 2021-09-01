# Disaster Response Pipeline Project
In a terminal that contains this README file, run commands in the following sequence:

        `python data/process_data.py` `data/disaster_messages.csv` `data/disaster_categories.csv` `data/DisasterResponse.db`
    
        `python models/train_classifier.py` `data/DisasterResponse.db` `models/classifier.pkl`

        `python run.py`

Go to the website http://0.0.0.0:3001/ and if facing problems try http://localhost:3001 in a browser

In the web app, you may input any text message (in English) and it will categorize it among 36 categories, as in this [notebook](https://github.com/Nhan121/Udacity_project2_disaster_response/blob/main/ML%20Pipeline%20Preparation.ipynb).

# Webapp screenshoot
![alt text](https://github.com/Nhan121/Udacity_project2_disaster_response/blob/main/Fig_1_1.jpg)

![alt text](https://github.com/Nhan121/Udacity_project2_disaster_response/blob/main/Fig_1_2.jpg)

![alt text](https://github.com/Nhan121/Udacity_project2_disaster_response/blob/main/Fig_1_3.jpg)

# Installation
This project requires Python 3.x and the following Python libraries:

`NumPy`
`Pandas`
`Matplotlib`
`Json`
`Plotly`
`Nltk`
`Flask`
`Sklearn`
`Sqlalchemy`
`Sys`
`Re`
`Pickle`
`Seaborn`

You will also need to have software installed to run and execute an iPython Notebook.


# Code and data
The coding for this project can be completed using the Project Workspace IDE provided. Here's the file structure of the project:

        - app
        | - template
        | |- master.html  # main page of web app
        | |- go.html  # classification result page of web app
        |- run.py  # Flask file that runs app

        - data
        |- disaster_categories.csv  # data to process 
        |- disaster_messages.csv  # data to process
        |- process_data.py
        |- InsertDatabaseName.db   # database to save clean data to

        - models
        |- train_classifier.py
        |- classifier.pkl  # saved model 

        - ETL Pipeline Preparation.ipynb # notebook file of Project Workspace - ETL
        - ML Pipeline Preparation.ipynb # notebook file of Project Workspace - Machine Learning Pipeline.
        - README.md
