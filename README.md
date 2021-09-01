# Disaster Response Pipeline Project
In a terminal that contains this README file, run commands in the following sequence:
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database:
    
        `python data/process_data.py` `data/disaster_messages.csv` `data/disaster_categories.csv` `data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves
    
        `python models/train_classifier.py` `data/DisasterResponse.db` `models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

In the web app, you may input any text message (in English) and it will categorize it among 36 categories, as in this notebook.

# Webapp screenshoot
![alt text](https://github.com/Nhan121/Udacity_project2_disaster_response/blob/main/Fig_1_1.jpg)

![alt text](https://github.com/Nhan121/Udacity_project2_disaster_response/blob/main/Fig_1_2.jpg)


![alt text](https://github.com/Nhan121/Udacity_project2_disaster_response/blob/main/Fig_1_3.jpg)
