# Disaster Response Pipeline Project

### Summary:
This a project done as a part of udacity data science nano degree, it is focus is machine learning and data engineering aspects of data science. The project also shows how to present your data science work to the user in the form of a web app that the user can interact with easily.

### Files:
The project files are divided on three folders: app, data and models. The app folder includes the web app files incuding python script that runs the flask web app, while data folder includes the data files and python scripts that cleans the data. Finally, models folder includes the machine learning script written in python which uses sklearn library.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
