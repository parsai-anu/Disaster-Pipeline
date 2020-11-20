# Disaster Response Pipeline Project

This project analyses messages sent during disaster event. It classifies messgaes into categories so that the message reaches to the correct agency handling the disaster response.

The script contains two python scripts - process_data.py and train_classifier.py, it also visualizes the application on the Flask App.

The python_data.py script loads the raw data, merges datasets, cleans data and saves it to sql database.
The train_classifier.py imports the sql table, splits as train-test, perfoms text processing, runs an ML model on the text to classify, seraches for the optimal model using grid search, outputs and saves result.
Finally the flask app displays simple visualization.

# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/




