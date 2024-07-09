# Garbage-Classification-Using-CNN
Garbage Classification Using CNN

Garbage Classification Project
Overview
	This project aims to classify six classes of garbage using a Convolutional Neural Network (CNN) model. 
	The model is trained on a dataset containing images of cardboard, glass, metal, paper, plastic, and trash.

Project Structure
	data/: Contains the dataset used for training and testing.
		"./data/Training_Data" and
		"./data/Testing_Data"
	model_training.py: Python script for training the CNN model.
	model_testing.py: Python script for evaluating the trained model on test data.
	requirements.txt: Lists the Python packages required to run the project.
	
	
Installation and Environment setup:

**Create a virtual environment**

    python -m venv venv
    
**Activate the virtual environment**

    - **On Windows:**bash
    
        venv\Scripts\activate
        
    - **On macOS and Linux:**bash
    
        venv/bin/activate

**Install the dependencies**

    pip install -r requirements.txt


Training the Model: Run model_training.py to train the CNN model using the dataset in the data/ directory.
	python model_training.py
	python model_testing.py

Testing the Model: After training, run model_testing.py to evaluate the model's performance on the test dataset.


Files Description
model_training.py: Defines the CNN architecture, trains the model, and saves it as garbage_classifier.h5.
model_testing.py: Loads the trained model, performs predictions on test data, and evaluates its performance using 
		  metrics such as accuracy, precision, recall, and F1-score.
For dataset there are alot of datas with the same class in Kaggle also here. 
