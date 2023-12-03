# MusicGenreClassification - IBM Machine Learning Service
Music is like a mirror, and it tells people a lot about who you are and what you care about, whether you like it or not. We love to say “you are what you stream,”.Companies nowadays use music classification, either to be able to place recommendations to their customers (such as Spotify, Soundcloud) or simply as a product (for example Shazam). Determining music genres is the first step in that direction. Machine Learning techniques have proved to be quite successful in extracting trends and patterns from the large pool of data. The same principles are applied in Music Analysis also.This project is aimed at using the KNN classification algorithm to detect the genre of music from an audio file. The ability to classify an audio file and categorise them according to their genres, has proven to put a huge impact on services mentioned above. This way, they engage their customers more by predicting what type of music a particular customer is interested in and further applying state of the art deep learning methods to give recommendations.
# Technical Architecture:
![image7](https://github.com/mdaliuddinhyder04/MusicGenreClassification/assets/106607934/8693f523-168e-4828-bca5-38d5c60f97b4)
# Project Flow:
Download/Create the dataset.
Process the audio files and load the data into Numpy Arrays.
Train and Test split on complete dataset
Build the classifier : Defining Functions and calling them as needed
Finding accuracy on Test Data 
Build a Web application using flask for the same.
# Project Structure
![image1](https://github.com/mdaliuddinhyder04/MusicGenreClassification/assets/106607934/a17715ce-be95-42f2-acab-24d83bfc6b77)

Flask App : has all the files necessary to build the flask application. 
templates folder has the HTML page.
uploads folder has the uploads made by the user.
app.py is the python script for server side computing.
my.dat is the file that contains the saved data (machine-readable format).
Dataset :  The dataset can be easily downloaded from the link given in below and we need to just put the genre folders as our dataset. Here we have named it as Music Genres and there are 10 subclasses which contain audio files (our actual data).
Python file : these are the files which we have used to train the model, named as “music genre classification train.py”. It contains all the necessary functions required for the KNN as well as handling audio data.
Data file :  this is originally created data file after training and then put into the flask folder for our flask app to access it and run the model while deployment. It contains information about all the audio clips present and stored as 
Wav files : These are some test files which we have used for testing the final application.
# Dataset Used:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
# Model Building:
WE use  KNN Algorithm to classify the music type. so this activity allows you to build machine learning Model
1. Importing The Libraires
   Import all the required libraries. Make sure to check you’ve everything installed on your system which is mentioned in the Pre-Requisites in order to not face any problems while doing this.
2. Define The Functions Required For KNN
   We need to define a few functions that’ll do our work in order to be able to apply  KNN algorithms on a dataset.

Step 1 :  Is to define a function to calculate distance between two instances.
Here one instance will be fixed and second will be changed one by one from the entire dataset and we’ll calculate distances between our input audio file and all the data we have.
Step 2 : is to define a function to Get Neighbors 
It returns a list of top k neighbors which means, the top k minimum distances between our given file and  files from the entire dataset are found out and their corresponding labels are returned in a list.
Step 3 : is to define a function to predict the nearest class.

It is built to choose the class that got max votes from the top k neighbors list.
We have created a dict which stores key value pairs as class and no. of votes received. We then sort the dict and return the first element which is the class that got max votes.
# Loading The Data From Audio Files To Machine Readable Format:
Create a binary “.dat” file to add data generated from the audio files.
Opening a file with write permissions. We’ll write our data into this file
Read data from all folders using OS library to parse through folders.

We make a list of all the genres available and parse through each folder to read all the files and extract the data from them. 
We are using Python_speech_features to calculate MFCC for the audio file. Along with MFCC, Covariance of MFCC features is also calculated. 
Now these two along with the label (that is the number associated to the folder we’re currently having the audio file) are combined to form a tuple and that tuple is added into the “.dat” file.
Loading the complete dataset created as a python readable object.
Here we created an empty list named “dataset”, added the my.dat file’s data to it and then converted it to an numpy array to proceed further.
# Train Test Split And Model Evaluation:
Splitting the data

We split the X data only, as we have defined the algorithm on our own, so we do not need to break the data into x and y manually.

We then make predictions by the function call.
Here, every single file’s data from the X Test is being matched with all the train data and then, the final prediction is being stored into predictions.
 Evaluation of the model

We first define a function to evaluate the accuracy by counting total no. of correct predictions divided by the total no. of predictions made.
# Application Building:
After the model is built, we will be integrating it to a web application so that normal users can also use it. The users need to give the microscopic image of the tissue/tumor to know the predictions.
1. Build A Flask Application
2. Build a flask application

Step 1: Load the required packages
Step 2: Initialize flask app and load the saved data file
An instance of Flask is created and the model is loaded using pickle into a list named dataset.
Step 3: Define functions required to calculate distances, neighbors and make predictions.
Step 3(A) :  Define functions to calculate distance between the input from user and the saved data files, and then find top ‘k’ nearest neighbours
Step 3(B) :  Define function to get the result which is the class that got max votes.
Step 4: Configure the home page
Step 5: Pre-process the frame and run

Pre-process the captured audio file and give it to the model for prediction. Based on the prediction the output text is generated and sent to the HTML to display.
The output produced by the model is a number which is used to get the genre name from the dictionary named as results. This class name is printed as output on the html page along with a sentence that describes the output more.

Run the flask application using the run method. By default the flask runs on Port number 5000. If the port is to be changed, an argument can be passed to do so.
# Build The HTML Page And Execute:
Build an html page to take an audio file as an input and display the output that is passed from the flask app.
Please refer to project structure section to download HTML files
# Run The App
In anaconda prompt, navigate to the folder in which the flask app is present. When the python file is executed the localhost is activated on 5000 port and can be accessed through it.
Step 2: Open the browser and navigate to localhost:5000 to check your application
The home page looks like this. When you click on the button “Drop in the song you want to classify!”, you’ll be redirected to the predict section
In this section you can browse and choose the audio file you want to predict genre for and then click on predict to get the predictions.

> Thank you
