# ML_Project
This Repo contains my proposal for my project on ML master's degree class, wichi consists in a classificator to see if the stellar object is or not an exoplannet, and in the case it is, predict if it could be habitable.

This contains:

- Linear classificator
- Logistic classificator
- Multilayer Perceptron
- Classification (Decision) Trees
- Mixture Models (EM + Naive Bayes)
- AdaBoost

## Dataset
I'm using the dataset 
[Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results) provided by NASA.

## Instructions
Run "pip install -r requirements" to install neesary libraries

Execute uvicorn "api:app --reload" to enable uvicorn server

Execute index.html

There are some test cases to validate de data on test_cases.txt