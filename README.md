# ucb_mids_w207_project

W207 Final Project: Facepoint Detection

Kaggle Link: <https://www.kaggle.com/c/facial-keypoints-detection>

Expected file structure:

```path
.
|-- data
|   |-- *.csv
├── db
|   |-- *.db
|-- model_trainer
|   |-- all python codes that builds, tunes and visualize the models
|-- models
|   |-- all python files that actually trains the models and predict the test data
└── README.md
```

## First time creating the database

- Make sure all csv files are in the data directory.
- cd to the root directory of the project, then run:
  > `python db/create_db.py`

## If you already created the database

- you want to now delete it and recreate it, then add the -f flag
  > `python db/create_db.py -f`

## Check that the EDA tools work

- cd to the root directory of the project, then run:
  > `python image_utils.py`
