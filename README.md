# ucb_mids_w207_project

W207 Final Project

Expected file structure:

```{bash}
.
|-- python scripts
|-- data
|   |-- *.csv
├── db
|   |-- *.db
└── README.md

```

Only do one of these
first time creating the database. Pass in `-d path/to/data` to specify the data location. Default is `./data`.

`python create_db.py`

alternatively if you already created the database, but you want to now delete it and recreate it add the -f flag

`python create_db.py -f`

Check that the utility tools work. Pass in `-d path/to/testing.db` to specify the database location. Default is `./db/testing.db`.

`python image_utils.py`
