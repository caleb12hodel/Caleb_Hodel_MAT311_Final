# Customer Churn Machine Learning Final Project

The MAT311 Final Project repository is my Data Science final project. The repository leverages scikit learn's machine learning classification models in the context of binary prediction. 


## Purpose and Context

This project aims to predict whether a given customer for a fictitious company will churn or not churn. We have been given a `test.csv` dataset which contains various features that must be experimented with in order to create a machine learning model to predict whether or not a given customer churns. 

## Project layout

```
.
├── main.py                 (Entry point that runs the entire pipeline)
├── requirements.txt        (Python dependencies)
├── data/
    ├── processed/          (Created after running the pipeline)
    └── raw/
        └── train.csv
        └── test.csv        (Used to generate Kaggle probabilities)
├── notebooks/
    └── dumb_model.ipynb
    └── eda_test.ipynb
    └── eda_train.ipynb
    └── ideas.ipynb
    └── knn_idea.ipynb
    └── neural_network.ipynb
    └── random_forest.ipynb
└── src/
    ├── data/
    │   └── load_data.py        (just for eda in main)
    ├── features/
    │   └── build_features.py
    ├── MAT311/
        └── clean_data.py       (clean data)
        └── encode_cat.py       (encode categorical variables)
        └── load.py             (load data and split data)
    ├── models/
        └── dumb_model.py
        └── knn_model.py
        └── random_forest.py
    ├── utils/
        └── helper_functions.py
    └── visualization/
        └── eda.py
        └── performance.py
├── submission                  (contains csv files for kaggle submissions)
```

`main.py` imports the modules from `src/` and utilizes `MAT311` for a custom python library, enabling seamless function integration into both notebooks and `main.py`. Using these modules we are able to executes models in a reproducible way that enables easier analysis. Jupyter notebooks are provided only for prototyping and exploration—they are **not** meant to be the main entry point of the project.

Some directories such as `data/external/`, `src/utils/` and `tests/` may be empty, but the folder structure is provided to illustrate how a complete project should look.

## Running the project

Install the dependencies and run the pipeline. You should use the versions of the dependencies as specified by the `requirements.txt` file:

```bash
conda create -n final_project --file requirements.txt
conda activate final_project
python main.py
```

This will load the dataset, perform basic feature engineering, train 3 models and produce visualizations.
The cleaned data will be written to `data/processed/` and all plots will be displayed interactively. To create a pdf based report use our functions in a notebook and export to either pdf or html.
