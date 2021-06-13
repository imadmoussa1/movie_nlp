# Movie Genre Classification
## Project details
### Data Set
For this project we will be using the data set movies_metadata from [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv) an open source data on Kaggle.

we will be using the columns: overview, title and genres
### Frameworks
We will be using `Python 3.9` with specific library predefined in the requirements.txt
The Main Training library for training will be Tensorflow.
### Data preparing for Training
we will prepare the training data by running the `prepare_data.py` script.
This script is used to:
1) read the `movies_metadata.csv`
2) select the fields used we need for training (genres, overview, title)
3) Get the first genre of the genre list to be used as label
4) Generate the genre taxonomy (list of labels)
5) before training we generate a vocab, and we do a tokenization based on the vocab (input for the training model)
The data are saved as text file in the `data` folder each represent the descriptions of the genre

#### genre Taxonomy
0) Animation
1) Adventure
2) Romance
3) Comedy
4) Action
5) Family
6) History
7) Drama
8) Crime
9) Fantasy
10) Science Fiction
11) Thriller
12) Music
13) Horror
14) Documentary
15) Mystery
16) Western
17) TV Movie
18) War
19) Foreign

### Data cleaning
We are using spacy and regex to clean the text from repeated character, punctuation, and stop words.
This process will used when preparing the training data set and when predicting the genre

### Model
We are using Tensorflow for library for this training, we will build this model based on 2 main layer:
1) The embedding layer, to generate a vocab (The data encoding are based on vocabs vectors )
2) LSTM layer as bidirectional (Reading the text in both direction)
3) Dense layer for the prediction
4) Also we added the preprocess layer to make sure the data is cleaned

The important part in the workflow is to save the best models and loaded in the application.

### Training
We did the training using the google colab service,
we open the Training notebook in colab (movie_classification.ipynb) after that we need to upload the text file from the data folder. finally we can the training.
we have cell to model the best model that we added to project, and we will be using it for prediction ( `model` directory in the project)
## Setup
### Dataset
Download the data set [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv) and add the movies_metadata.csv to the project under movie_nlp

### Install library
```
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm
```
### Prepare data
Prepare the data for training by running
```
python3 prepare_data.py
```
Data will be saved in `data` directory
### Training
Run the `movie_classification.ipynb`
1) on colab you need to upload the data text files from data directory. and download the model
2) locally you need to update the data path `parent_dir = 'data/'` in the cell
We already have a trained model in model directory

### run the prediction (directly)
```
python3 main.py --title "Sicario" --description "During a dangerous mission to stop a drug cartel operating between the US and Mexico, Kate Macer, an FBI agent, is exposed to some harsh realities."
```
### run the prediction (Docker)

```
docker build --pull --rm -f "Dockerfile" -t movienlp:latest "."
docker run -e "title=Sicario" -e "description=During a dangerous mission to stop a drug cartel operating between the US and Mexico, Kate Macer, an FBI agent, is exposed to some harsh realities."
```

## TFX
I added [TFX](https://www.tensorflow.org/tfx) to the project a new framework used for ML Workflow with the automated test, It's using the same procedure already used in manual Training model and Same prepared data. ( it's based on TFX imdb review workflow)
All the details of this project are under tfx, for Faster debugging we can run it using the the `tfx.ipynb` notebook
tfx library installation:
```
pip3 install -r tfx/requirements.txt
```
### Model
TFX generate tf model (Same model type used before) that can be used in the main.py to predict the movie genre.
