# Music classifier
Classifes songs in the dataset from https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db
as either pop-music or classical music.

## Installing dependencies

### Make a virtual environment and activate it

```
python3 -m venv venv-ml
source venv-ml/bin/activate
```

If you are using another shell than regular `bash` you may need to change the second line. E.g for `fish` it will be 

```
source venv-ml/bin/activate.fish
```

You should now type `which python` and and `which pip` to see that their paths are now in the virtual environment directory you just created.

### Install dependencies

With your virtual environment acitvated, install required packages by running

```
pip install -r requirements.txt
```


## Running the model

The dataset is provided as a zip-file in the kaggle link. Download it if you don't have it. 
The program will open it and and extract files if necessary, i.e if they are not extracted before.
Run 

```
python spotify_classifier.py
``` 
to run the model.
