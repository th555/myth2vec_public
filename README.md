Code to train a word2vec vector-space model on the MythFic corpus (not included), plot emotion associations per character (see `/radarplots/`), emotion associations per character comparing different genres (see `/genreradarplots/`), and distribution of emotion associations (over a larger set of characters) differentiated by gender and genre (see `/violinplots/`)

Installation and running
------------

`pip install -r requirements.txt`

Separately also make sure you have sklearn.

MythFic_txt directory containing \[id\].txt files should be at the same level as this repository (i.e. one level up from this README), as well as the file `fanfics_Greek_myth_metadata.csv` [available here](https://doi.org/10.34973/2MYE-8468). Then from within myth2vec just run `python myth2vec.py`. Training takes a while so the model is cached using pickle; therefore to retrain (after changing corpus or model parameters) remove all pickle files.
