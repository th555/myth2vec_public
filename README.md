pip install -r requirements.txt

separately also make sure you have sklearn

MythFic_txt directory containing .txt files should be at the same level as this repository, then from within myth2vec just run `python myth2vec.py`. Training takes a while so the model is cached using pickle; therefore to retrain (after changing corpus or model parameters) remove all .corpuspickle and .modelpickle files.
