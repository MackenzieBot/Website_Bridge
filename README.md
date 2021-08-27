# ChatBot-Backend
 Mackenzie ChatBot Backend Flask Project

## Directions
1. Create the virtual environment by running `python3 -m venv venv`.
2. Activate the virtual environment.
   ```bash
    cd venv
    source bin/activate
    ```
3. Install dependencies using the provided requirements:
    ```bash
    pip install -r requirements.txt
    ```
4. Add the nltk_data folder inside venv.
   1. Get files from http://www.nltk.org/nltk_data/
   2. Add the stopwords folder inside `venv/nltk_data/corpora`
   3. Add the punkt folder inside `venv/nltk_data/tokenizers`
5. Run `app.py`.