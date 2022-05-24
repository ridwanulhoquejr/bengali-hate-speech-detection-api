from fastapi import FastAPI
from hate_detection import schemas
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from bangla_stemmer.stemmer import stemmer
import joblib


# instances
stmr = stemmer.BanglaStemmer()
app = FastAPI()


#ML models
model = joblib.load('RF_model.joblib')
vector = joblib.load('Tf_Idf_vector.joblib')


@app.get('/')
def home():
    return 'Hello from fastAPI'


def preprocessing(comment):

  comments = []

  pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
  text = pattern.sub('', comment)

  # removing #mentions
  text = re.sub(r"@+","", text)

  # removing punctuations
  text = re.sub(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~।।ঃ\t\n]", "", text) 


  tokens = word_tokenize(text)
  stop_words = set(stopwords.words("bengali"))
  #stop_words.discard('না')
  words = [w for w in tokens if not w in stop_words] # removing stop words

  words = stmr.stem(words)
  words = ' '.join(words)
  comments.append(words)

  return comments


@app.post('/prediction')
def make_prediction(request: schemas.Comment):
    
    comment = request.message
    comment = preprocessing(comment)
    text_vec = vector.transform(comment)

    prediction = model.predict(text_vec)
    print(prediction[0])

    if prediction[0] == 1:
        return 'Hateful Comment - class: 1'
    else:
        return 'Non Hate Comment - class: 0'