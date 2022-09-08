from cProfile import label
from email import message
from fastapi import FastAPI, Depends
import schemas
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from bangla_stemmer.stemmer import stemmer
import joblib
import uvicorn
from . import  models
from .database import SessionLocal, engine
from sqlalchemy.orm import Session


# instances
stmr = stemmer.BanglaStemmer()
app = FastAPI()

models.Base.metadata.create_all(engine)




def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


#ML models
model = joblib.load('prediction/RF_model.joblib')
vector = joblib.load('prediction/Tf_Idf_vector.joblib')




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


@app.post('/prediction',  tags=['Bangla hate Speech Detection'])
def make_prediction(request: schemas.Comment, db: Session=Depends(get_db)):
    
    comment = request.message
    comment = preprocessing(comment)
    text_vec = vector.transform(comment)

    prediction = model.predict(text_vec)
    #print(prediction[0])


    new_prediction = models.Prediction(message=request.message,label=prediction[0])
    db.add(new_prediction)
    db.commit()
    db.refresh(new_prediction)


    if prediction[0] == 1:
        return 'Hateful Comment - class: 1'
    else:
        return 'None Hate Comment - class: 0'


# Get all the prediction
@app.get('/show_all_predictions', tags=['Bangla hate Speech Detection'])
def all(db: Session=Depends(get_db)):

    predictions = db.query(models.Prediction).all()

    return predictions


# Get predictions by ID
@app.get('/get_prediction_by_id/{id}', tags=['Bangla hate Speech Detection'])
def showParticularPrediction(id, db: Session=Depends(get_db)):
    
    pred = db.query(models.Prediction).filter(models.Prediction.id == id).first()
    return pred


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)