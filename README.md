# Popularity Prediction Framework

### Start the framework
1. Clone the repository: `git clone https://github.com/CarlAmbroselli/Popularity_Predictor`
2. Open repository: `cd Popularity_Predictor`
3. (optional) Checkout paper/napoles branch: `git checkout paper/napoles`
4. Install requirements: `pip install -r requirements.txt #might be done in a virtualenv`
5. Add `comments.csv` to the `data/datasets/YNACC-Evaluation/train/` and `data/datasets/YNACC-Evaluation/test` folder with the following format: 
```sdid,commentindex,headline,url,guid,commentid,timestamp,thumbs-up,thumbs-down,text,parentid,constructiveclass,sd_agreement,sd_type,sentiment,tone,commentagreement,topic,intendedaudience,persuasiveness,y_persuasive,y_audience,y_agreement_with_commenter,y_informative,y_mean,y_controversial,y_disagreement_with_commenter,y_off_topic_with_article,y_sentiment,y_sentiment_neutral,y_sentiment_positive,y_sentiment_negative,y_sentiment_mixed```
6. Run using `python .`
