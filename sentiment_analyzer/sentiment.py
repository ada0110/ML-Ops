import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_sentiment(text = "This is a very nice day"):
    sid = SentimentIntensityAnalyzer()
    score = ((sid.polarity_scores(str(text))))['compound']

    if(score > 0):
        label = 'Sentence is Positive'
    elif(score == 0):
        label = 'Sentence is Neutral'
    else:
        label = 'This sentence is negative'

    print(f"i/p: {text}")
    return f"i/p: {text}    o/p: {label}" 


if __name__ == "__main__":
    print(get_sentiment())