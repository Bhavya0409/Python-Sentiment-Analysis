"I pledge that I have abided by the Stevens Honor System" - Bhavya Shah

## Work
Sentiment Analysis on Movie review dataset from Kaggle

## Dataset URL
https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data

## Steps
1) Read TSV into Python -> list of reviews, or it can be in dataframe structure
  - import_dataset(file_path: string) -> array[array[any]]
    - inner array[any]: structured similar to the columns
      - Index 0: Phrase
      - Index 1: Sentiment
2) Iterate each review and for each "phrase" column, clean up stop words and punctuations
  - def clean_stop_words(review_text: string) -> string
3) Normalize the text (stemming or lemmatizing)
  - Takes a string of non stop words in the sentence
  - def stemming_text(nonstop_words_sentence: string) -> string
  - def lemmatize_text(nonstop_words_sentence: string) -> string
4) Run model
