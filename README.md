# Short text classification with CNNs

Preliminary experiments on classifying short texts (web queries) using ConvNets (via Keras). Performance with a bit of tuning looks to be similar to what I've gotten with a fairly optimized `sklearn` pipeline of tfidf-weighted bag-of-ngrams and logistic regression.

## The task

Classify short texts as "adult" in content or not. Pretty simple.

## The models

### Word-based

Word-based classification, leveraging GloVe vectors, derived from the Keras `pretrain_embeddings` and `imdb_cnn` examples. (TODO: link those)

### Character-based

Derived from previous, learning character embeddings.

## The data

The data are user search queries and are proprietary, hence not posted here.
TODO: post some summary stats about the queries.

## Performance

TODO: write stuff here about Prec/Rec/F metrics, as well as training time (why does no one ever discuss this?!) and model size.
