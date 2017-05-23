# cnn-short-text-classification

Preliminary experiments on classifying short texts using ConvNets (via Keras). Performance with a bit of tuning looks to be similar to what I've gotten with a fairly optimized `sklearn` pipeline of tfidf-weighted bag-of-ngrams and logistic regression.

## Word-based

Word-based classification, leveraging GloVe vectors, derived from the Keras `pretrain_embeddings` and `imdb_cnn` examples. (TODO: link those)

## Character-based

Derived from previous, learning character embeddings.

## Performance

TODO: write stuff here about Prec/Rec/F metrics, as well as training time (why does no one ever discuss this?!) and model size.
