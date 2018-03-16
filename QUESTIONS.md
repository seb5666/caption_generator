# Questions
1. The model is uses 2 different LSTMs: one to generate the output sequences and one as a language model:
    ```python
    image_model = Sequential()
    image_model.add(Dense(EMBEDDING_DIM, input_dim = 4096, activation='relu'))
    image_model.add(RepeatVector(self.max_cap_len))

    # First LSTM
    lang_model = Sequential()
    lang_model.add(Embedding(self.vocab_size, 256, input_length=self.max_cap_len))
    lang_model.add(LSTM(256, return_sequences=True))
    lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))

    # Second LSTM
    model = Sequential()
    model.add(Merge([image_model, lang_model], mode='concat'))
    model.add(LSTM(1000, return_sequences=False))
    model.add(Dense(self.vocab_size))
    model.add(Activation('softmax'))
    ```
    What is the purpose of the first LSTM? Why not just use the embeddings of the words directly?
    
 2. When generating the captions, to predict the next word, the current sequences is padded with zeroes (see `generate_captions` in `test_model.py`):
    ```python
    partial_caption = sequence.pad_sequences([caption[0]], maxlen=cg.max_cap_len, padding='post')
    ```
    Is that necessary?
    
 3. To create the captions, the probabilities of the words are added together in this implementation (see `generate_captions` in `test_model.py`)::
    ```python
    new_partial_caption_prob+=next_words_pred[word]
    ```
    Shouldn't the probablilities be multiplied together? Adding them would only make sense if we compute the log probability of the sentences but that doesn't seem to be the case here.
    