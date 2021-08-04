# NLP
### The School of Computer Science & Engineering at The Hebrew University of Jerusalem
This exercise deals with comparing 3 different models for a simple sentiment analysis task, implemented using using PyTorch to train and test all 3 models.
The dataset:  
- The Sentiment Treebank dataset by Stanford.  
*This dataset consists of sentences, taken from movie reviews, and their sentiment value.*

Models Description:  
- Simple log linear model:  
This model use a simple one-hot embedding for the words in order to perform the task.
The input to the model is the average over all the one-hot embeddings of the words in the
sentence. After receiving the average embedding, this model operates a single linear layer (fully-connected layer), followed by a
sigmoid in order to predict 𝑝(𝑦 = 𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒|𝑥).  

- Word2Vec log linear model:
This model is almost identical to the simple log-linear model, except it uses pre-trained Word2Vec embeddings instead of a simple one-hot embedding.
The input to the model is the average over all the Word2Vec embeddings of the words in the sentence. After receiving the average embedding, this model operates a single linear layer, followed by a sigmoid in order to predict the correct sentiment.  

- LSTM model:
This model allows us to learn long-term dependencies in our data with a lesser risk of vanishing gradients.
This model architecture is bidirectional LSTM, where each LSTM cell receives as input, the Word2Vec embedding of a word in the input sentence.
The two hidden states of the LSTM layer (the last hidden state of both directions of the bi-LSTM layer) – are concatenated.
Finally, this concatenation goes through a linear layer and output the sigmoid of the result (representing again 𝑝(𝑦 = 𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒|𝑥))  
![image](https://user-images.githubusercontent.com/83977654/128139398-1c7ff74f-830d-48a7-b63e-1e68fbca144a.png)
