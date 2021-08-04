import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"
RARE = "rare"
NEG = "neg"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------
#
def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim=300):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    sum_vec = np.zeros((embedding_dim,))
    words_num = len(sent.text)
    for word in sent.text:
        vec = word_to_vec.get(word)
        if vec is not None:
            sum_vec += vec
        else:
            words_num -= 1
    if not words_num:
        words_num = 1
    return sum_vec / words_num


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    vec = np.zeros((size, 1))
    vec[ind] = 1
    return vec


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    size = max(word_to_ind.values()) + 1
    all_vec = np.zeros((size, 1))
    for word in sent.text:
        all_vec += get_one_hot(size, word_to_ind[word])

    return all_vec / len(sent.text)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    word2ind = {}
    for i in range(len(words_list)):
        word2ind[words_list[i]] = i
    return word2ind


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    words_list = sent.text
    col_lst = []
    if len(words_list) < seq_len:
        words_list = words_list + [None for _ in
                                   range(seq_len - len(words_list))]
    for i in range(seq_len):
        word = words_list[i]
        word_vec = word_to_vec.get(word)
        if word_vec is not None:
            col_lst.append(word_to_vec[word])
        else:
            col_lst.append(np.zeros((embedding_dim,)))
    return np.vstack(col_lst)


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True,
                 dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None, load_special=False):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path,
                                                               split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[
                TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()
        if load_special:
            rare_ind = data_loader.get_rare_words_examples(self.sentences[TEST], dataset=self.sentiment_dataset)
            neg_ind = data_loader.get_negated_polarity_examples(self.sentences[TEST])
            self.sentences[RARE] = [self.sentences[TEST][i] for i in rare_ind]
            self.sentences[NEG] = [self.sentences[TEST][i] for i in neg_ind]

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {
                "word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(
                                         words_list),
                                     "embedding_dim": embedding_dim}
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {
                "word_to_vec": create_or_load_slim_w2v(words_list),
                "embedding_dim": embedding_dim}
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {
            k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs)
            for k, sentences in self.sentences.items()}
        self.torch_iterators = {
            k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
            for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array(
            [sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                            dropout=dropout, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(in_features=2 * hidden_dim, out_features=1)
        self.sigi = nn.Sigmoid()
        return

    def forward(self, text):
        out = self.lstm(text)[0]
        return self.linear(out)

    def predict(self, text):
        out = self.lstm(text)
        return self.sigi(self.linear(out))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.layer = nn.Linear(in_features=embedding_dim, out_features=1)

    def forward(self, x):
        return self.layer(x)

    def predict(self, x):
        return nn.Sigmoid()(self.layer(x))


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """

    return np.mean(preds.detach().numpy() == y.detach().numpy())


def get_batch_pred(model, x, lstm):
    if not lstm:
        x = x.reshape((x.shape[0], -1))
    pred = model(x.float())
    if not lstm:
        pred = pred[:, 0]
    else:
        pred = pred[:, 0, 0]
    return pred


def get_correct_pred_num(pred, y, sig=True):
    if sig:
        pred = nn.Sigmoid()(pred)
    pred = pred >= 0.5
    return (pred == y).sum().item()


def train_epoch(model, data_iterator, optimizer, criterion, lstm=False):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    corrects = 0
    samples = 0
    for x, y in data_iterator:
        x= x.to(device=get_available_device())
        y= y.to(device=get_available_device())
        optimizer.zero_grad()
        pred = get_batch_pred(model, x, lstm)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        samples += y.size(0)
        corrects += get_correct_pred_num(pred, y)
    return loss.item(), corrects / samples


def evaluate(model, data_iterator, criterion, lstm=False):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    corrects = 0
    samples = 0
    for x, y in data_iterator:
        x= x.to(device=get_available_device())
        y= y.to(device=get_available_device())
        pred = get_batch_pred(model, x, lstm)
        loss = criterion(pred, y)
        samples += y.size(0)
        corrects += get_correct_pred_num(pred, y)
    loss, acc = loss.item(), corrects / samples
    return loss, acc


def get_predictions_for_data(model, data_iter, lstm=False):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    all_preds = []
    for x, y in data_iter:
        pred = get_batch_pred(model, x, lstm)
        pred = nn.Sigmoid()(pred)
        pred = pred >= 0.5
        all_preds.extend(pred.detach.numpy())

    return all_preds


def train_model(model, data_manager, n_epochs, lr, weight_decay, lstm=False, name=None,gpu=False):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_iter = data_manager.get_torch_iterator(VAL)
    train_iter = data_manager.get_torch_iterator()
    train_iter = train_iter
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    model = model.float()
    model = model.to(device=get_available_device())
    for i in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_iter, optimizer,
                                            criterion, lstm)
        val_loss, val_acc = evaluate(model, val_iter, criterion, lstm)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(
            f"{name} epoch {i+1}/{n_epochs}: train loss {train_loss}, train accuracy: {train_acc},val loss {val_loss}"
            f",val accuracy: {val_acc}")
    save_model(model, name, n_epochs, optimizer)
    return train_accuracies, train_losses, val_accuracies, val_losses


def eval_on_test_set(lstm, data_type=None, path=None, dm_model=None):
    if dm_model is None:
        data_manager = DataManager(batch_size=DEFAULT_BATCH_SIZE, data_type=data_type, embedding_dim=W2V_EMBEDDING_DIM,
                                   load_special=True)
        if lstm:
            model = LSTM(W2V_EMBEDDING_DIM, hidden_dim=DEFUALT_HIDDEN, n_layers=2, dropout=0.5)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=LSTM_LT,
                                         weight_decay=LSTM_WD)
        else:
            model = LogLinear(data_manager.get_input_shape()[0])
            optimizer = torch.optim.Adam(params=model.parameters(), lr=DEFAULT_LT,
                                         weight_decay=DEFAULT_WD)
        model, optimizer, e = load(model, path, optimizer)
    else:
        data_manager, model = dm_model
    test_iter = data_manager.get_torch_iterator(TEST)
    full_loss, full_acc = evaluate(data_iterator=test_iter, model=model,criterion=nn.BCEWithLogitsLoss(), lstm=lstm)
    print(f"{path} Full test accuracy: {full_acc},Loss: {full_loss}")
    rare_iter = data_manager.get_torch_iterator(RARE)
    rare_loss, rare_acc = evaluate(data_iterator=rare_iter, model=model,criterion=nn.BCEWithLogitsLoss(), lstm=lstm)
    print(f"{path} Rare words accuracy: {rare_acc},Loss: {rare_loss}")
    neg_iter = data_manager.get_torch_iterator(NEG)
    neg_loss, neg_acc = evaluate(model,data_iterator=neg_iter,criterion=nn.BCEWithLogitsLoss(), lstm=lstm)
    print(f"{path} Negated polarity accuracy: {neg_acc},Loss: {neg_loss}")
    return full_loss, full_acc, rare_loss, rare_acc, neg_acc, neg_loss

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 20
DEFAULT_LT = 0.01
DEFAULT_WD = 0.0001


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manager = DataManager(batch_size=DEFAULT_BATCH_SIZE,load_special=True)
    print("Training Loglin with one hot repr:")
    model = LogLinear(data_manager.get_input_shape()[0])
    train_accuracies, train_losses, val_accuracies, val_losses = train_model(
        model, data_manager, DEFAULT_EPOCHS, DEFAULT_LT, DEFAULT_WD, name="one_hot")
    test_results = eval_on_test_set(False, dm_model=(data_manager, model))
    return train_accuracies, train_losses, val_accuracies, val_losses,test_results



def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    print("Training Loglin with w2v:")
    data_manager = DataManager(data_type=W2V_AVERAGE,
                               batch_size=DEFAULT_BATCH_SIZE,
                               embedding_dim=W2V_EMBEDDING_DIM,
                               load_special=True)
    print(data_manager.get_input_shape())
    model = LogLinear(data_manager.get_input_shape()[0])
    train_accuracies, train_losses, val_accuracies, val_losses = train_model(
        model, data_manager, DEFAULT_EPOCHS, DEFAULT_LT, DEFAULT_WD, name="w2v_lin")
    test_results = eval_on_test_set(False, dm_model=(data_manager, model),path="w2v_lin")
    return train_accuracies, train_losses, val_accuracies, val_losses, test_results


DEFUALT_HIDDEN = 100
LSTM_LT = 0.001
LSTM_WD = 0.0001
LSTM_EPOCHS = 4


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(data_type=W2V_SEQUENCE,
                               batch_size=DEFAULT_BATCH_SIZE,
                               embedding_dim=W2V_EMBEDDING_DIM,
                               load_special=True)
    print("Training LSTM:")
    model = LSTM(W2V_EMBEDDING_DIM, hidden_dim=DEFUALT_HIDDEN, n_layers=2, dropout=0.5)

    train_accuracies, train_losses, val_accuracies, val_losses = train_model(
        model, data_manager, LSTM_EPOCHS, LSTM_LT, LSTM_WD, lstm=True, name="lstm_mod")
    test_results = eval_on_test_set(True, dm_model=(data_manager, model),path="lstm")
    return train_accuracies, train_losses, val_accuracies, val_losses,test_results




