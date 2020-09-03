import io
import torch
import tensorflow as tf
from sklearn import matrics

import config
import dataset
import engine
import lstm

def load_vector(fname):
    fin = io.open(fname, 'r', encoding = 'utf-8', newline='\n', errors= 'ignore')
    n, d = map(int, fin.readline().split())

    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, token[1:]))

    return data


def create_embedding_matrix(word_index, embedding_dict):
    '''
    This function creates the embedding matrix.
    :param word_index: a dictionary with word:index_value
    :param embedding_dict: a dictionary with word:embedding_vector
    :return: a numpy array with embedding vectors for all known words
    '''
    #intialize matrix with 0's
    embedding_matrix = np.zeros((len(word_index)+1, 300))
    
    #loop over all words
    for word, i in word_index.items():
        #if word found in pre-trained embeddings
        #update the matrix
        #else vector is zeros
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]

        return embedding_matrix



def run(df, fold):
    '''
    Run training and validation for given fold and dataset
    :param df: pandas dataframe with kfold column
    :param fold: current fold, int
    '''
    #fetching training dataframe
    train_df = df[df.kfold != fold].reset_index(drop= True)
    #fetch validation dataframe
    valid_df = df[df.kfold == fold].reset_index(drop= True)
    

    print('Fitting Tokenizer')
    #Using tf.keras for tokenization
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.reviews.values.tolist())

    # convert training and validation data to sequences
    # for example : "bad movie" gets converted to
    # [24, 27] where 24 is the index for bad and 27 is  index for movie
    xtrain = tokenizer.text_to_sequences(train.df.reviews.values)
    xtest = tokenizer.texts_to_sequences(valid_df.reviews.values)


    # zero pad the training  & validation sequences given the maximum length
    # done on left hand side
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=config.MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=config.MAX_LEN)



    #intialize datset class for training
    train_Dataset = dataset.IMDDataset(reviews=xtrain, target= train_df.sentiment.values)
    # dataloader for training
    trian_data_loader = torch.utils.data.DataLoader(tain_dataset,
                                            batch_size=config.TRAIN_BATCH_SIZE,
                                            num_workers = 2)

    
    valid_Dataset = dataset.IMDDataset(reviews=xtest, target= valid_df.sentiment.values)
    # dataloader for training
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                            batch_size=config.VALID_BATCH_SIZE,
                                            num_workers = 2)


    print('Loading Embeddings')
    #load embedding
    embedding_dict = load_vectors('crawl-300d-2M.vec')
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict)


    device = torch.device('cuda')
    model = lstm.LSTM(embedding_matrix)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)


    print('Training Model')
    #set best accuracy & early stoping ouenter to 0
    best_accuracy = 0
    early_stopping_counter = 0

    #train and validate for all epochs
    for epoch in range(config.EPOCHS):
        #train & validate one epoch
        engine.train(train_data_loader, model, optimizer, device)
        outputs, targets = engine.evaluate(valid_data_loader, model, device)


        #use threshold 0.5 not sigmoid
        output = np.array(outputs)>=0.5

        #calculate accuracy
        accuracy = metrics.accuracy_score(target, outputs)
        print('Fold:{}, Epoch:{}, Accuracy Score:{}'.format(fold, epoch, accuracy))

        #simple early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1

        if early_stopping_counter > 2:
            break


if __name__ = "__main__":
    
    #load data
    df = pd.read_csv('imdb_folds.csv')

    run(df, fold= 0)
    run(df, fold= 1)
    run(df, fold= 2)
    run(df, fold= 3)
    run(df, fold= 4)
        


















