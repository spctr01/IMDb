import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv('imbd.csv')

    #map positive = 1 & negative = 0
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == 'positive' else 0)

    #new column kfold and fill with -1
    df['kfold'] = -1

    #randomize the rows of data
    df =  df.sample(frac =1).reset_index(drop = True)

    #fetch labels
    y = df.sentiment.values

    #intialize kfold class
    kf = model_Selection.StratifiedKFold(n_splits = 5)

    #fill new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    #new csv with kfold column
    df.to_csv('imbd_folds.csv', index = False)