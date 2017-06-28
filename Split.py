import pandas as pd
import numpy as np

def split(dataset, num_inputs, t, v):
    # normalize dataset into range between -1 and 1

    norm = dataset.ix[:, slice(0,num_inputs)].apply(lambda x: -1 + 2 * (x - x.min()) / (x.max() - x.min()), axis=0)
    dataset = pd.concat([norm, dataset.ix[:,'window':]], axis=1)

    # split dataset by target class
    dataset0 = dataset.ix[dataset.ix[:, 'window'] == 0, :]
    dataset1 = dataset.ix[dataset.ix[:, 'window'] == 1, :]

    # build datasets by target class
    train0, validate0, test0 = np.split(dataset0.sample(frac=1),
                                        [int(t * len(dataset0)), int((1 - v) * len(dataset0))])
    train1, validate1, test1 = np.split(dataset1.sample(frac=1),
                                        [int(t * len(dataset1)), int((1 - v) * len(dataset1))])

    # re-combine by target class
    train = pd.concat([train0, train1])
    validate = pd.concat([validate0, validate1])
    test = pd.concat([test0, test1])

    # shuffle order and reset index numbers
    train = train.sample(frac=1)
    train.reset_index(inplace=True)
    validate = validate.sample(frac=1)
    validate.reset_index(inplace=True)
    test = test.sample(frac=1)
    test.reset_index(inplace=True)

    return train.iloc[:, 1:], validate.iloc[:, 1:], test.iloc[:, 1:]
