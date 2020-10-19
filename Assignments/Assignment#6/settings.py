"""
SG = 1 : Use Skip-gram / SG = 0 : Use CBOW
HS = 1 : Hierarchical softmax / HS = 0 : Negative sampling
NEGATIVE : # of negative samples
ALPHA : Learning rate
MIN_ALPHA : Minimum learning rate when using LR decay
DIMENSION : size of the hidden layer
ITER : # of epochs
"""
settings = {
    'SET#1': {
        'SG': 0,
        'HS': 0,
        'NEGATIVE': 15,
        'ALPHA': 0.01,
        'MIN_ALPHA': 0.0001,
        'DIMENSION': 300,
        'ITER': 5,
    },
    'SET#2': {
        'SG': 1,
        'HS': 0,
        'NEGATIVE': 15,
        'ALPHA': 0.01,
        'MIN_ALPHA': 0.0001,
        'DIMENSION': 300,
        'ITER': 5,
    },
    'SET#3': {
        'SG': 0,
        'HS': 1,
        'NEGATIVE': 0,
        'ALPHA': 0.01,
        'MIN_ALPHA': 0.0001,
        'DIMENSION': 300,
        'ITER': 5,
    },
    'SET#4': {
        'SG': 1,
        'HS': 1,
        'NEGATIVE': 0,
        'ALPHA': 0.01,
        'MIN_ALPHA': 0.0001,
        'DIMENSION': 300,
        'ITER': 5,
    },
}
