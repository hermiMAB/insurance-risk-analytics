from src.model.py import split_data

def test_split_data():
    X = [1, 2, 3, 4, 5]
    y = [0, 1, 0, 1, 0]
    train, test, _, _ = split_data(X, y, test_size=0.2)
    assert len(test) == 1
    assert len(train) == 4