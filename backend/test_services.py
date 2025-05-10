from services import *

def test_predict_ANN():
    fakedata = [42,1,2,130,180,0,1,150,0,0,2,0,2]
    result = predict_ANN(fakedata)
    assert result == 1, f'expected 1 got: {result}'