import dill

# load the pickled model
clf = dill.load(open('iris.pkl', 'r'))

def iris_predict(data):
    # return predictions, transform to list so jsonify works
    # numpy arrays cannot be jsonified
    return clf.predict(data).tolist()
