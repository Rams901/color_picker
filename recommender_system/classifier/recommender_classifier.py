import json
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class Rec_Model():
    def __init__(self):
        
        # data = pd.read_csv('app/assets/final_cats.csv')
        data = pd.read_csv('assets/final_cats.csv')
        self.data = data
        
    def recommend(self, id):
        data = self.data
        # To be optimized, try putting the mx over in the __init__ or at least putting the y,x.. It will be costly in memory
        # But way faster..
        #We didn't yet integrate dogs and other pets.
        #Adding the specie weight is far more important than the recommendations themselves..
        y = data[data.id == id].drop(columns = ['breeds', 'colors', 'coat', 'id','published_at', 'Unnamed: 0', 'size'])
        
        X = data.drop(columns = ['breeds', 'colors', 'coat', 'id','published_at', 'Unnamed: 0', 'size'])
        
        mx = cosine_similarity(X, y)
        #How to extract ids
        top_5_pets = data.iloc[np.argsort(mx.reshape(1, -1))[0, -6:-1]][['id', 'breeds', 'colors', 'size', 'coat']]
        
        top_5_ids = top_5_pets[['id']]
        
        return (top_5_pets, top_5_ids)

model = Rec_Model()
        
def get_model():
    return model












    # init
    #     choose right data, prepare matrix
    # predict
    #     identify id, return top 5 results

# result=cosine_similarity(features)
# np.argsort(result[id])
# 