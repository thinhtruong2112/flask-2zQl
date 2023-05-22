from flask import Flask, request
import os
import numpy as np
from pandas import json_normalize
import requests
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/',methods = ['GET'])
def login():
    id = request.args.get('id', default = 1, type = int)
    type = request.args.get('type', default = 1, type = int)
    if type==1:
        api_url = "https://backend-production-d598.up.railway.app/api/data/product/gen"
    else:
        if type == 2:
            api_url = "https://backend-production-d598.up.railway.app/api/data/sample/gen"
        else: 
            api_url = "https://backend-production-d598.up.railway.app/api/data/real/gen"
        
    response = requests.get(api_url)
    print(response)
    data = response.json()
    ratings = json_normalize(data['rating']) 
    ratings.head()
    
    items = json_normalize(data['itemList']) 
    items.head()
    # Now, we create user-item matrix using scipy csr matrix
    
    def create_matrix(df):
        
        N = len(df['userId'].unique())
        M = len(df['itemId'].unique())
        
        # Map Ids to indices
        user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
        movie_mapper = dict(zip(np.unique(df["itemId"]), list(range(M))))
        
        # Map indices to IDs
        user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
        movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["itemId"])))
        
        user_index = [user_mapper[i] for i in df['userId']]
        movie_index = [movie_mapper[i] for i in df['itemId']]
    
        X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
        
        return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper
    
    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)
    
    def find_similar_items(movie_id, X, k, metric='cosine', show_distance=False):
        
        neighbour_ids = []
        
        movie_ind = movie_mapper[movie_id]
        movie_vec = X[movie_ind]
        k+=1
        kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
        kNN.fit(X)
        movie_vec = movie_vec.reshape(1,-1)
        neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
        for i in range(0,k):
            n = neighbour.item(i)
            neighbour_ids.append(movie_inv_mapper[n])
        neighbour_ids.pop(0)
        return neighbour_ids
    
    similar_ids = find_similar_items(id, X, k=6)
    listid = []
    for i in similar_ids:
        listid.append(int(i))
    
    return listid

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
