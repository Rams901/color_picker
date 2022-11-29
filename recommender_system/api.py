from typing import Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel
import os


# from .classifier.extract_colors import Ext_Model, get_model_ext
from typing import List
print(os.getcwd())
from recommender_system.classifier.recommender_classifier import Rec_Model, get_model
from recommender_system.classifier.ext_colors import Ext_Model, get_model_c
app = FastAPI()

class RecommendRequest(BaseModel):
    id: str

class ColorRequest(BaseModel):
    url: str

class RecommendResponse(BaseModel):
    top_5_ids: List 
    top_5: List

class ColorResponse(BaseModel):
    colors: List
    new_url: str

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest, model: Rec_Model = Depends(get_model)):
    
    pets, ids = model.recommend(int(request.id))
    
    return RecommendResponse(
       top_5 = pets.values.tolist(),

        top_5_ids = ids.values.tolist()
    )

@app.post("/identify_colors", response_model=ColorResponse)
def recommend(request: ColorRequest, model: Ext_Model = Depends(get_model_c)):
    
    new_url, colors = model.predict(request.url)
    
    return ColorResponse(
       new_url = new_url,
        colors = colors
    )

# @app.post("/color", response_model=ColorResponse)
# def color(request: ColorRequest, model: Ext_Model = Depends(get_model_ext)):
#     colors = model.predict(request.url)
    
#     return ColorResponse(
#        colors = colors
#     )






#LATEST GET MODEL WORKING.

#Currently at a cross road. Should we return ids only or photo included?
# @app.post("/recommend", response_model=RecommendResponse)
# def predict(request: RecommendRequest, model: Model = Depends(get_model)):
#     top_5_ids, top_5 = model.predict(request.text)
#     return RecommendResponse(
#         top_5_ids=top_5_ids, top_5 = top_5
#     )

# reformulate types to be compatible, add dataset try sample calling to model
# putting dataset as asset and implementing the model. Might need libs in pipfile
