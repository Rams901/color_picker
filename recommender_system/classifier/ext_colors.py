import requests
import io
import os
import pandas as pd
import tensorflow as tf
import extcolors
from rembg import remove
from rembg.session_factory import new_session
from PIL import Image
import numpy as np
import shutil
from sklearn import preprocessing
import pickle
from google.cloud import storage
import os

fixed_colors = ['Black', 'Black & White / Tuxedo', 'Blue Cream', 'Blue Point',
       'Brown / Chocolate', 'Buff & White', 'Buff / Tan / Fawn', 'Calico',
       'Chocolate Point', 'Cream / Ivory', 'Cream Point', 'Dilute Calico',
       'Dilute Tortoiseshell', 'Flame Point', 'Gray & White',
       'Gray / Blue / Silver', 'Leopard / Spotted', 'Lilac Point',
       'Orange & White', 'Orange / Red', 'Seal Point', 'Smoke',
       'Tiger Striped', 'Torbie', 'Tortoiseshell', 'White']
learning_rate = 0.09

# DEEP LEARNING
def create_model(my_learning_rate):
  """Create and compile a deep neural net."""
  
  # All models in this course are sequential.
  model = tf.keras.models.Sequential()

  # The features are stored in a two-dimensional 28X28 array. 
  # Flatten that two-dimensional array into a one-dimensional 
  # 784-element array.
  # model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

  # Define the first hidden layer.   
  # model.add(tf.keras.layers.Dense( input_shape = (13, ), units=512, activation='relu'))
  # model.add(tf.keras.layers.Dense( units=512, activation='relu'))
  # model.add(tf.keras.layers.Dropout(rate=0.15))
  model.add(tf.keras.layers.Dense(input_shape =(12, ), units=512, activation='relu'))
  model.add(tf.keras.layers.Dense( units=512, activation='relu'))
  model.add(tf.keras.layers.Dropout(rate=0.1))
  model.add(tf.keras.layers.Dense( units=512, activation='relu'))
  model.add(tf.keras.layers.Dense( units=128, activation='relu'))
  model.add(tf.keras.layers.Dropout(rate=0.15))
  
  # model.add(tf.keras.layers.Dropout(rate=0.05))
  # model.add(tf.keras.layers.Dense( units=16, activation='relu'))
  # model.add(tf.keras.layers.Dropout(rate=0.025))

  # Define a dropout regularization layer. 
  

  # Define the output layer. The units parameter is set to 10 because
  # the model must choose among 10 possible output values (representing
  # the digits from 0 to 9, inclusive).
  #
  # Don't change this layer.
  model.add(tf.keras.layers.Dense(units=26, activation='softmax'))     
                           
  # Construct the layers into a model that TensorFlow can execute.  
  # Notice that the loss function for multi-class classification
  # is different than the loss function for binary classification.  
  model.compile(optimizer=tf.keras.optimizers.Adam(my_learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
  
  return model    


def ext_color_bg_improved(x):
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "assets/test-22b1e-c81625d6f0c9.json"
  
  try:
    response = requests.get(x['url'])
    image_bytes = io.BytesIO(response.content)
    img1 = Image.open(image_bytes).resize((256, 256))
    #array /255
    img = remove(img1, session = new_session('u2netp'))
    # Upload nto firebase, grab new_url
    
    x['new_url'] = store_firebase(img)
  
  except Exception as e:
    x['rgb1'] = None
    x['rgb2'] = None
    x['rgb3'] = None
    print(e)
    return False
 
  #Ext colors
  colors_x = extcolors.extract_from_image(img, tolerance = 12, limit = 4)
  colors_pre_list = str(colors_x).replace('([(','').split(', (')[0:-1]
  df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
  df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]
  df_color_up = [(int(i.replace('[', '').replace(']','').split(", ")[0].replace("(","")),int(i.split(", ")[1]), int(i.split(", ")[2].replace(")",""))) for i in df_rgb]
  
  # if(i[0] == ('[')):
  #   i = i[1:-1]
  # # print(i)
  # return [int(i.split(", ")[0].replace("(","")),int(i.split(", ")[1]), int(i.split(", ")[2].replace(")",""))]

  if len(df_color_up) == 3:
    v = []
    # x['rgb1'] = df_color_up[0]
    # x['q1'] = df_percent[0]
    # # x['rgb2'] = df_color_up[1]
    # x['q2'] = df_percent[1]
    # # x['rgb3'] = df_color_up[2]
    # x['q3'] = df_percent[2]
    # x["breeds"] = pd.Categorical(x['breeds'])
    
    for j in range(3):
      i = df_color_up[j]
      v.append(int(df_percent[j]))
      v.append(int(i[0]))
      v.append(int(i[1]))
      v.append(int(i[2]))
      # v[f'q{j+1}'] = df_percent[j]
      # v[f'r{j}'] = i[0]
      # v[f'g{j}'] = i[1]
      # v[f'b{j}'] = i[2]
    x['features'] = v
    # X = x[['r1', 'g1', 'b1','r2', 'g2', 'b2', 'r3', 'g3', 'b3', 'q1', 'q2', 'q3']].values
    # X = X.reshape(1, -1)
    # X= preprocessing.StandardScaler().fit(X).transform(X)
    
    
    # pred = color_model.predict(X)

    # preds = np.argsort(pred)[0][-3:]
    # print(preds)
    # x['suggested_colors'] = [colors[i] for i in preds]
  # else:
  #   x['rgb1'] = df_color_up[0]
  #   x['q1'] = df_percent[0]
  else:
        return False
  return x  

#Fix storing to firebase
def store_firebase(response):
        # try:
        #     app = firebase_admin.get_app()

        # except ValueError as e:
      # print(app)
      # try:
      #   cred = credentials.Certificate("app/assets/test-22b1e-c81625d6f0c9.json")
      #   firebase_admin.initialize_app(cred)
      # except Exception as e:
      #   print(e)ù
      #   app = firebase_admin.get_app()
      os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "assets/test-22b1e-c81625d6f0c9.json"

      storage_client = storage.Client()
      img_byte_arr = io.BytesIO()
      response.save(img_byte_arr, format='PNG')
      img_byte_arr = img_byte_arr.getvalue()
      bucket = storage_client.get_bucket('test-22b1e.appspot.com')
      # bucket = storage.bucket('test-2e207.appspot.com')
      blob = bucket.blob('test_pets.jpg')
      # Turn image into bytes

      #Randomly generate name for now or get it from front end
      blob.upload_from_string(img_byte_arr, content_type = "image")
      blob.make_public()
      print("pet", blob.public_url)
      return blob.public_url

class Ext_Model():

    def __init__(self):
        
        

        self.colors =['Black', 'Black & White / Tuxedo', 'Blue Cream', 'Blue Point',
       'Brown / Chocolate', 'Buff & White', 'Buff / Tan / Fawn', 'Calico',
       'Chocolate Point', 'Cream / Ivory', 'Cream Point', 'Dilute Calico',
       'Dilute Tortoiseshell', 'Flame Point', 'Gray & White',
       'Gray / Blue / Silver', 'Leopard / Spotted', 'Lilac Point',
       'Orange & White', 'Orange / Red', 'Seal Point', 'Smoke',
       'Tiger Striped', 'Torbie', 'Tortoiseshell', 'White']

        ROOT_DIR = os.path.expanduser('~')
        print(ROOT_DIR)
        #Find an optimized solution here, must not reiterate through this everytime
        #declaration as well unefficient
        # shutil doesn't give an error, it may recopy again and again.. 
        # We'll have to find a solution later on..
        #TO-DO: Save an absolute variable that will always save the latest update. Copying is a one-time and should not be executed more.
        turn_root = True 
        self.turn_root = turn_root
        if (self.turn_root):
            try:
              os.mkdir(ROOT_DIR+'\\.u2net')
            except:
              pass
            shutil.copy('assets/u2netp.onnx', ROOT_DIR+'\\.u2net\\u2netp.onnx')
            self.turn_root = False

        # unet = 
        learning_rate = 0.009
        color_model = create_model(learning_rate)
        color_model.build((12, ))
        color_model.load_weights('assets/final_classifier_v2.h5')
        self.color_model = color_model
        sc= pickle.load( open('assets/scaler.pkl','rb'))
        self.scaler = sc

        # try:
        #     app = firebase_admin.get_app()

        # except ValueError as e:
        #     cred = credentials.Certificate("assets/test-22b1e-c81625d6f0c9.json")
        #     firebase_admin.initialize_app(cred)


    def predict(self, url):
        
        x = {'url': url}
        x = ext_color_bg_improved(x)
       
        if not x:
            return "Please retry again. There was a problem."
        print(np.expand_dims(np.array(x['features']), axis = 0))
        v = np.expand_dims(np.array(x['features']), axis = 0)
        print(v)

        X= self.scaler.transform(v)
        X.shape
        preds = np.argsort(self.color_model.predict(X))[0, -3:]
        print(preds)
        
        colors = [self.colors[i] for i in preds]
        print(colors)
        
        return x['new_url'], colors

model = Ext_Model()

def get_model_c():
    return model

        #grab the predicted 
# def final_ext_colors(x):  
#     x = ext_color_bg_improved(x)
#     if not x:
#         return "Please retry again. There was a problem."
#     model = create_model(learning_rate)
#     model.build((12, ))
#     model.load_weights('final_classifier_v2.h5')



