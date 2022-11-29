# Create class to initate the bucket storage and only call the store image
# Try uploading to the server no_bg img 
import firebase_admin
from firebase_admin import credentials, initialize_app, storage
from io import BytesIO
class firebase_store:
    def __init__(self):

        try:
            app = firebase_admin.get_app()
        except ValueError as e:
            cred = credentials.Certificate("assets/test-22b1e-c81625d6f0c9.json")
            firebase_admin.initialize_app(cred)

    def store_firebase(self, response):
        bucket = storage.bucket('test-2e207.appspot.com')
        blob = bucket.blob(x['headline'])
        blob.upload_from_string(response, content_type='image/jpeg')
        blob.make_public()
        print("roli", blob.public_url)