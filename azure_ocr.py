import os
import time
import pickle
import argparse
import requests

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--images_path', type=str, required=True)

# Replace <Subscription Key> with your valid subscription key.
# subscription_key = "<Subscription Key>"
parser.add_argument('-k', '--subscription_key', type=str, required=True)
args = parser.parse_args()
globals().update(vars(args))

images_names = [os.path.join(images_path, img_name) for img_name in os.listdir(images_path)]

# You must use the same region in your REST call as you used to get your
# subscription keys. For example, if you got your subscription keys from
# westus, replace "westcentralus" in the URI below with "westus".
#
# Free trial subscription keys are generated in the "westus" region.
# If you use a free trial subscription key, you shouldn't need to change
# this region.
#vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"
vision_base_url = "https://westeurope.api.cognitive.microsoft.com/vision/v2.0/"
text_recognition_url = vision_base_url + "recognizeText"

headers = {'Ocp-Apim-Subscription-Key': subscription_key}
# Note: The request parameter changed for APIv2.
# For APIv1, it is 'handwriting': 'true'.
params  = {'mode': 'Handwritten'}

# Set image_url to the URL of an image that you want to analyze.
headers['Content-Type'] = 'application/octet-stream'

transcription, exceptions = {}, {}

for image_path in tqdm(images_names):
    
    try:
        image_data = open(image_path, "rb").read()
        response = requests.post(
            text_recognition_url, headers=headers, params=params, data=image_data)
        response.raise_for_status()

        operation_url = response.headers["Operation-Location"]

        # The recognized text isn't immediately available, so poll to wait for completion.
        analysis = {}
        poll = True
        while (poll):
            response_final = requests.get(
                response.headers["Operation-Location"], headers=headers)
            analysis = response_final.json()
            time.sleep(1)
            if ("recognitionResult" in analysis):
                poll= False 
            if ("status" in analysis and analysis['status'] == 'Failed'):
                poll= False
        transcription[image_path] = analysis
    except Exception as e:
        print(e)
        exceptions[image_path] = e
        pass
    time.sleep(60)

pickle.dump(transcription, open("./flipchart_photos_transcription.pickle.dat", "wb"))
pickle.dump(exceptions, open("./flipchart_photos_exceptions.pickle.dat", "wb"))