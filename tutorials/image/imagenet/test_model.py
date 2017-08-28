import argparse
import requests
import base64
import simplejson

parser = argparse.ArgumentParser(description='Inception-v3 image classification')
parser.add_argument('--img_in',dest='img_in',help='Path to input image to be classified', type=str)
args = parser.parse_args()

URL = 'http://dev.moxel.ai/model/strin/inception-v3/latest'
# URL = 'http://localhost:5900'

with open('./cropped_panda.jpg', 'rb') as f:
    result = requests.post(URL, json={
        'img_in': base64.b64encode(f.read()).decode('utf-8')
    })

    try:
        results = result.json()
    except simplejson.scanner.JSONDecodeError:
        print('Cannot decode JSON: ')
        print(result.text)
        exit(1)

    print(results)
