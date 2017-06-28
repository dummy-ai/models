import requests
import base64

#URL = 'http://localhost:5900'
URL = 'http://35.190.30.217/tensorflow/object-detection'
#URL = 'http://35.186.196.237'

with open('test_images/image1.jpg', 'rb') as f:
    result = requests.post(URL, json={
        'image': base64.b64encode(f.read()).decode('utf-8'),
        'ext': 'jpg'
    }).text
    print(result)

    image_binary = base64.b64decode(result['vis'])
    with open('output.png', 'wb') as f:
        f.write(image_binary)

