import requests
import base64


with open('lenna.png', 'rb') as f:
    requests.post('http://localhost:5900/encode', json={
        'iteration': 1,
        'image': base64.b64encode(f.read()),
        'ext': 'png'
    })
