import requests
import base64


with open('test_images/image1.jpg', 'rb') as f:
    result = requests.post('http://localhost:5900/encode', json={
        'image': base64.b64encode(f.read()).decode('utf-8'),
        'ext': 'jpg'
    }).json()

    image_binary = base64.b64decode(result['vis'])
    with open('output.png', 'wb') as f:
        f.write(image_binary)

