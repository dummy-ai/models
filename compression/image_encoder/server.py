from flask import Flask, request, jsonify
import os
import subprocess
import uuid
import base64

app = Flask(__name__)

@app.route('/encode', methods=['POST'])
def encode():
    data = request.json

    uid = str(uuid.uuid4())[:10]
    if not os.path.exists('encoder/input'): os.makedirs('encoder/input')
    if not os.path.exists('encoder/output'): os.makedirs('encoder/output')
    image = base64.b64decode(data['image'])

    # TODO: what if users create too many files?
    input_path = 'encoder/input/{}.{}'.format(uid, data['ext'])
    output_path = 'encoder/output/{}.pkl'.format(uid)

    with open(input_path, 'wb') as f:
        f.write(image)

    print('running encoder')
    subprocess.check_output('python encoder.py --input_image={input_path} \
                            --iteration={iteration} \
                            --output_codes={output_path} \
                            --model=compression_residual_gru/residual_gru.pb'
                            .format(input_path=input_path, output_path=output_path,
                                    iteration=data['iteration']),
                            shell=True)

    print('getting output')
    with open(output_path, 'rb') as f:
        result = f.read()

    return jsonify({'data': base64.b64encode(result)})


if __name__ == '__main__':
    app.run(debug=True, port=5900)
