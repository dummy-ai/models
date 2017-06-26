FROM dummyai/py3-tf-cpu

RUN apt-get install -y protobuf-compiler python-pil python-lxml \
    && pip install jupyter matplotlib Pillow

COPY . /app/object_detection

ENV PYTHONPATH $PYTHONPATH:/app:/app/object_detection/slim

RUN cd /app \
    && protoc object_detection/protos/*.proto --python_out=. \
    && cd /app/object_detection \
    && python download_model.py

# install webserver
RUN pip install flask \
    && apt-get install -y htop tmux

# create symlinks
RUN ln -fs /usr/bin/python3 /usr/bin/python \
    && ln -fs /usr/bin/pip3 /usr/bin/pip

WORKDIR /app/object_detection
CMD python serve_model.py
    
