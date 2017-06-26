URL="dummyai/tf-object-detection"

dev:
	docker run -v `pwd`:/app/object_detection -it $(URL) bash

run:
	docker run -p 5900:5900 -it $(URL)

jupyter:
	docker run -p 8889:8889 -it $(URL) bash # jupyter notebook --allow-root --ip 0.0.0.0 --port 8889

push:
	docker push $(URL)

build:
	docker build -t $(URL):`cat VERSION` .
	docker tag $(URL):`cat VERSION` $(URL):latest

.PHONY: build
