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

deploy-ingress-controller:
	kubectl create configmap nginx-template --from-file=nginx.tmpl=./nginx.tmpl
	kubectl create -f ./ingress-controller.yaml

clean-ingress-controller:
	kubectl delete service nginx-ingress
	kubectl delete deploy nginx-ingress-controller
	kubectl delete configmap --all

clean-ingress:
	kubectl delete ing dummy-ing

deploy-ingress:
	kubectl create -f ingress.yaml

deploy-app:
	kubectl create -f pod.yaml
	kubectl expose deploy tf-object-detection --type=NodePort --name=tf-object-detection
	
clean-app:
	kubectl delete deploy tf-object-detection
	kubectl delete service tf-object-detection

.PHONY: build
