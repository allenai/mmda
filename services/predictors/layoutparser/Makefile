PROJECT_NAME=$(shell cat PROJECT_NAME.txt)

ifdef TAG
	OPTIONAL_TAG = -$(TAG)
endif

DEV_IMAGE_NAME=$(PROJECT_NAME)-dev$(OPTIONAL_TAG)
SERVER_IMAGE_NAME=$(PROJECT_NAME)-server$(OPTIONAL_TAG)

dev-image:
	docker build -f Dockerfile --target dev -t $(DEV_IMAGE_NAME) ../../..

server-image:
	docker build \
	-f Dockerfile --target server \
	-t $(SERVER_IMAGE_NAME) ../../..

serve: server-image
	docker run --rm \
		-p 8080:8080 \
		--env-file $(shell pwd)/docker.env \
		--name $(SERVER_IMAGE_NAME) \
		$(SERVER_IMAGE_NAME) serve

background-serve: server-image
	docker run -d --rm \
		-p 8080:8080 \
		--env-file $(shell pwd)/docker.env \
		--name $(SERVER_IMAGE_NAME) \
		-v $(shell pwd)/artifacts:/opt/ml/model:ro \
		$(SERVER_IMAGE_NAME) serve

type-check: dev-image
	docker run --rm $(DEV_IMAGE_NAME) -c 'mypy .'

format-check: dev-image
	docker run --rm $(DEV_IMAGE_NAME) -c 'black --check --exclude mmda .'

format: dev-image
	docker run --rm -v $(shell pwd):/opt/ml/code $(DEV_IMAGE_NAME) -c 'black --exclude mmda .'

unit: dev-image
	docker run --rm $(DEV_IMAGE_NAME) -c 'pytest tests'

kill-server:
	docker kill $(SERVER_IMAGE_NAME)

integration-tests: dev-image
	docker run --rm \
	--env-file docker.env \
	$(DEV_IMAGE_NAME) -c 'pytest integration_tests'

