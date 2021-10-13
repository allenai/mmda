PROJECT_NAME=$(shell cat PROJECT_NAME.txt)

ifdef TAG
	OPTIONAL_TAG = -$(TAG)
endif

IMAGE_NAME=$(PROJECT_NAME)$(OPTIONAL_TAG)

build-image: model-weights
	docker build -f Dockerfile -t $(IMAGE_NAME) ../../..

serve: build-image
	docker run --rm \
		-p 8080:8080 \
		--name $(IMAGE_NAME) \
		--env-file $(shell pwd)/docker.env \
		-v $(shell pwd):/opt/ml/code \
		-v $(shell cd ../../.. && pwd)/mmda:/opt/ml/code/mmda \
		$(IMAGE_NAME) serve

serve-background: build-image
	docker run -d --rm \
		-p 8080:8080 \
		--name $(IMAGE_NAME) \
		--env-file $(shell pwd)/docker.env \
		-v $(shell pwd):/opt/ml/code \
		-v $(shell cd ../../.. && pwd)/mmda:/opt/ml/code/mmda \
		-v $(shell pwd)/model-weights:/opt/ml/model:ro \
		$(IMAGE_NAME) serve

type-check: build-image
	docker run --rm --entrypoint /bin/bash $(IMAGE_NAME) -c 'mypy --exclude "mmda/.*" --follow-imports skip --ignore-missing-imports .'

format-check: build-image
	docker run --rm --entrypoint /bin/bash $(IMAGE_NAME) -c 'black --exclude "mmda/.*" --check .'

reformat: build-image
	docker run -v $(shell pwd):/opt/ml/code --rm --entrypoint /bin/bash $(IMAGE_NAME) -c 'black --exclude "mmda/.*" .'

format: build-image
	docker run --rm -v $(shell pwd):/opt/ml/code --entrypoint bash $(IMAGE_NAME) -c 'black .'

model-weights:
#   Grotoap layoutLM
#	aws s3 cp --recursive s3://ai2-s2-research/scienceparseplus/models/grotoap/baseline-long-run/microsoft-layoutlm-base-uncased/ model-weights
#	Docbank layoutLM
#	aws s3 cp --recursive s3://ai2-s2-research/scienceparseplus/models/docbank/baseline-long-run/microsoft-layoutlm-base-uncased/ model-weights
#   HVILA S2Hard
	aws s3 cp --recursive s3://ai2-s2-research/scienceparseplus/temporary-experiment-results/s2-hard-visionline-few-shot-cv-v2/hierarchical_model-block/weak-strong-layoutlm-average-first/70/0/ model-weights

clean-artifacts:
	rm -rf model-weights

kill-server:
	docker kill $(IMAGE_NAME)

integration-tests: build-image
	docker run --network host --rm \
	--env-file docker.env \
	--entrypoint /bin/bash \
	$(IMAGE_NAME) -c 'pytest integration_tests'

