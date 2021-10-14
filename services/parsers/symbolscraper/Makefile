ifdef TAG
	optional_tag = -$(TAG)
endif

image := mmda-symbolscraper$(optional_tag)
mkfile_dir := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

build-image:
	cd $(mkfile_dir)/../../.. &&\
		docker build -t $(image) . -f services/parsers/symbolscraper/Dockerfile

serve: build-image
	docker run --rm \
		-p 8080:8080 \
		--env-file $(shell pwd)/docker.env \
		-v $(shell pwd)/artifacts:/opt/ml/model:ro \
		$(image) serve

type-check: build-image
	docker run --rm --entrypoint /bin/bash $(image) -c 'mypy .'

format-check: build-image
	docker run --rm --entrypoint /bin/bash $(image) -c 'black --check .'

format: build-image
	docker run --rm -v $(shell pwd):/opt/ml/code --entrypoint bash $(image) -c 'black .'

unit: build-image
	docker run --rm --entrypoint /bin/bash $(image) -c 'pytest tests'
