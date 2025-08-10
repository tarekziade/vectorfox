
PYTHON=python3

VENV=.venv
REQS=requirements.txt
SCRIPT=index.py
QDRANT_CONTAINER_NAME=qdrant
QDRANT_PORT=6333

.PHONY: all setup run clean qdrant-up 

all: setup

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip 
	$(VENV)/bin/pip install -r $(REQS)
	$(VENV)/bin/pip install -e .
  
index:
	$(VENV)/bin/vectorfox-index

run:
	$(VENV)/bin/vectorweb

qdrant-up:
	docker run -d --rm \
		--name $(QDRANT_CONTAINER_NAME) \
		-p $(QDRANT_PORT):6333 \
		-p 6334:6334 \
		qdrant/qdrant

search:
	$(VENV)/bin/vectorfox "$(q)"

clean:
	rm -rf __pycache__ .cache $(VENV)
	docker stop $(QDRANT_CONTAINER_NAME) || true

