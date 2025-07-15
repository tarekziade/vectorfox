PYTHON=python3

VENV=.venv
REQS=requirements.txt
SCRIPT=index_firefox_docs.py
COLLECTION_INIT_SCRIPT=init_qdrant_collection.py

QDRANT_CONTAINER_NAME=qdrant
QDRANT_PORT=6333

.PHONY: all setup run clean qdrant-up init-collection

all: setup

setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r $(REQS)

run:
	. $(VENV)/bin/activate && $(PYTHON) $(SCRIPT)

qdrant-up:
	docker run -d --rm \
		--name $(QDRANT_CONTAINER_NAME) \
		-p $(QDRANT_PORT):6333 \
		-p 6334:6334 \
		qdrant/qdrant

init-collection:
	. $(VENV)/bin/activate && $(PYTHON) $(COLLECTION_INIT_SCRIPT)

clean:
	rm -rf __pycache__ .cache $(VENV)
	docker stop $(QDRANT_CONTAINER_NAME) || true

