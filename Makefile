
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
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r $(REQS)

run:
	. $(VENV)/bin/activate && $(PYTHON) $(SCRIPT)

qdrant-up:
	docker run -d --rm \
		--name $(QDRANT_CONTAINER_NAME) \
		-p $(QDRANT_PORT):6333 \
		-p 6334:6334 \
		qdrant/qdrant

search:
	. $(VENV)/bin/activate && $(PYTHON) cli.py "$(q)"

clean:
	rm -rf __pycache__ .cache $(VENV)
	docker stop $(QDRANT_CONTAINER_NAME) || true

