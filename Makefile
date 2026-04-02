# Environment variables
# ----------------------

SHELL = /bin/bash

# Get current git branch name
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

# The name of your expected environment folder (based on branch)
VENV_NAME := environments/$(GIT_BRANCH)
PY := $(VENV_NAME)/bin/python
PYTHON_VERSION := 3.14.3

setup:
	for folder in environments data ; \
		do if [ ! -e $$folder ] ; then mkdir $$folder ; fi ; \
	done ;

env:
	if [ -e $(VENV_NAME) ] ; then rm -r $(VENV_NAME) ; fi
	virtualenv -p $(PYTHON_VERSION) $(VENV_NAME) ; \
	$(MAKE) install

check-env:
	@if [ `which python | grep "$(VENV_NAME)" | wc -l` -eq 1 ]; then \
		echo "Virtual environment detected for branch $(GIT_BRANCH)."; \
	else \
		echo "Virtual environment not detected."; \
		echo "Please activate it by running: source $(VENV_NAME)/bin/activate"; \
		exit 1; \
	fi

install:
	$(PY) -m pip install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -f data/*

all: env install

full: clean env install