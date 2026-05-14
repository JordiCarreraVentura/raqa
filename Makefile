# Environment variables
# ----------------------

SHELL = /bin/bash

# Get current git branch name
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

# The name of your expected environment folder (based on branch)
VENV_NAME := environments/$(GIT_BRANCH)
PY := $(VENV_NAME)/bin/python

# Load environment variables from .env
ifneq ("$(wildcard .env)","")
    include .env
    export $(shell sed 's/=.*//' .env)
endif

setup:
	for folder in environments data ; \
		do if [ ! -e $$folder ] ; then mkdir $$folder ; fi ; \
	done ;

env: setup
	if [ -e $(VENV_NAME) ] ; then rm -r $(VENV_NAME) ; fi
	virtualenv $(VENV_NAME) ; \
	$(MAKE) install

install:
	$(PY) -m pip install -e .

run:
	$(PY) -m raqa

sync-data:
	@mkdir -p src/raqa/_sample
	@cp data/* src/raqa/_sample/

build: sync-data
	@echo "Cleaning old builds..."
	rm -rf dist/ build/ *.egg-info src/*.egg-info
	@echo "Building package..."
	$(PY) -m build

push:
	$(PY) -m twine upload --repository pypi dist/* -u __token__ -p $(PYPI_TOKEN)

publish: build push

clean:
	rm -rf __pycache__ src/raqa/__pycache__ .pytest_cache
	rm -rf dist/ build/ *.egg-info src/*.egg-info

all: env install run

full: clean all