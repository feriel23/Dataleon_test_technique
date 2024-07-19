install:
	pip install -r requirements.txt

test:
	pytest unit_tests

all: install test
