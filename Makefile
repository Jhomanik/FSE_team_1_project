all: hello

hello:
	g++ hello.cpp -o hello

setup:
	pip install -r requirements.txt

run_python:
	python main.py

test_python:
	python tests.py
