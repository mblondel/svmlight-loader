PYTHON ?= python
NOSETESTS ?= nosetests

build: _svmlight_loader.so

_svmlight_loader.so: _svmlight_loader.cpp
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf build *.so *.pyc

test:
	$(NOSETESTS)
