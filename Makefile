PYTHON ?= python
NOSETESTS ?= nosetests

build:
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf build *.so *.pyc

test:
	$(NOSETESTS)
