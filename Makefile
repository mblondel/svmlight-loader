build:
	python setup.py build_ext --inplace

clean:
	rm -rf build *.so *.pyc

test:
	nosetests
