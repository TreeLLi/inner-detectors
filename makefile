testloader:
	python 'test/data_loader_test.py' $(m)

testconfig:
	python 'test/config_test.py' $(m)

testreflector:
	python 'test/reflector_test.py' $(m)

testmatcher:
	python 'test/matcher_test.py' $(m)

testmodel:
	python 'test/model_test.py' $(m)

clean:
	rm -fr */__pycache__ */*/__pycache__
