testloader:
	python 'test/data_loader_test.py' $(m)

testmodel:
	python 'test/model_test.py' $(m)

clean:
	rm -fr */__pycache__ */*/__pycache__
