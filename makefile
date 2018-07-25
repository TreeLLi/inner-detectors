# main program
match:
	python -W ignore 'src/indr_matcher.py'

# preparation
data:
	python -W ignore 'datasets/prepare_datasets.py'

# test
testloader:
	python -W ignore 'test/data_loader_test.py' $(m)

testconfig:
	python -W ignore 'test/config_test.py' $(m)

testreflector:
	python -W ignore 'test/reflector_test.py' $(m)

testmatcher:
	python -W ignore 'test/matcher_test.py' $(m)

testmodel:
	python -W ignore 'test/model_test.py' $(m)

testplotter:
	python -W ignore 'test/plotter_test.py' $(m)

cleancache:
	rm -fr */__pycache__ */*/__pycache__

cleanmaps:
	rm -f datasets/class_maps.txt datasets/img_cls_maps.txt datasets/annos_statistics.txt

cleanresults:
	rm -f output/vgg16/*.txt output/vgg16/vgg16_concept_matches.pkl output/vgg16/vgg16_unit_matches.pkl
