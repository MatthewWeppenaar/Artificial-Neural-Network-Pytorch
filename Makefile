install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt

venv :
	test -d venv || virtualenv -p python3 venv

clean:
	rm -rf venv
	find -iname "*.pyc" -delete

mlp:
	python3 MLP.py -save

mlp_load: 
	python3 MLP.py -load

cnn:
	python3 CNN.py -save

cnn_load:
	python3 CNN.py -load

resnet:
	python3 RESNET.py -save

resnet_load:
	python3 RESNET.py -load
