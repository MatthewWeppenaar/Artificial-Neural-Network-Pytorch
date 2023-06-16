install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt

venv :
	test -d venv || virtualenv -p python3 venv

clean:
	rm -rf venv
	find -iname "*.pyc" -delete

run:
	python3 MLP.py -save
	python3 MLP.py -load

run2:
	python3 CNN.py -save
	python3 CNN.py -load

run3:
	python3 RESNET.py -save

run4:
	python3 RESNET.py -load
