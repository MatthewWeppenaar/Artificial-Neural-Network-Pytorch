This is a project that explores the differences in efficiency and accuracy between different Artificial Neural Network architectures. These are MLP(Multi-Layered-Perceptron), CNN(Convolutional Network-LaNet5) and RESNet(Residual Network-Resnet9).

The networks are created using Python, using the PyTorch framework, and is trained on and tested on the CIFAR-10 data set

Running program: To create a virtual environment: cd to working directory in terminal Enter "make" in terminal This will also download necessary modules, using "requirements.txt"

IMPORTANT NOTE: This make file will only work when running on Ubuntu If you want to run this on Mac change:

"test -d venv || virtualenv -p python3 venv"
To:

"test -d venv || python3 -m venv venv"
Trained networks for each architecture is included(files with ".pth" extention)

To run(example):

python3 MLP.py -load
To run you must activate your virtual environment in working directory: source ./venv/bin/activate

3 example runs are included for each network:

In working directory(after environment has been created) enter

make mlp: runs MLP.py with the save flag and trains a MLP and saves the best result.

make mlp_load : run MLP.py with the load flag and loads the best result

make cnn: runs CNN.py with the save flag and trains a CNN and saves the best result.

make cnn_load : run CNN.py with the load flag and loads the best result

make resnet: runs RESNET.py with the save flag and trains a RESNET and saves the best result.

make resnet_load : run RESNET.py with the load flag and loads the best resul

Make clean: removes virtual environment