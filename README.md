#Deep Autoencoder with TensorFlow

##Setup
It is expected that Python2.7 is installed and your default python version.
###Ubuntu/Linux
```bash
$ git clone https://github.com/cmgreen210/TensorFlowDeepAutoencoder
$ cd TensorFlowDeepAutoencoder
$ sudo chmod +x setup_linux
$ sudo ./setup_linux  # If you want GPU version specify -g or --gpu
$ source venv/bin/activate 
```
###Mac OS X
```bash
$ git clone https://github.com/cmgreen210/TensorFlowDeepAutoencoder
$ cd TensorFlowDeepAutoencoder
$ sudo chmod +x setup_mac
$ sudo ./setup_mac
$ source venv/bin/activate 
```
##Run
To run the default example execute the following command. 
NOTE: this will take a very long time if you are running on a CPU as opposed to a GPU
```bash
$ python code/run.py
```
Navigate to [http://localhost:6006](http://localhost:6006) to explore TensorBoard and view the training progress.
