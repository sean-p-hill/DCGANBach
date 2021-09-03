# Generating Bach Chorales with Deep Convlutional Generative Adversarial Networks

## Usage

python main.py gpu_num, input_data, num_epochs, params

gpu_num    : Number corresponding to the GPU device to use
input_data : Path of the images to be used in training
num_epochs : Number of Epochs to Run Training for
params     : Path corresponding to the Parameter .pkl file to use in training

E.G.
To run the training on GPU device 0, with the C Major/Minor dataset, for 1000 epochs, using the baseline model parameters:

python main.py 0, 'CMCm/Polyphonic', 1000, 'params_baseline.pkl'
