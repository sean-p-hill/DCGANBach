import os
import pickle
import sys

"""
This file is used to create .pkl files of parameters, so that different sets of 
parameters can be easilly interchanged in experimentation through the command line.
"""

params = {'input_data'      : '12keys/Polyphonic',
          'batch_size'      : 16, 
          'kernel_size'     : 4,
          'G_learning_rate' : 0.001,
          'D_learning_rate' : 0.0001,  
          'beta1'           : 0.5, 
          'beta2'           : 0.9, 
          'leaky_alpha'     : 0.2,  
          'real_label'      : 0.9,  
          'fake_label'      : 0,
          'loss'            : 'BCE',
          'gen_updates'     : 1,
          'mini_batch'      : True}  



file = sys.argv[1]

# Write new params from dictionary about
if '-w' in sys.argv:
	pfn = open(file, "wb")
	pickle.dump(params, pfn)
	pfn.close()

# Read parameters only
pfn = open(file, "rb")
params = pickle.load(pfn)
pfn.close()

# Modify certain parameters
if '-m' in sys.argv:
     # Modifications:
     # del params['ngpu']
     params['loss'] = 'BCE'
     params['gen_updates'] = 1

     # Modifying dict on file
     pfn = open(file, "wb")
     pickle.dump(params, pfn)
     pfn.close()

# Print Parameters
print(f'\n{file} Parameters:')
for k,v in params.items(): print('{}: {}'.format(k,v))
print(len(params))


