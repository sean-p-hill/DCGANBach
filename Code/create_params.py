import os
import pickle
import sys


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



# files = os.listdir('Params')

# for f in files:
#      print(f)
#      try:
#           # Modifying dict on file
#           pfn = open(os.path.join('Params',f), "rb")
#           p = pickle.load(pfn)
#           pfn.close()

#           # Print Parameters
#           print(f'\n{f} Parameters:')
#           for k,v in p.items(): print('{}: {}'.format(k,v))
#           print(len(p))
#      except:
#           raise

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


