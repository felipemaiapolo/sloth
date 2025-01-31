import numpy as np
import pandas as pd
from tqdm.auto import tqdm

encode = {'Llama-3-8B-Instruct':'meta-llama-3-8b-instruct',
          'Llama-3-70B-Instruct':'meta-llama-3-70b-instruct',
          'Llama-3-8B':'meta-llama-3-8b',
          'Gemma-2B':'gemma-2b', ###
          'Gemma-7B':'gemma-7b', ###
          'Pythia-160M':'pythia-160m',
          'Pythia-410M':'pythia-410m',
          'Pythia-2.8B':'pythia-2.8b',
          'Pythia-6.9B':'pythia-6.9b',
          'Pythia-12B':'pythia-12b'}

models = ['Llama-3-8B-Instruct',
          'Llama-3-8B',
          'Llama-3-70B-Instruct',
          'Gemma-2B',  
          'Gemma-7B',
          'Pythia-160M',
          'Pythia-410M',
          'Pythia-2.8B',
          'Pythia-6.9B',
          'Pythia-12B']

data = []
for model in tqdm(models):
    df = pd.read_json(f"hf://datasets/ScalingIntelligence/monkey_business/MATH_{model}.json")
    data.append(np.array([df.is_corrects[i] for i in range(df.shape[0])]).astype(int))

data = np.array(data)
data_test_time = data
np.save("../data/test_time_scaling_data.npy", {'data':data,'models':[encode[m] for m in models]})