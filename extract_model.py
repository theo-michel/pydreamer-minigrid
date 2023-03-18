import torch
import argparse
from pydreamer.models.dreamer import Dreamer
from pydreamer.tools import mlflow_load_checkpoint
from pydreamer.tools import (configure_logging, mlflow_log_params,
                             mlflow_init, print_once, read_yamls)
from distutils.util import strtobool
import os
import json
import mlflow
from torchsummary import summary
import pickle
from PIL import Image
import numpy as np


def get_worker_info():
    worker_type = None
    worker_index = None

    if 'TF_CONFIG' in os.environ:
        # TF_CONFIG indicates Google Vertex AI run
        tf_config = json.loads(os.environ['TF_CONFIG'])
        print_once('TF_CONFIG is set:', tf_config)
        if tf_config['cluster'].get('worker'):
            # If there are workers in the cluster, then it's a distributed run
            worker_type = {
                'chief': 'learner',
                'worker': 'generator',
            }[str(tf_config['task']['type'])]
            worker_index = int(tf_config['task']['index'])
            print_once('Distributed run detected, current worker is:', f'{worker_type} ({worker_index})')

    return worker_type, worker_index

configure_logging('[launcher]')
parser = argparse.ArgumentParser()
parser.add_argument('--configs', nargs='+', required=True)
args, remaining = parser.parse_known_args()
print("args you passed in")
print(args)
print(remaining)

# Config from YAML

conf = {}
configs = read_yamls('./config')
for name in args.configs:
    if ',' in name:
        for n in name.split(','):
            conf.update(configs[n])
    else:
        conf.update(configs[name])

# Override config from command-line

parser = argparse.ArgumentParser()
for key, value in conf.items():
    type_ = type(value) if value is not None else str
    if type_ == bool:
        type_ = lambda x: bool(strtobool(x))
    parser.add_argument(f'--{key}', type=type_, default=value)
conf = parser.parse_args(remaining)

print(conf)

worker_type, worker_index = get_worker_info()
is_main_worker = worker_type is None or worker_type == 'learner'
mlrun = mlflow_init(wait_for_resume=not is_main_worker)
artifact_uri = mlrun.info.artifact_uri
mlflow_log_params(vars(conf))

#We have to set everything to the right values for each
#TODO Get the conf

model = Dreamer(conf)
# model = torch.load("mlruns/0/6e7cd15f26854e42a458c358d21b65c9/artifacts/checkpoints/latest.pt")

# model_step = mlflow_load_checkpoint(model, map_location='cpu',artifact_path='/home/theomichel/work/pyDreamer/pydreamer/mlruns/0/6e7cd15f26854e42a458c358d21b65c9/artifacts/checkpoints/latest.pt')  # type: ignore #Modify's model by sideffect

optimizers=tuple()
map_location=None#Same place that it is loaded from
# PONG '/home/theomichel/work/pyDreamer/pydreamer/mlruns/0/a4efeae409604aa4a0f8455488dae462/artifacts/checkpoints/latest.pt'
# MINIGRID '/home/theomichel/work/pyDreamer/pydreamer/mlruns/0/342ccaea0b0b4812929cb5433bac3510/artifacts/checkpoints/latest.pt'
# Adventure 
# path = "/home/theomichel/work/pyDreamer/pydreamer-minigrid/results/baselines/atari/mlruns/0/261d3a26b2b842ec990a8d0a5d6111ac/artifacts/checkpoints/latest.pt"
# Alien  "/home/theomichel/work/pyDreamer/pydreamer-minigrid/mlruns/0/c1a7c69b35fa4570915c6be36f57f2c9/artifacts/checkpoints/latest.pt"
path = "/home/theomichel/work/pyDreamer/pydreamer-minigrid/mlruns/0/c1a7c69b35fa4570915c6be36f57f2c9/artifacts/checkpoints/latest.pt"
checkpoint = torch.load(path, map_location=map_location)
model.load_state_dict(checkpoint['model_state_dict'])
for i, opt in enumerate(optimizers):
    opt.load_state_dict(checkpoint[f'optimizer_{i}_state_dict'])
    checkpoint['epoch']


## Evaluate the model
from train import evaluate
from pydreamer.data import DataSequential, MlflowEpisodeRepository
from torch.utils.data import DataLoader
from pydreamer.preprocessing import Preprocessor, WorkerInfoPreprocess

preprocess = Preprocessor(image_categorical=conf.image_channels if conf.image_categorical else None,
                            image_key=conf.image_key,
                            map_categorical=conf.map_channels if conf.map_categorical else None,
                            map_key=conf.map_key,
                            action_dim=conf.action_dim,
                            clip_rewards=conf.clip_rewards,
                            amp=conf.amp and device.type == 'cuda')


device = torch.device(conf.device)
steps = 10
# eval_dirs = [
#             f'{artifact_uri}/episodes_eval/{i}'
#             for i in range(max(conf.generator_workers_eval, conf.generator_workers))
#         ]
model.to(device)#Important
# MINIGRID 'file:///home/theomichel/work/pyDreamer/pydreamer/mlruns/0/342ccaea0b0b4812929cb5433bac3510/artifacts/episodes_eval/0'
# PONG 'file:///home/theomichel/work/pyDreamer/pydreamer/mlruns/0/afc956ebbaa447008da84d7486c0c05a/artifacts/episodes_eval/0'
# ADVENTURE
# eval_dirs =['file:///home/theomichel/work/pyDreamer/pydreamer/mlruns/0/342ccaea0b0b4812929cb5433bac3510/artifacts/episodes_eval/0']
# ALIEN
eval_dirs = ['file:///home/theomichel/work/pyDreamer/pydreamer-minigrid/mlruns/0/c1a7c69b35fa4570915c6be36f57f2c9/artifacts/episodes_eval/0']

test_dirs = eval_dirs
data_test = DataSequential(MlflowEpisodeRepository(test_dirs), conf.batch_length, conf.test_batch_size, skip_first=False, reset_interval=conf.reset_interval)
print("Test Data")
print(data_test)
test_iter = iter(DataLoader(preprocess(data_test), batch_size=None))
#Here we launch an evaluation which will launch a training step and so save a model
evaluate('test', steps, model, test_iter, device, conf.test_batches, conf.iwae_samples, conf.keep_state, conf.test_save_size, conf) 

## Make it play in the gym

##TODO
## Play with the dreams 
#World Model.forward 
#then Modify the variables in input 
#See difference in output
#Find cool way to visualize
########################################### MODEL INFERENCE ###########################################
# exit()
# for state old
# with open('in_state_alien.pkl','rb') as f:
#     in_state = pickle.load(f)
# with open('obs_alien.pkl','rb') as f:
#     obs = pickle.load(f)
# For state that we extracted from inference
with open('in_state_alien_inf.pkl','rb') as f:
    in_state = pickle.load(f)
with open('obs_alien_inf.pkl','rb') as f:
    obs = pickle.load(f)

# print("In state")
# print(print(in_state))
# print("obs")
# print(type(obs))
for key in obs:
    obs[key] = obs[key].to(device)

in_state_new = (in_state[0].to(device),in_state[1].to(device))

#put data on gpu
# in_state = in_state.to(device)
#Forward is just used when the model is translating the world into features for the policy to take a decision
# features, out_state = model.wm.forward(obs,in_state_new)
print("obs")
print(obs.keys())
print(in_state_new)#(h,z) with h the forwarded state by the RSSM and z containing the latent variables
#Training step does all evem the image prediction

loss, features, states, out_state, metrics, tensors = model.wm.training_step(obs,in_state_new,do_image_pred=True)
print("tensors")
print(tensors.keys())
print(tensors['image_pred'].shape)
image_cpu = tensors['image_pred'].cpu().numpy()
image = image_cpu[0,0,:,:,:].transpose(1,2,0)
print("imshape")
print(image.shape)
print("min max")
print(np.min(image))
print(np.max(image))
ar = (image+0.5)* 255
print(np.min(ar))
print(np.max(ar))
image_final = ar.astype(np.uint8)
img_rgb = image_final#[...,::-1]
im = Image.fromarray(img_rgb)
im.save("predicted_dream_alien_2.jpeg")






""" MlFlow tries

# loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
model_step = mlflow_load_checkpoint(model, map_location='cpu',artifact_path='checkpoints/latest.pt')  # type: ignore #Modify's model by sideffect
 """