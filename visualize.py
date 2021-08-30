import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import copy
import re
from numpy import dot
from numpy.linalg import norm
# from tqdm import tqdm_notebook
from collections import defaultdict

# import utils
from envs import make_env, make_vec_envs
from arguments import get_args


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, gen_test=False, eval_epi=100):
    """
    Evaluate a given network with the env specified by env_name
    """
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, gen_test=gen_test)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    
    length = 0
    anim_frames = []
    anim_frames.append(obs.detach().cpu().numpy())
    while len(eval_episode_rewards) < eval_epi:
        if len(eval_episode_rewards) > length:
            length = len(eval_episode_rewards)
            print(length)
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)
        anim_frames.append(obs.detach().cpu().numpy())
        
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    return anim_frames


def hook_func(module, input, output):
    print(index)
    index += 1
    print('#######')
    print(module)
    print('#######')
    in_features.append(input)
    out_features.append(output)


def main():

    sys.argv = ['--use-gae --use-linear-lr-decay']
    args = get_args()

    args.num_processes = 1
    args.env_name = 'MountainCar-v0' 
    #M2 is `max` version, less fc weights
    args.log_dir = '/users/mli115/scratch/ppognnimpM2-logs'
    args.save_dir = '/users/mli115/scratch/ppognnimpM2-trained_models'
    args.use_gnn = True
    args.use_imp = True
    # if not os.path.exists(args.log_dir):
    #     os.makedirs(args.log_dir)
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    device = torch.device('cuda:0')
    eval_log_dir = args.log_dir
    print(args)

    torch.nn.Module.dump_patches = True
    actor_critic = torch.load(args.save_dir+"/ppo/MountainCar-v0.pt")[0].to(device)
    observations = evaluate(actor_critic, None, args.env_name, args.seed,
                        args.num_processes, eval_log_dir, device, gen_test=True, eval_epi=1)

    in_features = []
    out_features = []
    hook_handles = []


    actor_critic = torch.load(args.save_dir+"/ppo/MountainCar-v0.pt")[0].to(device)
    for module in actor_critic.base.named_modules():
        handle = module[1].register_forward_hook(hook_func)
        hook_handles.append(handle)

    # select a random observation frame to calculate feature activation map i.e. coord-dict
    actor_critic.act(torch.tensor(observations[44]).to(device))
    print(observations[44].shape)
    print(len(in_features))
    print(len(out_features))

    coord_dict = defaultdict(list)
    # baseline output of two conv layers with shape of [1, 510, 9, 9]
    # base = out_features[3] #vanilla gnn
    base = in_features[28][0] #impgnn
    # 4x84x84
    test_obs = observations[44].squeeze()
    summed_obs = np.sum(observations[44].squeeze(), axis=0)
    for i in range(84):
        print(i)
        for j in range(84):
            tmp = copy.copy(test_obs)
            tmp[:, i, j] = 0 if summed_obs[i, j] > 100 else 255
            in_features = []
            out_features = []
            actor_critic.act(torch.tensor(tmp).unsqueeze(0).to(device))
            a, b = np.where((in_features[28][0] - base).detach().cpu().numpy() != 0)[2:]
    #         print('index:\n', np.unique([*zip(a, b)], axis=0))
            coord_dict[str((i,j))] = np.unique([*zip(a, b)], axis=0)


if __name__ == "__main__":
    print('hello')
    index = 0
    main()