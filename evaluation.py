import numpy as np
import torch

import utils
from envs import make_vec_envs


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, gen_test=False):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, gen_test)

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
    while len(eval_episode_rewards) < 100:
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


# print(args.save_dir)
# actor_critic = torch.load(args.save_dir+"/ppo/CartPole-v0.pt")[0]
# # ob_rms = utils.get_vec_normalize(envs).ob_rms
# evaluate(actor_critic, None, args.env_name, args.seed,
#                      args.num_processes, eval_log_dir, device)