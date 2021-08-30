import copy
import glob
import os
import time
from collections import deque
from tqdm import tqdm

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# self defined functions
import utils
from ppo import PPO
from arguments import get_args
from envs import make_vec_envs
from model import Policy, GNNBase, ImpalaCNN, ImpalaGNN
from storage import RolloutStorage
from evaluation import evaluate


def main():

    # from pyvirtualdisplay import Display
    # display = Display(visible=0, size=(1400, 900))
    # display.start()

    args = get_args()
    print(args)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    if args.use_gnn:
        print("Using GNNBase!")
        if args.use_imp:
            print("Using IMPALA GNN")
            actor_critic = Policy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy},
                base=ImpalaGNN)
        else:
            print("Using GNN")
            actor_critic = Policy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy},
                base=GNNBase)
    else:
        if args.use_imp:
            print("Using IMPALA CNN")
            actor_critic = Policy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy},
                base=ImpalaCNN)  
        else:
            print("Using CNN")
            actor_critic = Policy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy})       

    # actor_critic.base = nn.DataParallel(actor_critic.base)
    actor_critic.to(device)

   
    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)


    # TODO: Modify here, similar to atari
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    start = time.time()
    # the total number of episodes = total steps / num_steps per episode / num processes
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    print("num_updates:%d num_env_steps:%d num_steps:%d num_processes:%d"%(num_updates, args.num_env_steps, args.num_steps, args.num_processes))

    writer = SummaryWriter(args.log_dir)
    tbar = tqdm(range(num_updates), desc='Train', leave=True)
    for j in tbar:

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        # if args.gail:
        #     if j >= 10:
        #         envs.venv.eval()

        #     gail_epoch = args.gail_epoch
        #     if j < 10:
        #         gail_epoch = 100  # Warm up
        #     for _ in range(gail_epoch):
        #         discr.update(gail_train_loader, rollouts,
        #                      utils.get_vec_normalize(envs)._obfilt)

        #     for step in range(args.num_steps):
        #         rollouts.rewards[step] = discr.predict_reward(
        #             rollouts.obs[step], rollouts.actions[step], args.gamma,
        #             rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            # tbar.refresh()
            writer.add_scalars(str(args.env_name)+str(args.len_eval)+' episode reward', {
                                  'mean': np.mean(episode_rewards),
                                  'median': np.median(episode_rewards),
                                  'max': np.max(episode_rewards),
                                  'min': np.min(episode_rewards),
                              }, total_num_steps)
            writer.add_scalars(str(args.env_name)+str(args.len_eval)+' episode loss', {
                                  'dist_entropy': dist_entropy,
                                  'value_loss': value_loss,
                                  'action_loss': action_loss,
                              }, total_num_steps)            

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

    writer.close()

if __name__ == "__main__":
    main()
