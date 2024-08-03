#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Game.py
@Time    :   2023/07/14 11:06:59
@Author  :   Zhou Zihao
@Version :   1.0
@Desc    :   None
'''

import os, json, sys
import numpy as np
import torch
import cv2
import time

import env
import algos
import utils
import cv2
from teacher_policy import TeacherPolicy

prefix = os.getcwd()
task_info_json = os.path.join(prefix, "prompt/task_info.json")

class Game:
    def __init__(self, args, training=True):
        # init seed
        self.seed = args.seed
        self.setup_seed(args.seed)
        
        # init env
        self.load_task_info(args.task, args.frame_stack, args.offline_planner, args.soft_planner)

        # init logger
        self.logger = utils.create_logger(args, training)
        
        # init policy
        policies = {}
        for agent in self.obs_spaces:
            if args.loaddir:
                model_dir = os.path.join(args.logdir, args.policy, args.task, args.loaddir, f"{agent}_", args.loadmodel)
                policies.append(torch.load(model_dir))
                policies[agent] = torch.load(model_dir)
            else:
                policies[agent] = None
        self.device = args.device
        self.batch_size = args.batch_size
        self.recurrent = args.recurrent
        # self.student_policy = policy
        self.student_policies = {}
        for agent, policy in policies.items():
            self.student_policies[agent] = (
                algos.PPO(
                    policy,
                    self.obs_spaces[agent],
                    self.action_spaces[agent],
                    self.device, 
                    self.logger.dir, 
                    batch_size=self.batch_size, 
                    recurrent=self.recurrent)
                )

        # init buffer
        self.gamma = args.gamma
        self.lam = args.lam
        self.buffers = {}
        for agent in self.obs_spaces:
            self.buffers[agent] = algos.Buffer(self.gamma, self.lam, self.device)

        # other settings
        self.n_itr = args.n_itr
        self.traj_per_itr = args.traj_per_itr
        self.num_eval = args.num_eval
        self.eval_interval = args.eval_interval
        self.save_interval = args.save_interval
        self.total_steps = 0
        
        
    def setup_seed(self, seed):
        # setup seed for Numpy, Torch and LLM, not for env
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        
    def load_task_info(self, task, frame_stack, offline, soft):
        print(f"[INFO]: resetting the task: {task}")
        with open(task_info_json, 'r') as f:
            task_info = json.load(f)
        task = task.lower()
        
        env_fn = utils.make_env_fn(task_info[task]['configurations'], 
                                   render_mode="rgb_array", 
                                   frame_stack = frame_stack)
        self.env = utils.WrapEnv(env_fn)
        self.obs_spaces = {k: utils.get_obss_preprocessor(v) for k, v in self.env.observation_spaces.items()}
        self.action_spaces = {k: v.n for k, v in self.env.action_spaces.items()}

        prefix = {}
        if isinstance(task_info[task]['description'], str):
            for agent in self.obs_spaces:
                prefix[agent] = task_info[task]['description'] + task_info[task]['example']
        else:
            for agent in task_info[task]['description']:
                prefix[agent] = task_info[task]['description'][agent] + task_info[task]['example'][agent]

        self.teacher_policies = {}
        for agent in self.obs_spaces:
            self.teacher_policies[agent] = TeacherPolicy(task, offline, soft, prefix[agent], self.action_spaces[agent])
            
    def train(self):
        start_time = time.time()
        for itr in range(self.n_itr):
            print("********** Iteration {} ************".format(itr))
            print("time elapsed: {:.2f} s".format(time.time() - start_time))

            ## collecting ##
            sample_start = time.time()
            for buffer in self.buffers.values():
                buffer.clear()
            n_traj = self.traj_per_itr
            for _ in range(n_traj):
                self.collect()
            while len(self.buffer[0]) < self.batch_size * 2:
                self.collect()
                n_traj += 1
            total_steps = len(self.buffers[0])    
            samp_time = time.time() - sample_start
            print("{:.2f} s to collect {:6n} timesteps | {:3.2f}sample/s.".format(samp_time, total_steps, (total_steps)/samp_time))
            self.total_steps += total_steps

            ## training ##
            optimizer_start = time.time()
            mean_losses = {}
            for agent, policy in self.student_policies.items():
                mean_losses[agent] = policy.update_policy(self.buffers[agent])
            opt_time = time.time() - optimizer_start
            try:
                print("{:.2f} s to optimizer|".format(opt_time))
                for agent in self.student_policies:
                    print("agent_{}: loss {:6.3f}, entropy {:6.3f}, kickstarting {:6.3f}.".format(agent, mean_losses[agent][0], mean_losses[agent][1], mean_losses[agent][2]))
            except:
                print(mean_losses)

            ## evaluate ##
            if itr % self.eval_interval == 0 and itr > 0:
                evaluate_start = time.time()
                eval_returns = {}
                eval_lens = []
                eval_success = {}
                for agent in self.student_policies:
                    eval_returns[agent] = []
                    eval_success[agent] = []
                for i in range(self.num_eval):
                    eval_outputs = self.evaluate(itr, record_frames=False)
                    eval_lens.append(eval_outputs[1])
                    for agent in self.student_policies:
                        eval_returns[agent].append(eval_outputs[0][agent])
                        eval_success[agent].append(eval_outputs[2][agent])
                eval_time = time.time() - evaluate_start
                print("{:.2f} s to evaluate.".format(eval_time))
            
            if itr % self.save_interval == 0 and itr > 0:
                for agent, policy in self.student_policies.items():
                    policy.save(f"{agent}_{itr}")
            
            ## log ##
            if self.logger is not None:
                # avg_len = np.mean(self.buffer.ep_lens)
                # avg_reward = np.mean(self.buffer.ep_returns)
                # std_reward = np.std(self.buffer.ep_returns)
                # success_rate = sum(i > 0 for i in self.buffer.ep_returns) / n_traj
                sys.stdout.write("-" * 49 + "\n")
                sys.stdout.write("| %25s | %15s |" % ('Timesteps', self.total_steps) + "\n")
                for agent, buffer in self.buffers.items():
                    avg_len = np.mean(buffer.ep_lens)
                    avg_reward = np.mean(buffer.ep_returns)
                    std_reward = np.std(buffer.ep_returns)
                    success_rate = sum(i > 0 for i in buffer.ep_returns) / n_traj
                    sys.stdout.write(f"Agent_{agent}:" + "\n")
                    sys.stdout.write("| %25s | %15s |" % ('Return (train)', round(avg_reward,2)) + "\n")
                    sys.stdout.write("| %25s | %15s |" % ('Episode Length (train)', round(avg_len,2)) + "\n")
                    sys.stdout.write("| %25s | %15s |" % ('Success Rate (train)', round(success_rate,2)) + "\n")
                    self.logger.add_scalar(f"Train/Return Mean {agent}", avg_reward, itr)
                    self.logger.add_scalar(f"Train/Return Std {agent}", std_reward, itr)
                    self.logger.add_scalar(f"Train/Eplen {agent}", avg_len, itr)
                    self.logger.add_scalar(f"Train/Success Rate {agent}", success_rate, itr)
                if itr % self.eval_interval == 0 and itr > 0:
                    avg_eval_len = np.mean(eval_lens)
                    sys.stdout.write("| %25s | %15s |" % ('Episode Length (eval) ', round(avg_eval_len,2)) + "\n")
                    self.logger.add_scalar("Test/Eplen", avg_eval_len, itr)
                    for agent in eval_returns:
                        avg_eval_reward = np.mean(eval_returns)
                        eval_success_rate = np.sum(eval_success) / self.num_eval
                        sys.stdout.write(f"Agent_{agent}:" + "\n")
                        sys.stdout.write("| %25s | %15s |" % ('Return (eval)', round(avg_eval_reward,2)) + "\n")
                        sys.stdout.write("| %25s | %15s |" % ('Success Rate (eval) ', round(eval_success_rate,2)) + "\n")
                        self.logger.add_scalar(f"Test/Return {agent}", avg_eval_reward, itr)
                        self.logger.add_scalar(f"Test/Success Rate {agent}", eval_success_rate, itr)
                sys.stdout.write("-" * 49 + "\n")
                sys.stdout.flush()

                for agent in mean_losses:
                    self.logger.add_scalar(f"Train/Loss {agent}", mean_losses[agent][0], itr)
                    self.logger.add_scalar(f"Train/Mean Entropy {agent}", mean_losses[agent][1], itr)
                    self.logger.add_scalar(f"Train/Kickstarting Loss {agent}", mean_losses[agent][2], itr)
                    self.logger.add_scalar(f"Train/Policy Loss {agent}", mean_losses[agent][3], itr)
                    self.logger.add_scalar(f"Train/Value Loss {agent}", mean_losses[agent][4], itr)
                    self.logger.add_scalar(f"Train/Kickstarting Coef {agent}", self.student_policies[agent].ks_coef, itr)
        for agent, policy in self.student_policies.items():
            policy.save(f"{agent}_acmodel")


    def collect(self):
        '''
        collect episodic data.
        ''' 
        with torch.no_grad():
            observations, _ = self.env.reset()
            
            # reset student policy
            masks = {}
            states = {}
            for agent, policy in self.student_policies.items():
                masks[agent] = torch.FloatTensor([1]).to(self.device) # not done until episode ends
                states[agent] = policy.model.init_states(self.device) if self.recurrent else None
            actions = {}
            values = {}
            log_probs = {}
            teacher_probs = {}
            
            # reset teacher policy
            for teacher_policy in self.teacher_policies.values():
                teacher_policy.reset()

            all_done = False
            while not all_done:
                # get action from student policy
                for agent, policy in self.student_policies.items():
                    dist, value, states[agent] = policy(torch.Tensor([observations[agent]]).to(self.device), masks[agent], states[agent])
                    action = dist.sample()
                    log_probs[agent] = dist.log_prob(action).to("cpu").numpy()
                    values[agent] = value.to("cpu").numpy()
                    actions[agent] = action.to("cpu").numpy()
                
                # get action from teacher policy
                # obs: ndarray, shape (1, 10, 10, 4)
                for agent, policy in self.teacher_policies.items():
                    teacher_probs[agent] = policy(observations[agent])
                
                # interact with env
                # next_obs, reward, done, info = self.env.step(actions)
                next_observations, rewards, terminations, truncations, all_done, _ = self.env.step(actions)
    
                # store in buffer
                for agent, buffer in self.buffers.items():
                    buffer.store(observations[agent], 
                                  actions[agent], 
                                  rewards[agent], 
                                  values[agent], 
                                  log_probs[agent], 
                                  teacher_probs[agent])
                observations = next_observations

            for agent, policy in self.student_policies.items():
                done = terminations[agent] or truncations[agent]
                if done:
                    values[agent] = 0.
                else:
                    values[agent] = policy(torch.Tensor(observations[agent]).to(self.device), 
                                                masks[agent], states[agent])[1].to("cpu").item()
            for agent, buffer in self.buffers.items():
                buffer.finish_path(last_val=values[agent])        
        
    def evaluate(self, itr=None, seed=None, record_frames=True, deterministic=False, teacher_policy=False):
        with torch.no_grad():
            # init env
            seed = seed if seed else np.random.randint(1000000)

            observations, _ = self.env.reset(seed=seed)
            all_done = False 
            ep_len = 0
            ep_return = {}
            for agent in self.obs_spaces:
                ep_return[agent] = 0.

            if teacher_policy:
                # init teacher policy
                for teacher_policy in self.teacher_policies.values():
                    teacher_policy.reset()
            else:
                # init student policy
                masks = {}
                states = {}
                for agent, policy in self.student_policies:
                    masks[agent] = torch.FloatTensor([1]).to(self.device) # not done until episode ends
                    states[agent] = policy.model.init_states(self.device) if self.recurrent else None
            
            actions = {}

            # init vedio directory
            if record_frames:
                img_array = []
                img = self.env.render()
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_array.append(img)
                
                dir_name = 'teacher_video' if teacher_policy else 'video'
                dir_path = os.path.join(self.logger.dir, dir_name)
                try:
                    os.makedirs(dir_path)
                except OSError:
                    pass

            while not all_done:
                if teacher_policy:
                    # get actions from teacher policy
                    for agent, policy in self.teacher_policies.items():
                        probs = policy(observations[agent])
                        if deterministic:
                            action = np.argmax(probs)
                        else:
                            action = np.random.choice(self.action_spaces[agent], p=probs)
                        actions[agent] = action
                else:
                    # get action from student policy
                    for agent, policy in self.student_policies.items():
                        dist, _, states[agent] = policy(torch.Tensor(observations[agent]).to(self.device), masks[agent], states[agent])
                        if deterministic:
                            action = torch.argmax(dist.probs).unsqueeze(0).to("cpu").numpy()
                        else:
                            action = dist.sample().to("cpu").numpy()
                        actions[agent] = action

                # interact with env
                observations, rewards, _, _, all_done, _ = self.env.step(actions)
                for agent, reward in rewards.items():
                    ep_return[agent] += reward

                ep_len += 1
                
                if record_frames:
                    img = self.env.render()
                    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img_array.append(img)
                
            ep_success = {}
            for agent, reward in ep_return.items():
                ep_success[agent] = 1 if reward > 0 else 0

            # save vedio
            if record_frames:
                height, width, layers = img.shape
                size = (width,height)
                video_name = "%s-%s.avi"%(itr, seed) if itr else "%s.avi"%seed
                video_path = os.path.join(dir_path, video_name)
                out = cv2.VideoWriter(video_path, 
                                      fourcc=cv2.VideoWriter_fourcc(*'DIVX'), 
                                      fps=3, 
                                      frameSize=size)

                for img in img_array:
                    out.write(img)
                out.release()
                
            return ep_return, ep_len, ep_success
    
        
if __name__ == '__main__':
    pass


