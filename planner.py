#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   planner.py
@Time    :   2023/05/16 09:12:11
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''


import requests
from mediator import *

from abc import ABC

class Base_Planner(ABC):
    """The base class for Planner."""

    def __init__(self, offline=True, soft=False, prefix=''):
        super().__init__()
        self.offline = offline
        self.soft = soft
        self.prompt_prefix = prefix
        self.plans_dict = {}
        self.mediator = None
        self.messages = [self.prompt_prefix]
        
        self.dialogue_system = ''               
        self.dialogue_user = ''
        self.show_dialogue = False
        
        if not offline:
            # self.llm_model = "vicuna-33b"
            # self.llm_url = 'http://localhost:3300/v1/chat/completions'
            self.llm_model = "GLM-4-AirX"
            self.llm_url = 'http://localhost:6000/v1/chat/completions'
            # self.llm_model = "openai"
            # self.llm_url = 'https://kimi.api.droneinterface.com/v1/chat/completions'
            self.plans_dict = {}
            if self.llm_model == "vicuna-33b":
                self.init_llm()
        
    def reset(self, show=False):
        self.messages = [self.prompt_prefix]
        self.dialogue_user = ''
        self.show_dialogue = show

        self.mediator.reset()
        # if not self.offline:
        #     self.online_planning("reset")
        
    def init_llm(self):
        ## set system part
        server_error_cnt = 0
        while server_error_cnt < 10:
            try:
                headers = {'Content-Type': 'application/json'}
                
                data = {'model': self.llm_model, "messages":[{"role": "system", "content": self.prompt_prefix}]}
                response = requests.post(self.llm_url, headers=headers, json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    break
                else:
                    assert False, f"fail to initialize: status code {response.status_code}"                
                    
            except Exception as e:
                server_error_cnt += 1
                print(f"fail to initialize: {e}")

    def query_codex(self, prompt_text):
        server_error_cnt = 0
        while server_error_cnt < 10:
            try:
                #response =  openai.Completion.create(prompt_text)
                headers = {'Content-Type': 'application/json'}
                
                # print(f"user prompt:{prompt_text}")
                if self.llm_model.startswith("GLM"):
                    data = {'model': self.llm_model, "prompt": self.messages}
                elif self.llm_model == "vicuna-33b":
                    data = {'model': self.llm_model, "messages":[{"role": "user", "content": prompt_text}]}
                response = requests.post(self.llm_url, headers=headers, json=data)

                if response.status_code == 200:
                    result = response.json()
                    break
                else:
                    print(response.text)
                    assert False, f"fail to query: status code {response.status_code}"
                    
            except Exception as e:
                server_error_cnt += 1
                print(f"fail to query: {e}")
                
        try:
            plan = re.search("Action[s]*\:\s*\{([\w\s\<\>\,]*)\}", result, re.I | re.M).group(1)
            return result, plan
        except:
            print(f"LLM response invalid format: '{result}'.")
            return self.query_codex(prompt_text)   
        
    def plan(self, text, n_ask=10):
        if self.llm_model.startswith("GLM"):
            self.messages.append(text)
        if text in self.plans_dict.keys():
            plans, probs = self.plans_dict[text]
        else:
            print(f"new obs: {text}")
            plans = {}
            plan_to_fulltext = {}
            
            for _ in range(n_ask):
                fulltext, plan = self.query_codex(text)
                plans[plan] = plans.get(plan, 0) + 1/n_ask
                plan_to_fulltext[plan] = fulltext
            
            plans, probs = list(plans.keys()), list(plans.values())
            self.plans_dict[text] = (plans, probs)
            
        max_prob_index = probs.index(max(probs))
        max_plan = plans[max_prob_index]
        self.messages.append(f"Action: {{{max_plan}}}")
            
            # for k, v in self.plans_dict.items():
            #     print(f"{k}:{v}")

        return plans, probs
    
    def __call__(self, obs):
        # self.mediator.reset()
        text = self.mediator.RL2LLM(obs)
        plans, probs = self.plan(text)
        self.dialogue_user = text + "\n" + str(plans) + "\n" + str(probs)
        if self.show_dialogue:
            print(self.dialogue_user)
        skill_list, probs = self.mediator.LLM2RL(plans, probs)
        
        return skill_list, probs

class TwoDoor_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix):
        super().__init__(offline, soft, prefix)
        self.mediator = TwoDoor_Mediator(soft)
        if offline:
            self.plans_dict = {
                "Agent sees <nothing>, holds <nothing>." : [["explore"], [1.0]],
                "Agent sees <door1>, holds <nothing>."  : [["explore"], [1.0]],
                "Agent sees <key>, holds <nothing>."   : [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <nothing>, holds <key>."     : [["explore"], [1.0]],
                "Agent sees <door1>, holds <key>."        : [["go to <door1>, open <door1>"], [1.0]],
                "Agent sees <key>, <door1>, holds <nothing>." : [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <door1>, <door2>, holds <nothing>."  : [["explore"], [1.0]],
                "Agent sees <key>, <door1>, <door2>, holds <nothing>.": [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <door1>, <door2>, holds <key>.": [["go to <door1>, open <door1>", "go to <door2>, open <door2>"], [0.5, 0.5]],
            } 

class Driving_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix):
        super().__init__(offline, soft, prefix)
        self.mediator = Driving_Mediator(soft)
        if offline:
            self.plans_dict = {} 
                                            
                                                            
def Planner(task, offline=False, soft=False, prefix=''):
    if task.lower() == "twodoor":
        planner = TwoDoor_Planner(offline, soft, prefix)
    elif task.lower() == "driving":
        planner = Driving_Planner(offline, soft, prefix)
    return planner
                                                            
                                                            