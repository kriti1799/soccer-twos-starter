import os
from typing import Dict
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface


class MyAgent(AgentInterface):
    """
    Agent wrapper for RLlib PPO multiagent checkpoint
    """
    
    def __init__(self, env):
        # Path to your checkpoint
        checkpoint_path = "checkpoint-1000"
        
       
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Create the trainer config (must match your training config)
        config = {
            "framework": "torch",
            "multiagent": {
                "policies": {
                    "default": (None, env.observation_space, env.action_space, {}),
                },
                "policy_mapping_fn": lambda agent_id: "default",
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "num_workers": 0,  # Use 0 for inference
            "explore": False,  # Disable exploration for evaluation
        }
        
        # Create trainer and restore checkpoint
        self.trainer = PPOTrainer(config=config, env=env)
        self.trainer.restore(checkpoint_path)
        
        # Get the trained policy
        self.policy = self.trainer.get_policy("default")
        
    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        The act method is called when the agent is asked to act.
        """
        actions = {}
        for player_id in observation:
            # Compute action using the policy
            action = self.policy.compute_single_action(
                observation[player_id],
                explore=False
            )
            actions[player_id] = action
        return actions