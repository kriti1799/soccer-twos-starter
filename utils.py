from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class RewardShapingWrapper(gym.Wrapper):
    """
    Optional reward shaping wrapper.

    Supported modes:
    - "none": no change (default)
    - "step_penalty": subtract a fixed penalty each step to encourage faster goals
    - "ball_progress": reward ball movement along a chosen axis
    """

    def __init__(self, env, step_penalty=0.0, ball_progress_weight=0.1, ball_to_goal_weight=0.1):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.ball_progress_weight = ball_progress_weight
        self.ball_to_goal_weight = ball_to_goal_weight
        self._last_ball_pos = None
        self._prev_dist_to_ball = {}
    
    def reset(self, **kwargs):
        self._last_ball_pos = None
        self._prev_dist_to_ball = {}
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        ball_pos = _extract_ball_position(info)

        for player_id in reward:
            if player_id not in info:
                continue

            player_pos = info[player_id]['player_info']['position']

            # Penalty each step to encourage faster goals
            reward[player_id] -= self.step_penalty

            # Reward moving toward ball
            if ball_pos is not None:
                curr_dist = np.linalg.norm(ball_pos - player_pos)
                prev_dist = self._prev_dist_to_ball.get(player_id, curr_dist)
                reward[player_id] += self.ball_to_goal_weight * (prev_dist - curr_dist)
                self._prev_dist_to_ball[player_id] = curr_dist

                # If the agent is very close to the ball, give a flat bonus.
                # You may need to tweak the '0.5' distance threshold based on the exact scale of your environment.
                if curr_dist < 1.5: 
                    reward[player_id] += 0.01

        # Reward ball moving toward opponent goal
        if ball_pos is not None and self._last_ball_pos is not None:
            delta = float(ball_pos[0] - self._last_ball_pos[0])
            for player_id in reward:
                if player_id in [0, 1]:  # Team 0 attacks +x direction
                    reward[player_id] += self.ball_progress_weight * delta
                else:  # Team 1 (players 2,3) attacks -x direction
                    reward[player_id] += self.ball_progress_weight * (-delta)

        self._last_ball_pos = ball_pos

        return obs, reward, done, info

def _apply_step_penalty(reward, penalty):
    if isinstance(reward, dict):
        return {k: v - penalty for k, v in reward.items()}
    if isinstance(reward, tuple):
        return tuple(_apply_step_penalty(list(reward), penalty))
    if isinstance(reward, list):
        return [r - penalty for r in reward]
    if isinstance(reward, np.ndarray):
        return reward - penalty
    return reward - penalty


def _apply_additive_reward(reward, bonus):
    if isinstance(reward, dict):
        return {k: v + bonus for k, v in reward.items()}
    if isinstance(reward, tuple):
        return tuple(_apply_additive_reward(list(reward), bonus))
    if isinstance(reward, list):
        return [r + bonus for r in reward]
    if isinstance(reward, np.ndarray):
        return reward + bonus
    return reward + bonus


def _axis_delta(prev_pos, curr_pos, axis="x"):
    axis_index = 0 if axis == "x" else 1
    return float(curr_pos[axis_index] - prev_pos[axis_index])


def _extract_ball_position(info):
    if info is None:
        return None
    if isinstance(info, dict):
        for key in ("ball_position", "ball_pos"):
            if key in info:
                return info[key]
        if "ball_info" in info and isinstance(info["ball_info"], dict):
            if "position" in info["ball_info"]:
                return info["ball_info"]["position"]
        if "ball" in info and isinstance(info["ball"], dict):
            if "position" in info["ball"]:
                return info["ball"]["position"]
        # Multi-agent info: recurse into per-agent dicts
        for value in info.values():
            pos = _extract_ball_position(value)
            if pos is not None:
                return pos
    if isinstance(info, (list, tuple)):
        for value in info:
            pos = _extract_ball_position(value)
            if pos is not None:
                return pos
    return None


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    
    step_penalty = env_config.pop("step_penalty", 0.0)
    ball_progress_weight = env_config.pop("ball_progress_weight", 0.0)
    ball_to_goal_weight = env_config.pop("ball_to_goal_weight", 0.1)

    env = soccer_twos.make(**env_config)
    env = RewardShapingWrapper(
        env,
        step_penalty=step_penalty,
        ball_progress_weight=ball_progress_weight,
        ball_to_goal_weight=ball_to_goal_weight
       
    )
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
