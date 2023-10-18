import numpy as np
import gym
# import gymnasium as gym
# from gym.envs.registration import register

import copy

from zone_envs.ZoneEnvBase import ZoneEnvBase, zone

key = zone.Yellow
sword = zone.Blue
shield = zone.Green
dragon = zone.Red
visited = zone.JetBlack

class DungeonQuestEnv(ZoneEnvBase):
    def __init__(self, config):
        self.new_zone_reached = None
        self.zones_dirty = None
        config = copy.deepcopy(config)

        self.time_saved_reward = config.pop("time_saved_reward", 0.05)

        if "reward_goal" in self.DEFAULT:
            self.DEFAULT.pop("reward_goal")
        if "num_cities" in config:
            config.pop("num_cities")
        
        config.update({"continue_goal": False})

        self.inventory = []

        self.zone_types = [key, sword, shield, dragon, visited]
        self.zones = [key, sword, shield, dragon]
        self.high_only_keys = ["remaining"] + [f"zones_lidar_{i}" for i in range(len(self.zones))]

        super().__init__(zones=self.zones, config=config)

    def build_zone_observation_space(self):
        for i, zone in enumerate(self.zones):
            space = gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)  # 3 = x,y,vis
            self.obs_space_dict.update({f"zones_lidar_{i}": space})

    def obs_zones(self, obs):
        for i, z in enumerate(self.zones):
            pos = self.data.get_body_xpos(f"zone{i}").copy()[:2] / 3.
            # vis = z == visited
            # obs[f"zones_lidar_{i}"] = np.concatenate([[vis], pos])
            kind = np.eye(len(self.zone_types))[self.zone_types.index(z)]
            obs[f"zones_lidar_{i}"] = np.concatenate([kind, pos])

    @property
    def reward_goal(self):
        # return (self.num_steps - self.steps) * self.time_saved_reward
        return 100

    def reward(self):
        return (1 if self.new_zone_reached else 0) - self.time_saved_reward

    def step(self, action):
        self.zones_dirty = True
        self.new_zone_reached = False

        return super().step(action)

    def set_mocaps(self):
        if not self.zones_dirty: return

        for i, pos in enumerate(self.zones_pos):
            if self.zones[i] != visited:
                dist = self.dist_xy(pos)
                if dist <= self.zones_size:
                    if self.zones[i] == key:
                        self.inventory.append("key")
                    
                    elif self.zones[i] == sword:
                        if "key" in self.inventory:
                            self.inventory.append("sword")
                        else:
                            continue
                    
                    elif self.zones[i] == shield:
                        self.inventory.append("shield")
                    
                    elif self.zones[i] == dragon:
                        if "sword" in self.inventory and "shield" in self.inventory:
                            self.inventory.append("scale")
                        else:
                            continue
                    
                    self.zones[i] = visited
                    
                    body_id = self.sim.model.geom_name2id(f"zone{i}")
                    self.sim.model.geom_rgba[body_id] = self._rgb[visited]
                    self.new_zone_reached = True

                    break

        self.zones_dirty = False

    def goal_met(self):
        return "scale" in self.inventory

    def reset(self, **kwargs):
        self.inventory = []
        self.zones = [key, sword, shield, dragon]

        return super().reset(**kwargs)
    
    def get_aps(self):
        aps = [item in self.inventory for item in ["key", "shield", "sword", "scale"]]
        return aps

class DiscreteDungeonQuestEnv(DungeonQuestEnv):
    def __init__(self, config):
        super().__init__(config)
        self.action_space = gym.spaces.Discrete(5)
    
    def step(self, action):
        action_values = [(1,0), (-1,0), (0.01,1), (0.01,-1), (0,0)]
        return super().step(action_values[action])

config_point = {
    'robot_base': 'xmls/point.xml',
    'walled': True,
    'observe_remaining': True,
    'observation_flatten': False,
    'num_steps': 4000
}

# register(id="PointDQ-v0", entry_point="DungeonQuestEnv:DungeonQuestEnv", kwargs={"config": copy.copy(config_point)})
# register(id="PointDDQ-v0", entry_point="DungeonQuestEnv:DiscreteDungeonQuestEnv", kwargs={"config": copy.copy(config_point)})
