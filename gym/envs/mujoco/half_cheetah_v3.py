import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}

TARGET_VELOCITY=0.1


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 target_velocity=0.1):
        utils.EzPickle.__init__(**locals())

        self.target_velocity = target_velocity
        
        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = np.linalg.norm(action)
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        alpha = 1.0
        beta = 0.05

        alive_bonus = 0.0
        reward = alive_bonus - alpha * np.abs(x_velocity - self.target_velocity) - beta * ctrl_cost
        #print(np.abs(x_velocity - self.target_velocity))
        #forward_reward = self._forward_reward_weight * (1. - np.tanh(10. * (np.abs(x_velocity - self.target_velocity)))) #np.square(x_velocity - TARGET_VELOCITY)
        #print(forward_reward, np.abs(x_velocity))
        observation = self._get_obs()
        #reward = forward_reward - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            #'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
