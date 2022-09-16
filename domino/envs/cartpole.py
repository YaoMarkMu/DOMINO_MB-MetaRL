import math
from collections import OrderedDict

from gym import spaces
# from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.classic_control.pendulum import PendulumEnv
import numpy as np
import tensorflow as tf

from .base import EnvBinarySuccessMixin
from gym import error, spaces


import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width /world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = spaces.Box(low, high)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

def uniform_exclude_inner(np_uniform, a, b, a_i, b_i):
    """Draw sample from uniform distribution, excluding an inner range"""
    if not (a < a_i and b_i < b):
        raise ValueError(
            "Bad range, inner: ({},{}), outer: ({},{})".format(a, b, a_i, b_i))
    while True:
        # Resample until value is in-range
        result = np_uniform(a, b)
        if (a <= result and result < a_i) or (b_i <= result and result < b):
            return result

################
### CARTPOLE ###
################
class ModifiableCartPoleEnv(CartPoleEnv, EnvBinarySuccessMixin):
    
    def _followup(self):
        """Cascade values of new (variable) parameters"""
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)

    def reset(self, new=True):
        """new is a boolean variable telling whether to regenerate the environment parameters"""
        """Default is to just ignore it"""
        self.nsteps = 0
        return super(ModifiableCartPoleEnv, self).reset()
    
    def get_sim_parameters(self):
        return np.array([self.force_mag/10.0, self.length])
    
    def reward(self, obs, action, next_obs):
        x_threshold = 2.4
        theta_threshold_radians = 12 * 2 * math.pi / 360

        cond = next_obs[...,0] > x_threshold
        cond += next_obs[...,0] < -x_threshold 
        cond += next_obs[...,2] > theta_threshold_radians
        cond += next_obs[...,2] < -theta_threshold_radians
        reward = 1 - cond*1
        
        return reward
    
    def log_diagnostics(self, paths, prefix):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
            ]

    def step(self, *args, **kwargs):
        """Wrapper to increment new variable nsteps"""
        self.nsteps += 1
        next=super().step(*args, **kwargs)
        return next[0],next[1],False,next[3]

    def is_success(self):

        """Returns True is current state indicates success, False otherwise
        Balance for at least 195 time steps ("definition" of success in Gym:
        https://github.com/openai/gym/wiki/CartPole-v0#solved-requirements)
        """
        target = 195
        
        if self.nsteps >= target:
            return True
        else:
            return False
    
    def obs_preproc(self, obs):
        return obs
    
    def obs_postproc(self, obs, pred):
        return obs + pred
    
    def targ_proc(self, obs, next_obs):
        return next_obs - obs

class RandomCartPole_Force_Length(ModifiableCartPoleEnv):

    def __init__(self, 
                 force_set=[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], 
                 length_set=[0.4, 0.45, 0.5, 0.55, 0.6]):
        super(RandomCartPole_Force_Length, self).__init__()
        self.proc_observation_space_dims = self.observation_space.shape[0]
        self.force_set = force_set
        self.length_set = length_set
        
        random_index = self.np_random.randint(len(self.force_set))        
        self.force_mag = self.force_set[random_index]
        self.masspole = 0.1
        
        random_index = self.np_random.randint(len(self.length_set))        
        self.length = self.length_set[random_index]

        self._followup()

    def num_modifiable_parameters(self):
        return 2

    def reset(self, new=True):
        self.nsteps = 0  # for super.is_success()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        
        random_index = self.np_random.randint(len(self.force_set))        
        self.force_mag = self.force_set[random_index]
        
        random_index = self.np_random.randint(len(self.length_set))        
        self.length = self.length_set[random_index]
        
        self.masspole = 0.1

        self._followup()
        
        return np.array(self.state)
    
    def reward(self, obs, action, next_obs):
        x_threshold = 2.4
        theta_threshold_radians = 12 * 2 * math.pi / 360

        cond = next_obs[...,0] > x_threshold
        cond += next_obs[...,0] < -x_threshold 
        cond += next_obs[...,2] > theta_threshold_radians
        cond += next_obs[...,2] < -theta_threshold_radians
        reward = 1 - cond*1
        
        return reward

    def tf_reward_fn(self):
        def _thunk(obs, act, next_obs):
            x_threshold = 2.4
            theta_threshold_radians = 12 * 2 * math.pi / 360

            cond1 = next_obs[...,0] > x_threshold
            cond2 = next_obs[...,0] < -x_threshold
            cond3 = next_obs[...,2] > theta_threshold_radians
            cond4 = next_obs[...,2] < -theta_threshold_radians
            cond = tf.cast(cond1, tf.float32) + tf.cast(cond2, tf.float32) + tf.cast(cond3, tf.float32) + tf.cast(cond4, tf.float32)
            reward = 1 - cond*1
            return reward
        return _thunk

    def seed(self, seed=None):
        if seed is None:
            self._seed = 0
        else:
            self._seed = seed
        super().seed(seed)

    def log_diagnostics(self, paths, prefix):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
            ]

################
### Pendulum ###
################
class ModifiablePendulumEnv(PendulumEnv):
    '''The pendulum environment without length and mass of object hard-coded.'''


    def __init__(self):
        super(ModifiablePendulumEnv, self).__init__()

        self.mass = 1.0
        self.length = 1.0
        self.proc_observation_space_dims = self.observation_space.shape[0]
        
    def get_sim_parameters(self):
        return np.array([self.mass, self.length])
    
    def log_diagnostics(self, paths, prefix):
        '''
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
            ]
        '''
        pass

    def reward(self, obs, action, next_obs):
        theta = np.arctan2(obs[...,1],obs[...,0]) # (batch_size,)
        theta_normalize = ((theta + np.pi) % (2 * np.pi)) - np.pi
        thetadot = obs[...,2] # (batch_size, )
        torque = np.clip(action, -self.max_torque, self.max_torque)
        torque = np.reshape(torque, torque.shape[:-1])
        cost = theta_normalize**2 + 0.1*(thetadot)**2 + 0.001*(torque**2) # original
        return -cost
    
    def tf_reward_fn(self):
        def _thunk(obs, action, next_obs):
            theta = tf.math.atan2(obs[...,1],obs[...,0]) # (batch_size,)
            theta_normalize = ((theta + np.pi) % (2 * np.pi)) - np.pi
            thetadot = obs[...,2] # (batch_size, )
            torque = tf.clip_by_value(action, -self.max_torque, self.max_torque)
            torque = tf.reshape(torque, tf.shape(torque)[:-1])
            cost = theta_normalize**2 + 0.1*(thetadot)**2 + 0.001*(torque**2)
            return -cost
        return _thunk

    def step(self, u):
        th, thdot = self.state
        g = 10.0
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        angle_normalize = ((th+np.pi) % (2*np.pi))-np.pi
        
        costs = angle_normalize**2 + .1*thdot**2 + .001*((u/2.0)**2) # original

        newthdot = thdot + (-3*g/(2*self.length) * np.sin(th + np.pi) + 3./(self.mass*self.length**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        normalized = ((newth+np.pi) % (2*np.pi))-np.pi

        self.state = np.array([newth, newthdot])

        # Extra calculations for is_success()
        # TODO(cpacker): be consistent in increment before or after func body
        self.nsteps += 1
        # Track how long angle has been < pi/3
        if -np.pi/3 <= normalized and normalized <= np.pi/3:
            self.nsteps_vertical += 1
        else:
            self.nsteps_vertical = 0
        # Success if if angle has been kept at vertical for 100 steps
        target = 100
        if self.nsteps_vertical >= target:
            #print("[SUCCESS]: nsteps is {}, nsteps_vertical is {}, reached target {}".format(
            #      self.nsteps, self.nsteps_vertical, target))
            self.success = True
        else:
            #print("[NO SUCCESS]: nsteps is {}, nsteps_vertical is {}, target {}".format(
            #      self.nsteps, self.nsteps_vertical, target))
            self.success = False

        return self._get_obs(), -costs, False, {}

    def reset(self, new=True):
        # Extra state for is_success()
        self.nsteps = 0
        self.nsteps_vertical = 0

        low = np.array([(7/8)*np.pi, -0.2])
        high = np.array([(9/8)*np.pi, 0.2])

        theta, thetadot = self.np_random.uniform(low=low, high=high)
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([theta, thetadot])

        self.last_u = None
        return self._get_obs()

    def is_success(self):
        """Returns True if current state indicates success, False otherwise

        Success: keep the angle of the pendulum at most pi/3 radians from
        vertical for the last 100 time steps of a trajectory with length 200
        (max_length is set to 200 in sunblaze_envs/__init__.py)
        """
        return self.success
    
    def obs_preproc(self, obs):
        return obs
    
    def obs_postproc(self, obs, pred):
        return obs + pred
    
    def targ_proc(self, obs, next_obs):
        return next_obs - obs

class RandomPendulumAll(ModifiablePendulumEnv):
    
    def __init__(self, 
                 mass_set=[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.20, 1.25], 
                 length_set=[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.20, 1.25]):

        super(RandomPendulumAll, self).__init__()    

        self.mass_set = mass_set
        self.length_set = length_set
        
        random_index = self.np_random.randint(len(self.mass_set))       
        self.mass = self.mass_set[random_index]

        random_index = self.np_random.randint(len(self.length_set))
        self.length = self.length_set[random_index]


    def seed(self, seed=None):
        if seed is None:
            self._seed = 0
        else:
            self._seed = seed
        super().seed(seed)

    def num_modifiable_parameters(self):
        return 2
    
    def reset(self):
        random_index = self.np_random.randint(len(self.mass_set))
        self.mass = self.mass_set[random_index]

        random_index = self.np_random.randint(len(self.length_set))
        self.length = self.length_set[random_index]

        return super(RandomPendulumAll, self).reset()
