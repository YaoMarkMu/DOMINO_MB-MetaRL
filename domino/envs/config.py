from domino.envs.normalized_env import normalize
from domino.envs import *


def get_environment_config(config):
    if config["dataset"] == "halfcheetah":
        train_mass_scale_set = [0.75,0.85,1.0,1.15,1.25]
        train_damping_scale_set = [0.75,0.85,1.0,1.15,1.25]
        config["num_test"] = 4
        config["test_range"] = [
            [[0.4, 0.5], [0.4, 0.5]],
            [[0.4, 0.5], [1.5, 1.6]],
            [[1.5, 1.6], [0.4, 0.5]],
            [[1.5, 1.6], [1.5, 1.6]],
        ]
        env = HalfCheetahEnv(
            mass_scale_set=train_mass_scale_set,
            damping_scale_set=train_damping_scale_set,
        )
        env.seed(config["seed"])
        env = normalize(env)
        config["n_itr"] = 10
        config["total_test"] = 10
        config["test_num_rollouts"] = 10
        config["max_path_length"] = 1000
        config["simulation_param_dim"] = 2

    elif config['dataset'] == 'ant':
        train_mass_scale_set = [0.75,0.85,1.0,1.15,1.25]
        train_damping_scale_set = [0.75,0.85,1.0,1.15,1.25]
        config['num_test'] = 4
        config["test_range"] = [
            [[0.4, 0.5], [0.4, 0.5]],
            [[0.4, 0.5], [1.5, 1.6]],
            [[1.5, 1.6], [0.4, 0.5]],
            [[1.5, 1.6], [1.5, 1.6]],
        ]
        config['total_test'] = 20
        config['test_num_rollouts'] = 20
        config['max_path_length'] = 1000
        config['total_timesteps'] = 5000000
        config["simulation_param_dim"] = 2

        env = AntEnv(mass_scale_set=train_mass_scale_set, damping_scale_set=train_damping_scale_set)
        env.seed(config['seed'])
        env = normalize(env)


    elif config["dataset"] == "cripple_ant":
        cripple_set = [0, 1, 2]
        extreme_set = [0]
        mass_scale_set= [0.75,0.85,1.0,1.15,1.25]
        config["num_test"] = 2
        config["test_range"] = [
            [[3], [0],[0.4, 0.5]],
            [[3], [0],[1.5, 1.6]],
        ]
        env = CrippleAntEnv(cripple_set=cripple_set, extreme_set=extreme_set,mass_scale_set=mass_scale_set)
        env.seed(config["seed"])
        env = normalize(env)
        config["n_itr"] = 10
        config["total_test"] = 10
        config["test_num_rollouts"] = 10
        config["max_path_length"] = 1000
        config["simulation_param_dim"] = 1

    elif config['dataset'] == 'cripple_halfcheetah':
        cripple_set = [0, 1, 2, 3]
        extreme_set = [0]
        mass_scale_set= [0.75,0.85,1.0,1.15,1.25]
        config['num_test'] = 2
        config['test_range'] = [
            [[4, 5], [0],[0.4, 0.5]],
            [[4, 5], [0],[1.5, 1.6]],
        ]
        config["n_itr"] = 10
        config['total_test'] = 20
        config['test_num_rollouts'] = 10
        config['max_path_length'] = 1000
        config['total_timesteps'] = 5000000
        config["simulation_param_dim"] = 1
        env = CrippleHalfCheetahEnv(cripple_set=cripple_set, extreme_set=extreme_set,mass_scale_set=mass_scale_set)
        env.seed(config['seed'])
        env = normalize(env)

    elif config["dataset"] == "slim_humanoid":
        train_mass_scale_set = [0.8, 0.9, 1.0, 1.15, 1.25]
        train_damping_scale_set = [0.8, 0.9, 1.0, 1.15, 1.25]
        config["num_test"] = 4
        config["test_range"] = [
            [[0.6, 0.7], [0.6, 0.7]],
            [[0.6, 0.7], [1.5, 1.6]],
            [[1.5, 1.6], [0.6, 0.7]],
            [[1.5, 1.6], [1.5, 1.6]],
        ] 
        env = SlimHumanoidEnv(
            mass_scale_set=train_mass_scale_set,
            damping_scale_set=train_damping_scale_set,
        )
        env.seed(config["seed"])
        env = normalize(env)
        config["n_itr"] = 10
        config["total_test"] = 10
        config["test_num_rollouts"] = 10
        config["max_path_length"] = 1000
        config["simulation_param_dim"] = 2

    elif config["dataset"] == "hopper":
        train_mass_scale_set = [0.5, 0.75, 1.0, 1.25, 1.5]
        train_damping_scale_set = [0.5, 0.75, 1.0, 1.25, 1.5]
        config["num_test"] = 4
        config["test_range"] = [
            [[0.25, 0.375], [0.25, 0.375]],
            [[0.25, 0.375], [1.75, 2.0]],
            [[1.75, 2.0], [0.25, 0.375]],
            [[1.75, 2.0], [1.75, 2.0]],
        ]
        env = HopperEnv(
            mass_scale_set=train_mass_scale_set,
            damping_scale_set=train_damping_scale_set,
        )
        env.seed(config["seed"])
        env = normalize(env)
        config["total_test"] = 10
        config["test_num_rollouts"] = 10
        config["max_path_length"] = 500
        config["simulation_param_dim"] = 2

    elif config["dataset"] == "cartpole":
        train_force_scale_set = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        train_length_scale_set = [0.4, 0.45, 0.5, 0.55, 0.6]
        config["num_test"] = 4
        config["test_range"] = [
            [[3.0, 3.5], [0.25, 0.3]],
            [[3.0, 3.5], [0.7, 0.75]],
            [[16.5, 17.0], [0.25, 0.3]],
            [[16.5, 17.0], [0.7, 0.75]],
        ]
        env = Cartpoleenvs(train_force_scale_set, train_length_scale_set,
        )
        env.seed(config["seed"])
        env = normalize(env)
        config["total_test"] = 10
        config["test_num_rollouts"] = 10
        config["max_path_length"] = 500
        config["simulation_param_dim"] = 2

    elif config["dataset"] == "pendulum":
        train_mass_set=[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.20, 1.25]
        train_length_set=[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.20, 1.25]

        # train_mass_scale_set = [0.5, 0.75, 1.0, 1.25, 1.5]
        # train_length_scale_set = [1.0]
        config["num_test"] = 4
        config["test_range"] = [
            [[0.5, 0.7], [0.5, 0.7]],
            [[0.5, 0.7], [1.3, 1.5]],
            [[1.3, 1.5], [0.5, 0.7]],
            [[1.3, 1.5], [1.3, 1.5]],
        ]
        env = Pendulumenvs(train_mass_set, train_length_set,
        )
        env.seed(config["seed"])
        env = normalize(env)
        config["total_test"] = 10
        config["test_num_rollouts"] = 10
        config["max_path_length"] = 500
        config["simulation_param_dim"] = 2

    else:
        raise ValueError(config["dataset"])

    return env, config
