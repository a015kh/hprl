import os
import time
import torch
import numpy as np
import random
import sys
import pandas as pd
import ipdb

# sys.path.insert(0, '..')
sys.path.insert(0, '.')

from pretrain.customargparse import CustomArgumentParser, args_to_dict
from fetch_mapping import fetch_mapping
from rl.envs import make_vec_envs


from karel_env.dsl import get_DSL_option_v2
from karel_env.generator_option_key2door import KarelStateGenerator
# from d2p_karel_env.dsl import get_KarelDSL
from karel_env import karel_option_key2door as karel
from pygifsicle import optimize
import imageio
from PIL import Image


def _temp(config, args):

    args.task_file = config['rl']['envs']['executable']['task_file']
    args.grammar = config['dsl']['grammar']
    args.use_simplified_dsl = config['dsl']['use_simplified_dsl']
    args.task_definition = config['rl']['envs']['executable']['task_definition']
    args.execution_guided = config['rl']['policy']['execution_guided']



if __name__ == "__main__":
    
    torch.set_num_threads(1)

    t_init = time.time()
    parser = CustomArgumentParser(description='syntax learner')

    # Add arguments (including a --configfile)
    parser.add_argument('-o', '--outdir',
                        help='Output directory for results', default='karel_demo')
    parser.add_argument('-c', '--configfile',
                        help='Input file for parameters, constants and initial settings')
    
    parser.add_argument('-v', '--verbose',
                        help='Increase output verbosity', action='store_true')
    parser.add_argument('--num_demo', type=int,
                        help='Demo number', default=10)

    parser.add_argument('--program_dir', type=str,
                        help='File path/name of the program', default="abs")
    
    parser.add_argument('--draw', action='store_true', help='draw the gif')

    parser.add_argument('--evaluate', action='store_true', help='evaluate the program similarity')
    parser.add_argument('--dsl', type=str, help='dsl type', default='hprl')
    tasks = "fourCorners topOff_sparse harvester randomMaze stairClimber_sparse doorkey oneStroke seeder cleanHouse snake"
    tasks = tasks.split()

    # ipdb.set_trace()

    write = 0
    result = {}

    missing_task = False

    for task_name in tasks:
        parser.add_argument("--env_task",help="task",default=task_name)
        
        args = parser.parse_args()
        if task_name == "cleanHouse":
            args.input_height = 14
            args.input_width = 22
        config = args_to_dict(args)
        config['args'] = args
        
        _temp(config, args)
        
        # TODO: shift this logic somewhere else
        # encode reward along with state and action if task defined by custom reward
        config['rl']['envs']['executable']['dense_execution_reward'] = config['rl']['envs']['executable'][
                                                                        'task_definition'] == 'custom_reward'
        # FIXME: This is only for backwards compatibility to old parser, should be removed once we change the original
        # args.outdir = os.path.join(args.outdir, '%s-%s-%s-%s' % (args.prefix, args.grammar, args.seed, time.strftime("%Y%m%d-%H%M%S")))
        if config['device'].startswith('cuda') and torch.cuda.is_available():
            device = torch.device(config['device'])
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')

        config["env_task"] = task_name
        custom = True if "karel" or "CartPoleDiscrete" in config['env_name'] else False

        # config['args'][]
        config['args'].env_task = task_name
        custom_kwargs = {"config": config['args']}

        exp_name = os.path.split(args.program_dir)[-1]

        output_dir = args.outdir
        args.outdir = os.path.join(args.outdir, exp_name)

        # fetch the mapping from prl tokens to dsl tokens
        if args.mapping_file is not None:
            args.dsl2prl_mapping, args.prl2dsl_mapping, args.dsl_tokens, args.prl_tokens = \
                fetch_mapping(args.mapping_file)
            args.use_simplified_dsl = True
            args.use_shorter_if = True if 'shorter_if' in args.mapping_file else False
        else:
            _, _, args.dsl_tokens, _ = fetch_mapping('mapping_karel2prl_new_vae_v2.txt')
            args.use_simplified_dsl = False

        # print("config['args'] :  ",config['args'].env_task)

        # Call the main method


        if config['seed'] is not None:
            seed = config['seed']
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)

        dsl = get_DSL_option_v2(seed=seed, environment=config['rl']['envs']['executable']['name'])
        config['dsl']['num_agent_actions'] = len(dsl.action_functions) + 1      # +1 for a no-op action, just for filling



        textfile = os.path.join(args.program_dir, f"{task_name}.txt")
        
        best_id = -1
        best_program = ""
        best_reward = -1

        if not os.path.exists(textfile):
            # print(f"{textfile} not exists")
            missing_task = True
            continue

        task_output_dir = os.path.join(args.outdir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)

        with open(textfile, 'r') as f:
            predictions = f.readlines()

        all_rewards = []
        for k, prediction in enumerate(predictions):
            prediction = prediction.strip()
            
            program_str = prediction
            rewards = float(0)
            
            # try:
            program_tokens = torch.from_numpy(np.array(dsl.str2intseq(program_str)[1:], dtype=np.int8))
            
            envs = make_vec_envs(config['env_name'], config['seed'], 1,
            config['gamma'], None, device, False, custom_env=custom,
            custom_kwargs=custom_kwargs)
            
            obs = envs.reset()

            for i in range(config['num_demo']):
                exec_dict = dict()
                for j in range(config['max_episode_steps']):
                    ### program_str = "DEF run m( WHILE c( rightIsClear c) w( turnRight w) WHILE c( noMarkersPresent c) w( WHILE c( frontIsClear c) w( move w) IF c( noMarkersPresent c) i( putMarker turnLeft move i) w) m)"
                    # program_tokens = torch.from_numpy(np.array(dsl.str2intseq(program_str)[1:], dtype=np.int8))
                    action = torch.unsqueeze(program_tokens, 0).repeat(1, 1).to(device)

                    obs, reward, done, infos = envs.step(action)
                    if 'episode' in infos[0].keys():
                        rewards += float(infos[0]['episode']['r'])
                        exec_dict['s_h'] = np.array(infos[0]['exec_data']['s_image_h_list'])

            average_rewards = rewards/config['num_demo']    

            if average_rewards > best_reward:
                best_reward = average_rewards
                best_id = k
                best_program = program_str

            # except RuntimeError as e:
            #     print(f"Error in {task_name} : {e}")
            #     average_rewards = -1
            #     if average_rewards > best_reward:
            #         best_reward = average_rewards
            #         best_id = k
            #         best_program = program_str
            all_rewards.append(average_rewards)
        
        # s_gen = KarelStateGenerator()
        # s, x, y, wall, metadata = s_gen.generate_single_state_seeder()
        # karel_world = karel.Karel_world(task_definition=config['env_task'], env_task=config['env_task'], reward_diff=True, make_error=False)
        # karel_world.set_new_state(s, metadata=metadata)
        # exe = dsl.parse(best_program)
        # exe(karel_world)
        
        # draw the GIF
        if best_program != "" and args.draw:
            program_tokens = torch.from_numpy(np.array(dsl.str2intseq(best_program)[1:], dtype=np.int8))
            obs = envs.reset()
            action = torch.unsqueeze(program_tokens, 0).repeat(1, 1).to(device)
            karel_world = karel.Karel_world()

            for i in range(config['num_demo']):
                exec_dict = dict()
                for j in range(config['max_episode_steps']):
                    obs, reward, done, infos = envs.step(action)
                    if 'episode' in infos[0].keys():
                        exec_dict['s_h'] = np.array(infos[0]['exec_data']['s_image_h_list'])
                frames = []
                for s in exec_dict['s_h']:
                    frames.append(Image.fromarray(karel_world.state2image(s=s, root_dir='.').squeeze()))

                path = os.path.join(task_output_dir, f"{task_name}-{i:02d}.gif")
                frames[0].save(path, save_all=True, append_images=frames[1:], optimize=False, duration=75, loop=0)
                print(f"Saved {path}")

        # print the result
        print(f"For task {task_name}: ")
        print(f"The {best_id}th program : {best_program} ")
        print(f"get highest rewards {best_reward}")



    