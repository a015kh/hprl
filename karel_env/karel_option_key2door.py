import os
import numpy as np
import scipy
from scipy import spatial
from collections import deque
from typing import Tuple
import ipdb


state_table = {
    0: 'Karel facing North',
    1: 'Karel facing East',
    2: 'Karel facing South',
    3: 'Karel facing West',
    4: 'Wall',
    5: '0 marker',
    6: '1 marker',
    7: '2 markers',
}

MAX_NUM_MARKER = len(state_table) - 6

action_table = {
    0: 'Move',
    1: 'Turn left',
    2: 'Turn right',
    3: 'Pick up a marker',
    4: 'Put a marker'
}

vars_table = {
    "crash_penalty": -1.0,
    "door_locked": True,
    "number_cells_visited": 0,
    "marker_position": (1, 1),
}


class Karel_world(object):

    def __init__(self, s=None, make_error=False, env_task="program", task_definition='program' ,reward_diff=False, final_reward_scale=True):
        if s is not None:
            self.set_new_state(s)
        self.make_error = make_error
        self.env_task = env_task
        self.task_definition = task_definition
        self.rescale_reward = True
        self.final_reward_scale = final_reward_scale
        self.reward_diff = reward_diff
        self.num_actions = len(action_table)
        self.elapse_step = 0
        self.progress_ratio = 0.0
        self.done = False
        self.actions = {
            0: self.move,
            1: self.turn_left,
            2: self.turn_right,
            3: self.pick_marker,
            4: self.put_marker
        }

        # list all variables for linting
        self.crash_penalty = -1.0
        self.door_locked = True
        self.number_cells_visited = 0
        self.marker_position = (1, 1)

        # variables for program task
        self.set_vars()


    def set_vars(self):
        for var, val in vars_table.items():
            setattr(self, var, val)
    
    def set_new_state(self, s, metadata=None):
        self.elapse_step = 0
        self.perception_count = 0
        self.progress_ratio = 0.0
        self.s = s.astype(bool)
        self.s_h = [self.s.copy()]
        self.a_h = []
        self.h = self.s.shape[0]
        self.w = self.s.shape[1]
        p_v = self.get_perception_vector()
        self.p_v_h = [p_v.copy()]
        self.pos_h = [tuple(self.get_location()[:2])]
        self.pos_h_set = set(self.pos_h)
        self.snake_body = [tuple(self.get_location()[:2])]
        self.snake_len  = 2
        self.set_vars()

        if self.task_definition != "program":
            self.r_h = []
            self.d_h = []
            self.progress_h = []
            self.program_reward = 0.0
            self.prev_pos_reward = 0.0
            self.init_pos_reward = 0.0
            self.done = False
            # self.stage = 0 # For key2door
            self.metadata = metadata
            self.total_markers = np.sum(s[:,:,6:])
            if self.env_task == "snake":
                r, c = np.where(self.s[:, :, 6])
                self.marker_position = (r[0], c[0])


    ###################################
    ###    Collect Demonstrations   ###
    ###################################

    def clear_history(self):
        self.perception_count = 0
        self.elapse_step = 0
        self.progress_ratio = 0.0
        self.s_h = [self.s.copy()]
        self.a_h = []
        self.p_v_h = []
        self.pos_h = [tuple(self.get_location()[:2])]
        self.pos_h_set = set(self.pos_h)
        self.snake_body = []
        self.snake_len  = 2
        self.set_vars()

        if self.task_definition != "program":
            self.r_h = []
            self.progress_h = []
            self.d_h = []
            self.program_reward = 0.0
            self.prev_pos_reward = 0.0
            self.init_pos_reward = 0.0
            self.done = False
            self.total_markers = np.sum(self.s_h[-1][:,:,6:])

    def add_to_history(self, a_idx, agent_pos, made_error=False):
        self.s_h.append(self.s.copy())
        self.a_h.append(a_idx)
        p_v = self.get_perception_vector()
        self.p_v_h.append(p_v.copy())

        self.elapse_step += 1

        if self.task_definition != "program":
            reward, done = self._get_state_reward(agent_pos, made_error)
            # log agent position
            pos_tuple = tuple(agent_pos[:2])
            self.pos_h_set.add(pos_tuple)
            self.pos_h.append(pos_tuple)           
            
            self.done = self.done or done
            self.r_h.append(reward)
            self.progress_h.append(self.progress_ratio)
            self.d_h.append(done)
            self.program_reward += reward

        self.total_markers = np.sum(self.s[:,:,6:]) 
        #if self.task_definition != 'program' and not made_error:
        #    if a_idx == 3: self.total_markers -= 1
        #    if a_idx == 4: self.total_markers += 1

    def _get_cleanHouse_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        total_markers = np.sum(self.s[:,:,6:])
        # terminate if put marker
        if total_markers > self.total_markers:
            self.done = True
            return self.crash_penalty, self.done

        done = False
        pick_marker = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        for mpos in self.metadata['marker_positions']:
            if state[mpos[0], mpos[1], 5] and not state[mpos[0], mpos[1], 6]:
                pick_marker += 1

        current_progress_ratio = pick_marker / float(len(self.metadata['marker_positions']))
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = pick_marker == len(self.metadata['marker_positions'])

        reward = reward if self.env_task == 'cleanHouse' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_harvester_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        # calculate total_1 marker in the state
        max_markers = (w-2)*(h-2)
        total_markers = np.sum(state[:,:,6:])
        if total_markers > self.total_markers:
            self.done = True
            return self.crash_penalty, self.done

        current_progress_ratio = (max_markers - total_markers) / float(max_markers)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = self.total_markers == 0

        reward = reward if self.env_task == 'harvester' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_randomMaze_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        distance_to_goal = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        # initial marker position
        init_state = self.s_h[0]
        x, y = np.where(init_state[:, :, 6] > 0)
        if len(x) != 1: assert 0, '{} markers found!'.format(len(x))
        marker_pos = np.asarray([x[0], y[0]])
        distance_to_goal = -1 * spatial.distance.cityblock(agent_pos[:2], marker_pos)

        done = distance_to_goal == 0
        reward = float(done)
        self.done = self.done or done
        return reward, done

    def _get_fourCorners_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]
        correct_markers = 0
        reward = 0.0
        
        assert not done and not self.done

        # calculate correct markers
        if state[1, 1, 6]:
            correct_markers += 1
        if state[h-2, 1, 6]:
            correct_markers += 1
        if state[h-2, w-2, 6]:
            correct_markers += 1
        if state[1, w-2, 6]:
            correct_markers += 1

        total_markers = np.sum(state[:,:,6:])
        if total_markers > correct_markers:
            self.done = True
            return self.crash_penalty, self.done
        
        current_progress_ratio = correct_markers / 4.0
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio
        done = correct_markers == 4
        
        if self.env_task == 'fourCorners_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward, done

    def _get_stairClimber_task_reward(self, agent_pos):
        # check if already done
        assert self.reward_diff == True

        if self.done:
            return 0.0, self.done

        done = False
        state = self.s_h[-1]

        # initial marker position
        init_state = self.s_h[0]
        x, y = np.where(init_state[:, :, 6] > 0)
        if len(x) != 1: assert 0, '{} markers found!'.format(len(x))
        marker_pos = np.asarray([x[0], y[0]])
        reward = -1 * spatial.distance.cityblock(agent_pos[:2], marker_pos)
        
        # initial agent position
        x, y, z = np.where(self.s_h[0][:, :, :4] > 0)
        init_pos = np.asarray([x[0], y[0], z[0]])
        longest_distance = spatial.distance.cityblock(init_pos[:2], marker_pos)
        assert longest_distance >= 1.0

        # NOTE: need to do this to avoid high negative reward for first action
        if len(self.s_h) == 2:
            x, y, z = np.where(self.s_h[0][:, :, :4] > 0)
            init_pos = np.asarray([x[0], y[0], z[0]])
            self.prev_pos_reward = -1 * spatial.distance.cityblock(init_pos[:2], marker_pos)

        if not self.reward_diff:
            # since reward is based on manhattan distance, rescale it to range between 0 to 1
            if self.rescale_reward:
                from_min, from_max, to_min, to_max = -(sum(self.s.shape[:2])), 0, -1, 0
                reward = ((reward - from_min) * (to_max - to_min) / (from_max - from_min)) + to_min
            if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions']:
                reward = -0.2 # -1.0
            done = reward == 0
        else:
            abs_reward = reward
            reward = self.prev_pos_reward-1.0 if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions'] else reward
            reward = (reward - self.prev_pos_reward) / longest_distance
            assert reward < 1.0, "agent pos: {}, marker_pos: {}, reward: {}, prev_pos_reward: {}".format(agent_pos[:2], marker_pos, reward, self.prev_pos_reward)
            #reward = -1.0 if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions'] else reward
            self.prev_pos_reward = abs_reward
            done = abs_reward == 0

        # # calculate previous distance to the goal
        # # TODO: check why this work
        # x, y, z = np.where(self.s_h[-2][:, :, :4] > 0)
        # prev_pos = np.asarray([x[0], y[0], z[0]])
        # self.prev_pos_reward = -1 * spatial.distance.cityblock(prev_pos[:2], marker_pos)


        # current_progress_ratio = (distance_to_goal - self.prev_pos_reward) / (self.init_pos_reward * -1)
        # reward = current_progress_ratio - self.progress_ratio
        # self.progress_ratio = current_progress_ratio
        # reward = -1.0 if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions'] else reward
        # self.prev_pos_reward = distance_to_goal
        # done = distance_to_goal == 0

        reward = reward if self.env_task == 'stairClimber' else float(done)
        if self.env_task == 'stairClimber_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward, done

    def _get_topOff_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done       

        
        done = False
        score = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        for c in range(1, w-1):
            if (h-2, c) in self.metadata['not_expected_marker_positions']:
                if state[h-2, c, 7]:
                    score += 1
                elif state[h-2, c, 5]:
                    self.done = True
                    return self.crash_penalty, self.done
                

        current_progress_ratio = score / len(self.metadata['not_expected_marker_positions'])
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio
        total_markers = np.sum(state[:,:,6:])
        
        if total_markers > score + len(self.metadata['not_expected_marker_positions']):
            
            self.done = True
            return self.crash_penalty, self.done

        done = score == len(self.metadata['not_expected_marker_positions'])

        reward = reward if self.env_task == 'topOff' else float(done)
        if self.env_task == 'topOff_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward, done


    def _get_randomMaze_key2door_task_reward(self, agent_pos): ## key2door
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        '''
        # initial marker position
        init_state = self.s_h[0]
        x, y = np.where(init_state[:, :, 6] > 0)
        if len(x) != 2: assert 0, '{} markers found!'.format(len(x))

        mxs, mys = np.where(state[:, :, 6] > 0)
        if len(mxs):
            for mx, my in zip(mxs, mys):
                if (mx, my) != (x[0], y[0]) and (mx, my) != (x[1], y[1]):
                    self.done = 1
                    if self.stage == 0:
                        return -0.1, self.done
                    else:
                        return -0.1, self.done

        prev_stage = self.stage
        if self.stage == 0:
            if state[x[0], y[0], 7] or state[x[1], y[1], 7]:
                self.done = 1
                return -0.1, self.done
            elif len(mxs) < 2:
                self._door = (mxs[0], mys[0])
                self._key = (x[0], y[0]) if (self._door == (x[1], y[1])) else (x[1], y[1])
                self.stage = 1
        elif self.stage == 1:
            if not state[self._key[0], self._key[1], 5]:
                self.done = 1
                return -0.1, self.done
            if state[self._door[0], self._door[1], 5]:
                self.done = 1
                return -0.1, self.done
            if state[self._door[0], self._door[1], 7]:
                self.stage = 2
                done = True
        '''
        total_markers = np.sum(state[:,:,6:])
        error_markers = total_markers - 2
        score = 0
        if state[6, 3, 5]: # [1, 3, 7]
            score += 0.5
        if state[6, 3, 5] and state[1, 6, 7]:
            score += 0.5
        
        #for y in range(1, 6):
        #    if state[1, y, 6] or state[1, y, 7]:
        #        score -= 0.1
        #for x in range(2, 6):
        #    if state[x, 3, 6] or state[x, 3, 7]:
        #        score -= 0.1
        if error_markers > 0:
            score -= error_markers * 0.0001

        #if state[6, 3, 7]:
        #    score -= 0.1
        #if state[1, 6, 5]:
        #    score -= 0.1

        current_progress_ratio = score
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = (current_progress_ratio==1.0)

        reward = reward if self.env_task == 'randomMaze_key2door' else float(done)
        if self.env_task == 'randomMaze_key2door_sparse':
            reward = reward if done and not self.done else 0
                
        self.done = self.done or done
        return reward, done

   
    def _get_randomMaze_key2doorSpace_task_reward(self, agent_pos): 
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        total_markers = np.sum(state[:,:,6:])
        error_markers = total_markers - 2
        score = 0
        if state[6, 3, 5]: # [1, 3, 7]
            score += 0.5
        if state[6, 3, 5] and state[1, 6, 7]:
            score += 0.5
        
        #for y in range(1, 6):
        #    if state[1, y, 6] or state[1, y, 7]:
        #        score -= 0.1
        #for x in range(2, 6):
        #    if state[x, 3, 6] or state[x, 3, 7]:
        #        score -= 0.1
        if error_markers > 0:
            score -= error_markers * 0.0001 # penalty

        #if state[6, 3, 7]:
        #    score -= 0.1
        #if state[1, 6, 5]:
        #    score -= 0.1

        current_progress_ratio = score
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = (current_progress_ratio==1.0)

        reward = reward if self.env_task == 'randomMaze_key2doorSpace' else float(done)
        if self.env_task == 'randomMaze_key2doorSpace_sparse':
            reward = reward if done and not self.done else 0
                
        self.done = self.done or done
        return reward, done


    def _get_oneStroke_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done
 
        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]
        pos_tuple = tuple(agent_pos[:2])

        # calculate total_1 marker in the state
        max_markers = (w-2)*(h-2) - 1

        # position is logged after the reward function is called
        prev_pos = self.pos_h[-1]
        if pos_tuple == prev_pos:
            reward = 0.0
        else:
            self.number_cells_visited += 1
            reward = 1.0 / max_markers
            # Place a wall where the agent was
            self.s[prev_pos[0], prev_pos[1], 4] = True

        done = self.number_cells_visited == max_markers
        reward = reward if self.env_task == 'oneStroke' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_doorkey_task_reward(self, agent_pos): ## doorkey
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        reward = 0.0
        total_markers = np.sum(self.s[:,:,6:])
        if self.door_locked:
            if total_markers > 2:
                done = True
                reward = self.crash_penalty
            # Check if key has been picked up
            elif self.s[self.metadata['key'][0], self.metadata['key'][1], 5]:
                self.door_locked = False
                for door_pos in self.metadata['door_positions']: 
                    self.s[door_pos[0], door_pos[1], 4] = False
                reward = 0.5
        else:
            if total_markers > 1:
                # Check if end marker has been topped off
                if self.s[self.metadata['target'][0], self.metadata['target'][1], 7]:
                    reward = 0.5
                else:
                    reward = self.crash_penalty
                done = True
            elif total_markers == 0:
                done = True
                reward = self.crash_penalty
                
        self.done = self.done or done
        return reward, done

    def _get_seeder_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        # calculate total_1 marker in the state
        existing_marker_num = len(self.metadata['existing_marker'])
        max_markers = (w-2)*(h-2) - existing_marker_num
       
        total_one_markers = np.sum(self.s[:,:,6])
        total_two_markers = np.sum(self.s[:,:,7])
        

        if total_two_markers > 0:
            self.done = True
            return self.crash_penalty, self.done
        
        if total_one_markers < self.total_markers:
            self.done = True
            return self.crash_penalty, self.done
        
        score = total_one_markers - existing_marker_num # - total_two_markers * 3

        current_progress_ratio = score / float(max_markers)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = (total_one_markers == max_markers)

        reward = reward if self.env_task == 'seeder' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_snake_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done
 
        done = False
        reward = 0.0

        # Update body and check if it reached marker
        agent_y, agent_x, d = agent_pos
        if (agent_y == self.marker_position[0]) and (
            agent_x == self.marker_position[1]
        ):
            self.snake_len += 1
            self.pick_marker_loc(agent_pos)
            reward = 1 / 20
            if self.snake_len == 20 + 2:
                done = True
            else:
                valid_loc = False
                while not valid_loc:
                    ym = np.random.randint(1, self.h - 1)
                    xm = np.random.randint(1, self.w - 1)
                    if np.sum(self.s[ym, xm, :5]) <= 0 and ((ym, xm) not in self.snake_body):
                        valid_loc = True
                        self.put_marker_loc((ym, xm, d))
                        self.marker_position = (ym, xm)

        last_y, last_x = self.snake_body[-1]
        if (agent_y, agent_x) in self.snake_body[:-1]:
            done = True
            reward = self.crash_penalty
        elif agent_y != last_y or agent_x != last_x:
            self.put_marker_loc((last_y, last_x, d))
            self.snake_body.append((agent_y, agent_x))
            if len(self.snake_body) > self.snake_len:
                first_y, first_x = self.snake_body.pop(0)
                self.pick_marker_loc((first_y, first_x, d))

        reward = reward if self.env_task == 'snake' else float(done)
        self.done = self.done or done
        return reward, done


    def _get_state_reward(self, agent_pos: Tuple[int, int, int], made_error=False):
        if self.env_task == 'cleanHouse' or self.env_task == 'cleanHouse_sparse':
            reward, done = self._get_cleanHouse_task_reward(agent_pos)
        elif self.env_task == 'harvester' or self.env_task == 'harvester_sparse':
            reward, done = self._get_harvester_task_reward(agent_pos)
        elif self.env_task == 'fourCorners' or self.env_task == 'fourCorners_sparse':
            reward, done = self._get_fourCorners_task_reward(agent_pos)
        elif self.env_task == 'randomMaze' or self.env_task == 'randomMaze_sparse':
            reward, done = self._get_randomMaze_task_reward(agent_pos)
        elif self.env_task == 'stairClimber' or self.env_task == 'stairClimber_sparse':
            reward, done = self._get_stairClimber_task_reward(agent_pos)
        elif self.env_task == 'topOff' or self.env_task == 'topOff_sparse':
            reward, done = self._get_topOff_task_reward(agent_pos)
        elif self.env_task == 'randomMaze_key2door' or self.env_task == 'randomMaze_key2door_sparse': 
            reward, done = self._get_randomMaze_key2door_task_reward(agent_pos)
        elif self.env_task == 'randomMaze_key2doorSpace' or self.env_task == 'randomMaze_key2doorSpace_sparse': 
            reward, done = self._get_randomMaze_key2doorSpace_task_reward(agent_pos)
        elif self.env_task == 'oneStroke' or self.env_task == 'oneStroke_sparse': ## oneStroke
            reward, done = self._get_oneStroke_task_reward(agent_pos)
        elif self.env_task == 'doorkey' or self.env_task == 'doorkey_sparse': ## doorkey
            reward, done = self._get_doorkey_task_reward(agent_pos)
        elif self.env_task == 'seeder' or self.env_task == 'seeder_sparse':
            reward, done = self._get_seeder_task_reward(agent_pos)
        elif self.env_task == 'snake' or self.env_task == 'snake_sparse':
            reward, done = self._get_snake_task_reward(agent_pos)
        else:
            raise NotImplementedError('{} task not yet supported'.format(self.env_task))

        return reward, done

    def print_state(self, state=None):
        agent_direction = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        state = self.s_h[-1] if state is None else state
        state_2d = np.chararray(state.shape[:2])
        state_2d[:] = '.'
        state_2d[state[:,:,4]] = 'x'
        state_2d[state[:,:,6]] = 'm'
        state_2d[state[:,:,7]] = 'M'
        x, y, z = np.where(state[:, :, :4] > 0)
        state_2d[x[0], y[0]] = agent_direction[z[0]]

        state_2d = state_2d.decode()
        for i in range(state_2d.shape[0]):
            print("".join(state_2d[i]))

    def render(self, mode='rgb_array'):
        return self.s_h[-1]

    # get location (x, y) and facing {north, east, south, west}
    def get_location(self):
        x, y, z = np.where(self.s[:, :, :4] > 0)
        return np.asarray([x[0], y[0], z[0]])

    # get the neighbor {front, left, right} location
    def get_neighbor(self, face):
        loc = self.get_location()
        if face == 'front':
            neighbor_loc = loc[:2] + {
                0: [-1, 0],
                1: [0, 1],
                2: [1, 0],
                3: [0, -1]
            }[loc[2]]
        elif face == 'left':
            neighbor_loc = loc[:2] + {
                0: [0, -1],
                1: [-1, 0],
                2: [0, 1],
                3: [1, 0]
            }[loc[2]]
        elif face == 'right':
            neighbor_loc = loc[:2] + {
                0: [0, 1],
                1: [1, 0],
                2: [0, -1],
                3: [-1, 0]
            }[loc[2]]
        return neighbor_loc

    ###################################
    ###    Perception Primitives    ###
    ###################################
    # return if the neighbor {front, left, right} of Karel is clear
    def neighbor_is_clear(self, face):
        self.perception_count += 1
        neighbor_loc = self.get_neighbor(face)
        if neighbor_loc[0] >= self.h or neighbor_loc[0] < 0 \
                or neighbor_loc[1] >= self.w or neighbor_loc[1] < 0:
            return False
        return not self.s[neighbor_loc[0], neighbor_loc[1], 4]

    def front_is_clear(self):
        return self.neighbor_is_clear('front')

    def left_is_clear(self):
        return self.neighbor_is_clear('left')

    def right_is_clear(self):
        return self.neighbor_is_clear('right')

    # return if there is a marker presented
    def marker_present(self):
        self.perception_count += 1
        loc = self.get_location()
        return np.sum(self.s[loc[0], loc[1], 6:]) > 0

    def no_marker_present(self):
        self.perception_count += 1
        loc = self.get_location()
        return np.sum(self.s[loc[0], loc[1], 6:]) == 0

    def get_perception_list(self):
        vec = ['frontIsClear', 'leftIsClear',
               'rightIsClear', 'markersPresent',
               'noMarkersPresent']
        return vec

    def get_perception_vector(self):
        vec = [self.front_is_clear(), self.left_is_clear(),
               self.right_is_clear(), self.marker_present(),
               self.no_marker_present()]
        return np.array(vec)

    ###################################
    ###       Action Privitives     ###
    ###################################

    def move(self) -> Tuple[int, int, int]:
        # move
        loc = self.get_location()
        r, c, d = loc
        new_r, new_c = r, c
        if(d == 0): new_r = new_r - 1
        if(d == 1): new_c = new_c + 1
        if(d == 2): new_r = new_r + 1
        if(d == 3): new_c = new_c - 1

        if self.front_is_clear():
            self.s[new_r, new_c, d] = True
            self.s[r, c, d] = False
            return (new_r, new_c, d)
        else:
            if self.make_error:
                raise RuntimeError("Failed to move.")
            self.s[r, c, d] = False
            d = (d + 2) % 4 # Turn 180
            self.s[r, c, d] = True
            return (r, c, d)

    def turn_left(self) -> Tuple[int, int, int]:
        # turn left
        loc = self.get_location()
        r, c, d = loc
        self.s[r, c, d] = False
        d = (d - 1) % 4
        self.s[r, c, d] = True
        return (r, c, d)

    def turn_right(self) -> Tuple[int, int, int]:
        # turn right
        loc = self.get_location()
        r, c, d = loc
        self.s[r, c, d] = False
        d = (d + 1) % 4
        self.s[r, c, d] = True
        return (r, c, d)

    def pick_marker(self) -> Tuple[int, int, int]:
        # pick up a marker
        loc = self.get_location()
        return self.pick_marker_loc(loc)

    def put_marker(self) -> Tuple[int, int, int]:
        # put down a marker
        loc = self.get_location()
        return self.put_marker_loc(loc)

    def pick_marker_loc(self, loc) -> Tuple[int, int, int]:
        r, c, d = loc
        num_marker = np.sum(self.s[r, c, 6:])
        if num_marker > 0:
            self.s[r, c, num_marker + 5] = False # pick up the marker
        else:
            if self.make_error:
                raise RuntimeError("Failed to pick up a marker.")
        self.s[r, c, 5] = np.sum(self.s[r, c, 6:]) == 0
        return (r, c, d)

    def put_marker_loc(self, loc) -> Tuple[int, int, int]:
        r, c, d = loc
        num_marker = np.sum(self.s[r, c, 6:])
        if num_marker < MAX_NUM_MARKER:
            self.s[r, c, num_marker + 5 + 1] = True # put down the marker
        else:
            if self.make_error:
                raise RuntimeError("Failed to put down a marker.")
        self.s[r, c, 5] = np.sum(self.s[r, c, 6:]) == 0
        return (r, c, d)

    ###################################
    ###       State Transition      ###
    ###################################
    # given a state and a action, return the next state
    def state_transition(self, a):
        if self.done:
            raise RuntimeError("Cannot take action in a terminal state.")
        a_idx = np.argmax(a)
        loc = self.actions[a_idx]()
        self.add_to_history(a_idx, loc)

    # given a karel env state, return a visulized image
    def state2image(self, s=None, grid_size=100, root_dir='./'):
        h = s.shape[0]
        w = s.shape[1]
        img = np.ones((h*grid_size, w*grid_size, 1))
        import pickle
        from PIL import Image
        import os.path as osp
        f = pickle.load(open(osp.join(root_dir, 'karel_env/asset/texture.pkl'), 'rb'))
        wall_img = f['wall'].astype('uint8')
        marker_img = f['marker'].astype('uint8')
        agent_0_img = f['agent_0'].astype('uint8')
        agent_1_img = f['agent_1'].astype('uint8')
        agent_2_img = f['agent_2'].astype('uint8')
        agent_3_img = f['agent_3'].astype('uint8')
        blank_img = f['blank'].astype('uint8')
        #blanks
        for y in range(h):
            for x in range(w):
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = blank_img
        # wall
        y, x = np.where(s[:, :, 4])
        for i in range(len(x)):
            img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = wall_img
        # marker
        y, x = np.where(np.sum(s[:, :, 6:], axis=-1))
        for i in range(len(x)):
            img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = marker_img
        # karel
        y, x = np.where(np.sum(s[:, :, :4], axis=-1))
        if len(y) == 1:
            y = y[0]
            x = x[0]
            idx = np.argmax(s[y, x])
            marker_present = np.sum(s[y, x, 6:]) > 0
            if marker_present:
                extra_marker_img = Image.fromarray(f['marker'].squeeze()).copy()
                if idx == 0:
                    extra_marker_img.paste(Image.fromarray(f['agent_0'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_0'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_0'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
                elif idx == 1:
                    extra_marker_img.paste(Image.fromarray(f['agent_1'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_1'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_1'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
                elif idx == 2:
                    extra_marker_img.paste(Image.fromarray(f['agent_2'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_2'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_2'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
                elif idx == 3:
                    extra_marker_img.paste(Image.fromarray(f['agent_3'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_3'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_3'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
            else:
                if idx == 0:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_0']
                elif idx == 1:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_1']
                elif idx == 2:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_2']
                elif idx == 3:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_3']
        elif len(y) > 1:
            raise ValueError
        return img
