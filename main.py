import math
from time import sleep
import numpy as np
import gymnasium as gym
import rubiks
from rubiks import cube, clear_terminal, scramble_cube, print_cube, moveit, up, down, left, right, front, back, up_prime, down_prime, left_prime, right_prime, front_prime, back_prime
from stable_baselines3 import PPO
import torch as th
import datetime
from gym import spaces

date_time = datetime.datetime.now()
date = date_time.strftime('%m%d%y')

# Define colors and faces
face_keys = ['F', 'R', 'B', 'L', 'U', 'D']
colors = ['White', 'Red', 'Yellow', 'Orange', 'Blue', 'Green']

observation_space = spaces.Box(low=0, high=5, shape=(54,), dtype=np.uint8)

# Use observation space definition to numerically represent Rubik's cube
# def flatten_cube(cube):
#     flattened_cube = np.zeros((54, 6), dtype=np.uint8)
#     for i in range(6):
#         for j in range(9):
#             face = cube[i]
#             color = face[j]
#             flattened_cube[i*9 + j] = color_encoding(color)
#     flattened_cube = flattened_cube.flatten()
#     print(f"flattened_cube: {flattened_cube}")
#     print(f"flattened_cube shape: {flattened_cube.shape}")
#     return flattened_cube

def flatten_cube(cube):
    flattened_cube = np.zeros(54, dtype=np.uint8)
    face_order = ['F', 'R', 'B', 'L', 'U', 'D']
    for face_idx, face in enumerate(face_order):
        for row in range(3):
            for col in range(3):
                color = cube[face][row, col]
                flattened_cube[face_idx * 9 + row * 3 + col] = color
    print(f"flattened_cube: {flattened_cube}")
    print(f"flattened_cube shape: {flattened_cube.shape}")
    return flattened_cube

# NOT NECESSARY SINCE NOT USING ONE HOT
# def color_encoding(color):
#     #gets a one hot encoding for each color numpy
#     if color == "white":
#         return np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
#     elif color == "red":
#         return np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
#     elif color == "yellow":
#         return np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
#     elif color == "orange":
#         return np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
#     elif color == "blue":
#         return np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)
#     elif color == "green":
#         return np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)

class RubiksCubeEnv(gym.Env):
    def __init__(self, scramble=0, time_limit=10):
        self.action_space = spaces.Discrete(12)  # Assuming 12 possible rotations
        self.observation_space = observation_space
        # self.observation_space = gym.spaces.MultiBinary(324)
        self.current_state = np.zeros((324,), dtype=np.uint8)  # Initial solved state
        self.action_to_function = [up, down, left, right, front, back, up_prime, down_prime, left_prime, right_prime, front_prime, back_prime]
        self.cube = self.initialize_cube()
        self.totalsteps = 0

        self.scrambles = scramble
        self.time_limit = time_limit

        self.prev_numscrambles = 0
        self.prev_totalsteps = 0
        self.solved = 0
        self.timedout = 0

    def initialize_cube(self):
        # Initialize the cube
        cube = {
            'F': np.array([[0, 0, 0] for _ in range(3)], dtype=np.uint8),
            'R': np.array([[1, 1, 1] for _ in range(3)], dtype=np.uint8),
            'B': np.array([[2, 2, 2] for _ in range(3)], dtype=np.uint8),
            'L': np.array([[3, 3, 3] for _ in range(3)], dtype=np.uint8),
            'U': np.array([[4, 4, 4] for _ in range(3)], dtype=np.uint8),
            'D': np.array([[5, 5, 5] for _ in range(3)], dtype=np.uint8)
        }
        return cube

    # Used to retrieve the cube's state, ensuring that the state representation is consistent and encapsulated within the class
    def get_state(self):
        flattened_state = []
        for face in ['F', 'R', 'B', 'L', 'U', 'D']:
            flattened_state.extend(self.state[face].flatten())
        return np.array(flattened_state, dtype=np.uint8)

    def reset(self, seed=None):
        numscrambles = self.scrambles

        self.cube = self.initialize_cube()
        self.cube = scramble_cube(self.cube, numscrambles, False)
        state = np.array(list(self.cube.values())).flatten()
        self.current_state = state
        self.time = 0
        self.prev_numscrambles = numscrambles

        return self.current_state, {}

    def is_solved(self):
        for face in self.cube.values():
            # Convert face to a NumPy array if it's not already one
            if isinstance(face, list):
                face = np.array(face)

            # Proceed with the original logic
            reference_encoding = face[0, 0]
            for row in face:
                for color_vector in row:
                    if not np.array_equal(color_vector, reference_encoding):
                        return False  # Found a square that doesn't match the reference, so the cube isn't solved

        return True  # All squares on all faces match their respective references, so the cube is solved

    def step(self, action):
        self.time += 1
        self.totalsteps += 1
        # nummoves = math.floor(min(11 + (self.totalsteps * math.e ** (-self.totalsteps/3000000)) / 40000, 30))
        reward = 0
        # prev_state = self.manhattan_distance(self.cube)

        action = int(action)
        self.action_to_function[action](self.cube) # retrieves action from action_to_function list, passing the current state as an argument
        
        done = self.is_solved()
        time_out = self.time >= self.time_limit  # Limit to this many moves
        
        #if nummoves > math.floor(min(11 + (self.prev_totalsteps * math.e ** (-self.prev_totalsteps/3000000)) / 40000, 30)):
        #    print(f"Allowed {nummoves} steps")

        state = np.array(list(self.cube.values())).flatten()
        
        self.prev_totalsteps = self.totalsteps

        if done:
            reward = 0 # maybe positive reward for solving the cube
            # reward = (1500 / math.log(self.time + 1)) - 300 # Reward for solving the cube based on number of steps
            self.solved += 1
            return state, reward, True, False, {}
        
        # if self.manhattan_distance(self.cube) > prev_state:
        #     reward = (prev_state - self.manhattan_distance(self.cube)) * 0.6 - 2 # need to change this function to increase magnitude as ep_len_mean drops

        # if time_out:
            # print("Cube not solved")

        # elif self.manhattan_distance(self.cube) < prev_state:
        #     reward = (prev_state - self.manhattan_distance(self.cube))

        # reward -= abs(prev_state - self.manhattan_distance(self.cube)) / 12 # max manhattan distance is 26
        reward -= 1
        
        self.timedout += 1
        # max or min manhattan distance for each face might allow it to learn one face at a time
        # bigger NN

        # if time_out:
        #     reward = -100  # Penalty for exceeding the time limit

        return state, reward, time_out, False, {}

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError("Only 'human' mode is currently implemented for rendering.")
        print_cube(self.cube)
    
    def manhattan_distance(self, solved_state):
        total_distance = 0

        # Iterate over each face and calculate discrepancies
        for face in ['F', 'R', 'B', 'L', 'U', 'D']:
            # Count how many facelets per face are not matching the solved state
            face_distance = np.sum(np.argmax(cube[face], axis=2) != np.argmax(solved_state[face], axis=2))
            total_distance += face_distance

        return total_distance

def train_rubiks_cube_solver():
    # Create Rubik's Cube environment
    env = RubiksCubeEnv()
    env.reset()

    # Custom Neural Network Architecture (more complex)
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64, 64], vf=[128, 64, 64]))

    # Create PPO agent
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    print(model.policy)

    training = True
    if training:
        for scrambles in range(1, 6):
            env.scrambles = scrambles
            env.time_limit = scrambles ** 2
            print(f"training with {scrambles} scrambles, time limit: {env.time_limit}")
            env.reset()
            model.learn(total_timesteps=20000 + 20000 * scrambles)

            model.save("models/"f"model-{date}--{scrambles}s")

        # Save the trained model
        model.save("models/" + f"model-{date}-complete")

    testing = False
    if testing:
        stats = []
        for _ in range(1, 11):
            # Load a trained agent to run it
            reloaded_model = PPO.load("models/" + f"model-{date}-complete")

            # Enjoy trained agent
            env.scrambles = 5
            env.time_limit = 10
            obs, _ = env.reset()
            clear_terminal()
            print("Scrambled cube")
            env.render()
            sleep(0.1)
            
            for i in range(100):
                action_index, _ = reloaded_model.predict(obs)
                obs, rewards, done, _, info = env.step(action_index)
                action_function = env.action_to_function[action_index]

                clear_terminal()
                # Apply the action to the environment
                print(action_function.__name__.capitalize().replace("_", " "))

                # Step through the environment using the selected action
                env.render()

                sleep(0.1)

                if done:
                    if env.is_solved():
                        print(f"Rubik's Cube solved in {i+1} moves!")
                        stats.append("1")
                    else:
                        print(f"Agent timed out after {i+1} moves.")
                        stats.append("0")
                    sleep(0.2)
                    break
        print(f"Solves: {stats.count('1')}/{len(stats)}")

if __name__ == "__main__":
    train_rubiks_cube_solver()