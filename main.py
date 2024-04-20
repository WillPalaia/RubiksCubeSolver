import math
import numpy as np
import gymnasium as gym
import rubiks
from rubiks import cube, scramble_cube, print_cube, moveit, onehotstate, up, down, left, right, front, back, up_prime, down_prime, left_prime, right_prime, front_prime, back_prime
from stable_baselines3 import PPO

# Define colors and faces
face_keys = ['F', 'R', 'B', 'L', 'U', 'D']
colors = ['White', 'Red', 'Yellow', 'Orange', 'Blue', 'Green']

# Use observation space definition to numerically represent Rubik's cube
def flatten_cube(cube):
    flattened_cube = np.zeros((54, 6), dtype=np.uint8)
    for i in range(6):
        for j in range(9):
            face = cube[i]
            color = face[j]
            flattened_cube[i*9 + j] = color_encoding(color)
    flattened_cube = flattened_cube.flatten()
    print(f"flattened_cube: {flattened_cube}")
    print(f"flattened_cube shape: {flattened_cube.shape}")
    return flattened_cube

def color_encoding(color):
    #gets a one hot encoding for each color numpy
    if color == "white":
        return np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    elif color == "red":
        return np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
    elif color == "yellow":
        return np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
    elif color == "orange":
        return np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    elif color == "blue":
        return np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)
    elif color == "green":
        return np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)

class RubiksCubeEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)  # Assuming 12 possible rotations
        self.observation_space = gym.spaces.MultiBinary(324)
        # self.observation_space = gym.spaces.MultiBinary(shape=(324,), dtype=np.uint8)  # 6 faces, 3x3 each
        self.current_state = np.zeros((324,), dtype=np.uint8)  # Initial solved state
        self.action_to_function = [up, down, left, right, front, back, up_prime, down_prime, left_prime, right_prime, front_prime, back_prime]
        self.cube = self.initialize_cube()
        self.totalsteps = 0
        self.prev_numscrambles = 0
        self.prev_totalsteps = 0

    def initialize_cube(self):
        # Initialize the cube
        cube = {
            'F': np.array([[[1, 0, 0, 0, 0, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # White
            'R': np.array([[[0, 1, 0, 0, 0, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # Red
            'B': np.array([[[0, 0, 1, 0, 0, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # Yellow
            'L': np.array([[[0, 0, 0, 1, 0, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # Orange
            'U': np.array([[[0, 0, 0, 0, 1, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # Blue
            'D': np.array([[[0, 0, 0, 0, 0, 1] for _ in range(3)] for _ in range(3)], dtype=np.uint8)  # Green
        }
        return cube

    def reset(self, seed=None):
        # print(f"Resetting...")
        numscrambles = min(1 + self.totalsteps // 30000, 20)
        if numscrambles > self.prev_numscrambles:
            print(f"Scrambling {numscrambles} times")
        self.cube = self.initialize_cube()
        self.cube = scramble_cube(self.cube, numscrambles)
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
        
        prev_state = self.manhattan_distance(self.cube)

        action = int(action)
        self.action_to_function[action](self.cube) # retrieves action from action_to_function list, passing the current state as an argument
        # print(f"Action: {self.action_to_function[action]}")

        done = self.is_solved()
        time_out = self.time >= min(5 + self.totalsteps // 30000, 30)  # Limit to this many moves
        if min(5 + self.totalsteps // 30000, 30) > min(5 + self.prev_totalsteps // 30000, 30):
            print(f"Allowed {min(5 + self.totalsteps // 30000, 30)} steps")

        state = np.array(list(self.cube.values())).flatten()
        
        reward = -1 # changed from 0

        self.prev_totalsteps = self.totalsteps

        if done:
            reward = 0
            # reward = (1500 / math.log(self.time + 1)) - 300 # Reward for solving the cube based on number of steps
            return state, reward, done or time_out, False, {}
        
        # if self.manhattan_distance(self.cube) > prev_state:
        #     reward = (prev_state - self.manhattan_distance(self.cube)) * 0.6 - 2 # need to change this function to increase magnitude as ep_len_mean drops

        # elif self.manhattan_distance(self.cube) < prev_state:
        #     reward = (prev_state - self.manhattan_distance(self.cube))
        reward = self.manhattan_distance(self.cube) * -1

        # if time_out:
        #     reward = -100  # Penalty for exceeding the time limit

        return state, reward, done or time_out, False, {}

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError("Only 'console' mode is currently implemented for rendering.")
        # Call the print_cube function with the current cube state
        render = print_cube(self.cube)
        # Print the rendered cube to the console
        print(render)
    
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

    # Create PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    total_timesteps = 1000000
    model.learn(total_timesteps=total_timesteps)

    # scramble_cube(env.cube, 10) # Why is this here?
    # env.reset()

    # Save the trained model
    model.save("rubiks_cube_model-2")

    # model.learn(total_timesteps=total_timesteps) # continue training the model

    # Load the trained agent
    model.load("rubiks_cube_model-2", env=env)

    # Enjoy trained agent
    vec_env = model.get_env()   
    obs = vec_env.reset()
    for i in range(1000):
        action = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
    # print_cube(env.cube)

# env = RubiksCubeEnv()
# solved_state = env.initialize_cube()

# print(f"Is solved? {env.is_solved()}")
# up(env.cube)
# print(f"Is solved? {env.is_solved()}")
# left_prime(env.cube)
# print(f"Is solved? {env.is_solved()}")
# print_cube(env.cube)
# moveit(env.cube, left)

# print(RubiksCubeEnv().manhattan_distance(env.cube))

if __name__ == "__main__":
    train_rubiks_cube_solver()
