import random
import numpy as np
import gymnasium as gym
import main
from main import cube, print_cube, up, down, left, right, front, back, up_prime, down_prime, left_prime, right_prime, front_prime, back_prime
from stable_baselines3 import PPO

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

#flattened_cube = rubikscube.print_cube(cube)
#flattened_observation = flatten_cube(cube)
#print(flattened_observation)

class RubiksCubeEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)  # Assuming 12 possible rotations
        self.observation_space = gym.spaces.MultiBinary(324)
        # self.observation_space = gym.spaces.MultiBinary(shape=(324,), dtype=np.uint8)  # 6 faces, 3x3 each
        self.current_state = np.zeros((324,), dtype=np.uint8)  # Initial solved state
        self.action_to_function = [up, down, left, right, front, back, up_prime, down_prime, left_prime, right_prime, front_prime, back_prime]
        self.cube = self.initialize_cube()

    def initialize_cube(self):
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
        print(f"RESET CUBE")
        cube = self.initialize_cube()
        state = np.array(list(cube.values())).flatten()
        self.current_state = state
        self.time = 0
        return self.current_state, {}

    def is_solved(self):
        # Check if the cube is solved
        for face in self.cube.values():
            # Sum over the squares of the face. For a solved face, one color should sum to 9, and others should be 0.
            if not all (np.sum(face, axis=(0, 1)) == 1):
                return False
           #    color_counts = np.sum(face, axis=(0, 1))
        
            # if not np.any(color_counts == 9) or not np.all((color_counts == 9) | (color_counts == 0)):
            #     return False  # Cube is not solved
        return True  # Cube is solved

    def step(self, action):
        self.time += 1
        self.action_to_function[action](self.cube)
        done = self.is_solved()
        time_out = self.time >= 1000  # Limit to 100 moves
        reward = 1 if done else 0  # Simple reward: 1 if solved, 0 otherwise; figure out reward later
        # cube = self.cube() # niko thinks comment this out (gpt told me to)
        state = np.array(list(self.cube.values())).flatten()
        return state, reward, done or time_out, False, {}

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError("Only 'console' mode is currently implemented for rendering.")
        
        # Call the print_cube function with the current cube state
        self.print_cube(self.cube)
        rendered_cube = self.generate_rendered_cube()
    
        # Print the rendered cube to the console
        print(rendered_cube)
    
        # Optionally, return the rendered cube as a string
        return rendered_cube

def train_rubiks_cube_solver():
    # Create Rubik's Cube environment
    env = RubiksCubeEnv()

    # Create PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    total_timesteps = 100000  # Adjust as needed
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    model.save("rubiks_cube_model")

    # Load the trained agent
    model.load("rubiks_cube_model", env=env)

# Enjoy trained agent
    vec_env = model.get_env()   
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")

if __name__ == "__main__":
    #new_cube = RubiksCubeEnv()
    #new_cube.reset()
    train_rubiks_cube_solver()


# from niko:
# lets use moveit() function in render() to visualize the cube
# lets make sure is_solved() works
