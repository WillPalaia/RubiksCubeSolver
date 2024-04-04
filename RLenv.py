import random
import rubikscube
import numpy as np
import gymnasium as gym
from rubikscube import cube
from rubikscube import print_cube
from stable_baselines3 import PPO



# Use observation space definition to numerically represent Rubik's cube
def flatten_cube(cube):
    flattened_cube = np.zeros(54, dtype=np.uint8)
    for i in range(6):
        for j in range(9):
            face = cube[i]
            color = face[j]
            flattened_cube[i*9 + j] = color_encoding(color)
    return flattened_cube

def color_encoding(color):
    color_map = {'white': 0, 'red': 1, 'blue': 2, 'orange': 3, 'green': 4, 'yellow': 5}
    return color_map[color]

#flattened_cube = rubikscube.print_cube(cube)
#flattened_observation = flatten_cube(cube)
#print(flattened_observation)


class RubiksCubeEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)  # Assuming 12 possible rotations
        self.observation_space = gym.spaces.MultiBinary(324)
        # self.observation_space = gym.spaces.MultiBinary(shape=(324,), dtype=np.uint8)  # 6 faces, 3x3 each
        self.current_state = np.zeros((324,), dtype=np.uint8)  # Initial solved state
        self.action_to_function = [move_up, ] #TODO add other moves, import from their code

    def initialize_cube(self):
        cube = {
            'U': np.array([[[1, 0, 0, 0, 0, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # White
            'L': np.array([[[0, 1, 0, 0, 0, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # Red
            'F': np.array([[[0, 0, 1, 0, 0, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # Yellow
            'R': np.array([[[0, 0, 0, 1, 0, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # Orange
            'B': np.array([[[0, 0, 0, 0, 1, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # Blue
            'D': np.array([[[0, 0, 0, 0, 0, 1] for _ in range(3)] for _ in range(3)], dtype=np.uint8)  # Green
        }
        return cube

    def reset(self, seed):
        print(f"RESET CUBE")
        cube = self.initialize_cube()
        state = np.array(list(cube.values())).flatten()
        print(f"state: {state}")
        print(f"state shape: {state.shape}")
        self.current_state = np.zeros((324,), dtype=np.uint8)  # Reset to solved state
        return self.current_state, {}

    def is_solved(self):
        # Check if the cube is solved
        for face in self.cube.values():
            if not all(np.sum(face, axis=(0, 1)) == 1):  # Check if each color appears exactly once on each face
                return False  # Cube is not solved
        return True  # Cube is solved

    def step(self, action):
        self.action_to_function[action](self.cube)
        done = self.is_solved()
        reward = 1 if done else 0  # Simple reward: 1 if solved, 0 otherwise; figure out reward later
        cube = self.cube()
        state = np.array(list(cube.values())).flatten()
        return state, reward, done, False, {}

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
    total_timesteps = 10000  # Adjust as needed
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    model.save("rubiks_cube_model")

    # Load the trained agent
    model.load("rubiks_cube_model", env=env)

# Enjoy trained agent
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")

if __name__ == "__main__":
    #new_cube = RubiksCubeEnv()
    #new_cube.reset()
    train_rubiks_cube_solver()