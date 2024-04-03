import os
import platform
import random
from time import sleep
import numpy as np
import gymnasium as gym
from gymnasium import spaces

def clear_terminal():
    os_name = platform.system()
    if os_name == "Windows":
        os.system('cls')
    else:
        os.system('clear')

def rotate_face_clockwise(face):
    """Rotates a face 90 degrees clockwise."""
    return [list(row) for row in zip(*face[::-1])]

def rotate_face_counter_clockwise(face):
    """Rotates a face 90 degrees counter-clockwise."""
    return [list(row)[::-1] for row in zip(*face)]

class RubiksCubeEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        # Define colors and faces
        face_keys = ['F', 'R', 'B', 'L', 'U', 'D']
        colors = ['White', 'Red', 'Yellow', 'Orange', 'Blue', 'Green']

        super(RubiksCubeEnv, self).__init__()
        self.action_space = spaces.Discrete(12)  # 6 faces * 2 directions
        self.observation_space = spaces.Box(low=0, high=1, shape=(324,), dtype=np.float32)  # One-hot encoding of the cube state
        self.cube = self.initialize_cube()

        # Initialize the cube
        self.cube = {face: [[colors[i] for _ in range(3)] for _ in range(3)] for i, face in enumerate(face_keys)}
 
    def front_prime(self):
        # Rotate the front face counter-clockwise
        self.cube['F'] = rotate_face_clockwise(self.cube['F'])

        # Temporarily store the edges that will be moved
        temp_top = self.cube['U'][2].copy()  # Copy is used to prevent aliasing
        temp_left = [row[2] for row in self.cube['L']]  # Right column of the left face
        temp_right = [row[0] for row in self.cube['R']]  # Left column of the right face
        temp_bottom = self.cube['D'][0].copy()  # Bottom row of the down face

        # Swap the edges
        self.cube['U'][2] = temp_right[::-1]
        self.cube['D'][0] = temp_left[::-1]

        for i in range(3):
            self.cube['L'][i][2] = temp_top[i]  # Move the top row of the up face to the right column of the left face
            self.cube['R'][i][0] = temp_bottom[2 - i]  # Move the bottom row of the down face to the left column of the right face

    def front(self):
        # Rotate the front face clockwise
        self.cube['F'] = rotate_face_clockwise(self.cube['F'])

        # Temporarily store the edges that will be moved, inverting the process of front_prime
        temp_top = self.cube['U'][2].copy()  # Copy is used to prevent aliasing
        temp_left = [row[2] for row in self.cube['L']]  # Right column of the left face
        temp_right = [row[0] for row in self.cube['R']]  # Left column of the right face
        temp_bottom = self.cube['D'][0].copy()  # Bottom row of the down face

        # Invert the edge swaps compared to front_prime
        self.cube['U'][2] = temp_left
        self.cube['D'][0] = temp_right[::-1]

        for i in range(3):
            self.cube['L'][i][2] = temp_bottom[i]  # Move the bottom row of the down face to the right column of the left face
            self.cube['R'][i][0] = temp_top[2 - i]  # Move the top row of the up face to the left column of the right face, in reverse

    def back_prime(self):
        # Rotate the back face counter-clockwise
        self.cube['B'] = rotate_face_clockwise(self.cube['B'])

        # Temporarily store the edges that will be moved
        temp_top = self.cube['U'][0].copy()  # Top row of the up face, to be moved to the right
        temp_left = [row[0] for row in self.cube['L']]  # Left column of the left face, to be moved down
        temp_right = [row[2] for row in self.cube['R']]  # Right column of the right face, to be moved up
        temp_bottom = self.cube['D'][2].copy()  # Bottom row of the down face, to be moved to the left

        # Swap the edges
        self.cube['U'][0] = temp_left  # Move the left column of the left face to the top row of the up face
        self.cube['D'][2] = temp_right  # Move the right column of the right face to the bottom row of the down face

        for i in range(3):
            self.cube['L'][i][0] = temp_bottom[2 - i]  # Move the bottom row of the down face to the left column of the left face, in reverse order
            self.cube['R'][i][2] = temp_top[i]  # Move the top row of the up face to the right column of the right face

    def back(self):
        # Rotate the back face clockwise
        self.cube['B'] = rotate_face_clockwise(self.cube['B'])

        # Temporarily store the edges that will be moved
        temp_top = self.cube['U'][0].copy()  # Top row of the up face
        temp_left = [row[0] for row in self.cube['L']]  # Left column of the left face
        temp_right = [row[2] for row in self.cube['R']]  # Right column of the right face
        temp_bottom = self.cube['D'][2].copy()  # Bottom row of the down face

        # Swap the edges, adjusting for the clockwise rotation
        self.cube['U'][0] = temp_right[::-1]  # Move the right column of the right face to the top row of the up face, reversed
        self.cube['D'][2] = temp_left[::-1]  # Move the left column of the left face to the bottom row of the down face, reversed

        for i in range(3):
            self.cube['L'][i][0] = temp_top[i]  # Move the top row of the up face to the left column of the left face
            self.cube['R'][i][2] = temp_bottom[2 - i]  # Move the bottom row of the down face to the right column of the right face, in reverse order

    def right(self):
        # Rotate the right face clockwise
        self.cube['R'] = rotate_face_clockwise(self.cube['R'])

        # Temporarily store the edges that will be moved
        temp_top = [row[2] for row in self.cube['U']]  # Right column of the up face
        temp_front = [row[2] for row in self.cube['F']]  # Right column of the front face
        temp_back = [row[2] for row in self.cube['B']]  # Left column of the back face (noted change)
        temp_down = [row[2] for row in self.cube['D']]  # Right column of the down face

        # Swap the edges
        for i in range(3):
            self.cube['U'][i][2] = temp_front[i]
            self.cube['F'][i][2] = temp_down[i]
            self.cube['D'][i][2] = temp_back[2 - i]  # Correctly reverse the order for the back face
            self.cube['B'][2 - i][2] = temp_top[i]  # Correct orientation for the up face to the back face

    def right_prime(self):
        # Rotate the right face counterclockwise
        for _ in range(3):  # Rotate counterclockwise by rotating clockwise 3 times
            self.cube['R'] = rotate_face_clockwise(self.cube['R'])

        # Temporarily store the edges that will be moved, in reverse order compared to 'right'
        temp_top = [row[2] for row in self.cube['U']]  # Right column of the up face
        temp_front = [row[2] for row in self.cube['F']]  # Right column of the front face
        temp_back = [row[2] for row in self.cube['B']]  # Right column of the back face, corrected from left column
        temp_down = [row[2] for row in self.cube['D']]  # Right column of the down face

        # Swap the edges in the reverse direction of the 'right' function
        for i in range(3):
            self.cube['U'][i][2] = temp_back[2 - i]  # Reverse the order for the back face to up face
            self.cube['B'][2 - i][2] = temp_down[i]  # Correct orientation for down face to back face
            self.cube['D'][i][2] = temp_front[i]  # Move the front face to down face directly
            self.cube['F'][i][2] = temp_top[i]  # Move the top face to front face directly

    def left(self):
        # Rotate the left face clockwise
        self.cube['L'] = rotate_face_clockwise(self.cube['L'])

        # Temporarily store the edges that will be moved
        temp_top = [row[0] for row in self.cube['U']]  # Left column of the up face
        temp_front = [row[0] for row in self.cube['F']]  # Left column of the front face
        temp_back = [row[0] for row in self.cube['B']]  # Left column of the back face
        temp_down = [row[0] for row in self.cube['D']]  # Left column of the down face

        # Swap the edges
        for i in range(3):
            self.cube['U'][i][0] = temp_back[2 - i]  # Reverse the order for the back face
            self.cube['F'][i][0] = temp_top[i]
            self.cube['D'][i][0] = temp_front[i]
            self.cube['B'][i][0] = temp_down[2 - i]  # Reverse the order for the down face

    def left_prime(self):
        # Rotate the left face counter-clockwise
        self.cube['L'] = rotate_face_clockwise(self.cube['L'])

        # Temporarily store the edges that will be moved
        temp_top = [row[0] for row in self.cube['U']]  # Left column of the up face
        temp_front = [row[0] for row in self.cube['F']]  # Left column of the front face
        temp_back = [row[0] for row in self.cube['B']]  # Left column of the back face
        temp_down = [row[0] for row in self.cube['D']]  # Left column of the down face

        # Swap the edges in the opposite direction
        for i in range(3):
            self.cube['U'][i][0] = temp_front[i]
            self.cube['F'][i][0] = temp_down[i]
            self.cube['D'][i][0] = temp_back[2 - i]  # Reverse the order for the back face
            self.cube['B'][i][0] = temp_top[2 - i]  # Reverse the order for the up face

    def up(self):
        # Rotate the up face clockwise
        self.cube['U'] = rotate_face_clockwise(self.cube['U'])

        # Temporarily store the edges that will be moved
        temp_front = self.cube['F'][0].copy()  # Top row of the front face
        temp_right = self.cube['R'][0].copy()  # Top row of the right face
        temp_back = self.cube['B'][0].copy()  # Top row of the back face
        temp_left = self.cube['L'][0].copy()  # Top row of the left face

        # Swap the edges
        self.cube['F'][0] = temp_right  # Move the top row of the left face to the front
        self.cube['R'][0] = temp_back  # Move the top row of the front face to the right
        self.cube['B'][0] = temp_left  # Move the top row of the right face to the back
        self.cube['L'][0] = temp_front  # Move the top row of the back face to the left

    def up_prime(self):
        # Rotate the up face counter-clockwise
        self.cube['U'] = rotate_face_clockwise(self.cube['U'])

        # Temporarily store the edges that will be moved
        temp_front = self.cube['F'][0].copy()  # Top row of the front face
        temp_right = self.cube['R'][0].copy()  # Top row of the right face
        temp_back = self.cube['B'][0].copy()  # Top row of the back face
        temp_left = self.cube['L'][0].copy()  # Top row of the left face

        # Swap the edges in the opposite direction
        self.cube['F'][0] = temp_left  # Move the top row of the left face to the front
        self.cube['R'][0] = temp_front  # Move the top row of the front face to the right
        self.cube['B'][0] = temp_right  # Move the top row of the right face to the back
        self.cube['L'][0] = temp_back

    def down_prime(self):
        # Rotate the down face clockwise
        self.cube['D'] = rotate_face_clockwise(self.cube['D'])

        # Temporarily store the edges that will be moved
        temp_front = self.cube['F'][2].copy()  # Bottom row of the front face
        temp_right = self.cube['R'][2].copy()  # Bottom row of the right face
        temp_back = self.cube['B'][2].copy()  # Bottom row of the back face
        temp_left = self.cube['L'][2].copy()  # Bottom row of the left face

        # Swap the edges
        self.cube['F'][2] = temp_right  # Move the bottom row of the right face to the front
        self.cube['R'][2] = temp_back  # Move the bottom row of the back face to the right
        self.cube['B'][2] = temp_left  # Move the bottom row of the left face to the back
        self.cube['L'][2] = temp_front  # Move the bottom row of the front face to the left

    def down(self):
        # Rotate the down face counter-clockwise
        self.cube['D'] = rotate_face_clockwise(self.cube['D'])

        # Temporarily store the edges that will be moved
        temp_front = self.cube['F'][2].copy()  # Bottom row of the front face
        temp_right = self.cube['R'][2].copy()  # Bottom row of the right face
        temp_back = self.cube['B'][2].copy()  # Bottom row of the back face
        temp_left = self.cube['L'][2].copy()  # Bottom row of the left face

        # Swap the edges in the opposite direction
        self.cube['F'][2] = temp_left  # Move the bottom row of the left face to the front
        self.cube['R'][2] = temp_front  # Move the bottom row of the front face to the right
        self.cube['B'][2] = temp_right  # Move the bottom row of the right face to the back
        self.cube['L'][2] = temp_back  # Move the bottom row of the back face to the left

    def step(self, action):
        #given an action, call the correct action function
        #action will be a number between 0-11
        return convert_to_onehot(self.cube)

    def reset(self):
        #make the cube randomized
        return convert_to_onehot(self.cube)

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError("Only 'console' mode is currently implemented for rendering.")
        # Call the print_cube function with the current cube state
        self.print_cube(self.cube)

    def print_cube(cube):
        color_codes = {'White': '\033[97m', 'Yellow': '\033[93m', 'Red': '\033[91m', 'Green': '\033[92m', 'Blue': '\033[94m', 'Orange': '\033[38;2;255;165;0m'}
        reset_code = '\033[0m'
        
        # Helper function to print a single row of three faces horizontally
        def print_row(face1, face2, face3, row):
            for face in [face1, face2, face3]:
                for color in cube[face][row]:
                    print(f"{color_codes[color]}■{reset_code}", end=" ")
                print(" ", end="")  # Space between faces
            print()  # New line after each row

        # Print the Up face (U) alone
        for row in range(3):
            print("       ", end="")
            for color in cube['U'][row]:
                print(f"{color_codes[color]}■{reset_code}", end=" ")
            print("     ")  # New lines and spaces for alignment

        # Print Left (L), Front (F), Right (R), and Back (B) faces
        for row in range(3):
            print_row('L', 'F', 'R', row)
        
        # Print the Down face (D) alone
        for row in range(3):
            print("       ", end="")
            for color in cube['D'][row]:
                print(f"{color_codes[color]}■{reset_code}", end=" ")
            print("     ")  # New lines and spaces for alignment

        # Print the Back face (B) alone for completeness
        for row in cube['B']:
            print("       ", end="")  # Alignment for the Back face
            for color in row:
                print(f"{color_codes[color]}■{reset_code}", end=" ")
            print()  # New line after each row

    move_functions = {
    'F': front,
    'F_prime': front_prime,
    'B': back,
    'B_prime': back_prime,
    'L': left,
    'L_prime': left_prime,
    'R': right,
    'R_prime': right_prime,
    'U': up,
    'U_prime': up_prime,
    'D': down,
    'D_prime': down_prime
}

    def close(self):
        pass

    # Additional move functions would be defined here...

    def is_solved(self):
        # Assuming self.cube is a dictionary where keys are face identifiers
        # and values are 2D arrays representing the colors on each face
        for face in self.cube.values():
            # Get the color of the first square as the reference color for this face
            reference_color = face[0][0]
            # Check if all squares on this face are the same color as the reference square
            if not all(square == reference_color for row in face for square in row):
                return False  # Found a face that is not uniform, cube is not solved
        return True  # All faces are uniform, cube is solved

    @staticmethod
    def initialize_cube():
        # Initialize the cube's state
        pass

rubiks = RubiksCubeEnv()

def scramble_cube(rubiks, num_moves):
    moves = list(rubiks.move_functions.keys())
    movelist = []
    timestep = 0.5

    cubeinit(rubiks.cube)

    for _ in range(num_moves):
        clear_terminal()
        move = random.choice(moves)
        movelist.append(move)
        rubiks.move_functions[move](rubiks.cube)  # Apply the selected move to the cube
        
        print(f"Scrambling {num_moves} times... \n")
        rubiks.print_cube(rubiks.cube)
        sleep(timestep)

    print()
    print(', '.join(movelist))
    return rubiks.cube

def moveit(cube, action, times=1):
    for x in range(times):
        clear_terminal()
        action(cube)
        rubiks.print_cube(cube)
        sleep(0.5)

def cubeinit(rubiks):
    clear_terminal()
    rubiks.print_cube(rubiks.cube)
    sleep(1)

def convert_to_onehot(cube):
    # Define a mapping of colors to one-hot encoding vectors
    color_to_onehot = {
        'White': [1, 0, 0, 0, 0, 0],
        'Red': [0, 1, 0, 0, 0, 0],
        'Yellow': [0, 0, 1, 0, 0, 0],
        'Orange': [0, 0, 0, 1, 0, 0],
        'Blue': [0, 0, 0, 0, 1, 0],
        'Green': [0, 0, 0, 0, 0, 1]
    }
    
    # Initialize an empty dictionary to hold the one-hot encoded cube
    onehot_cube = {}
    
    # Iterate through each face and facelet, converting colors to one-hot encoding
    for face, grid in cube.items():
        onehot_face = []
        for row in grid:
            onehot_row = [color_to_onehot[color] for color in row]
            onehot_face.append(onehot_row)
        onehot_cube[face] = onehot_face
    
    return onehot_cube

def update_and_encode():
    onehot_cube = convert_to_onehot(rubiks.cube)
    print()
    for face, grid in onehot_cube.items():
        print(face)
        for row in grid:
            print(' '.join([''.join(map(str, color)) for color in row]))

''' ACTIONS BELOW '''

# cubeinit(rubiks.cube)
# moveit(rubiks.cube, rubiks.up)
# moveit(rubiks.cube, rubiks.up_prime)
# moveit(rubiks.cube, rubiks.down)
# moveit(rubiks.cube, rubiks.down_prime)
# moveit(rubiks.cube, rubiks.front)
# moveit(rubiks.cube, rubiks.front_prime)
# moveit(rubiks.cube, rubiks.back)
# moveit(rubiks.cube, rubiks.back_prime)
# moveit(rubiks.cube, rubiks.left)
# moveit(rubiks.cube, rubiks.left_prime)
# moveit(rubiks.cube, rubiks.right)
# moveit(rubiks.cube, rubiks.right_prime)

scramble_cube(rubiks, 4)

# update_and_encode()