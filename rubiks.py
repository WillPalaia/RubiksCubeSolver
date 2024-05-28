# hello world
import numpy as np
import random
import os
import platform
from time import sleep

# Define colors and faces
face_keys = ['F', 'R', 'B', 'L', 'U', 'D']
colors = ['White', 'Red', 'Yellow', 'Orange', 'Blue', 'Green']
# Initialize the cube
# cube = {face: [[colors[i] for _ in range(3)] for _ in range(3)] for i, face in enumerate(face_keys)}
# change cube to nparray and fix all functions
cube = {
    'F': np.array([[[1, 0, 0, 0, 0, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # White
    'R': np.array([[[0, 1, 0, 0, 0, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # Red
    'B': np.array([[[0, 0, 1, 0, 0, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # Yellow
    'L': np.array([[[0, 0, 0, 1, 0, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # Orange
    'U': np.array([[[0, 0, 0, 0, 1, 0] for _ in range(3)] for _ in range(3)], dtype=np.uint8),  # Blue
    'D': np.array([[[0, 0, 0, 0, 0, 1] for _ in range(3)] for _ in range(3)], dtype=np.uint8)  # Green
}

''' EPIC DESIGN '''
# cube = {
#     'F': np.array([
#         [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]],
#         [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
#         [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]]
#     ], dtype=np.uint8),
#     'R': np.array([
#         [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
#         [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]],
#         [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
#     ], dtype=np.uint8),
#     'B': np.array([
#         [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
#         [[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
#         [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]]
#     ], dtype=np.uint8),
#     'L': np.array([
#         [[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
#         [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
#         [[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
#     ], dtype=np.uint8),
#     'U': np.array([
#         [[0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]],
#         [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0]],
#         [[0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
#     ], dtype=np.uint8),
#     'D': np.array([
#         [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0]],
#         [[0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]],
#         [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0]]
#     ], dtype=np.uint8)
# }


def clear_terminal():
    os_name = platform.system()
    if os_name == "Windows":
        os.system('cls')
    else:
        os.system('clear')

def rotate_face_clockwise(face):
    """Rotates a face 90 degrees clockwise using numpy."""
    return np.rot90(face, k=-1)

def front_prime(cube):
    # Rotate the front face counter-clockwise
    for _ in range(3):
        cube['F'] = rotate_face_clockwise(cube['F'])

    # Temporarily store the edges that will be moved
    temp_top = cube['U'][2].copy()  # Copy is used to prevent aliasing
    temp_left = cube['L'][:, 2].copy()  # Last column of the left face
    temp_right = cube['R'][:, 0].copy()  # First column of the right face
    temp_bottom = cube['D'][0].copy()  # Bottom row of the down face

    # Swap the edges
    cube['U'][2] = temp_right
    cube['D'][0] = temp_left

    # In numpy, you can directly assign the reversed slices
    cube['L'][:, 2] = temp_top[::-1]
    cube['R'][:, 0] = temp_bottom[::-1]

def front(cube):
    # Rotate the front face clockwise
    cube['F'] = rotate_face_clockwise(cube['F'])

    # Temporarily store the edges that will be moved, inverting the process of front_prime
    temp_top = cube['U'][2].copy()  # Copy is used to prevent aliasing
    temp_left = cube['L'][:, 2].copy()  # Last column of the left face
    temp_right = cube['R'][:, 0].copy()  # First column of the right face
    temp_bottom = cube['D'][0].copy()  # Top row of the down face

    # Invert the edge swaps compared to front_prime
    cube['U'][2] = temp_left[::-1]
    cube['D'][0] = temp_right[::-1]

    cube['L'][:, 2] = temp_bottom  # Move the bottom row of the down face to the right column of the left face
    cube['R'][:, 0] = temp_top  # Move the top row of the up face to the left column of the right face

def back(cube):
    # Rotate the back face counter-clockwise
    cube['B'] = rotate_face_clockwise(cube['B'])
    # Temporarily store the edges that will be moved
    temp_top = cube['U'][0].copy()  # Copy to prevent aliasing
    temp_left = cube['L'][:, 0].copy()  # First column of the left face
    temp_right = cube['R'][:, 2].copy()  # Last column of the right face
    temp_bottom = cube['D'][2].copy()  # Top row of the down face

    # Adjust the edges according to the back move
    cube['U'][0] = temp_right[::]
    cube['D'][2] = temp_left[::]

    for i in range(3):
        cube['L'][i][0] = temp_top[2 - i]
        cube['R'][i][2] = temp_bottom[2 - i]

# Correct orientation and swapping of edges for back_prime
def back_prime(cube):
    # Rotate the back face clockwise
    for _ in range(3):
        cube['B'] = rotate_face_clockwise(cube['B'])

    # Temporarily store the edges that will be moved
    temp_top = cube['U'][0].copy()  # Copy to prevent aliasing
    temp_left = cube['L'][:, 0].copy()  # First column of the left face
    temp_right = cube['R'][:, 2].copy()  # Last column of the right face
    temp_bottom = cube['D'][2].copy()  # Top row of the down face

    # Adjust the edges according to the back_prime move
    cube['U'][0] = temp_left[::-1]
    cube['D'][2] = temp_right[::-1]

    for i in range(3):
        cube['L'][i][0] = temp_bottom[i]
        cube['R'][i][2] = temp_top[i]

def right(cube):
    cube['R'] = rotate_face_clockwise(cube['R'])

    # Temporarily store the edges that will be moved
    temp_top = cube['U'][:, 2].copy()  # Right column of the up face
    temp_front = cube['F'][:, 2].copy()  # Right column of the front face
    temp_back = cube['B'][:, 0].copy()  # Left column of the back face
    temp_down = cube['D'][:, 2].copy()  # Right column of the down face

    # Swap the edges
    cube['U'][:, 2] = temp_front
    cube['F'][:, 2] = temp_down
    cube['D'][:, 2] = temp_back[::-1]  # Reverse the order for the back face
    cube['B'][::-1, 0] = temp_top  # Reverse the rows and assign the top values

def right_prime(cube):
    # Rotate the right face counterclockwise
    for _ in range(3):
        cube['R'] = rotate_face_clockwise(cube['R'])

    # Temporarily store the edges that will be moved
    temp_top = cube['U'][:, 2].copy()  # Right column of the up face
    temp_front = cube['F'][:, 2].copy()  # Right column of the front face
    temp_back = cube['B'][:, 0].copy()  # Left column of the back face
    temp_down = cube['D'][:, 2].copy()  # Right column of the down face

    # Swap the edges in the reverse direction
    cube['U'][:, 2] = temp_back[::-1]  # Reverse the order for the back face to the up face
    cube['B'][::-1, 0] = temp_down  # Reverse the rows and assign the down values
    cube['D'][:, 2] = temp_front  # Move the front face to the down face directly
    cube['F'][:, 2] = temp_top

def left(cube):
    # Rotate the left face counter-clockwise (equivalent to 3 clockwise rotations)
    cube['L'] = rotate_face_clockwise(cube['L'])

    # Temporarily store the edges that will be moved using numpy slicing
    temp_top = cube['U'][:, 0].copy()  # Left column of the up face
    temp_front = cube['F'][:, 0].copy()  # Left column of the front face
    temp_back = cube['B'][:, 2].copy()  # Right column of the back face
    temp_down = cube['D'][:, 0].copy()  # Left column of the down face

    # Swap the edges
    cube['U'][:, 0] = temp_back[::-1]
    cube['F'][:, 0] = temp_top
    cube['D'][:, 0] = temp_front
    cube['B'][::-1, 2] = temp_down  # Reverse the rows and assign the down values

def left_prime(cube):
    # Rotate the left face clockwise
    for _ in range(3):
        cube['L'] = rotate_face_clockwise(cube['L'])

    # Temporarily store the edges that will be moved using numpy slicing
    temp_top = cube['U'][:, 0].copy()  # Left column of the up face
    temp_front = cube['F'][:, 0].copy()  # Left column of the front face
    temp_back = cube['B'][:, 2].copy()  # Right column of the back face
    temp_down = cube['D'][:, 0].copy()  # Left column of the down face

    # Swap the edges in the reverse direction
    cube['U'][:, 0] = temp_front
    cube['F'][:, 0] = temp_down
    cube['D'][:, 0] = temp_back[::-1]
    cube['B'][::-1, 2] = temp_top  # Reverse the rows and assign the top values

def up(cube):
    # Rotate the up face clockwise
    cube['U'] = rotate_face_clockwise(cube['U'])

    # Temporarily store the edges that will be moved
    temp_front = cube['F'][0].copy()  # Top row of the front face
    temp_right = cube['R'][0].copy()  # Top row of the right face
    temp_back = cube['B'][0].copy()  # Top row of the back face
    temp_left = cube['L'][0].copy()  # Top row of the left face

    # Swap the edges
    cube['F'][0] = temp_right  # Move the top row of the left face to the front
    cube['R'][0] = temp_back  # Move the top row of the front face to the right
    cube['B'][0] = temp_left  # Move the top row of the right face to the back
    cube['L'][0] = temp_front  # Move the top row of the back face to the left

def up_prime(cube):
    for _ in range(3):
        cube['U'] = rotate_face_clockwise(cube['U'])

    # Temporarily store the edges that will be moved
    temp_front = cube['F'][0].copy()  # Top row of the front face
    temp_right = cube['R'][0].copy()  # Top row of the right face
    temp_back = cube['B'][0].copy()  # Top row of the back face
    temp_left = cube['L'][0].copy()  # Top row of the left face

    # Swap the edges in the opposite direction
    cube['F'][0] = temp_left  # Move the top row of the left face to the front
    cube['R'][0] = temp_front  # Move the top row of the front face to the right
    cube['B'][0] = temp_right  # Move the top row of the right face to the back
    cube['L'][0] = temp_back

def down_prime(cube):
    # Rotate the down face clockwise
    for _ in range(3):
        cube['D'] = rotate_face_clockwise(cube['D'])

    # Temporarily store the edges that will be moved
    temp_front = cube['F'][2].copy()  # Bottom row of the front face
    temp_right = cube['R'][2].copy()  # Bottom row of the right face
    temp_back = cube['B'][2].copy()  # Bottom row of the back face
    temp_left = cube['L'][2].copy()  # Bottom row of the left face

    # Swap the edges
    cube['F'][2] = temp_right  # Move the bottom row of the right face to the front
    cube['R'][2] = temp_back  # Move the bottom row of the back face to the right
    cube['B'][2] = temp_left  # Move the bottom row of the left face to the back
    cube['L'][2] = temp_front  # Move the bottom row of the front face to the left

def down(cube):
    # Rotate the down face counter-clockwise
    cube['D'] = rotate_face_clockwise(cube['D'])

    # Temporarily store the edges that will be moved
    temp_front = cube['F'][2].copy()  # Bottom row of the front face
    temp_right = cube['R'][2].copy()  # Bottom row of the right face
    temp_back = cube['B'][2].copy()  # Bottom row of the back face
    temp_left = cube['L'][2].copy()  # Bottom row of the left face

    # Swap the edges in the opposite direction
    cube['F'][2] = temp_left  # Move the bottom row of the left face to the front
    cube['R'][2] = temp_front  # Move the bottom row of the front face to the right
    cube['B'][2] = temp_right  # Move the bottom row of the right face to the back
    cube['L'][2] = temp_back  # Move the bottom row of the back face to the left

color_codes = {
    'White': '\033[97m',  # White
    'Green': '\033[92m',  # Green
    'Red': '\033[91m',  # Red
    'Blue': '\033[94m',  # Blue
    'Orange': '\033[38;2;255;165;0m',  # Orange
    'Yellow': '\033[93m'   # Yellow
}

reset_code = '\033[0m'

def print_cube(cube):
    # Map one-hot indices to color names for printing
    colors = ['White', 'Red', 'Yellow', 'Orange', 'Blue', 'Green']
    color_codes = {'White': '\033[97m', 'Yellow': '\033[93m', 'Red': '\033[91m', 'Green': '\033[92m', 'Blue': '\033[94m', 'Orange': '\033[38;2;255;165;0m'}
    reset_code = '\033[0m'

    # Helper function to find the color from one-hot encoding
    def get_color(one_hot_vector):
        index = np.argmax(one_hot_vector)
        return colors[index]

    # Helper function to print a single row of three faces horizontally
    def print_row(face1, face2, face3, row):
        for face in [face1, face2, face3]:
            for color_vector in cube[face][row]:
                color = get_color(color_vector)
                print(f"{color_codes[color]}■{reset_code}", end=" ")
            print(" ", end="")  # Space between faces
        print()  # New line after each row

    # Print the Up face (U) alone
    for row in range(3):
        print("       ", end="")
        for color_vector in cube['U'][row]:
            color = get_color(color_vector)
            print(f"{color_codes[color]}■{reset_code}", end=" ")
        print("     ")  # New lines and spaces for alignment

    # Print Left (L), Front (F), Right (R), and Back (B) faces
    for row in range(3):
        print_row('L', 'F', 'R', row)
    
    # Print the Down face (D) alone
    for row in range(3):
        print("       ", end="")
        for color_vector in cube['D'][row]:
            color = get_color(color_vector)
            print(f"{color_codes[color]}■{reset_code}", end=" ")
        print("     ")  # New lines and spaces for alignment

    # Print the Back face (B) alone for completeness
    for row in cube['B']:
        print("       ", end="")  # Alignment for the Back face
        for color_vector in row:
            color = get_color(color_vector)
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

def scramble_cube(cube, num_moves, printscramble=True):
    moves = list(move_functions.keys())
    movelist = []
    timestep = 0.5

    # cubeinit(cube)

    for _ in range(num_moves):
        # clear_terminal()
        move = random.choice(moves)
        movelist.append(move)
        move_functions[move](cube)  # Apply the selected move to the cube
        
        # print(f"Scrambling {num_moves} times... \n")
        # print_cube(cube)
        # sleep(timestep)

    # print()
    if printscramble:
        print("Scramble:", ', '.join(movelist))

    # Invert and reverse the move list
    # inverted_movelist = [invert_move(move) for move in reversed(movelist)]
    # Create a string representation of the reversed and inverted move list
    # reversed_moves_str = ', '.join(inverted_movelist)
    # print("Undo:", reversed_moves_str)

    return cube

def invert_move(move):
    if "_prime" in move:
        return move.replace("_prime", "")
    else:
        return move + "_prime"

def moveit(cube, action, times=1):
    for x in range(times):
        # clear_terminal()
        action(cube)
        # print_cube(cube)
        # sleep(0.5)

def cubeinit(cube):
    # clear_terminal()
    print_cube(cube)
    sleep(1)
    None

def onehotstate(cube):
    concatenated_array = np.concatenate([cube[face].flatten() for face in ['U', 'D', 'F', 'B', 'L', 'R']])
    return concatenated_array

''' ACTIONS BELOW '''

print_cube(cube)
# cubeinit(cube)
# print_cube(cube)

# DESIGNED CUBE
# moveit(cube, right, 2)
# moveit(cube, left, 2)
# moveit(cube, up, 2)
# moveit(cube, down, 2)
# moveit(cube, front, 2)
# moveit(cube, back, 2)

# moveit(cube, down_prime)
# moveit(cube, up)
# moveit(cube, up_prime)
# moveit(cube, front)
# moveit(cube, front_prime)
# moveit(cube, back)
# moveit(cube, back_prime)
# moveit(cube, right)
# moveit(cube, right_prime)
# moveit(cube, left)
# moveit(cube, left_prime)

# scramble_cube(cube, 5)
# print_cube(cube)
# print(onehotstate(cube))
