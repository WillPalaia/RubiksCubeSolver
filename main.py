import random
import os
import platform
from time import sleep

# Define colors and faces
face_keys = ['F', 'R', 'B', 'L', 'U', 'D']
colors = ['White', 'Red', 'Yellow', 'Orange', 'Blue', 'Green']
# Initialize the cube
cube = {face: [[colors[i] for _ in range(3)] for _ in range(3)] for i, face in enumerate(face_keys)}

def clear_terminal():
    os_name = platform.system()
    if os_name == "Windows":
        os.system('cls')
    else:
        os.system('clear')

def rotate_face_clockwise(face):
    """Rotates a face 90 degrees clockwise."""
    return [list(row) for row in zip(*face[::-1])]

def front_prime(cube):
    # Rotate the front face counter-clockwise
    for _ in range(3):
        cube['F'] = rotate_face_clockwise(cube['F'])

    # Temporarily store the edges that will be moved
    temp_top = cube['U'][2].copy()  # Copy is used to prevent aliasing
    temp_left = [row[2] for row in cube['L']]  # Right column of the left face
    temp_right = [row[0] for row in cube['R']]  # Left column of the right face
    temp_bottom = cube['D'][0].copy()  # Bottom row of the down face

    # Swap the edges
    cube['U'][2] = temp_right[::-1]
    cube['D'][0] = temp_left[::-1]

    for i in range(3):
        cube['L'][i][2] = temp_top[i]  # Move the top row of the up face to the right column of the left face
        cube['R'][i][0] = temp_bottom[2 - i]  # Move the bottom row of the down face to the left column of the right face

def front(cube):
    # Rotate the front face clockwise
    cube['F'] = rotate_face_clockwise(cube['F'])

    # Temporarily store the edges that will be moved, inverting the process of front_prime
    temp_top = cube['U'][2].copy()  # Copy is used to prevent aliasing
    temp_left = [row[2] for row in cube['L']]  # Right column of the left face
    temp_right = [row[0] for row in cube['R']]  # Left column of the right face
    temp_bottom = cube['D'][0].copy()  # Bottom row of the down face

    # Invert the edge swaps compared to front_prime
    cube['U'][2] = temp_left
    cube['D'][0] = temp_right[::-1]

    for i in range(3):
        cube['L'][i][2] = temp_bottom[i]  # Move the bottom row of the down face to the right column of the left face
        cube['R'][i][0] = temp_top[2 - i]  # Move the top row of the up face to the left column of the right face, in reverse

def back_prime(cube):
    # Rotate the back face counter-clockwise
    for _ in range(3):
        cube['B'] = rotate_face_clockwise(cube['B'])

    # Temporarily store the edges that will be moved
    temp_top = cube['U'][0].copy()  # Top row of the up face, to be moved to the right
    temp_left = [row[0] for row in cube['L']]  # Left column of the left face, to be moved down
    temp_right = [row[2] for row in cube['R']]  # Right column of the right face, to be moved up
    temp_bottom = cube['D'][2].copy()  # Bottom row of the down face, to be moved to the left

    # Swap the edges
    cube['U'][0] = temp_left  # Move the left column of the left face to the top row of the up face
    cube['D'][2] = temp_right  # Move the right column of the right face to the bottom row of the down face

    for i in range(3):
        cube['L'][i][0] = temp_bottom[2 - i]  # Move the bottom row of the down face to the left column of the left face, in reverse order
        cube['R'][i][2] = temp_top[i]  # Move the top row of the up face to the right column of the right face

def back(cube):
    # Rotate the back face clockwise
    cube['B'] = rotate_face_clockwise(cube['B'])

    # Temporarily store the edges that will be moved
    temp_top = cube['U'][0].copy()  # Top row of the up face
    temp_left = [row[0] for row in cube['L']]  # Left column of the left face
    temp_right = [row[2] for row in cube['R']]  # Right column of the right face
    temp_bottom = cube['D'][2].copy()  # Bottom row of the down face

    # Swap the edges, adjusting for the clockwise rotation
    cube['U'][0] = temp_right[::-1]  # Move the right column of the right face to the top row of the up face, reversed
    cube['D'][2] = temp_left[::-1]  # Move the left column of the left face to the bottom row of the down face, reversed

    for i in range(3):
        cube['L'][i][0] = temp_top[i]  # Move the top row of the up face to the left column of the left face
        cube['R'][i][2] = temp_bottom[2 - i]  # Move the bottom row of the down face to the right column of the right face, in reverse order

def right(cube):
    # Rotate the right face clockwise
    cube['R'] = rotate_face_clockwise(cube['R'])

    # Temporarily store the edges that will be moved
    temp_top = [row[2] for row in cube['U']]  # Right column of the up face
    temp_front = [row[2] for row in cube['F']]  # Right column of the front face
    temp_back = [row[2] for row in cube['B']]  # Left column of the back face (noted change)
    temp_down = [row[2] for row in cube['D']]  # Right column of the down face

    # Swap the edges
    for i in range(3):
        cube['U'][i][2] = temp_front[i]
        cube['F'][i][2] = temp_down[i]
        cube['D'][i][2] = temp_back[2 - i]  # Correctly reverse the order for the back face
        cube['B'][2 - i][2] = temp_top[i]  # Correct orientation for the up face to the back face

def right_prime(cube):
    # Rotate the right face counterclockwise
    for _ in range(3):
        cube['R'] = rotate_face_clockwise(cube['R'])

    # Temporarily store the edges that will be moved, in reverse order compared to 'right'
    temp_top = [row[2] for row in cube['U']]  # Right column of the up face
    temp_front = [row[2] for row in cube['F']]  # Right column of the front face
    temp_back = [row[2] for row in cube['B']]  # Right column of the back face, corrected from left column
    temp_down = [row[2] for row in cube['D']]  # Right column of the down face

    # Swap the edges in the reverse direction of the 'right' function
    for i in range(3):
        cube['U'][i][2] = temp_back[2 - i]  # Reverse the order for the back face to up face
        cube['B'][2 - i][2] = temp_down[i]  # Correct orientation for down face to back face
        cube['D'][i][2] = temp_front[i]  # Move the front face to down face directly
        cube['F'][i][2] = temp_top[i]  # Move the top face to front face directly

def left(cube):
    # Rotate the left face clockwise
    cube['L'] = rotate_face_clockwise(cube['L'])

    # Temporarily store the edges that will be moved
    temp_top = [row[0] for row in cube['U']]  # Left column of the up face
    temp_front = [row[0] for row in cube['F']]  # Left column of the front face
    temp_back = [row[0] for row in cube['B']]  # Left column of the back face
    temp_down = [row[0] for row in cube['D']]  # Left column of the down face

    # Swap the edges
    for i in range(3):
        cube['U'][i][0] = temp_back[2 - i]  # Reverse the order for the back face
        cube['F'][i][0] = temp_top[i]
        cube['D'][i][0] = temp_front[i]
        cube['B'][i][0] = temp_down[2 - i]  # Reverse the order for the down face

def left_prime(cube):
    # Rotate the left face counter-clockwise
    for _ in range(3):
        cube['L'] = rotate_face_clockwise(cube['L'])

    # Temporarily store the edges that will be moved
    temp_top = [row[0] for row in cube['U']]  # Left column of the up face
    temp_front = [row[0] for row in cube['F']]  # Left column of the front face
    temp_back = [row[0] for row in cube['B']]  # Left column of the back face
    temp_down = [row[0] for row in cube['D']]  # Left column of the down face

    # Swap the edges in the opposite direction
    for i in range(3):
        cube['U'][i][0] = temp_front[i]
        cube['F'][i][0] = temp_down[i]
        cube['D'][i][0] = temp_back[2- i]  # Reverse the order for the back face
        cube['B'][2 - i][0] = temp_top[i]  # Reverse the order for the up face

def up(cube):
    # Rotate the up face clockwise
    cube['U'] = rotate_face_clockwise(cube['U'])

    # Temporarily store the edges that will be moved
    temp_front = cube['F'][0].copy()  # Top row of the front face
    temp_right = cube['R'][2].copy()  # Top row of the right face
    temp_back = cube['B'][2].copy()  # Top row of the back face
    temp_left = cube['L'][2].copy()  # Top row of the left face

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

def scramble_cube(cube, num_moves):
    moves = list(move_functions.keys())
    movelist = []
    timestep = 1

    cubeinit(cube)

    for _ in range(num_moves):
        clear_terminal()
        move = random.choice(moves)
        movelist.append(move)
        move_functions[move](cube)  # Apply the selected move to the cube
        
        print(f"Scrambling {num_moves} times... \n")
        print_cube(cube)
        sleep(timestep)

    print()
    print(', '.join(movelist))
    return cube

def moveit(cube, action, times=1):
    for x in range(times):
        clear_terminal()
        action(cube)
        print_cube(cube)
        sleep(0.5)

def cubeinit(cube):
    clear_terminal()
    print_cube(cube)
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
    onehot_cube = convert_to_onehot(cube)
    print()
    for face, grid in onehot_cube.items():
        print(face)
        for row in grid:
            print(' '.join([''.join(map(str, color)) for color in row]))

''' ACTIONS BELOW '''

cubeinit(cube)

moveit(cube, up)
moveit(cube, left_prime)
# moveit(cube, up)
#bug description:
#R and L sides reflected across y axis
#D is flipped x axis
#Problem is most likely with the cube environment, not the actions themselves

# moveit(cube, up_prime)

# moveit(cube, down)
# moveit(cube, down_prime)
# moveit(cube, front)
# moveit(cube, front_prime)
# moveit(cube, back)
# moveit(cube, back_prime)
# moveit(cube, left)
# moveit(cube, left_prime)
# moveit(cube, right)
# moveit(cube, right_prime)

# scramble_cube(cube, 2)

# update_and_encode()
