# Rubik's Cube Solver

A machine learning project that implements a digital Rubik's Cube solver using reinforcement learning techniques, specifically the Proximal Policy Optimization (PPO) algorithm. The solver creates a virtual 3×3 Rubik's Cube environment, scrambles it, and employs deep reinforcement learning to find optimal solutions.

## 🎯 Features

- **Digital Rubik's Cube Environment**: Complete 3D cube simulation with one-hot encoded color representation
- **Reinforcement Learning**: PPO algorithm implementation with custom neural network architecture
- **Progressive Training**: Curriculum learning approach with increasing scramble complexity
- **Visual Feedback**: Colored terminal output for cube visualization
- **Model Persistence**: Save and load trained models for continued training or testing
- **Performance Metrics**: Success rate tracking and solving statistics

## 🚀 Quick Start

### Prerequisites

Install the required packages:

```bash
pip install gymnasium
pip install stable-baselines3
pip install numpy
pip install torch
```

### Running the Project

Execute the main script to start training or testing:

```bash
python main.py
```

## 📁 Project Structure

```
RubiksCubeSolver/
├── main.py           # Main training and testing script
├── rubiks.py         # Rubik's cube implementation and move functions
├── models/           # Directory containing trained model files
│   ├── model-*.zip   # Saved PPO models for different scramble levels
├── README.md         # Project documentation
└── LICENSE           # MIT License
```

## 🧠 Technical Implementation

### Cube Representation

The Rubik's Cube is represented using a dictionary structure where each face is a 3×3 NumPy array with one-hot encoded colors:

- **White**: `[1, 0, 0, 0, 0, 0]`
- **Red**: `[0, 1, 0, 0, 0, 0]`
- **Yellow**: `[0, 0, 1, 0, 0, 0]`
- **Orange**: `[0, 0, 0, 1, 0, 0]`
- **Blue**: `[0, 0, 0, 0, 1, 0]`
- **Green**: `[0, 0, 0, 0, 0, 1]`

### Available Moves

The implementation supports all standard Rubik's Cube moves:
- **Face Rotations**: F, R, B, L, U, D (clockwise)
- **Prime Moves**: F', R', B', L', U', D' (counter-clockwise)

### Reinforcement Learning Environment

The [`RubiksCubeEnv`](main.py) class implements a Gymnasium environment with:
- **Action Space**: 12 discrete actions (6 face rotations + 6 prime moves)
- **Observation Space**: 324-dimensional binary vector (54 squares × 6 colors)
- **Reward System**: Negative reward per step (-1) to encourage efficiency
- **Episode Termination**: Success (cube solved) or timeout (step limit reached)

### Neural Network Architecture

The PPO agent uses a custom neural network with:
- **Policy Network**: 5 hidden layers of 256 neurons each
- **Value Network**: 5 hidden layers of 256 neurons each
- **Activation Function**: ReLU
- **Algorithm**: Proximal Policy Optimization (PPO)

## 🎮 Usage Examples

### Training a New Model

To train a model with progressive difficulty:

```python
# Set training = True in main.py
training = True
if training:
    for scrambles in range(1, 21):
        env.scrambles = scrambles
        env.time_limit = scrambles ** 2
        model.learn(total_timesteps=50000 * scrambles)
        model.save(f"models/model-{date}--50k-{scrambles}s")
```

### Testing a Trained Model

To test a model's performance:

```python
# Set testing = True in main.py
testing = True
if testing:
    # Load a trained model
    reloaded_model = PPO.load("models/model-050824--4s")
    
    # Test on 4-move scrambles
    env.scrambles = 4
    env.time_limit = 16
    # ... testing loop
```

### Manual Cube Manipulation

You can also manually interact with the cube:

```python
from rubiks import cube, front, right, up, print_cube

# Perform moves
front(cube)
right(cube)
up(cube)

# Display the cube
print_cube(cube)
```

## 📊 Key Functions

### Core Cube Operations

- [`initialize_cube()`](main.py): Creates a solved cube state
- [`scramble_cube()`](rubiks.py): Randomly scrambles the cube with N moves
- [`is_solved()`](main.py): Checks if the cube is in solved state
- [`print_cube()`](rubiks.py): Displays the cube with colored output

### Move Functions

All move functions are available in [`rubiks.py`](rubiks.py):
- [`front()`](rubiks.py), [`front_prime()`](rubiks.py)
- [`right()`](rubiks.py), [`right_prime()`](rubiks.py)
- [`back()`](rubiks.py), [`back_prime()`](rubiks.py)
- [`left()`](rubiks.py), [`left_prime()`](rubiks.py)
- [`up()`](rubiks.py), [`up_prime()`](rubiks.py)
- [`down()`](rubiks.py), [`down_prime()`](rubiks.py)

### Utility Functions

- [`onehotstate()`](rubiks.py): Converts cube to flattened observation vector
- [`clear_terminal()`](rubiks.py): Cross-platform terminal clearing
- [`rotate_face_clockwise()`](rubiks.py): NumPy-based face rotation

## 🔧 Configuration

### Environment Parameters

- `scramble`: Number of scramble moves (default: 0)
- `time_limit`: Maximum steps per episode (default: 10)

### Training Parameters

- `total_timesteps`: Training duration per difficulty level
- `policy_kwargs`: Neural network architecture settings
- `verbose`: Training output verbosity

## 📈 Model Performance

The project includes pre-trained models for different scramble complexities:
- `model-*--1s.zip`: 1-move scrambles
- `model-*--2s.zip`: 2-move scrambles
- ...up to 8+ move scrambles

Success rates vary by scramble complexity, with simpler scrambles achieving higher solve rates.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Reward Engineering**: Implement Manhattan distance or other heuristics
2. **Advanced Algorithms**: Experiment with A3C, SAC, or other RL algorithms
3. **Curriculum Learning**: Improve training progression strategies
4. **Performance Optimization**: Enhance solving efficiency and success rates

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Gymnasium for the RL environment framework
- Stable-Baselines3 for the PPO implementation
- NumPy for efficient array operations

---

**Note**: This is an educational project demonstrating the application of reinforcement learning to combinatorial puzzles. The current implementation focuses on learning and experimentation rather than optimal solving performance.
