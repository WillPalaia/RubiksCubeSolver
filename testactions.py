import unittest
from main import RubiksCubeEnv  # Adjust this import statement as necessary
from rubiks import up
import numpy as np

class TestRubiksCubeActions(unittest.TestCase):
    
    def setUp(self):
        """Setup method to create a fresh environment for each test."""
        self.env = RubiksCubeEnv()

    def test_up_action(self):
        """Test the 'up' action to ensure it correctly manipulates the cube's state."""
        # Step 1: Initialize the cube to a specific state
        # For simplicity, starting with a solved state might make it easier to predict the outcome.
        self.env.reset()  # Assuming this sets the cube to a solved state
        
        # Step 2: Apply the 'up' action
        # Assuming your actions are indexed in self.action_to_function, find the index for 'up'
        up_action_index = self.env.action_to_function.index(up)  # Adjust as necessary
        self.env.step(up_action_index)
        
        # Step 3: Check the cube's state
        # This part depends on your cube's representation and what you expect to happen after an 'up' action
        # Example:
        expected_state = ... # Define the expected state of the cube here
        current_state = self.env.cube  # Adjust if your state is retrieved differently
        self.assertEqual(current_state, expected_state, "Up action did not result in the expected state.")

if __name__ == "__main__":
    unittest.main()
