import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import re
import os
import sys

def parse_all_digits_file(filename):
    """Parse the all_digits.txt file and organize samples by digit."""
    digit_samples = {i: [] for i in range(10)}
    
    with open(filename, 'r') as f:
        current_digit = None
        current_matrix = []
        
        for line in f:
            line = line.strip()
            
            # Check if this is a new digit sample header
            digit_match = re.match(r'# Digit (\d+) \(Sample (\d+)\)', line)
            if digit_match:
                if current_digit is not None and current_matrix:
                    # Save the previous matrix
                    digit_samples[current_digit].append(current_matrix)
                
                current_digit = int(digit_match.group(1))
                current_matrix = []
                continue
            
            # Check if this is a matrix row
            if line.startswith('.float'):
                # Extract all float values (more precise pattern)
                float_values = []
                # Split the line after '.float' and process the numbers
                number_part = line[6:].strip()
                # Split on commas and strip whitespace
                number_strings = [s.strip() for s in number_part.split(',')]
                for num_str in number_strings:
                    try:
                        float_val = float(num_str)
                        float_values.append(float_val)
                    except ValueError:
                        # Skip any malformed entries
                        continue
                if float_values:
                    current_matrix.append(float_values)
        
        # Add the last matrix if exists
        if current_digit is not None and current_matrix:
            digit_samples[current_digit].append(current_matrix)
    
    return digit_samples

def display_digit_image(matrix, digit):
    """Display the digit as a grayscale image."""
    # Set the backend based on the OS
    if os.name == 'nt':  # Windows
        matplotlib.use('TkAgg')
    else:  # Linux/Unix
        matplotlib.use('Agg')  # Non-interactive backend
    
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Test Image - Digit {digit}")
    plt.axis('off')
    
    # Save the figure if we can't display it
    if os.name != 'nt':
        plt.savefig(f'digit.png')
        print(f"\n Image saved to digit.png (display not available)")
    else:
        plt.show()
    plt.close()

def save_to_inc_file(matrix, digit, filename='inputs.inc'):
    """Save the matrix to inputs.inc in the specified format."""
    with open(filename, 'w') as f:
        f.write(f"input_matrix:\n")
        f.write(f"    # Digit {digit}\n")
        for row in matrix:
            row_str = ', '.join([f"{x:.6f}" for x in row])
            f.write(f"    .float {row_str}\n")
    print(f"\n Matrix saved to {filename}\n\n")

def main():
    # Parse the all_digits.txt file
    try:
        digit_samples = parse_all_digits_file('all_digits.txt')
    except FileNotFoundError:
        print("Error: all_digits.txt not found in current directory.")
        print("Current directory:", os.getcwd())
        return
    
    # Prompt user for digit
    while True:
        try:
            print("\n")
            digit = int(input("Enter digit to test (0-9): "))
            if 0 <= digit <= 9:
                break
            else:
                print("Please enter a digit between 0 and 9.")
        except ValueError:
            print("Please enter a valid integer.")
    
    # Get all samples for the selected digit
    samples = digit_samples.get(digit, [])
    if not samples:
        print(f"No samples found for digit {digit}")
        return
    
    # Select a random sample
    selected_matrix = random.choice(samples)
    
    # Convert to numpy array for display
    matrix_array = np.array(selected_matrix)
    
    # Verify matrix dimensions
    if matrix_array.shape != (28, 28):
        print(f"Warning: Matrix shape is {matrix_array.shape}, expected (28, 28)")
    
    # Display the image (handles both interactive and non-interactive cases)
    display_digit_image(matrix_array, digit)
    
    # Save to inputs.inc
    save_to_inc_file(selected_matrix, digit)

if __name__ == "__main__":
    main()
