import sys
import random
import tensorflow as tf
import numpy as np
import os
import struct
from sklearn.model_selection import train_test_split

# Function to generate a random matrix
def generate_matrix(size):
    """Generates a square matrix of given size with random floats between -1000 and 1000, rounded to 0.25 increments."""
    matrix = [[round(random.uniform(-1000, 1000) * 4) / 4 for _ in range(size)] for _ in range(size)]
    return matrix

# Function to read and modify the assembly file
def read_and_modify_file(filename, matrix, size, type):
    """Reads the file, replaces the matrix data, and writes a modified version."""
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Find the markers
    start_idx = end_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "## DATA DEFINE START":
            start_idx = i
        elif line.strip() == "## DATA DEFINE END":
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        raise ValueError("Start or end marker not found in the file.")

    # Create the matrix data
    matrix_data = [f".equ MatrixSize, {size}\n", "matrix:\n"]
    for row in matrix:
        row_data = ", ".join(map(str, row))
        matrix_data.append(f"    .float {row_data}\n")

    # Replace the data between the markers
    modified_lines = lines[:start_idx + 1] + matrix_data + lines[end_idx:]

    # Write the modified file
    savefilename = "assembly/VectorizedModified.s" if type == "V" else "assembly/NonVectorizedModified.s"
    
    with open(savefilename, 'w') as file:
        file.writelines(modified_lines)

# # Function to load and preprocess MNIST data
# def load_mnist_data(*, test_size=0.1):
#     base_path = '/home/ubuntu/IBA-Project/MNIST/archive'
#     train_images_path = os.path.join(base_path, 'train-images.idx3-ubyte')
#     train_labels_path = os.path.join(base_path, 'train-labels.idx1-ubyte')
#     test_images_path = os.path.join(base_path, 't10k-images.idx3-ubyte')
#     test_labels_path = os.path.join(base_path, 't10k-labels.idx1-ubyte')

#     train_images = idx3_ubyte_to_numpy(train_images_path)
#     train_labels = idx1_ubyte_to_numpy(train_labels_path)
#     test_images = idx3_ubyte_to_numpy(test_images_path)
#     test_labels = idx1_ubyte_to_numpy(test_labels_path)

#     train_images, val_images, train_labels, val_labels = train_test_split(
#         train_images, train_labels, test_size=test_size, random_state=42)

#     print(test_images[0])
#     train_images, test_images = train_images / 255.0, test_images / 255.0
#     val_images = val_images / 255.0
#     print(test_images[0])

#     return train_images, train_labels, val_images, val_labels, test_images, test_labels

def write_input(test_image_index=0, output_file='inputs.txt'):
    """
    Write the test image at `test_image_index` from `test_images` into an assembly-style text file.
    
    Args:
    - test_images: The dataset of test images (normalized).
    - test_image_index: Index of the test image to write.
    - output_file: The file to write the image to.
    """
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_mnist_data()

    # Select the test image (28x28)
    test_image = test_images[test_image_index]
    test_label = test_labels[test_image_index]
    
    # Flatten the image to 1D array
    flattened_image = test_image.flatten()  # Shape: (28*28,)
    
    # Open the output file for writing
    with open(output_file, 'w') as f:
        # Write the header for the assembly data
        f.write('.section .data\n')
        f.write('.global input_image\n')
        f.write('input_image:\n')
        
        # Write each pixel as a byte (if values are between 0 and 1, scale to 0-255)
        for pixel_value in flattened_image:
            byte_value = int(pixel_value * 255)  # Scale from [0, 1] to [0, 255]
            f.write(f'    .byte {byte_value}\n')
        f.write(f'## Label: {test_label}\n')

    print(f"Test image {test_image_index} written to {output_file}")


# # Function to read IDX file for images
# def idx3_ubyte_to_numpy(file_path):
#     with open(file_path, 'rb') as f:
#         magic, num_items, rows, cols = struct.unpack('>IIII', f.read(16))
#         data = np.fromfile(f, dtype=np.uint8).reshape(num_items, rows, cols)
#     return data

# # Function to read IDX file for labels
# def idx1_ubyte_to_numpy(file_path):
#     with open(file_path, 'rb') as f:
#         magic, num_items = struct.unpack('>II', f.read(8))
#         labels = np.fromfile(f, dtype=np.uint8)
#     return labels

# Function to train the CNN and extract weights
def train_and_extract_weights():
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_mnist_data()

    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        tf.keras.layers.Conv2D(8, (5, 5), activation='relu', padding='valid'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=5, batch_size=128)

    conv_layer = model.layers[1]
    conv_weights, conv_biases = conv_layer.get_weights()  # conv_weights shape: (5, 5, 1, 8)

    fc_layer = model.layers[-1]
    W_fc, b_fc = fc_layer.get_weights()

    np.save('weights.npy', W_fc)
    np.save('biases.npy', b_fc)
    print("Weights and biases saved.")
    print("Weights Shape:", W_fc.shape)
    print("Biases Shape:", b_fc.shape)

    model.save('mnist_cnn_model.h5')
    print("Model saved as 'mnist_cnn_model.h5'.")

    generate_weights_biases_text_file(W_fc, b_fc, conv_weights)

# # Function to generate the text file with weights, biases, and convolutional filters
# def generate_weights_biases_text_file(weights, biases, filters):
#     weights_file = 'weights_biases_matrices.txt'
#     with open(weights_file, 'w') as f:
#         f.write(".section .data\n")
#         f.write("## Weights and Biases\n")
#         f.write(".global weights\n")
#         f.write("weights:\n")

#         flat_weights = weights.flatten()
#         for i in range(0, len(flat_weights), 10):
#             line_vals = flat_weights[i:i+10]
#             line = "    .float " + ", ".join(f"{val:.6f}" for val in line_vals)
#             f.write(line + "\n")

#         f.write("\n.global biases\n")
#         f.write("biases:\n")
#         bias_line = "    .float " + ", ".join(f"{b:.6f}" for b in biases)
#         f.write(bias_line + "\n\n")

#         f.write(".global weights_size\n")
#         f.write(f"weights_size: .word {weights.size}\n")
#         f.write(".global biases_size\n")
#         f.write(f"biases_size: .word {biases.size}\n")

#         # --- Add convolution filters ---
#         f.write("\n## Convolution Filters 5x5x1 (8 total)\n")
#         f.write(".global filters\n")
#         f.write("filters:\n")

#         # filters shape: (5, 5, 1, 8) → 8 filters of shape (5,5)
#         for i in range(filters.shape[-1]):
#             f.write(f"\n## Filter {i + 1}\n")
#             filter_matrix = filters[:, :, 0, i]
#             for row in filter_matrix:
#                 row_line = "    .float " + ", ".join(f"{val:.6f}" for val in row)
#                 f.write(row_line + "\n")

#     print(f"Text file '{weights_file}' generated with weights, biases, and filters.")


# Function to write the CNN weights and biases to the assembly file
def write_to_file(size, type):
    """Generates matrix, modifies assembly file, and writes it out."""
    try:
        if size <= 0:
            raise ValueError("Size must be a positive integer.")
        
        matrix = generate_matrix(size)
        filename = "assembly/Vectorized.s" if type == "V" else "assembly/NonVectorized.s"
        
        read_and_modify_file(filename, matrix, size, type)

        print(f"File successfully modified and saved as '{filename}'.")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    except FileNotFoundError:
        print("Error: The specified file was not found.")
        sys.exit(1)

# === Load IDX Files ===
def idx3_ubyte_to_numpy(file_path):
    with open(file_path, 'rb') as f:
        _, num_items, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.fromfile(f, dtype=np.uint8).reshape(num_items, rows, cols)
    return data

def idx1_ubyte_to_numpy(file_path):
    with open(file_path, 'rb') as f:
        _, num_items = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def load_mnist_data():
    # Get the directory where this script lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Walk up one level into the project, then into MNIST/archive
    base_path = os.path.join(script_dir, '..', 'MNIST', 'archive')

    train_images = idx3_ubyte_to_numpy(os.path.join(base_path, 'train-images.idx3-ubyte')) / 255.0
    train_labels = idx1_ubyte_to_numpy  (os.path.join(base_path, 'train-labels.idx1-ubyte'))

    return train_images, train_labels

#!!!!!!!!!!!!!! Relevant Code here GRINDER bhai!!!!!!!!!!!!!!!!!!!!!!!!

# === Assembly Writing Helpers ===
def format_matrix(matrix, label, comment=""):
    lines = [f"# {comment}\n", f"{label}:\n"]
    
    # Check if the matrix is 1D, 2D, or 4D
    if matrix.ndim == 1:
        # Handle 1D arrays
        float_values = ' '.join(f"{val:.6f}" for val in matrix)
        lines.append(f"    .float {float_values}  # {label}[0:{len(matrix)}]\n")
    elif matrix.ndim == 2:
        # Handle 2D arrays (rows and columns)
        for row_idx, row in enumerate(matrix):
            float_values = ' '.join(f"{val:.6f}" for val in row)
            lines.append(f"    .float {float_values}  # {label}[{row_idx}]\n")
    elif matrix.ndim == 4:
        # Handle 4D arrays (e.g., 5x5x1x8) - filters with 8 channels
        for filter_idx in range(matrix.shape[3]):
            lines.append(f"# Filter {filter_idx + 1} Weights:\n")
            filter_matrix = matrix[..., filter_idx]  # Shape: (5, 5, 1)
            for row_idx in range(filter_matrix.shape[0]):
                # Flatten the 5x5 matrix row into a single line of floats
                float_values = ' '.join(f"{filter_matrix[row_idx, col_idx, 0]:.6f}"
                                       for col_idx in range(filter_matrix.shape[1]))
                lines.append(f"    .float {float_values}  # {label}[{row_idx}, {filter_idx}]\n")
            lines.append("")  # Blank line after each filter for clarity

    return lines


def inject_into_asm(template_path, output_path, replacements):
    with open(template_path, 'r') as file:
        lines = file.readlines()

    for tag, data_lines in replacements.items():
        start_tag = f"## {tag} BEGIN"
        #print (start_tag)
        end_tag = f"## {tag} END"
        start_idx = end_idx = None

        for i, line in enumerate(lines):
            if line.strip() == start_tag:
                start_idx = i
            elif line.strip() == end_tag:
                end_idx = i
                break

        if start_idx is None or end_idx is None:
            print(f"[!] Tag not found: {tag}")
            continue

        lines = lines[:start_idx+1] + data_lines + lines[end_idx:]
       
    with open(output_path, 'w') as file:
        file.writelines(lines)

# === CNN Training and Extraction ===
def train_and_export():
    # Load MNIST data
    train_images, train_labels = load_mnist_data()

    # Normalize the images to the range [0, 1]
    train_images_resized = train_images

    # Define CNN with 5x5 filter, 8 filters, and Dense(10)
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        tf.keras.layers.Conv2D(8, (5, 5), activation='relu', padding='valid'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images_resized, train_labels, epochs=4, batch_size=128)

    # Extract weights
    conv_layer = model.layers[1]
    dense_layer = model.layers[-1]

    conv_weights, conv_biases = conv_layer.get_weights()  # Shape: (5, 5, 1, 8)
    dense_weights, dense_biases = dense_layer.get_weights()  # Shape: (1152, 10), (10,)

    # Extract one sample input image
    sample_input = train_images_resized[0].squeeze()

    # === Format for Assembly Output ===
    # EXPECTS TAGS IN ORDER
    replacements = {
        "DENSE_WEIGHTS": format_matrix(dense_weights, "dense_weights", "1152x10 Fully Connected Weights"),
        "DENSE_BIAS": format_matrix(dense_biases, "dense_bias", "Dense Bias"),
        "FILTER": format_matrix(conv_weights, "filter", "5x5x1x8 Convolution Filter Weights"),
        "FILTER_BIAS": format_matrix(conv_biases, "filter_bias", "Bias for each of 8 filters")
    }

    # Write into modified file
    inject_into_asm("assembly/Vectorized.s", "assembly/buffer.txt", replacements)
    inject_inputs(0, train_images_resized, train_labels, "assembly/buffer.txt", "assembly/VectorizedModified.s")


def inject_inputs(index, train_images_resized, train_labels, template_path, output_path):
    # Select the image and its label
    input_image = train_images_resized[index]  # Shape: (24, 24)
    label = train_labels[index]

    # Format the input image
    input_lines = format_matrix(input_image, "input_matrix", f"Sample 24x24 Input Image (Label: {label})")

    # Create replacements dictionary
    replacements = {
        "INPUT_MATRIX": input_lines
    }

    # Inject into the assembly template
    inject_into_asm(template_path, output_path, replacements)



if __name__ == "__main__":
    # Example usage for modifying assembly file (commented for testing)
    # write_to_file(5, "V")  # Modify as needed

    # Train the CNN and extract weights
    #train_and_extract_weights()

    #write_input()

    #load_mnist_data()
    train_and_export()

    # if len(sys.argv) != 3:
    #     print("Usage: python script.py <size> <type>")
    #     sys.exit(1)

    # size = int(sys.argv[1])
    # type = sys.argv[2]

    # # Call the function to modify the assembly files
    # write_to_file(size, type)

    # # After that, train the CNN model and extract weights
    # train_and_extract_weights()