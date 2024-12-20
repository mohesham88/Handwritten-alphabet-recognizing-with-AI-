import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


def load_data(filepath):
    """
    Load a dataset from a CSV file.
    Parameters:
    filepath (str): The path to the CSV file to be loaded.
    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    Raises:
    FileNotFoundError: If the file at the specified path does not exist.
    pd.errors.EmptyDataError: If the file is empty or not formatted correctly.
    Exception: For any other exceptions that occur during loading.
    """
    
    try:
        print("Loading dataset...")
        dataset = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with shape: {dataset.shape}")
        return dataset
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The file is empty or not formatted correctly.")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def process_data(dataset):
    """Process dataset: extract features, labels, and normalize pixel values."""
    print("Processing dataset...")

    
    # labels are from 0 to 25 donating the alphabets A to Z (first column)
    labels = dataset.iloc[:, 0]
    
    
    data = dataset.iloc[:, 1:]

    # Map numeric labels to characters (A-Z)
    labels = labels.map(lambda x: chr(65 + x))

    # Normalize data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    print("Data normalization complete.")

    return normalized_data, labels

def plot_label_distribution(labels):
    """Visualize the label distribution."""
    print("Analyzing class distribution...")
    label_counts = labels.value_counts().sort_index()

    plt.figure(figsize=(16, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.title("Class Distribution", fontsize=16)
    plt.xlabel("Characters", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    for index, value in enumerate(label_counts.values):
        plt.text(index, value + 10, str(value), ha='center', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def show_samples(data, labels, num_samples=3):
    """Display sample images for each label."""
    print(f"Displaying {num_samples} samples per class...")

    unique_labels = sorted(labels.unique())
    fig, axs = plt.subplots(num_samples, len(unique_labels), figsize=(20, 2 * num_samples))

    for col, label in enumerate(unique_labels):
        label_indices = labels[labels == label].index

        for row in range(num_samples):
            if len(label_indices) > 0:
                random_idx = np.random.choice(label_indices)
                image = data[random_idx].reshape(28, 28)
                axs[row, col].imshow(image, cmap='gray')
            else:
                axs[row, col].imshow(np.zeros((28, 28)), cmap='gray')

            axs[row, col].axis('off')
            if row == 0:
                axs[row, col].set_title(label, fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    # File path to dataset
    dataset_path = 'A_Z Handwritten Data.csv'

    try:
        # Load and process the data
        dataset = load_data(dataset_path)
        features, targets = process_data(dataset)

        # Visualize label distribution
        plot_label_distribution(targets)

        # Display samples for each class
        show_samples(features, targets)

        # Dataset overview
        print("\nDataset Overview:")
        print(f"Total samples: {len(targets)}")
        print(f"Feature dimensions: {features.shape[1]}")
        print(f"Value range: {features.min():.2f} to {features.max():.2f}")
        print(f"Mean: {features.mean():.2f}, Std Dev: {features.std():.2f}")

    except FileNotFoundError:
        print(f"Error: The file '{dataset_path}' does not exist.")
    except Exception as error:
        print(f"An unexpected error occurred: {error}")