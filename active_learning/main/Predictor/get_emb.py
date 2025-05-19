# Import necessary libraries
import numpy as np  # Used for numerical operations, especially for handling the embeddings as arrays
import pandas as pd # Imported but not used in this specific script snippet. It's good practice to remove unused imports.

# --- Part 1: Prepare the text data ---

# Define a string containing a list of metals, separated by commas
metal_list_string = 'Palladium, Platinum, Copper, Gold, Iridium, Cerium, Niobium, Chromium'

# Split the string into a list of individual metal names
metal_names = metal_list_string.split(',') # Result: ['Palladium', ' Platinum', ' Copper', ...]

# Define a prefix string to add context to each metal name
# This helps the embedding model understand the role or category of these metals
descriptive_prefix = 'The hydrodesulfurization and hydrodenitrogenation catalytic metal: '

# Create a new list where each metal name is prepended with the descriptive prefix
# For example, "Palladium" becomes "The hydrodesulfurization and hydrodenitrogenation catalytic metal: Palladium"
texts_to_embed = [descriptive_prefix + metal.strip() for metal in metal_names]
# .strip() is used to remove any leading/trailing whitespace from metal names after splitting

# --- Part 2: Generate Embeddings using OpenAI API ---

# Import the OpenAI library
from openai import OpenAI

# Initialize the OpenAI client
# IMPORTANT: You need to replace 'YOUR_API_KEY_HERE' with your actual OpenAI API key.
# It's best to manage API keys securely, for example, using environment variables.
client = OpenAI(api_key='YOUR_API_KEY_HERE') # Replace with your actual key

def get_embedding(text_input, model_name="text-embedding-3-large"):
    """
    Generates a numerical embedding for a given text string using the specified OpenAI model.

    Args:
        text_input (str or list of str): The text(s) for which to generate embeddings.
        model_name (str): The identifier of the OpenAI embedding model to use.

    Returns:
        list: A numerical vector representing the embedding of the input text.
              If the input was a list, it returns a list of embedding data objects.
    """
    response = client.embeddings.create(input=text_input, model=model_name)
    # The API can return multiple embeddings if the input is a list.
    # For a single string input, we expect one embedding.
    return response.data[0].embedding

# Generate an embedding for each of the prepared descriptive metal texts
# This will make a separate API call to OpenAI for each text
embeddings_list = [get_embedding(text) for text in texts_to_embed]

# --- Part 3: Save the Embeddings ---

# Convert the list of embedding vectors into a NumPy array
# NumPy arrays are efficient for numerical computations and easy to save/load
embeddings_array = np.array(embeddings_list)

# Save the NumPy array to a file named 'emb.npy'
# '.npy' is a standard file format for saving NumPy arrays
np.save('catalyst_metal_embeddings.npy', embeddings_array) # Changed filename for clarity

print(f"Embeddings generated for {len(texts_to_embed)} texts and saved to 'catalyst_metal_embeddings.npy'")
print(f"Shape of the saved embeddings array: {embeddings_array.shape}")