import requests
import json

# Step 1: Define source and destination
source_url = "https://raw.githubusercontent.com/Sanchit-404/Reasoing-Towards-Fairness/refs/heads/main/outputs/processed_bbq_checkpoint_results/entire_dataset/checkpoint_10689/religion/checkpoint_10689_final_with_context.json"
destination_path = "outputs/processed_bbq_checkpoint_results/entire_dataset/checkpoint_10689/religion/checkpoint_10689_final_with_context.json"

# Step 2: Download and parse JSON
response = requests.get(source_url)
response.raise_for_status()  # raise error if download fails
data = response.json()

# Step 3: Save to local file
with open(destination_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved to {destination_path}")