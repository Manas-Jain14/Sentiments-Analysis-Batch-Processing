import os
import json
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# --- Configuration ---
# Set MOCK_MODE to True to test the pipeline locally without an OpenAI API key.
# Set MOCK_MODE to False to run the real batch job on OpenAI's servers.
MOCK_MODE = True

# Load environment variables for API access
load_dotenv()

# Initialize OpenAI client only if not in MOCK_MODE
if not MOCK_MODE:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file. Please add it to run in Real Mode.")
    client = OpenAI(api_key=api_key)
else:
    client = None
    print("Running in MOCK_MODE: No API key required. Generating local predictions.")

# --- Agent Configuration ---
AGENT_CONFIG = {
    "model": "gpt-4.1-mini",
    "temperature": 0,
    "system_prompt": (
        "You are an intent classification agent for automotive customer conversations. "
        "Classify each record into exactly one of these labels: "
        "CONFIDENCE, CONFUSION, TRUST, INTEREST, DOUBT, URGENCY, FRUSTRATION, NONE. "
        "Return the result as a JSON object."
    ),
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "sentiment_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": ["CONFIDENCE", "CONFUSION", "TRUST", "INTEREST", "DOUBT", "URGENCY", "FRUSTRATION", "NONE"]
                    }
                },
                "required": ["label"],
                "additionalProperties": False
            }
        }
    }
}


# Step 1: Read the ground truth dataset (CSV file)
def read_ground_truth(file_path):
    """Load dataset with tab separator and handle potential encoding issues."""
    return pd.read_csv(file_path, sep='\t')


# Step 2: Convert each record into a JSONL batch request file
def create_jsonl_file(df, output_file):
    """Convert each record into a JSONL batch request file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            # Define the API request structure for each row
            request = {
                "custom_id": f"request-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": AGENT_CONFIG["model"],
                    "temperature": AGENT_CONFIG["temperature"],
                    "response_format": AGENT_CONFIG["response_format"],
                    "messages": [
                        {
                            "role": "system",
                            "content": AGENT_CONFIG["system_prompt"]
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {row['question']}\n"
                                f"Response: {row['response']}\n"
                                "Classify the intent."
                            )
                        }
                    ]
                }
            }
            # Write each request as a new line in the JSONL file
            f.write(json.dumps(request) + '\n')
    return output_file


# Step 3: Upload the JSONL file to OpenAI
def upload_file(file_path):
    """Upload the JSONL file to OpenAI."""
    if MOCK_MODE:
        return "mock_file_id"
    
    # Assert client is not None to satisfy type checkers (Pylance)
    assert client is not None
    
    with open(file_path, "rb") as f:
        # Upload the file to OpenAI's servers
        response = client.files.create(file=f, purpose="batch")
    return response.id


# Step 4: Create a batch job
def create_batch_job(file_id):
    """Create a batch job using the returned file_id."""
    if MOCK_MODE:
        return "mock_batch_id"
    
    assert client is not None
    
    # Submit the batch processing job to OpenAI
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return batch.id


# Step 5: Poll the batch status every 30 seconds
def poll_batch_status(batch_id):
    """Poll the batch status every 30 seconds until it is completed."""
    if MOCK_MODE:
        return "mock_output_file_id"
    
    assert client is not None
    
    while True:
        # Retrieve the latest status of the batch job
        batch = client.batches.retrieve(batch_id)
        print(f"Batch status: {batch.status}")
        # Stop polling if the job is completed
        if batch.status == "completed":
            return batch.output_file_id
        # Raise error if the job failed or was cancelled
        if batch.status in ["failed", "cancelled", "expired"]:
            raise Exception(f"Batch failed with status: {batch.status}")
        # Wait 30 seconds before checking again
        time.sleep(30)


# Step 6: Download the output file
def download_output(file_id, output_path, df=None):
    """Download the output file once batch is completed."""
    if MOCK_MODE:
        if df is None:
            raise ValueError("df must be provided in MOCK_MODE")
        # Generate fake model responses for testing evaluation logic
        labels = [
            "CONFIDENCE", "CONFUSION", "TRUST", "INTEREST",
            "DOUBT", "URGENCY", "FRUSTRATION", "NONE"
        ]
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(len(df)):
                # Return structured JSON output in mock mode
                mock_json_content = json.dumps({"label": random.choice(labels)})
                mock_res = {
                    "custom_id": f"request-{i}",
                    "response": {
                        "body": {
                            "choices": [
                                {"message": {"content": mock_json_content}}
                            ]
                        }
                    }
                }
                f.write(json.dumps(mock_res) + '\n')
        return output_path

    assert client is not None
    
    # Retrieve the content of the completed batch output
    content = client.files.content(file_id).text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return output_path


# Step 7: Extract predictions from the output file
def extract_predictions(output_file):
    """Extract predictions from the output file."""
    predictions = {}
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Link each prediction to its custom_id
            custom_id = data['custom_id']
            # Parse the JSON content and extract the label
            content = data['response']['body']['choices'][0]['message']['content']
            parsed_content = json.loads(content)
            prediction = parsed_content.get("label", "NONE").strip().upper()
            predictions[custom_id] = prediction
    return predictions


# Step 8: Match predictions with original records
def match_results(df, predictions):
    """Match predictions with original records using custom_id."""
    df['prediction'] = df.index.map(lambda i: predictions.get(f"request-{i}"))
    return df


# Step 9: Generate an 8x8 confusion matrix
def generate_confusion_matrix(df):
    """Generate an 8x8 confusion matrix."""
    labels = [
        "CONFIDENCE", "CONFUSION", "TRUST", "INTEREST",
        "DOUBT", "URGENCY", "FRUSTRATION", "NONE"
    ]
    # Normalize ground truth labels to uppercase
    df['sentiment'] = df['sentiment'].str.upper()
    # Compute the cross-tabulation of actual vs predicted
    matrix = pd.crosstab(
        df['sentiment'],
        df['prediction'],
        rownames=['Actual'],
        colnames=['Predicted']
    )
    # Force the matrix to be 8x8 with all specified labels
    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)
    return matrix


# Step 10: Calculate overall accuracy
def calculate_accuracy(df):
    """Calculate overall accuracy = correct predictions / total predictions."""
    # Count how many predictions match the ground truth
    correct = (df['sentiment'] == df['prediction']).sum()
    total = len(df)
    # Return the ratio of correct predictions to total
    return correct / total if total > 0 else 0


# Step 11: Save evaluation results to a JSON file
def save_results(matrix, accuracy, output_path):
    """Save evaluation results to a JSON file."""
    # Structure the evaluation results for JSON export
    results = {
        "accuracy": accuracy,
        "confusion_matrix": matrix.to_dict()
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write results to file with clean indentation
        json.dump(results, f, indent=4)


# Step 12: Generate and save a heatmap visualization of the confusion matrix
def plot_confusion_matrix(confusion_matrix):
    """
    Create a heatmap visualization of the confusion matrix using Matplotlib.
    """
    # Define the 8 target labels
    labels = [
        "CONFIDENCE", "CONFUSION", "TRUST", "INTEREST",
        "DOUBT", "URGENCY", "FRUSTRATION", "NONE"
    ]
    
    # Set the figure size to be readable
    plt.figure(figsize=(10, 8))
    
    # Convert pandas DataFrame to numpy array for plotting
    matrix_values = confusion_matrix.values
    
    # Display the matrix as an image (heatmap)
    # Using 'Blues' color map for better visibility
    plt.imshow(matrix_values, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix - GPT-4.1 Mini")
    plt.colorbar() # Add color bar to show scale
    
    # Set the tick positions for the axes
    tick_marks = np.arange(len(labels))
    
    # Set X-axis (Predicted) and Y-axis (Actual) labels
    # Rotating X-axis labels to prevent overlap
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # Add the numeric count values inside each cell
    # Logic to change text color based on cell background for readability
    threshold = matrix_values.max() / 2.
    for i in range(matrix_values.shape[0]):
        for j in range(matrix_values.shape[1]):
            plt.text(j, i, format(matrix_values[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if matrix_values[i, j] > threshold else "black")
    
    # Label the axes
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    
    # Use tight_layout to ensure everything fits without cutoff
    plt.tight_layout()
    
    # Save the plot as a PNG file
    plt.savefig("confusion_matrix.png")
    print("Step 12: Confusion matrix visualization saved to 'confusion_matrix.png'")


def main():
    """Main execution function for the pipeline."""
    # Define file paths for the pipeline
    input_csv = "ground_truth.csv"
    batch_input = "batch_input.jsonl"
    batch_output = "batch_output.jsonl"
    eval_results = "evaluation_results.json"

    print(f"--- Pipeline Started (Mock Mode: {MOCK_MODE}) ---")

    print("Step 1: Reading dataset...")
    df = read_ground_truth(input_csv)

    print("Step 2: Converting records to JSONL...")
    create_jsonl_file(df, batch_input)

    print("Step 3: Uploading input file...")
    file_id = upload_file(batch_input)

    print("Step 4: Creating OpenAI batch job...")
    batch_id = create_batch_job(file_id)
    print(f"Batch ID assigned: {batch_id}")

    print("Step 5: Polling batch for completion...")
    output_file_id = poll_batch_status(batch_id)

    print("Step 6: Downloading processing results...")
    download_output(output_file_id, batch_output, df=df)

    print("Step 7 & 8: Extracting and matching predictions...")
    predictions = extract_predictions(batch_output)
    df = match_results(df, predictions)

    print("Step 9: Generating 8x8 confusion matrix...")
    matrix = generate_confusion_matrix(df)
    print("\nConfusion Matrix Results:")
    print(matrix)

    print("\nStep 10: Calculating overall accuracy...")
    accuracy = calculate_accuracy(df)
    print(f"Final Accuracy: {accuracy:.2%}")

    print("Step 11: Saving results to JSON...")
    save_results(matrix, accuracy, eval_results)

    print("Step 12: Generating confusion matrix plot...")
    plot_confusion_matrix(matrix)

    print(f"\n--- Pipeline Completed Successfully! Results saved to {eval_results} ---")


if __name__ == "__main__":
    main()
