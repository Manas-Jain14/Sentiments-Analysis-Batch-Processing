# OpenAI Batch Processing Evaluation Pipeline

This project implements a complete pipeline for 8-label intent classification using OpenAI's Batch API.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your OpenAI API Key:
   Update the `.env` file with your actual API key:
   ```
   OPENAI_API_KEY=your_actual_key_here
   ```

3. Ensure `ground_truth.csv` is in the root directory with the following columns:
   - `question`
   - `response`
   - `sentiment` (This is used as the ground truth label)

## Usage

Run the pipeline:
```bash
python pipeline.py
```

## Steps performed:
1. **Read Ground Truth**: Loads the CSV dataset.
2. **Convert to JSONL**: Prepares the batch request file.
3. **Upload**: Sends the file to OpenAI.
4. **Create Batch**: Starts the batch processing job.
5. **Poll Status**: Waits for completion (checks every 30s).
6. **Download Output**: Retrieves results from OpenAI.
7. **Extract Predictions**: Parses the model responses.
8. **Match Records**: Aligns predictions with original ground truth.
9. **Confusion Matrix**: Generates an 8x8 matrix of results.
10. **Calculate Accuracy**: Computes the overall performance.
11. **Save Results**: Exports evaluation data to `evaluation_results.json`.
