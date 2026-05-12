# Task 5: Mental Health Support Chatbot

## Overview
This project is a fine-tuned conversational chatbot designed to provide supportive and empathetic responses around stress, anxiety, and emotional wellness.

## Features
- Fine-tunes `DistilGPT2` on counseling-style conversation pairs
- Uses the `Amod/mental_health_counseling_conversations` dataset
- Includes training artifacts and visualizations for model review
- Demonstrates chatbot generation with a simple notebook workflow

## Technologies Used
- Python
- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch
- Jupyter Notebook

## Project Structure
- `task5_mental_health_chatbot.ipynb` - main notebook for training and demo
- `task5_model/` - saved model artifacts
- `task5_loss_curve.png` - training loss visualization
- `task5_data_distribution.png` - dataset distribution visualization

## Screenshots
Training and analysis visuals are included below:

![Loss Curve](task5_loss_curve.png)

![Data Distribution](task5_data_distribution.png)

## How to Run
1. Open `task5_mental_health_chatbot.ipynb` in Jupyter Notebook or VS Code.
2. Install the required Python packages if they are not already available.
3. Run the notebook cells from top to bottom to reproduce training and chatbot generation.

## Notes
- Keep the project folder tidy and avoid committing temporary or unnecessary files.
- Add more screenshots or evaluation results if you extend the project further.