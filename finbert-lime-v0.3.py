# V3: Added GUI

# Goal: Apply a LIME text explainer to finBERT, "a pre-trained NLP model to analyze sentiment of financial text"
# finBERT repository: https://huggingface.co/ProsusAI/finbert

# Import PyTorch for tensor functionality
import torch
# Import the PyTorch quantization submodule for resource usage optimization (might not be needed if enough resources are available)
import torch.quantization
# Import the text explainer from LIME
from lime.lime_text import LimeTextExplainer
# Import Huggingface Transformers library for loading the model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Import signal and time for "emergency termination" functionality (might not be needed if enough resources are available)
import signal
import time
# Import Tkinter for simple GUI functionality
import tkinter as tk
from tkinter import messagebox

# Setting up a function for getting user input from a simple GUI
def run_model():
    # Get the user's input from the text field
    user_input = text_field.get("1.0", "end-1c")
    if user_input:
        try:
            # Setting up the "emergency termination" handler
            # Define the handler
            def handler(signum, frame):
                raise Exception("Execution is taking too long due to too much resource usage and the script was stopped.")
            # Register the signal function handler
            signal.signal(signal.SIGALRM, handler)
            # Define a time limit
            time_limit = 3600  # 1 hour
            signal.alarm(time_limit)
            try:

                # Select tokenizer
                tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                # Select model
                model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                # Input the text sequence we want finBERT to classify into the tokenizer
                # Example taken from: https://www.forbes.com/sites/dereksaul/2024/02/12/nvidia-is-now-more-valuable-than-amazon-and-google/
                input = tokenizer(user_input, return_tensors="pt")
                # Tokenizing: Splitting text into tokens (words, subwords, characters) and then mapping each token to a unique integer (its ID in the model’s vocabulary)
                # Output: PyTorch tensor where each element is the ID of a token in the input text

                # Quantize the model for optimized resource usage
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )

                # Get the quantized model's output (as raw logits)
                output = quantized_model(**input)
                # The model function takes the tensor of token IDs and feeds it through the model 
                # The model outputs logits (raw, unnormalized scores) for each class in the model’s classification task
                # Output logits are also stored in a PyTorch tensor
                # If quantization is not needed, change quantized_model to model

                # Get the probabilities by applying softmax function to logits
                probs = torch.nn.functional.softmax(output.logits, dim=-1)
                # The softmax function converts the logits into probabilities (easier to interpret) 
                # Output is a tensor of probabilities

                # Get the predicted class
                predicted_class = torch.argmax(probs)
                # Argmax function takes the tensor of probabilities and finds the index (class) with the highest probability
                # This is the model’s predicted class for the input text

                # Initialize LIME text explainer
                explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])
                # Define a prediction function
                def predictor(texts):
                    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                    outputs = quantized_model(**inputs)
                # Change quantized_model to model if quantization is not needed!
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    return probs.detach().numpy()
                # Explain the model's prediction
                explanation = explainer.explain_instance(user_input, 
                                                        predictor, 
                                                        num_features=6, 
                                                        labels=(predicted_class.item(),))
                # Print the explanation
                print(explanation)

            # Continuation of the emergency termination handler
            except Exception as exc:
                print(exc)

            result = "Model + LIME output for: " + user_input
            messagebox.showinfo("Result", result)
        
        except Exception as e:
            messagebox.showerror("Error", str(e))
    else:
        messagebox.showwarning("Missing input!", "Please enter a text sequence for the model to classify.")

# Create a new Tkinter window
window = tk.Tk()
# Create a new text field
text_field = tk.Text(window, width=50, height=10)
text_field.pack()
# Create a new button that will run the model and LIME explainer when clicked
run_button = tk.Button(window, text="Run model + LIME explainer on user input", command=run_model)
run_button.pack()
# Start the Tkinter event loop
window.mainloop()