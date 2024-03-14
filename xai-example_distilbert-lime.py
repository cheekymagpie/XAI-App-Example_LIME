# Overarching goal of this script: Apply a LIME text explainer to distilBERT

# Abstract description of what this script does, as summarized by Copilot:
# 1. distilBERT Model: The script uses the distilBERT model to classify a given user input as positive, neutral, or negative.
# 2. LIME Text Explainer: The script uses LIME (Local Interpretable Model-Agnostic Explanations) to explain the classification made by the finBERT model. 
#    It identifies which parts (words or phrases) of the user input contribute most to the classification.
# 3. Emergency Termination: To prevent excessive resource usage that could freeze your device, the script includes a feature that stops the script if it runs for longer than a set amount of time without producing a result.
# 4. GUI: The script includes a graphical user interface (GUI) where the user can enter their text. 
#    The GUI displays the output (the classification and the LIME explanation) in the same window.
# 5. Visualization: The GUI also displays a graphical representation of the LIME explanation in the form of a bar chart plot showing how much each word in the user input text contributes to the finBERT classification.

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
# Import matplotlib and a Tkinter-Canvas for graphical display of results in the GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Import necessary libraries for performance monitoring in its own thread
import threading
import psutil
import GPUtil

start_time = time.time()
# Returns the current time in seconds since the epoch (Jan 1, 1970?)
# By calling this function before and after the code, we calculate the elapsed time (execution time of the script)

# Defining a global flag used for stopping the separate monitoring thread when the main script gets stopped
stop_monitoring = False
# Setting up resource monitoring functionality
def monitor_resources():
    global stop_monitoring
    while not stop_monitoring:
        # For CPU usage measuring
        cpu_usage = psutil.cpu_percent()
        # For RAM usage measuring
        ram_usage = psutil.virtual_memory().percent
        # For GPU usage measuring
        gpus = GPUtil.getGPUs()
        # Check whether one or more GPUs are available
        if gpus:
            gpu = gpus[0]
            gpu_load = gpu.load
            gpu_memory_usage = gpu.memoryUtil

            print(f"CPU Usage: {cpu_usage} %")
            print(f"RAM Usage: {ram_usage} %")
            print(f"GPU Load: {gpu_load * 100} %")
            print(f"GPU Memory Usage: {gpu_memory_usage * 100} %")
        else:
            print("No GPU available.")

        # Sleep for 300 seconds (5 minutes); we want to print resource monitoring output every 5 minutes
        time.sleep(300)

# Starting resource monitoring in a separate thread
monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.start()        

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

                # Save model name to a variable
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                # Select tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Select model
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                # Input the text sequence we want distilBERT to classify into the tokenizer
                # Example article to get text from: https://www.forbes.com/sites/dereksaul/2024/02/12/nvidia-is-now-more-valuable-than-amazon-and-google/
                tokenized_input = tokenizer(user_input, return_tensors="pt")
                # Tokenizing: Splitting text into tokens (words, subwords, characters) and then mapping each token to a unique integer (its ID in the model’s vocabulary)
                # Output: PyTorch tensor where each element is the ID of a token in the input text

                # Quantize the model for optimized resource usage
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )

                # Get the quantized model's output (as raw logits)
                output = quantized_model(**tokenized_input)
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

            finally:
                # Stop resource monitoring when the main script stops, whether by design or exception
                stop_monitoring = True
                monitor_thread.join()

            # Display result in the same Tkinter window as the user input prompt    
            result_label.config(text=str(explanation))

            # Prepare for graphical display of LIME explainer results
            # Get the features and their weights (from the user's input, as weighted by distilBERT)
            features = explanation.as_list()
            # Separate the feature names and weights
            feature_names = [feature[0] for feature in features]
            feature_weights = [feature[1] for feature in features]
            # Create a new figure via matplotlib library
            fig = plt.Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)
            # Plot the feature weights as a bar chart
            ax.bar(feature_names, feature_weights)
            # Create a new canvas and draw the plot onto it
            canvas = FigureCanvasTkAgg(fig, master=window)
            canvas.draw()
            # Add the canvas to the Tkinter window
            canvas.get_tk_widget().pack()
        
        # Catch any/all exceptions from trying to run the model or explainer
        except Exception as e:
            messagebox.showerror("Error (catching exception related to model execution or explanation generation):", str(e))
    # Error for when there is no user-entered text when the button is clicked        
    else:
        messagebox.showwarning("Missing input!", "Please enter a text sequence for the model to classify.")

run_model()

# Create a new Tkinter window
window = tk.Tk()
# Create a new text field
text_field = tk.Text(window, width=50, height=10)
text_field.pack()
# Create a new label to display the result
result_label = tk.Label(window, text="")
result_label.pack()
# Create a new button that will run the model and LIME explainer when clicked
run_button = tk.Button(window, text="Run model + LIME explainer on user input", command=run_model)
run_button.pack()
# Start the Tkinter event loop
window.mainloop()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Script execution time: {elapsed_time} seconds")