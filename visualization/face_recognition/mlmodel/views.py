from django.shortcuts import render

import os
import nbformat
from nbconvert import NotebookExporter


# Get the current working directory
current_path = os.getcwd()

# Print the current working directory
print("Current Working Directory:", current_path+"\mlmodel\gaemain.ipynb")

# Set the path to the notebook file
notebook_file = current_path+"\mlmodel\gaemain.ipynb"

# Create a NotebookExporter object
exporter = NotebookExporter()

# Read the notebook file
with open(notebook_file, "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

# Export the notebook to a Python script
(script, _) = exporter.from_notebook_node(notebook)

# Save the Python script to a file
script_file = current_path+"\mlmodel\gaemain.py"
with open(script_file, "w", encoding="utf-8") as f:
    f.write(script)

# Run the Python script
os.system(f"python {script_file}")


# from gaemain import gae_for
# Create your views here.
def trainModel(request):
    # gae_for(current_path+"\mlmodel\datasets\yale\person1\\train_images\subject10_normal.jpg")
    return render(request,'ml.html')