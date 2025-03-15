import os
import subprocess

# Install requirements
subprocess.run(["pip", "install", "--no-cache-dir", "--no-deps", "-r", "requirements.txt"])
subprocess.run(["pip", "install", "--no-cache-dir", "bertopic"])
subprocess.run(["pip", "install", "--no-cache-dir", "-U", "magic-pdf[full]", "--extra-index-url", "https://wheels.myhloli.com"])
subprocess.run(["pip", "install", "--no-cache-dir", "unstructured==0.16.10"])
subprocess.run(["pip", "install", "--no-cache-dir", "huggingface_hub"])
subprocess.run(["pip", "uninstall", "-y", "pdfminer"])
subprocess.run(["pip", "uninstall", "-y", "pdfminer-six"])
subprocess.run(["pip", "install", "pdfminer-six"])
subprocess.run(["pip", "uninstall", "-y", "opencv-python-headless"])
subprocess.run(["pip", "uninstall", "-y", "opencv-python"])
subprocess.run(["pip", "install", "opencv-python-headless"])
subprocess.run(["python", "-m", "nltk.downloader", "punkt"])

# Download model files
subprocess.run(["wget", "https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py", "-O", "download_models_hf.py"])
subprocess.run(["python", "download_models_hf.py"])

print("Environment setup complete!")