<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/TechnicolorGUO/InteractiveSurvey/tree/main/resources/logo1.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/TechnicolorGUO/InteractiveSurvey/tree/main/resources/logo1.svg">
    <img src="https://github.com/TechnicolorGUO/InteractiveSurvey/tree/main/resources/logo1.svg" alt="Logo" width="50%" height="50%">
  </picture>
</p>

<p align="center">An <b>Interactive</b> and <b>Automatic</b> literature survey generator.
</p>
<p align="center">
<img alt="python" src="https://img.shields.io/badge/python-3.10-blue">
<img alt="Static Badge" src="https://img.shields.io/badge/license-MIT-green">
</p>
<div align="center">
<hr>

[Quick Start](#quick-start) | [Use Docker(Recommended)](#use-docker) | [Paper]()

</div>

---

## Introduction

**Auto Literature Survey Generator** is an **interactive** and **automated** tool designed to help researchers efficiently conduct **literature reviews**. By leveraging **natural language processing (NLP)** and **AI-powered summarization**, it enables users to collect, organize, and generate structured literature surveys **effortlessly**.

### 🔥 Key Features:
- **📝 Automatic Literature Review Generation**: Extract key insights from papers and generate structured summaries.  
- **💡 Interactive Exploration**: Dynamically filter, refine, and customize your survey in real time.  
- **📄 PDF Export**: Easily generate high-quality literature surveys in PDF format.  
- **⚡ AI-Powered Insights**: Identify trends, categorize research, and highlight important findings.  
- **🐳 Docker Support**: Quickly deploy and run the application in a containerized environment.  

---

<hr>

## Quick Start

LiveSurvey requires Python 3.10.

### 1️⃣ Clone the Repository  
Clone the repository to your local machine:  
```sh
git clone https://github.com/TechnicolorGUO/Auto_Survey_Generator_pdf.git
cd Auto_Survey_Generator_pdf
```

### 2️⃣ Set Up the Environment
Create a virtual environment and activate it:
```sh
conda create -n livesurvey python=3.10
conda activate livesurvey
```
Install the required dependencies:
```sh
python scripts/setup_env.py
```

For the `ConnectTimeout` error when downloading Huggingface models, please run the following script:
```bash
pip install modelscope
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py
python download_models.py
```

### 3️⃣ Configure Environment Variables
Create a  `.env` file in the root directory of the project and add the following configurations:
```env
OPENAI_API_KEY=<your_openai_api_key_here>
OPENAI_API_BASE=<your_openai_api_base_here>
MODEL=<your_preferred_model_here>
```
Replace the placeholders with your actual OpenAI API key, API base URL, and preferred model.

To test the system on GPU, you also need to follow the instructions provided by [MinerU](https://github.com/opendatalab/MinerU/tree/master):
- [Ubuntu 22.04 LTS + GPU](https://github.com/opendatalab/MinerU/blob/master/docs/README_Ubuntu_CUDA_Acceleration_en_US.md)
- [Windows 10/11 + GPU](https://github.com/opendatalab/MinerU/blob/master/docs/README_Windows_CUDA_Acceleration_en_US.md)

### 4️⃣ Run the Application
Start the development server by running the following command:
```sh
python src/manage.py runserver 0.0.0.0:8001
```
_(Replace 8001 with any available port number of your choice.)_

### 5️⃣ Access the Application
Once the server is running, open your browser and navigate to:
```
http://localhost:8001
```
You can now use the ​Auto Literature Survey Generator to upload, analyze, and generate literature surveys!

<hr>

## Use Docker
Before proceeding, ensure you have cloned the repository and configured your `.env` file in the root directory of the project. The `.env` file must include the following configurations:
```env
OPENAI_API_KEY=<your_openai_api_key_here>
OPENAI_API_BASE=<your_openai_api_base_here>
MODEL=<your_preferred_model_here>
```
Replace the placeholders with your actual OpenAI API key, API base URL, and preferred model.

### GPU Version
If you have GPU support, you can build and run the GPU version of the Docker container using the following commands:
```bash
# Build the Docker image
docker build -t my-docker-app .

# Run the Docker container (with GPU support)
docker run --gpus all -p 8001:8001 my-docker-app
```

### CPU Version
If you do not have GPU support, you can run the CPU version of the Docker container. *​Note*: Before building and running, you need to manually remove the following line from the `scripts/additional_scripts.py` file:
```python
"device-mode": "cuda",
```
Then run the following commands:
```bash
# Build the Docker image
docker build -t my-docker-app .

# Run the Docker container (with CPU support)
docker run -p 8001:8001 my-docker-app
```

After starting the container, access http://localhost:8001 to confirm that the application is running correctly.

<hr>

## Direct Survey Generation Without Frontend

If you want to generate surveys directly without using the frontend, follow these steps:

1. Navigate to the `src/demo/survey_generation_pipeline` directory:
```bash
cd src/demo/survey_generation_pipeline
```
2. Copy the `.env` file to this directory. If you already have a .env file in the root of your project, you can copy it like this:
```bash
cp ../../../.env .
```
*Note*: Ensure the `.env` file contains the required configurations (e.g., `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `MODEL`).

3. Run the pipeline directly:
```bash
python main.py
```
This will execute the survey generation pipeline on our sample PDFs and output the results (.md and .pdf) to the `result` folder directly.

4. Modify the script for your own sample
The `main.py` contains the following code to generate a survey:

```python
if __name__ == "__main__":
    root_path = "."
    pdf_path = "./sample_pdfs" #Set this to the path of the folder containing your PDF files.
    survey_title = "Automating Literature Review Generation with LLM" #Set this to the title of your survey.
    cluster_standard = "method" #Set this to the clustering standard you want to use.
    asg_system = ASG_system(root_path, 'test', pdf_path, survey_title, cluster_standard) #test refers to the survey_id which prevent you from parsing pdfs again.
    asg_system.parsing_pdfs()
    asg_system.description_generation()
    asg_system.agglomerative_clustering()
    asg_system.outline_generation()
    asg_system.section_generation()
    asg_system.citation_generation()
```

## Contact
If youhave any enquiries, please email [guobeichen0228@gmail.com](guobeichen0228@gmail.com)

## License

[Apache License 2.0](LICENSE)
