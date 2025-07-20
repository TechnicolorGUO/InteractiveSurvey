<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/TechnicolorGUO/InteractiveSurvey/blob/main/resources/logo1.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/TechnicolorGUO/InteractiveSurvey/blob/main/resources/logo1.svg">
    <img src="https://github.com/TechnicolorGUO/InteractiveSurvey/blob/main/resources/logo1.svg" alt="Logo" width="50%" height="50%">
  </picture>
</p>

<p align="center">An <b>Interactive</b> and <b>Personalized</b> literature survey generation system.</p>
<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10-blue">
  <img alt="Static Badge" src="https://img.shields.io/badge/license-apache-green">
  <a href="https://arxiv.org/abs/2504.08762" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2504.08762-red" />
  </a>
</p>
<div align="center">
<hr>

[Quick Start](#quick-start) | [Use Docker(Recommended)](#use-docker) | [Paper](https://arxiv.org/abs/2504.08762)

</div>

---

## 📰 News  
<table>
  <tr>
    <td style="white-space: nowrap;"><strong>2025/7/8</strong></td>
    <td>🚀 Our <strong>InteractiveSurvey demo is now officially online!</strong> Experience the power of AI-driven survey paper generation with our interactive web platform. Try it now! 🌟</td>
  </tr>
  <tr>
    <td style="white-space: nowrap;"><strong>2025/4/15</strong></td>
    <td>The paper <strong><a href="https://arxiv.org/abs/2504.08762">InteractiveSurvey: An LLM-based Personalized and Interactive Survey Paper Generation System</a></strong> is now released! 🎉✨</td>
  </tr>
</table>

---

## Introduction

**InteractiveSurvey** is an **interactive** and **personalized** tool designed to help researchers efficiently conduct **literature reviews**. By leveraging **natural language processing (NLP)** and **Large Language Models (LLMs)**, it enables users to collect, organize, and generate structured literature surveys **effortlessly**.

### 🔥 Key Features:
- **📝 Automatic Literature Review Generation**: Extract key insights from papers and generate structured literature surveys.  
- **💡 Interactive Exploration**: Dynamically filter, refine, and customize your survey in real-time.  
- **📄 PDF Export**: Easily generate high-quality literature surveys in PDF format with either **Markdown** or **LaTeX**.  
- **⚡ Multimodality**: Extract figures from references and insert customized figures by yourself.
- **🐳 Docker Support**: Quickly deploy and run the application in a containerized environment.  


![flochart](/resources/flowchart.png)


https://github.com/user-attachments/assets/15beefae-3b85-453e-a10d-3c210a80933b

**📺 Demo Video**: You can watch the demo video at [https://www.bilibili.com/video/BV1dju2z8ESw/?spm_id_from=333.1387.homepage.video_card.click](https://www.bilibili.com/video/BV1dju2z8ESw/?spm_id_from=333.1387.homepage.video_card.click)

---

<hr>

## Examples

### Markdown Example
https://github.com/user-attachments/assets/fdf48927-ae0f-4040-9595-4a509ea62f08

### LaTeX Example


https://github.com/user-attachments/assets/db2b08f5-a328-43e1-9ae9-41c09b54214b


<hr>

## Quick Start

Interactive requires Python 3.10. A minimum 20G disk space is required

### 1️⃣ Clone the Repository  
Clone the repository to your local machine:  
```sh
git clone https://github.com/TechnicolorGUO/InteractiveSurvey
cd InteractiveSurvey
```

### 2️⃣ Set Up the Environment
Create a virtual environment and activate it:
```sh
conda create -n interactivesurvey python=3.10
conda activate interactivesurvey
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
docker build -t interactivesurvey .

# Run the Docker container (with GPU support)
docker run --gpus all -p 8001:8001 interactivesurvey
```

### CPU Version
If you do not have GPU support, you can run the CPU version of the Docker container. *​Note*: Before building and running, you need to manually remove the following line from the `scripts/additional_scripts.py` file:
```python
"device-mode": "cuda",
```
Then run the following commands:
```bash
# Build the Docker image
docker build -t interactivesurvey-cpu .

# Run the Docker container (with CPU support)
docker run -p 8001:8001 interactivesurvey-cpu
```

After starting the container, access (http://localhost:8001)[http://localhost:8001] to confirm that the application is running correctly.

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
    asg_system.download_pdf() 
    # Downloads PDFs to "./sample_pdfs". Add with your own files for upload.
    asg_system.parsing_pdfs()
    asg_system.description_generation()
    asg_system.agglomerative_clustering()
    asg_system.outline_generation()
    asg_system.section_generation()
    asg_system.citation_generation()
```

## Contact
If you have any enquiries, please email [guobeichen0228@gmail.com](guobeichen0228@gmail.com)

## License

[Apache License 2.0](LICENSE)
