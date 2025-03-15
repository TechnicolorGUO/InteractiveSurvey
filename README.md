<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/TechnicolorGUO/Auto_Survey_Generator_pdf/blob/main/resources/logo.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/TechnicolorGUO/Auto_Survey_Generator_pdf/blob/main/resources/logo.svg">
    <img src="https://github.com/TechnicolorGUO/Auto_Survey_Generator_pdf/blob/main/resources/logo.svg" alt="Logo" width="50%" height="50%">
  </picture>
</p>

<p align="center">A <b>Interactive</b> and <b>Automatic</b> literature survey generator.
</p>
<p align="center">
<img alt="python" src="https://img.shields.io/badge/python-3.11-blue">
<img alt="Static Badge" src="https://img.shields.io/badge/license-MIT-green">
</p>
<div align="center">
<hr>

[Quick Start](#quick-start) | [Use Docker](#use-docker) | [Paper]()

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
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
Install the required dependencies:
```sh
python setup_env.py
```

### 3️⃣ Configure Environment Variables
Create a  `.env` file in the root directory of the project and add the following configurations:
```env
OPENAI_API_KEY=<your_openai_api_key_here>
OPENAI_API_BASE=<your_openai_api_base_here>
MODEL=<your_preferred_model_here>
```
Replace the placeholders with your actual OpenAI API key, API base URL, and preferred model.

### 4️⃣ Run the Application
Start the development server by running the following command:
```sh
python src/manage.py runserver 0.0.0.0:8001
```
_(Replace 8001 with any available port number of your choice.)_

### 5️⃣ Access the Application
Once the server is running, open your browser and navigate to:
```
http://localhost:8001```
```
You can now use the ​Auto Literature Survey Generator to upload, analyze, and generate literature surveys!

## Use Docker

To run Auto Literature Survey Generator with Docker, follow these steps:


## Cite

Please cite the following 

```
```

## Contact



## Contributing



## License

[MIT](LICENSE)
