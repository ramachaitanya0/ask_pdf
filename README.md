# Ask_Pdf

This chatbot allows users to upload documents(pdf only), and then it leverages OPEN AI ChatGPT and Langchain
to answer questions based on the content of the uploaded documents.
Users can engage in interactive Q&A sessions with the chatbot, making it a powerful tool for document 
exploration and retrieval.

Key Features
1. Document upload capability.
2. Interactive and user-friendly chat interface
3. Potential for customization and extension.

# Installation

1. Clone the Git Hub Repo into your local workspace using the below code.
```sh
git clone https://github.com/ramachaitanya0/ask_pdf.git 
```

2. Create a Conda Environment.
```sh
conda create -n <env_name> python=3.11.4
```

3. Install all the required Packages using requirements.txt file.
```sh
pip install -r requirements.txt
```
4. Add .env file in the Repo and add your OPEN AI Key in .env file.

```sh
OPENAI_API_KEY=<OPENAI_API_KEY>
```

# Usage

Run the Stream lit app using below code.
```sh
streamlit run app.py
```




