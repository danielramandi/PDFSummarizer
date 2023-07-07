# Streamlit PDF Summarizer App

This Streamlit application provides a user-friendly interface to perform complex text summarization tasks on PDF documents using OpenAI's powerful language models. With this application, users can summarize lengthy PDFs into bite-sized, easily digestible pieces of information.
Overview

The app takes advantage of the langchain Python package, a toolchain that integrates OpenAI's models with document loading, text splitting, and text summarization capabilities.

The summarization process starts by breaking the input PDF into chunks, with each chunk being of user-defined size. The text in each chunk is then fed into the selected language model (either GPT4 or ChatGPT) for summarization. The model generates a summary of the chunk and this process is repeated for all the chunks, thereby creating a summarized version of the entire document.

The application uses PyPDFLoader from the langchain.document_loaders package to load the PDF documents. The RecursiveCharacterTextSplitter from langchain.text_splitter splits the loaded text into chunks, and the load_summarize_chain function from langchain.chains.summarize applies the OpenAI model to summarize the chunks.

This application provides an intuitive interface for users to fine-tune the summarization process. Users can select the type of summarization chain, choose the size of chunks, and set the overlap size between the chunks. Moreover, they can select the model (ChatGPT or GPT4) to be used for summarization, adjust the temperature setting for the model, and specify the number of summaries to be generated.
Prerequisites

Before you can run this application, you need:

    Python 3.6 or later.

    An OpenAI API key. You can get one by signing up on the OpenAI website.

    The following Python libraries: openai, streamlit, langchain, os, tempfile. These libraries can be installed using pip:

bash

pip install openai streamlit langchain

Cloning the Repository

You can clone this repository by following these steps:

    Open the terminal/command prompt.

    Navigate to the directory where you want to clone the repository.

    Run the following command:

bash

git clone https://github.com/danielramandi/PDFSummarizer.git

Installation and Usage

To install and run this application locally:

    Navigate to the directory containing the app.py file.

    Run the following command to start the Streamlit server:

bash

streamlit run app.py

    Open your web browser and go to localhost:8501 (or the URL provided in the terminal) to view the application.

    Enter your OpenAI API key in the sidebar, choose your desired settings, and upload a PDF file to summarize.

    Click the "Summarize" button to get your summaries.

Running on Google Colab

Running Streamlit apps on Google Colab requires tunneling the Streamlit server through a public URL. Here's how to do it:

    Open a new Google Colab notebook.

    Install necessary packages:

bash

!pip install openai streamlit langchain pyngrok

    Pull the code from GitHub:

bash

!git clone https://github.com/danielramandi/PDFSummarizer.git

    Navigate to the directory containing app.py.

bash

%cd PDFSummarizer

    Run the Streamlit app.

bash

!streamlit run app.py &>/dev/null&

    Tunnel the app through a public URL using pyngrok.

python

from pyngrok import ngrok
# Setup a tunnel to the streamlit port 8501
public_url = ngrok.connect(port='8501')
public_url

    Click the generated public URL to view the Streamlit app.

Contribution

Contributions are warmly welcomed. Feel free to fork this project, make changes according to your needs, and create a pull request.
License

This project is licensed under the terms of the MIT license.
