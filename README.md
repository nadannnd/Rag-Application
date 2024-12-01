# Multi-Model Local Document Q&A Application

## Overview

This Streamlit application provides an interactive, local document question-answering system using Ollama's large language models. Users can upload PDF and text documents, choose from multiple local models, and ask questions about the uploaded content.

## Features

- 🚀 Multiple Local LLM Model Support
  - Llama 3.1
  - Mistral
  - Gemma 7b

- 📄 Document Upload
  - PDF and Text file support
  - Multiple file upload
  - Automatic document processing

- 🤖 Local AI-Powered Q&A
  - Retrieval-augmented generation
  - Context-aware responses
  - Runs entirely on local machine

## Prerequisites

### Software Requirements
- Python 3.8+
- Ollama
- pip

### Installation Steps

1. **Install Ollama**
   - Download from: https://ollama.com/
   - Follow installation instructions for your operating system

2. **Pull Required Models**
   ```bash
   ollama pull llama3.1
   ollama pull mistral
   ollama pull gemma:7b
