# Model Layer and Concept Information Extraction

This repository provides an implementation of a system that interacts with a database and an external language model API to extract information related to machine learning model layers and their captured concepts. The code also supports semantic search and prompt generation for question-answering based on model-specific context.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Docker Setup for pgvector](#docker-setup-for-pgvector)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
  - [Prompt Generation](#prompt-generation)
  - [API Request and Response Handling](#api-request-and-response-handling)
  - [Database Connection and Search](#database-connection-and-search)
- [File Descriptions](#file-descriptions)
- [Contributing](#contributing)
- [License](#license)

## Requirements

You can install the required dependencies by running:

```bash
pip install -r requirements.txt

# Sample .env

```bash
API_URL=<your-api-url>
DB_HOST=<your-db-host>
DB_NAME=<your-db-name>
DB_USER=<your-db-user>
DB_PASSWORD=<your-db-password>
DB_PORT=<your-db-port>


# Docker Setup for pgvector

Check DB_setup.pdf
