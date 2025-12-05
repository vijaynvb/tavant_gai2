# SLM model with LangChain

This repository contains an implementation of a Second Language Model (SLM) using LangChain. The SLM is designed to assist with language learning and practice through interactive conversations and exercises.

## Install Ollama 

Follow the instructions on the [Ollama website](https://ollama.com/download) to download and install Ollama for your operating system.

## Download a Model

To download a model, open your terminal and run the following command:

```bash
ollama pull <model-name>
```

Replace `<model-name>` with the name of the model you wish to download.

In this example, we will use the [`smollm:360m`](https://ollama.com/library/smollm) model:

```bash
ollama pull smollm:360m
```

## Integrate with LangChain

To use the downloaded model with LangChain, you need to set up the LangChain environment and create a script to interact with the model.

1. Install LangChain:

```bash
pip install langchain_ollama
```

2. Create a Python script (e.g., `slm_langchain.py`) and add the following code:

```python
# Import necessary libraries
from langchain_ollama import ChatOllama

# Initialize the Ollama model
llm = ChatOllama(
    model="smollm:360m",
    temperature=0
)

# Define the conversation messages
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

# Invoke the model with the messages
ai_msg = llm.invoke(messages)

# Print the model's response
print(ai_msg.content)
```

3. Run the script:

```bash
python slm_langchain.py
```

This will output the French translation of the sentence "I love programming."

---