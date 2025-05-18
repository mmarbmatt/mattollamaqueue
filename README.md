# Ollama Queue with Multi-Model Priority

A robust, asynchronous Ollama client with priority-based model selection and queue management system. This library provides a clean interface for managing multiple Ollama model requests with automatic fallback mechanisms and priority-based selection.

## Features

- **Priority-Based Model Selection**: Automatically select models based on configured priorities
- **Intelligent Model Fallback**: If preferred models are busy or fail, automatically try alternatives
- **Asynchronous Queue Management**: Process requests in queue order with proper error handling
- **Request Deduplication**: Only keep the latest request for each unique key
- **Persistent Memory and State**: Save and load state between sessions
- **Comprehensive Monitoring**: Track model performance, errors, and usage statistics
- **Highly Configurable**: Customize model configurations, response formats, and more
- **Easy Integration**: Simple API that works with any Python async application

## Installation

```bash
pip install ollama
```

This library requires the `ollama` Python client to be installed.

## Quick Start

```python
import asyncio
from llmlogic import OllamaQueue

async def main():
    # Get the singleton instance
    queue = OllamaQueue.get_instance(
        model_config=[
            {"name": "llama3:latest", "priority": 1},
            {"name": "mistral:latest", "priority": 2}
        ]
    )
    
    # Start the worker
    queue.start()
    
    try:
        # Generate a response
        response = await queue.generate(
            "What is the capital of France? Respond in JSON format.",
            key="geography-question"
        )
        print("Response:", response)
    finally:
        # Stop the worker
        queue.stop()

asyncio.run(main())
```

## Detailed Usage

### Managing Models

```python
from llmlogic import OllamaQueueManager

# Create a queue manager with custom models and priorities
manager = OllamaQueueManager(
    model_config=[
        {"name": "llama3:latest", "priority": 1},
        {"name": "mistral:latest", "priority": 2},
        {"name": "phi3:latest", "priority": 3}
    ]
)

# Start the worker
manager.start()
```

### Generating Responses

```python
# Basic generation
response = await manager.generate("What is the capital of France?")

# Specify a model explicitly
response = await manager.generate(
    "What is the capital of France?",
    model="llama3:latest"
)

# With custom parameters
response = await manager.generate(
    "What is the capital of France?",
    parameters={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 200
    }
)

# With cancellation support
stop_event = asyncio.Event()
task = asyncio.create_task(
    manager.generate(
        "Long running request...",
        stop_event=stop_event
    )
)

# Later, to cancel:
stop_event.set()
```

### Parallel Requests

```python
# Process multiple requests in parallel
tasks = [
    asyncio.create_task(manager.generate(
        f"What is {i} + {i}?",
        key=f"math-{i}"
    ))
    for i in range(1, 4)
]

# Wait for all to complete
results = await asyncio.gather(*tasks)
```

### Working with Memory and State

```python
# Store parameters
manager.set_parameter("temperature", 0.7)
temperature = manager.get_parameter("temperature", default=0.5)

# Work with memory
manager.add_memory({"event": "interesting_observation", "data": "..."})
recent_memories = manager.get_memory(max_items=10)
manager.clear_memory()  # Clear all memory
```

### Monitoring Model Performance

```python
# Get statistics about model usage
stats = manager.get_model_statistics()
print("Model statistics:", stats)

# Get the currently most successful model
active_model = manager.get_active_model()
print(f"Most successful model: {active_model}")
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, BackgroundTasks
from llmlogic import OllamaQueue

app = FastAPI()
queue = OllamaQueue.get_instance()

# Start the queue when the app starts
@app.on_event("startup")
async def startup_event():
    queue.start()

# Stop the queue when the app shuts down
@app.on_event("shutdown")
async def shutdown_event():
    queue.stop()

@app.post("/generate")
async def generate_text(prompt: str):
    response = await queue.generate(prompt)
    return response
```

### Integration with Discord Bot

```python
import discord
from discord.ext import commands
from llmlogic import OllamaQueue

bot = commands.Bot(command_prefix='!')
queue = OllamaQueue.get_instance()

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    queue.start()

@bot.command()
async def ask(ctx, *, question):
    await ctx.send("Thinking...")
    response = await queue.generate(question)
    await ctx.send(f"Answer: {response}")

# Start the bot
bot.run('YOUR_TOKEN')
```

## Advanced Configuration

### Custom Logging

```python
import logging

# Configure logging before importing the library
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ollama_queue.log'
)

# Then import and use the library
from llmlogic import OllamaQueueManager
```

### Custom Client Options

```python
manager = OllamaQueueManager(
    client_options={
        "host": "http://localhost:11434",
        "timeout": 120
    }
)
```

## Best Practices

1. **Always start() and stop() the worker**: The worker needs to be running to process requests, and should be properly stopped when done.

2. **Use unique keys for related requests**: Group related requests with the same key to ensure only the latest is processed.

3. **Handle Ollama errors**: The library has built-in retries, but be prepared to handle exceptions.

4. **Monitor model statistics**: Regularly check model statistics to identify issues.

5. **Configure model priorities thoughtfully**: Place more reliable and faster models at higher priority.