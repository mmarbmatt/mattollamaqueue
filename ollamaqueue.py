#!/usr/bin/env python3
# ───────────────────────────── llmlogic.py ─────────────────────────────
"""
Ollama Queue with Multi-Model Priority

A robust, asynchronous Ollama client with priority-based model selection and queue management.
This module provides a way to manage multiple Ollama model requests with automatic fallback
and priority-based selection.
"""

import asyncio
import json
import os
import random
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable, Tuple, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llmlogic")

try:
    from ollama import AsyncClient
    _has_ollama = True
except ImportError:
    logger.warning("Ollama Python client not found. Please install with: pip install ollama")
    _has_ollama = False
    # Mock class for type checking when ollama is not available
    class AsyncClient:
        async def generate(self, *args, **kwargs):
            raise NotImplementedError("Ollama is not installed")

# ───────────────────────── Configuration Options ────────────────────────

# Enable FlashAttention + KV-cache quantization if supported by the hardware
os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
os.environ["OLLAMA_KV_CACHE_TYPE"] = "q8_0"


class OllamaQueueManager:
    """
    Manages a queue of Ollama requests with priority-based model selection.
    Supports multiple models with fallback capabilities.
    """

    def __init__(
        self,
        model_config: List[Dict[str, Union[str, int]]] = None,
        memory_file: str = "llmlogic_memory.json",
        client_options: Dict[str, Any] = None
    ):
        """
        Initialize the Ollama Queue Manager.
        
        Args:
            model_config: List of model configurations with name and priority.
                Example: [{"name": "llama3:latest", "priority": 1}, 
                         {"name": "mistral:latest", "priority": 2}]
            memory_file: Path to file for persisting memory/state
            client_options: Options to pass to the Ollama AsyncClient
        """
        if not _has_ollama:
            logger.error("Ollama Python client not installed. Queue manager will fail.")
        
        # Initialize the Ollama client with optional parameters
        self._client = AsyncClient(**(client_options or {}))
        
        # Set default model configuration if none provided
        if not model_config:
            self._models = [
                {"name": "llama3:latest", "priority": 1},
                {"name": "mistral:latest", "priority": 2}
            ]
        else:
            self._models = model_config
        
        # Initialize model status tracking
        self._model_status = {}
        for model in self._models:
            self._model_status[model["name"]] = {
                "busy": False,
                "last_used": 0,
                "errors": 0,
                "success": 0
            }
        
        # Lock for model selection to prevent race conditions
        self._model_lock = asyncio.Lock()
        
        # Queue for managing requests
        self._request_queue = asyncio.Queue()
        
        # Dictionary of pending requests
        self._pending = {}  # key -> Request
        
        # Memory/state management
        self._memory_file = memory_file
        self._state = {
            "mode": "default",
            "parameters": {},
            "memory": []
        }
        self._load_state()
        
        # Worker task
        self._worker_task = None
    
    def _load_state(self) -> None:
        """Load state from disk if available."""
        if os.path.exists(self._memory_file):
            try:
                with open(self._memory_file, 'r') as f:
                    self._state.update(json.load(f))
                logger.debug(f"Loaded state from {self._memory_file}")
            except Exception as e:
                logger.warning(f"Failed to load state from {self._memory_file}: {e}")
    
    def _save_state(self) -> None:
        """Save current state to disk."""
        try:
            with open(self._memory_file, 'w') as f:
                json.dump(self._state, f, indent=2)
            logger.debug(f"Saved state to {self._memory_file}")
        except Exception as e:
            logger.warning(f"Failed to save state to {self._memory_file}: {e}")
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set a parameter in the state."""
        self._state["parameters"][key] = value
        self._save_state()
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a parameter from the state."""
        return self._state["parameters"].get(key, default)
    
    def add_memory(self, item: Dict[str, Any]) -> None:
        """Add an item to memory."""
        self._state["memory"].append({**item, "timestamp": time.time()})
        self._prune_memory()
        self._save_state()
    
    def _prune_memory(self, max_items: int = 100) -> None:
        """Ensure memory doesn't exceed max_items."""
        if len(self._state["memory"]) > max_items:
            self._state["memory"] = self._state["memory"][-max_items:]
    
    def get_memory(self, max_items: int = None) -> List[Dict[str, Any]]:
        """Get items from memory, optionally limited to max_items."""
        if max_items is None:
            return self._state["memory"]
        return self._state["memory"][-max_items:]
    
    def clear_memory(self) -> None:
        """Clear all memory."""
        self._state["memory"] = []
        self._save_state()
    
    async def _select_model(self) -> str:
        """
        Select the highest priority available model.
        Uses a lock to prevent race conditions when multiple calls request a model simultaneously.
        """
        async with self._model_lock:
            # First try to find a non-busy model
            available_models = [m for m in self._models 
                               if not self._model_status[m["name"]]["busy"]]
            
            if available_models:
                # Sort by priority (lowest number is highest priority)
                selected = min(available_models, key=lambda m: m["priority"])
                model_name = selected["name"]
            else:
                # If all models are busy, select the one with fewest errors and used longest ago
                def score(m):
                    model_name = m["name"]
                    status = self._model_status[model_name]
                    # Higher score = less desirable
                    error_penalty = status["errors"] * 10  # Heavily penalize error-prone models
                    recency_penalty = (time.time() - status["last_used"]) * -0.01
                    return error_penalty - recency_penalty  # Lower score is better
                    
                selected = min(self._models, key=score)
                model_name = selected["name"]
                logger.warning(f"All models busy, selecting {model_name} as fallback")
                
            # Mark the model as busy and update last used time
            self._model_status[model_name]["busy"] = True
            self._model_status[model_name]["last_used"] = time.time()
            return model_name
    
    async def generate(
        self,
        prompt: str,
        key: str = None,
        model: str = None,
        retries: int = 3,
        format: str = "json",
        parameters: Dict[str, Any] = None,
        stop_event: asyncio.Event = None
    ) -> Dict[str, Any]:
        """
        Generate a response from Ollama.
        
        Args:
            prompt: The prompt to send to the model
            key: A unique key for this request (optional, defaults to random)
            model: Specific model to use (optional, auto-selected if not specified)
            retries: Number of retries on failure
            format: Response format (default: "json")
            parameters: Additional parameters to pass to Ollama
            stop_event: Event to signal cancellation
            
        Returns:
            The parsed response as a dictionary
        """
        if not _has_ollama:
            logger.error("Ollama not installed. Cannot generate response.")
            return {}
            
        request_key = key or f"request-{random.randint(1000000, 9999999)}"
        
        # Create a future for getting the result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # Build request object
        req = _OllamaRequest(
            key=request_key,
            prompt=prompt,
            future=future,
            model=model,
            format=format,
            parameters=parameters or {},
            stop_event=stop_event
        )
        
        # Cancel any previous request with the same key
        if request_key in self._pending:
            self._pending[request_key].future.cancel()
        
        # Add to pending and the queue
        self._pending[request_key] = req
        self._request_queue.put_nowait(req)
        
        # Ensure the worker is running
        self._ensure_worker_running()
        
        # Return the awaitable future
        return await future
    
    def _ensure_worker_running(self) -> None:
        """Ensure the worker task is running."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())
    
    async def _worker(self) -> None:
        """
        Worker coroutine that processes the queue of requests.
        """
        while True:
            try:
                # Use a timeout to make cancellation more responsive
                req = await asyncio.wait_for(self._request_queue.get(), timeout=0.5)
                
                # Skip if already cancelled
                if req.stop_event and req.stop_event.is_set():
                    req.future.cancel()
                    continue
                
                try:
                    # Process the request
                    logger.debug(f"Processing request {req.key}")
                    model_name = req.model
                    
                    # Auto-select model if none specified
                    if model_name is None:
                        model_name = await self._select_model()
                        model_was_auto_selected = True
                    else:
                        # Mark the specified model as busy if it's not already
                        if self._model_status[model_name]["busy"]:
                            logger.warning(f"Model {model_name} already busy but specifically requested")
                        self._model_status[model_name]["busy"] = True
                        self._model_status[model_name]["last_used"] = time.time()
                        model_was_auto_selected = False
                    
                    logger.info(f"Using model {model_name} for request {req.key}")
                    
                    # Generate response with retries
                    result = await self._generate_with_retries(
                        model_name=model_name,
                        prompt=req.prompt,
                        retries=3,
                        format=req.format,
                        parameters=req.parameters,
                        model_was_auto_selected=model_was_auto_selected
                    )
                    
                    # Set the result if the future isn't cancelled
                    if not req.future.cancelled():
                        req.future.set_result(result)
                    
                    # Log model usage statistics occasionally
                    if random.random() < 0.05:  # 5% chance
                        stats = ", ".join([
                            f"{m}: {s['success']}✓ {s['errors']}✗"
                            for m, s in self._model_status.items()
                        ])
                        logger.info(f"Model usage stats: {stats}")
                    
                except Exception as e:
                    if not req.future.cancelled():
                        req.future.set_exception(e)
                finally:
                    # Remove from pending
                    if req.key in self._pending and self._pending[req.key] is req:
                        del self._pending[req.key]
            
            except asyncio.TimeoutError:
                # Just retry the loop, makes cancellation more responsive
                continue
            except asyncio.CancelledError:
                # Worker is being cancelled, exit cleanly
                break
            except Exception as e:
                # Unexpected error, log and continue
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1.0)  # Prevent rapid error loops
    
    async def _generate_with_retries(
        self,
        model_name: str,
        prompt: str,
        retries: int,
        format: str,
        parameters: Dict[str, Any],
        model_was_auto_selected: bool
    ) -> Dict[str, Any]:
        """
        Generate a response from Ollama with retries and fallbacks.
        
        Args:
            model_name: Name of model to use
            prompt: The prompt to send
            retries: Number of retries
            format: Response format
            parameters: Additional parameters for Ollama
            model_was_auto_selected: Whether the model was auto-selected
            
        Returns:
            Parsed response as dictionary
        """
        for attempt in range(1, retries + 1):
            try:
                # Generate response from Ollama
                resp = await self._client.generate(
                    model=model_name,
                    prompt=prompt,
                    format=format,
                    keep_alive="60s",  # Keep model loaded for 60s
                    stream=False,
                    **parameters
                )
                
                # Record successful use
                self._model_status[model_name]["success"] += 1
                
                # Extract and parse response
                result = self._parse_response(resp)
                
                # Mark model as not busy
                self._model_status[model_name]["busy"] = False
                
                return result
                
            except Exception as e:
                # Handle "loading model" error by retrying
                if "loading model" in str(e).lower() and attempt < retries:
                    logger.warning(f"Loading model {model_name} (attempt {attempt}/{retries})")
                    await asyncio.sleep(1.0)
                    continue
                
                # Try different model if auto-selected
                if model_was_auto_selected and attempt < retries:
                    self._model_status[model_name]["errors"] += 1
                    logger.warning(f"Error with model {model_name}: {e} - trying different model")
                    
                    # Mark current model as not busy
                    self._model_status[model_name]["busy"] = False
                    
                    # Select a different model
                    async with self._model_lock:
                        available_models = [
                            m for m in self._models
                            if m["name"] != model_name and not self._model_status[m["name"]]["busy"]
                        ]
                        if available_models:
                            new_model = min(available_models, key=lambda m: m["priority"])["name"]
                            model_name = new_model
                            self._model_status[model_name]["busy"] = True
                            self._model_status[model_name]["last_used"] = time.time()
                            logger.info(f"Switching to model {model_name} after error")
                            continue
                
                # All retries failed
                logger.error(f"Error with model {model_name}: {e}")
                self._model_status[model_name]["errors"] += 1
                self._model_status[model_name]["busy"] = False
                return {}
        
        # We shouldn't reach here, but just in case
        logger.error(f"All retries failed for model {model_name}")
        self._model_status[model_name]["busy"] = False
        return {}
    
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """
        Parse the response from Ollama into a dictionary.
        
        Args:
            response: The response from Ollama
            
        Returns:
            The parsed dictionary
        """
        # Extract the raw text from different response formats
        if hasattr(response, "choices") and response.choices:
            raw = response.choices[0].message.content.strip()
        elif hasattr(response, "message") and hasattr(response.message, "content"):
            raw = response.message.content.strip()
        elif hasattr(response, "text"):
            raw = response.text.strip()
        elif isinstance(response, str):
            raw = response.strip()
        else:
            raw = str(response).strip()
        
        # Try to extract JSON from the response
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.error(f"Failed to find JSON in output: {raw[:200]!r}")
            return {}
            
        json_str = raw[start:end + 1]
        
        try:
            # Try to parse the JSON
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix unicode escapes and parse again
            try:
                fixed = json_str.encode().decode("unicode_escape")
                return json.loads(fixed)
            except Exception as e:
                logger.error(f"Failed to parse JSON: {json_str[:200]!r} ({e})")
                return {}
    
    def start(self) -> None:
        """Start the worker if not already running."""
        self._ensure_worker_running()
    
    def stop(self) -> None:
        """Stop the worker and clear the queue."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
        
        # Clear all pending requests
        while not self._request_queue.empty():
            try:
                self._request_queue.get_nowait()
            except:
                pass
        
        # Clear pending dictionary
        self._pending.clear()
        
        # Reset all models to not busy
        for model_name in self._model_status:
            self._model_status[model_name]["busy"] = False
        
        # Reset worker task
        self._worker_task = None
    
    def get_model_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current model usage statistics.
        
        Returns:
            Dict of models with their stats
        """
        return {model: {
            "busy": status["busy"],
            "success_rate": status["success"] / max(status["success"] + status["errors"], 1),
            "last_used": time.time() - status["last_used"] if status["last_used"] > 0 else float('inf'),
            "success": status["success"],
            "errors": status["errors"],
        } for model, status in self._model_status.items()}
    
    def get_active_model(self) -> str:
        """
        Get the name of the most successfully used model.
        
        Returns:
            The model name
        """
        # Find the model with highest usage
        if not any(status["success"] > 0 for status in self._model_status.values()):
            # If no model has been used yet, return the highest priority one
            return min(self._models, key=lambda m: m["priority"])["name"]
        
        return max(self._model_status.items(), key=lambda item: item[1]["success"])[0]


class _OllamaRequest:
    """
    Internal class representing a request to Ollama.
    """
    def __init__(
        self,
        key: str,
        prompt: str,
        future: asyncio.Future,
        model: str = None,
        format: str = "json",
        parameters: Dict[str, Any] = None,
        stop_event: asyncio.Event = None
    ):
        self.key = key
        self.prompt = prompt
        self.future = future
        self.model = model
        self.format = format
        self.parameters = parameters or {}
        self.stop_event = stop_event


# ─────────────────────── Simple convenience wrapper ───────────────────────

class OllamaQueue:
    """
    A singleton wrapper around OllamaQueueManager for easier use.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls, **kwargs) -> OllamaQueueManager:
        """
        Get or create the singleton instance.
        
        Args:
            **kwargs: Arguments to pass to OllamaQueueManager if creating a new instance
            
        Returns:
            The OllamaQueueManager instance
        """
        if cls._instance is None:
            cls._instance = OllamaQueueManager(**kwargs)
        return cls._instance
    
    @classmethod
    def generate(cls, prompt: str, **kwargs) -> Awaitable[Dict[str, Any]]:
        """
        Generate a response using the singleton instance.
        
        Args:
            prompt: The prompt to send
            **kwargs: Additional arguments for generate()
            
        Returns:
            Future that resolves to the response
        """
        return cls.get_instance().generate(prompt, **kwargs)


# ───────────────────────── Example usage ─────────────────────────
if __name__ == "__main__":
    async def main():
        # Example 1: Direct usage with OllamaQueueManager
        manager = OllamaQueueManager(
            model_config=[
                {"name": "llama3:latest", "priority": 1},
                {"name": "mistral:latest", "priority": 2}
            ]
        )
        
        # Start the worker
        manager.start()
        
        try:
            # Generate a response
            response = await manager.generate(
                "What is the capital of France? Respond in JSON format with a 'capital' field.",
                key="test-request"
            )
            print("Response:", response)
            
            # Check model statistics
            stats = manager.get_model_statistics()
            print("Model statistics:", stats)
            
        finally:
            # Stop the worker
            manager.stop()
            
        # Example 2: Using the singleton wrapper
        queue = OllamaQueue.get_instance(
            model_config=[
                {"name": "llama3:latest", "priority": 1},
                {"name": "mistral:latest", "priority": 2}
            ]
        )
        
        # Start the worker
        queue.start()
        
        try:
            # Process multiple requests in parallel
            tasks = [
                asyncio.create_task(queue.generate(
                    f"What is {i} + {i}? Respond with just the number in JSON format.",
                    key=f"math-{i}"
                ))
                for i in range(1, 4)
            ]
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks)
            for i, result in enumerate(results, 1):
                print(f"Result {i}:", result)
                
        finally:
            # Stop the worker
            queue.stop()
    
    # Run the example
    if _has_ollama:
        asyncio.run(main())
    else:
        print("Ollama Python client not installed. Cannot run example.")
        print("Install with: pip install ollama")
