import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from openai.types.shared.reasoning import Reasoning
from tqdm import tqdm
import sqlite3
import threading
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

# Pricing per 1M tokens (update these based on current pricing)
PRICING = {
    # OpenAI models
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    # Gemini models (example pricing - update with actual pricing)
    "gemini-2.5-flash": {"input": 0.3, "output": 2.5},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash-lite": {"input": 0.1, "output": 0.4},
    "gemini-3-pro-preview": {"input": 2, "output": 12},
}

class ParallelResponsesClient:
    """
    Fast, parallel client for OpenAI and Gemini APIs with SQLite caching and logging.
    Handles multiple prompts concurrently with configurable limits.
    Supports both OpenAI (GPT) and Google (Gemini) models.
    """

    def load_oai_key(self, keypath="/accounts/projects/sewonm/prasann/oaikey.sh"):
        with open(keypath, "r") as f:
            key = f.read().strip()
        return key

    def _is_gemini_model(self, model: str) -> bool:
        """Check if model is a Gemini model."""
        return model.startswith("gemini")
    
    def _is_openai_model(self, model: str) -> bool:
        """Check if model is an OpenAI model."""
        return model.startswith("gpt")
    
    def __init__(
        self,
        max_concurrent: int = 25,
        cache_db: str = "propercache/cache/response_cache.db",
        log_file: str = "propercache/cache/requests_log.jsonl",
        use_cache: bool = True,
        openai_key_path: Optional[str] = None,
        use_vertexai: bool = True
    ):
        
        # Initialize OpenAI client
        if openai_key_path:
            self.openai_client = AsyncOpenAI(api_key=self.load_oai_key(openai_key_path))
        else:
            self.openai_client = AsyncOpenAI()  # Will use OPENAI_API_KEY env var
        
        # Initialize Gemini ASYNC client
        self.gemini_client = None
        os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = str(use_vertexai).lower()
        # Use the async client instead of synchronous
        self.gemini_client = genai.Client(
            http_options=HttpOptions(api_version="v1"),
            vertexai=use_vertexai
        )
        
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.cache_db = Path(cache_db)
        self.log_file = Path(log_file)
        self.use_cache = use_cache
        self.total_cost = 0.0
        self.cache_hits = 0
        self.api_calls = 0
        
        # Thread-local storage for database connections
        self._local = threading.local()
        
        # Initialize database
        if self.use_cache:
            self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(str(self.cache_db), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self):
        """Initialize SQLite database with required tables."""
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()
        
        # Create cache table with indexed cache_key
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd REAL,
                success INTEGER,
                error TEXT,
                created_at TEXT,
                temperature REAL,
                max_output_tokens INTEGER
            )
        ''')
        
        # Create index on cache_key for faster lookups (though PRIMARY KEY already indexes)
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_cache_key ON response_cache(cache_key)
        ''')
        
        # Create index on created_at for potential time-based queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at ON response_cache(created_at)
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_cache_key(self, model: str, prompt: str, temperature: float, max_output_tokens: Optional[int], **kwargs) -> str:
        """Generate a cache key from request parameters."""
        cache_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            **kwargs
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response from SQLite."""
        if not self.use_cache:
            return None
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT model, prompt, response, input_tokens, output_tokens, 
                       total_tokens, cost_usd, success, error
                FROM response_cache
                WHERE cache_key = ?
            ''', (cache_key,))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    "model": row[0],
                    "prompt": row[1],
                    "response": row[2],
                    "usage": {
                        "input_tokens": row[3],
                        "output_tokens": row[4],
                        "total_tokens": row[5]
                    } if row[3] is not None else None,
                    "cost_usd": row[6],
                    "success": bool(row[7]),
                    "error": row[8],
                    "cached": True
                }
            
            return None
            
        except Exception as e:
            print(f"Warning: Could not read from cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any], temperature: float, max_output_tokens: Optional[int]):
        """Save a response to SQLite cache."""
        if not self.use_cache:
            return
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            usage = result.get("usage")
            
            cursor.execute('''
                INSERT OR REPLACE INTO response_cache 
                (cache_key, model, prompt, response, input_tokens, output_tokens, 
                 total_tokens, cost_usd, success, error, created_at, temperature, max_output_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cache_key,
                result["model"],
                result["prompt"],
                result["response"],
                usage["input_tokens"] if usage else None,
                usage["output_tokens"] if usage else None,
                usage["total_tokens"] if usage else None,
                result["cost_usd"],
                int(result["success"]),
                result["error"],
                datetime.now().isoformat(),
                temperature,
                max_output_tokens
            ))
            
            conn.commit()
            
        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate the cost of a request in USD."""
        pricing = PRICING[model]
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def _log_request(self, result: Dict[str, Any], cached: bool, model: str):
        """Log request details to JSONL file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt": result["prompt"],
            "success": result["success"],
            "cached": cached,
            "cost_usd": result.get("cost_usd", 0.0),
            "usage": result.get("usage"),
            "error": result.get("error")
        }
        
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    async def get_completion(
        self,
        model: str,
        prompt: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[str] = "minimal", # For OpenAI: "minimal", "low", "high", "maximal"
        thinking_budget: Optional[int] = 0,    # For Gemini: thinking budget (0 disables thinking)
        **kwargs
    ) -> Dict[str, Any]:
        assert model in PRICING, f"Model {model} not found in pricing"

        # Check cache
        cache_key = self._get_cache_key(model, prompt, temperature, max_output_tokens, reasoning=reasoning, thinking_budget=thinking_budget, **kwargs)
        
        cached_result = self._get_cached_response(cache_key)
        if cached_result:
            self.cache_hits += 1
            self._log_request(cached_result, cached=True, model=model)
            return cached_result
        
        # Make API call
        async with self.semaphore:
            try:
                self.api_calls += 1
                
                # Determine which API to use
                if self._is_openai_model(model):
                    result = await self._get_openai_completion(
                        model, prompt, temperature, max_output_tokens, reasoning, **kwargs
                    )
                elif self._is_gemini_model(model):
                    result = await self._get_gemini_completion(
                        model, prompt, temperature, max_output_tokens, thinking_budget, **kwargs
                    )
                else:
                    raise ValueError(f"Unknown model type: {model}. Model must start with 'gpt' or 'gemini'")
                
                # Cache the result
                self._save_to_cache(cache_key, result, temperature, max_output_tokens)
                
                self._log_request(result, cached=False, model=model)
                return result
                
            except Exception as e:
                result = {
                    "model": model,
                    "prompt": prompt,
                    "response": None,
                    "usage": None,
                    "cost_usd": 0.0,
                    "success": False,
                    "error": str(e),
                    "cached": False
                }
                print(f"Error: {e}")
                self._log_request(result, cached=False, model=model)
                return result
    
    async def _get_openai_completion(
        self,
        model: str,
        prompt: str,
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        reasoning: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get completion from OpenAI API."""
        response = await self.openai_client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning=Reasoning(effort=reasoning),
            **kwargs
        )
        
        # Calculate cost
        cost = self._calculate_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            model
        )
        self.total_cost += cost
        
        return {
            "model": model,
            "prompt": prompt,
            "response": response.output_text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "cost_usd": cost,
            "success": True,
            "error": None,
            "cached": False
        }
    
    async def _get_gemini_completion(
        self,
        model: str,
        prompt: str,
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        thinking_budget: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Get completion from Gemini API using async client."""
        if not self.gemini_client:
            raise ValueError("Gemini client not initialized")
        
        # Build config
        config_params = {}
        if temperature is not None:
            config_params["temperature"] = temperature
        if max_output_tokens is not None:
            config_params["max_output_tokens"] = max_output_tokens
        
        # Add thinking config
        config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
        
        config = types.GenerateContentConfig(**config_params)
        
        # Use async API call directly - no threads needed!
        response = await self.gemini_client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        
        # Extract token usage from response
        usage_metadata = response.usage_metadata
        input_tokens = usage_metadata.prompt_token_count
        output_tokens = usage_metadata.candidates_token_count
        total_tokens = usage_metadata.total_token_count
        
        # Calculate cost
        cost = self._calculate_cost(input_tokens, output_tokens, model)
        self.total_cost += cost
        
        return {
            "model": model,
            "prompt": prompt,
            "response": response.text,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            "cost_usd": cost,
            "success": True,
            "error": None,
            "cached": False
        }
    
    
    async def get_completions(
        self,
        model: str,
        prompts: List[str],
        temperature: float = 1.0,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[str] = "minimal",      # For OpenAI
        thinking_budget: Optional[int] = 0,         # For Gemini
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        
        # Create tasks with their original indices to maintain order
        indexed_tasks = [
            (i, self.get_completion(model, prompt, temperature, max_output_tokens, reasoning, thinking_budget, **kwargs))
            for i, prompt in enumerate(prompts)
        ]
        
        # Create progress bar
        if show_progress:
            pbar = tqdm(total=len(prompts), desc="Processing prompts", unit="prompt")
        
        # Store results with their indices
        indexed_results = []
        for coro in asyncio.as_completed([task for _, task in indexed_tasks]):
            result = await coro
            indexed_results.append(result)
            if show_progress:
                pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        # Match results back to their original indices
        # Build a map from prompt to list of results (handles duplicates)
        result_lists = {}
        for result in indexed_results:
            prompt = result['prompt']
            if prompt not in result_lists:
                result_lists[prompt] = []
            result_lists[prompt].append(result)
        
        # Reconstruct in original order, popping from lists for duplicates
        results = []
        for prompt in prompts:
            results.append(result_lists[prompt].pop(0))
        
        return results
    
    def run(
        self,
        model: str,
        prompts: List[str],
        temperature: float = 1.0,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[str] = "minimal",      # For OpenAI: "minimal", "low", "high", "maximal"
        thinking_budget: Optional[int] = 0,         # For Gemini: thinking budget
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for getting completions.
        
        Args:
            model: Model name (e.g., "gpt-5-nano" or "gemini-2.5-flash")
            prompts: List of prompt texts
            temperature: Sampling temperature
            max_output_tokens: Maximum tokens in response
            reasoning: For OpenAI models - reasoning effort level
            thinking_budget: For Gemini models - thinking budget (0 disables)
            **kwargs: Additional parameters for the API
            
        Returns:
            List of dicts containing responses, metadata, and costs
        """
        if model=="gemini-2.5-pro" or model=="gemini-3-pro-preview":
            thinking_budget = max(thinking_budget, 128)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        return asyncio.run(
            self.get_completions(model, prompts, temperature, max_output_tokens, reasoning, thinking_budget, **kwargs)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cache usage and costs."""
        cache_size = 0
        if self.use_cache:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM response_cache')
                cache_size = cursor.fetchone()[0]
            except Exception as e:
                print(f"Warning: Could not get cache size: {e}")
        
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hits / (self.api_calls + self.cache_hits), 2) if (self.api_calls + self.cache_hits) > 0 else 0,
            "cache_size": cache_size
        }
    
    def clear_cache(self):
        """Clear all entries from the cache."""
        if not self.use_cache:
            return
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM response_cache')
            conn.commit()
            print("Cache cleared successfully")
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")
    
    def close(self):
        """Close database connections, and close the clients."""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
        if self.gemini_client:
            self.gemini_client.close()
        if self.openai_client:
            self.openai_client.close()


# Example usage
if __name__ == "__main__":
    # Initialize the client
    client = ParallelResponsesClient(
        max_concurrent=50,  # Can now handle much higher concurrency!
        # cache_db="propercache/cache/response_cache.db",
        # log_file="propercache/cache/requests_log.jsonl",
        use_cache=True,
        use_vertexai=True
    )
    
    # Example with OpenAI models
    print("=" * 60)
    print("Testing OpenAI GPT models...")
    print("=" * 60)
    
    gpt_prompts = [
        "What is the capital of India?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about programming.",
    ]
    
    gpt_results = client.run(
        model="gpt-5-nano", 
        prompts=gpt_prompts, 
        max_output_tokens=100,
        reasoning="minimal"
    )
    
    for i, result in enumerate(gpt_results, 1):
        cached_label = " [CACHED]" if result.get("cached") else ""
        if result["success"]:
            print(f"\nGPT Prompt {i}{cached_label}: {result['prompt']}")
            print(f"Response: {result['response']}")
            print(f"Cost: ${result['cost_usd']:.6f}")
        else:
            print(f"\nGPT Prompt {i} failed: {result['error']}")
    
    # Example with Gemini models (now much faster!)
    print("\n" + "=" * 60)
    print("Testing Google Gemini models...")
    print("=" * 60)
    
    gemini_prompts = [
        "What is machine learning?",
        "Explain neural networks briefly.",
        "What are transformers in AI?",
    ]
    
    gemini_results = client.run(
        model="gemini-2.5-pro",
        prompts=gemini_prompts,
        max_output_tokens=100,
        thinking_budget=0
    )
    
    for i, result in enumerate(gemini_results, 1):
        cached_label = " [CACHED]" if result.get("cached") else ""
        if result["success"]:
            print(f"\nGemini Prompt {i}{cached_label}: {result['prompt']}")
            print(f"Response: {result['response']}")
            print(f"Cost: ${result['cost_usd']:.6f}")
        else:
            print(f"\nGemini Prompt {i} failed: {result['error']}")
    
    # Summary
    stats = client.get_stats()
    print(f"\n{'='*60}")
    print(f"Statistics:")
    print(f"  Total cost: ${stats['total_cost_usd']:.6f}")
    print(f"  API calls: {stats['api_calls']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.0%}")
    print(f"  Cache size: {stats['cache_size']} entries")
    print(f"\nLogs saved to: {client.log_file}")
    print(f"Cache saved to: {client.cache_db}")
    
    # Close connections
    client.close()