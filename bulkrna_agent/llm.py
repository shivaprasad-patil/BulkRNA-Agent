"""
LLM interface for BulkRNA Agent using Ollama
"""
import logging
from typing import List, Dict, Any, Optional
import requests
import json

logger = logging.getLogger(__name__)


class OllamaLLM:
    """Interface to Ollama LLM models"""
    
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 300
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        logger.info(f"Initialized OllamaLLM with model: {model}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """Generate completion from Ollama"""
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": stream,
                "options": {
                    "num_predict": self.max_tokens
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            logger.debug(f"Sending request to {url} with model {self.model}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate: {e}")
            raise
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> str:
        """Chat completion from Ollama"""
        try:
            url = f"{self.base_url}/api/chat"
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "stream": stream,
                "options": {
                    "num_predict": self.max_tokens
                }
            }
            
            logger.debug(f"Sending chat request to {url} with model {self.model}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama chat API: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in chat: {e}")
            raise


class DualLLMManager:
    """Manages two LLMs: one for reasoning, one for biomedical tasks"""
    
    def __init__(self, config):
        self.reasoning_llm = OllamaLLM(
            model=config.llm.reasoning_model,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout
        )
        
        self.biomedical_llm = OllamaLLM(
            model=config.llm.biomedical_model,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout
        )
        
        logger.info("Initialized DualLLMManager with reasoning and biomedical LLMs")
    
    def route_query(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Route query to appropriate LLM based on content
        
        Biomedical LLM for:
        - Gene interpretation
        - Pathway analysis
        - Biological significance
        
        Reasoning LLM for:
        - Tool selection
        - Analysis planning
        - Statistical decisions
        """
        # Simple keyword-based routing
        biomedical_keywords = [
            'gene', 'pathway', 'biological', 'function',
            'mechanism', 'protein', 'disease', 'phenotype',
            'enrichment', 'ontology', 'annotation'
        ]
        
        query_lower = query.lower()
        is_biomedical = any(kw in query_lower for kw in biomedical_keywords)
        
        if is_biomedical:
            logger.info("Routing to biomedical LLM")
            return "biomedical"
        else:
            logger.info("Routing to reasoning LLM")
            return "reasoning"
    
    def generate(
        self,
        prompt: str,
        llm_type: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response using appropriate LLM"""
        if llm_type is None:
            llm_type = self.route_query(prompt)
        
        if llm_type == "biomedical":
            return self.biomedical_llm.generate(prompt, system_prompt)
        else:
            return self.reasoning_llm.generate(prompt, system_prompt)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        llm_type: Optional[str] = None
    ) -> str:
        """Chat using appropriate LLM"""
        if llm_type is None and messages:
            last_message = messages[-1].get("content", "")
            llm_type = self.route_query(last_message)
        
        if llm_type == "biomedical":
            return self.biomedical_llm.chat(messages)
        else:
            return self.reasoning_llm.chat(messages)
