"""
BulkRNA Agent - ReAct-style agent for RNA-seq analysis
"""
import logging
from typing import Dict, List, Any, Optional
import re
import json

logger = logging.getLogger(__name__)


class BulkRNAAgent:
    """
    ReAct-style agent for bulk RNA-seq analysis
    
    Implements the ReAct (Reasoning + Acting) framework:
    - Thought: Reasoning about what to do
    - Action: Calling tools
    - Observation: Analyzing tool results
    """
    
    def __init__(self, config, llm_manager, tools: Dict):
        self.config = config
        self.llm_manager = llm_manager
        self.tools = tools
        self.conversation_history = []
        self.max_iterations = 10
        
        logger.info("Initialized BulkRNA Agent")
    
    def reset(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("Reset conversation history")
    
    def chat(self, user_message: str) -> str:
        """
        Main chat interface for the agent
        
        Args:
            user_message: User's question or request
        
        Returns:
            Agent's response
        """
        try:
            logger.info(f"User message: {user_message}")
            
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Run ReAct loop
            response = self._react_loop(user_message)
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            logger.info("Generated response")
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            return f"I encountered an error: {str(e)}"
    
    def _react_loop(self, query: str) -> str:
        """
        Execute ReAct loop to answer query
        """
        context = self._build_context()
        
        # Check if this is a simple question or requires tool use
        if self._is_simple_question(query):
            return self._answer_directly(query, context)
        
        # ReAct loop for complex queries requiring tools
        thoughts = []
        actions = []
        observations = []
        
        for iteration in range(self.max_iterations):
            logger.info(f"ReAct iteration {iteration + 1}")
            
            # Think: What should I do?
            thought = self._think(query, context, thoughts, actions, observations)
            thoughts.append(thought)
            
            # Check if done
            if "FINAL ANSWER:" in thought:
                final_answer = thought.split("FINAL ANSWER:")[-1].strip()
                return final_answer
            
            # Act: Choose and execute tool
            action = self._parse_action(thought)
            if action is None:
                # No action parsed, ask for clarification
                return self._answer_directly(query, context)
            
            actions.append(action)
            
            # Execute action
            observation = self._execute_action(action)
            observations.append(observation)
            
            # Check if we have enough information
            if self._can_answer(query, observations):
                final_answer = self._generate_final_answer(
                    query, thoughts, actions, observations
                )
                return final_answer
        
        # Max iterations reached
        logger.warning("Max iterations reached")
        return self._generate_final_answer(query, thoughts, actions, observations)
    
    def _build_context(self) -> str:
        """Build context from conversation history"""
        context = "Previous conversation:\n"
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            role = msg["role"]
            content = msg["content"]
            context += f"{role}: {content}\n"
        return context
    
    def _is_simple_question(self, query: str) -> bool:
        """Check if query is a simple question not requiring tools"""
        simple_keywords = [
            "what is", "explain", "define", "how does",
            "tell me about", "describe"
        ]
        
        query_lower = query.lower()
        
        # Check if it's asking about data analysis
        if any(kw in query_lower for kw in ["analyze", "run", "perform", "calculate"]):
            return False
        
        # Check if it's a simple question
        return any(kw in query_lower for kw in simple_keywords)
    
    def _answer_directly(self, query: str, context: str) -> str:
        """Answer directly without tools"""
        system_prompt = """
You are BulkRNA Agent, an expert in bulk RNA-seq data analysis.
You help researchers analyze their transcriptomics data and answer questions about RNA-seq methods.
Provide clear, accurate, and helpful responses.
"""
        
        prompt = f"""
{context}

User question: {query}

Please provide a helpful answer.
"""
        
        response = self.llm_manager.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        return response
    
    def _think(
        self,
        query: str,
        context: str,
        thoughts: List[str],
        actions: List[Dict],
        observations: List[str]
    ) -> str:
        """Generate thought about what to do next"""
        
        # Build tool descriptions
        tool_descriptions = self._get_tool_descriptions()
        
        # Build history of thoughts, actions, observations
        history = ""
        for i, (t, a, o) in enumerate(zip(thoughts, actions, observations)):
            history += f"\nThought {i+1}: {t}\n"
            history += f"Action {i+1}: {a}\n"
            history += f"Observation {i+1}: {o}\n"
        
        system_prompt = """
You are BulkRNA Agent using the ReAct framework for bulk RNA-seq analysis.

Think step-by-step about what to do next. You can:
1. Use available tools to analyze data
2. Answer the user's question based on observations

Format your thought as:
Thought: <your reasoning>
Action: <tool_name>(<parameters>)

Or if ready to answer:
Thought: <reasoning>
FINAL ANSWER: <your answer>
"""
        
        prompt = f"""
Available tools:
{tool_descriptions}

User query: {query}

{context}

{history}

What should you do next?
"""
        
        response = self.llm_manager.generate(
            prompt=prompt,
            llm_type="reasoning",
            system_prompt=system_prompt
        )
        
        return response
    
    def _get_tool_descriptions(self) -> str:
        """Get descriptions of available tools"""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
    
    def _parse_action(self, thought: str) -> Optional[Dict]:
        """Parse action from thought"""
        # Look for Action: pattern
        action_match = re.search(r'Action:\s*(\w+)\((.*?)\)', thought, re.DOTALL)
        
        if not action_match:
            return None
        
        tool_name = action_match.group(1)
        params_str = action_match.group(2)
        
        # Parse parameters (simple key=value parsing)
        params = {}
        if params_str.strip():
            # Try to parse as JSON-like structure
            try:
                # Simple parsing for key=value pairs
                for param in params_str.split(','):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        params[key] = value
            except Exception as e:
                logger.warning(f"Could not parse parameters: {e}")
        
        return {
            "tool": tool_name,
            "params": params
        }
    
    def _execute_action(self, action: Dict) -> str:
        """Execute tool action"""
        tool_name = action["tool"]
        params = action["params"]
        
        logger.info(f"Executing {tool_name} with params: {params}")
        
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        tool = self.tools[tool_name]
        
        try:
            result = tool.execute(**params)
            observation = json.dumps(result, indent=2)
            return observation
        except Exception as e:
            logger.error(f"Error executing {tool_name}: {e}", exc_info=True)
            return f"Error executing tool: {str(e)}"
    
    def _can_answer(self, query: str, observations: List[str]) -> bool:
        """Check if we have enough information to answer"""
        # Simple heuristic: if we have at least one successful observation
        for obs in observations:
            if "success" in obs.lower():
                return True
        return False
    
    def _generate_final_answer(
        self,
        query: str,
        thoughts: List[str],
        actions: List[Dict],
        observations: List[str]
    ) -> str:
        """Generate final answer based on observations"""
        
        # Build summary of what was done
        summary = "Analysis performed:\n"
        for i, (action, obs) in enumerate(zip(actions, observations)):
            summary += f"{i+1}. {action['tool']}: "
            try:
                obs_dict = json.loads(obs)
                if obs_dict.get("status") == "success":
                    summary += "✓ Completed successfully\n"
                else:
                    summary += f"✗ {obs_dict.get('message', 'Failed')}\n"
            except:
                summary += f"{obs[:100]}...\n"
        
        system_prompt = """
You are BulkRNA Agent. Summarize the analysis results for the user.
Be clear, concise, and highlight key findings.
"""
        
        prompt = f"""
User query: {query}

{summary}

Detailed observations:
{chr(10).join(observations)}

Please provide a final answer to the user's query based on these results.
"""
        
        response = self.llm_manager.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        return response
    
    def suggest_design_matrix(self, metadata_file: str) -> str:
        """
        Convenience method to suggest design matrix
        """
        if "design_matrix_suggestion" in self.tools:
            tool = self.tools["design_matrix_suggestion"]
            result = tool.execute(metadata_file=metadata_file)
            
            if result["status"] == "success":
                response = f"""
**Suggested Design Matrix**

Design Formula: `{result['suggested_design']}`

**Explanation:**
{result['explanation']}

**Possible Contrasts:**
"""
                for contrast in result.get('possible_contrasts', []):
                    response += f"\n- {contrast}"
                
                return response
            else:
                return f"Error: {result.get('message', 'Unknown error')}"
        else:
            return "Design matrix suggestion tool not available"
