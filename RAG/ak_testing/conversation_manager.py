"""
ConversationManager class for handling tutoring conversation flow and history management.
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import openai


class ConversationManager:
    """
    Manages conversation state, history, and turn flow for AI tutoring sessions.
    
    Key features:
    - Maintains conversation history for both student and tutor
    - Manages turn flow with configurable number of turns
    - Integrates with existing memory retrieval system
    - Uses proper chat completion format with message history
    """
    
    def __init__(self, openai_client, build_memory_function):
        """
        Initialize conversation manager.
        
        Args:
            openai_client: OpenAI client instance
            build_memory_function: Function to retrieve memory blocks (existing build_memory_block)
        """
        self.openai_client = openai_client
        self.build_memory_function = build_memory_function
        self.conversation_history = []
        self.memory_block = ""
        self.skill_block = ""
        self.prereq_block = ""
        
    def initialize_session(self, student_msg: str, kaid: str, skill_tag: str, k: int = 4):
        """
        Initialize a new conversation session with memory retrieval.
        
        Args:
            student_msg: Initial student message
            kaid: Student ID
            skill_tag: Skill being tutored
            k: Number of similar conversations to retrieve
        """
        # Get static memory from vector store (retrieved once per session)
        self.memory_block, self.skill_block, self.prereq_block = self.build_memory_function(
            student_msg, kaid, skill_tag, k=k
        )
        
        # Initialize conversation history with the first student message
        self.conversation_history = [
            {"role": "user", "content": student_msg}
        ]
        
    def get_system_messages(self, system_prompt: str) -> List[Dict[str, str]]:
        """
        Build system messages including context and memory blocks.
        
        Args:
            system_prompt: Base system prompt for the tutor
            
        Returns:
            List of system messages with context
        """
        # Format prerequisite information more clearly
        if self.prereq_block.strip():
            prereq_formatted = f"""PREREQUISITE SKILL STATES (use ONLY these skills):
{self.prereq_block}

Remember: ONLY reference prerequisite skills that are listed above. Do not invent additional prerequisites."""
        else:
            prereq_formatted = "PREREQUISITE SKILL STATES: No prerequisite skills provided for this session."
            
        system_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f'Current knowledge state on the relevant skill: {self.skill_block}'},
            {"role": "system", "content": f'Relevant past conversations: {self.memory_block}'},
            {"role": "system", "content": prereq_formatted}
        ]
        return system_messages
        
    def get_tutor_response(self, system_prompt: str, model: str = "gpt-4-khan", temperature: float = 0.0) -> str:
        """
        Generate tutor response based on current conversation history.
        
        Args:
            system_prompt: System prompt for tutor
            model: OpenAI model to use
            temperature: Temperature setting
            
        Returns:
            Tutor response string
        """
        # Build complete message history
        messages = self.get_system_messages(system_prompt) + self.conversation_history
        
        completion = self.openai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages
        )
        
        tutor_response = completion.choices[0].message.content
        
        # Add tutor response to conversation history
        self.conversation_history.append({"role": "assistant", "content": tutor_response})
        
        return tutor_response
        
    def get_student_response(self, student_system_prompt: str, model: str = "gpt-4-khan", temperature: float = 0.0) -> str:
        """
        Generate student response based on the last tutor message.
        
        Args:
            student_system_prompt: System prompt for student persona
            model: OpenAI model to use
            temperature: Temperature setting
            
        Returns:
            Student response string
        """
        # Get the last tutor message
        last_tutor_message = self.conversation_history[-1]["content"]
        
        # Generate student response
        completion = self.openai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": student_system_prompt + last_tutor_message}]
        )
        
        student_response = completion.choices[0].message.content
        
        # Add student response to conversation history
        self.conversation_history.append({"role": "user", "content": student_response})
        
        return student_response
        
    def run_conversation(self, 
                        student_msg: str, 
                        kaid: str, 
                        skill_tag: str,
                        tutor_system_prompt: str,
                        student_system_prompt: str,
                        max_turns: int = 3,
                        k: int = 4,
                        model: str = "gpt-4-khan",
                        temperature: float = 0.0) -> pd.DataFrame:
        """
        Run a complete conversation session.
        
        Args:
            student_msg: Initial student message
            kaid: Student ID  
            skill_tag: Skill being tutored
            tutor_system_prompt: System prompt for tutor
            student_system_prompt: System prompt for student
            max_turns: Maximum number of conversation turns
            k: Number of similar conversations to retrieve
            model: OpenAI model to use
            temperature: Temperature setting
            
        Returns:
            DataFrame with conversation turns
        """
        # Initialize session
        self.initialize_session(student_msg, kaid, skill_tag, k=k)
        
        # Track conversation for output
        student_questions = [student_msg]
        tutor_responses = []
        
        # Run conversation turns
        for turn in range(max_turns):
            # Get tutor response
            tutor_response = self.get_tutor_response(
                tutor_system_prompt, model, temperature
            )
            tutor_responses.append(tutor_response)
            
            # Get student response (except on last turn)
            if turn < max_turns - 1:
                student_response = self.get_student_response(
                    student_system_prompt, model, temperature
                )
                student_questions.append(student_response)
        
        # Return conversation as DataFrame
        return pd.DataFrame({
            'question': student_questions, 
            'responses': tutor_responses
        })
        
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
        
    def get_memory_context(self) -> Tuple[str, str, str]:
        """Get the current memory context (static for session)."""
        return self.memory_block, self.skill_block, self.prereq_block
