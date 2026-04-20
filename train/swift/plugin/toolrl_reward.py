# Copyright 2024 ToolRL Implementation for MS-Swift
# Based on: https://github.com/qiancheng0/ToolRL
# Paper: "ToolRL: Reward is All Tool Learning Needs" (arXiv:2504.13958)
#
# This file implements ToolRL's reward function design adapted for the MS-Swift framework.

import re
import json
import os
from typing import List, Dict, Any, Optional
from collections import Counter


class ToolRLReward:
    """
    ToolRL-style reward function for tool-calling agents.
    
    Based on the ToolRL paper (Section 3.3 Reward Design), the reward consists of:
    1. Format Reward: Checks if the output follows the expected format (<think>, <tool_call>, <response>)
    2. Tool Call Correctness Reward: Evaluates the correctness of tool calls (name + parameters)
    3. Length Reward (optional): Rewards appropriate thinking length
    
    Key differences from existing reward functions:
    - Uses ToolRL's multi-component reward (format + correctness + optional length)
    - Supports dynamic scheduling of rewards based on training steps
    - More granular tool call scoring (tool name + parameter matching)
    
    Environment Variables (for ablation studies):
    - TOOLRL_WITH_LENGTH: Enable length reward component (default: 0)
    - TOOLRL_SCHEDULE_LENGTH: Dynamic length scheduling (default: 0)
    - TOOLRL_SCHEDULE_REWARD: Dynamic reward scaling (default: 0)
    - TOOLRL_CORRECT_MAX1: Set max correctness reward to 1 instead of 3 (default: 0)
    - TOOLRL_REFINED_REWARD: Use strict matching for tool calls (default: 0)
    - TOOLRL_COARSE_REWARD: Use coarse reward (exact match or nothing) (default: 0)
    - TOOLRL_INTERMEDIATE_REWARD: Use intermediate reward granularity (default: 0)
    """
    
    def __init__(
        self,
        format_max: float = 1.0,
        format_min: float = 0.0,
        tool_max: float = 3.0,
        tool_min: float = -3.0,
        length_max: float = 1.0,
        length_min: float = 0.0,
        max_reward_len: int = 512,
        model_type: str = "llama",  # "llama" or "qwen"
    ):
        self.format_max = format_max
        self.format_min = format_min
        self.tool_max = tool_max
        self.tool_min = tool_min
        self.length_max = length_max
        self.length_min = length_min
        self.max_reward_len = max_reward_len
        self.model_type = model_type.lower()
        
        # Read environment variable overrides
        self.with_length = str(os.getenv("TOOLRL_WITH_LENGTH", "0")) == "1"
        self.schedule_length = str(os.getenv("TOOLRL_SCHEDULE_LENGTH", "0")) == "1"
        self.schedule_reward = str(os.getenv("TOOLRL_SCHEDULE_REWARD", "0")) == "1"
        self.correct_max1 = str(os.getenv("TOOLRL_CORRECT_MAX1", "0")) == "1"
        self.refined_reward = str(os.getenv("TOOLRL_REFINED_REWARD", "0")) == "1"
        self.coarse_reward = str(os.getenv("TOOLRL_COARSE_REWARD", "0")) == "1"
        self.intermediate_reward = str(os.getenv("TOOLRL_INTERMEDIATE_REWARD", "0")) == "1"
        
        if self.correct_max1:
            self.tool_max = 1.0
            self.tool_min = -1.0
    
    # ========== Parsing Functions ==========
    
    def _extract_assistant_content(self, full_response: str) -> str:
        """Extract the assistant's content from the full model response."""
        if self.model_type == "llama":
            # Llama format: <|start_header_id|>assistant<|end_header_id|>...<|eot_id|>
            if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                content = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                content = content.split("<|eot_id|>")[0].strip()
                return content
        elif self.model_type == "qwen":
            # Qwen format: <|im_start|>assistant...<|im_end|>
            if "<|im_start|>assistant" in full_response:
                content = full_response.split("<|im_start|>assistant")[-1]
                content = content.split("<|im_end|>")[0].strip()
                return content
        # Fallback: return the full response
        return full_response.strip()
    
    def _parse_think_content(self, text: str) -> str:
        """Extract content from <think>...</think> tags."""
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _parse_tool_calls_llama(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from llama-style format.
        Llama uses JSON objects directly: {"name": "...", "parameters": {...}}
        """
        calls = []
        # Find all JSON objects that look like tool calls
        i = 0
        while i < len(text):
            if text[i] == '{':
                stack = 1
                start = i
                i += 1
                in_string = False
                escape_next = False
                
                while i < len(text) and stack > 0:
                    char = text[i]
                    if escape_next:
                        escape_next = False
                    elif char == '\\':
                        escape_next = True
                    elif char == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            stack += 1
                        elif char == '}':
                            stack -= 1
                    i += 1
                
                if stack == 0:
                    json_str = text[start:i]
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict) and 'name' in parsed:
                            name = parsed.get('name', '')
                            params = parsed.get('parameters', {})
                            if not isinstance(params, dict):
                                params = {}
                            calls.append({'name': str(name).strip(), 'parameters': params})
                    except json.JSONDecodeError:
                        pass
            else:
                i += 1
        return calls
    
    def _parse_tool_calls_qwen(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from qwen-style format.
        Qwen uses <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        """
        calls = []
        # Remove think blocks first
        text = re.sub(r'(?is)<think>.*?</think>', '', text)
        
        blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL | re.IGNORECASE)
        for blk in blocks:
            blk = blk.strip()
            try:
                parsed = json.loads(blk)
                if isinstance(parsed, dict) and 'name' in parsed:
                    name = parsed.get('name', '')
                    args = parsed.get('arguments', {})
                    if not isinstance(args, dict):
                        args = {}
                    calls.append({'name': str(name).strip(), 'parameters': args})
            except json.JSONDecodeError:
                pass
        return calls
    
    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls based on model type."""
        if self.model_type == "qwen":
            return self._parse_tool_calls_qwen(text)
        else:
            return self._parse_tool_calls_llama(text)
    
    def _parse_response(self, text: str) -> str:
        """Extract <response>...</response> or <final>...</final> content."""
        # Try <response> first
        match = re.search(r'<response>(.*?)</response>', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Try <final>
        match = re.search(r'<final>(.*?)</final>', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
    
    def _parse_reflect(self, text: str) -> str:
        """Extract <reflect>...</reflect> content."""
        match = re.search(r'<reflect>(.*?)</reflect>', text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    # ========== Reward Computation Functions ==========
    
    def _match_score(self, list1: List, list2: List) -> float:
        """
        Compute a similarity score considering element frequency, ignoring order.
        This is from ToolRL's original implementation.
        """
        if list1 == list2:
            return 1.0
        
        if self.refined_reward:
            if list1 != list2:
                return 0.0
        
        if not list1 or not list2:
            return 0.0
        
        count1 = Counter(list1)
        count2 = Counter(list2)
        
        intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
        max_possible = len(list1) + len(list2) - intersection
        
        return intersection / max_possible if max_possible > 0 else 0.0
    
    def compute_format_reward(
        self, 
        response: str, 
        gt_answer: str,
        step: int = 0
    ) -> float:
        """
        ToolRL Format Reward: Check if the output follows expected format.
        
        Expected formats based on ground truth:
        - <think>...</think>\n<response>...</response> (for final answer)
        - <think>...</think>\n<tool_call>\n...\n</tool_call> (for tool call)
        - <think>...</think>\n<tool_call>\n...\n</tool_call>\n<response>...</response> (both)
        """
        max_reward = self.format_max
        min_reward = self.format_min
        
        # Dynamic scheduling
        if self.schedule_reward:
            max_reward = 2 - (2 - self.format_max) * step / 150
            min_reward = -2 + (2 + self.format_min) * step / 150
            max_reward = max(max_reward, 1.0)
            min_reward = min(min_reward, -1.0)
        
        # Determine expected format from ground truth
        has_response_gt = "<response>" in gt_answer or "<final>" in gt_answer
        has_tool_call_gt = "<tool_call>" in gt_answer or "{\"name\":" in gt_answer
        
        # Check format compliance
        has_think = "<think>" in response and "</think>" in response
        
        if has_response_gt and not has_tool_call_gt:
            # Expect: <think>...</think>\n<response>...</response>
            pattern = r"^<think>.*?</think>\s*(<response>.*?</response>|<final>.*?</final>)$"
            if re.search(pattern, response, re.DOTALL):
                return max_reward
        elif not has_response_gt and has_tool_call_gt:
            # Expect: <think>...</think>\n<tool_call>...\n</tool_call>
            # For llama, we check for JSON tool calls
            if self.model_type == "llama":
                if has_think and "{\"name\":" in response:
                    return max_reward
            else:
                pattern = r"^<think>.*?</think>\s*<tool_call>.*?</tool_call>$"
                if re.search(pattern, response, re.DOTALL):
                    return max_reward
        elif has_response_gt and has_tool_call_gt:
            # Expect both
            if self.model_type == "llama":
                if has_think and "{\"name\":" in response and ("<response>" in response or "<final>" in response):
                    return max_reward
            else:
                pattern = r"^<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*(<response>.*?</response>|<final>.*?</final>)$"
                if re.search(pattern, response, re.DOTALL):
                    return max_reward
        else:
            # Only think expected
            pattern = r"^<think>.*?</think>$"
            if re.search(pattern, response, re.DOTALL):
                return max_reward
        
        return min_reward
    
    def compute_tool_call_reward(
        self,
        gt_tools: List[Dict[str, Any]],
        pd_tools: List[Dict[str, Any]],
        step: int = 0
    ) -> float:
        """
        ToolRL Tool Call Correctness Reward.
        
        Computes reward based on:
        1. Tool name matching
        2. Parameter key matching
        3. Parameter value correctness
        """
        max_reward = self.tool_max
        min_reward = self.tool_min
        
        # Dynamic scheduling for later training stages
        if self.schedule_reward:
            max_reward = (self.tool_max - 2) * step / 150 + 2
            min_reward = (self.tool_min + 2) * step / 150 - 2
            max_reward = min(max_reward, 3.0)
            min_reward = max(min_reward, -3.0)
        
        # Exact match = max reward
        if gt_tools == pd_tools:
            return max_reward
        
        # Coarse reward: exact match or nothing
        if self.coarse_reward:
            return min_reward
        
        # Empty cases
        if not gt_tools:
            return 0.0  # No tool call expected
        if not pd_tools:
            return min_reward  # Tool call expected but not provided
        
        # Tool name matching
        gt_names = [tool.get("name", "") for tool in gt_tools]
        pd_names = [tool.get("name", "") for tool in pd_tools]
        score = self._match_score(gt_names, pd_names)
        
        # Local max possible score
        local_max_possible = 1.0
        used_pd_indices = set()
        
        for gt_tool in gt_tools:
            gt_name = gt_tool.get("name", "")
            gt_params = gt_tool.get("parameters", {})
            
            if self.intermediate_reward:
                local_max_possible += 1.0
            else:
                local_max_possible += 1.0 + len(gt_params)
            
            best_match_score = 0.0
            best_match_index = -1
            
            for i, pd_tool in enumerate(pd_tools):
                if i in used_pd_indices:
                    continue
                if pd_tool.get("name", "") != gt_name:
                    continue
                
                if self.intermediate_reward:
                    if gt_tool == pd_tool:
                        best_match_score = 1.0
                        best_match_index = i
                        break
                    continue
                
                pd_params = pd_tool.get("parameters", {})
                param_score = self._match_score(list(gt_params.keys()), list(pd_params.keys()))
                
                # Parameter value correctness
                correctness_score = sum(
                    1.0 for k, v in gt_params.items() 
                    if k in pd_params and pd_params[k] == v
                )
                
                total_score = param_score + correctness_score
                
                if total_score > best_match_score:
                    best_match_score = total_score
                    best_match_index = i
            
            if best_match_index != -1:
                used_pd_indices.add(best_match_index)
                score += best_match_score
        
        # Normalize score to reward range
        normalized = score / local_max_possible if local_max_possible > 0 else 0.0
        return (max_reward - min_reward) * normalized + min_reward
    
    def compute_length_reward(
        self,
        response: str,
        step: int = 0
    ) -> float:
        """
        ToolRL Length Reward: Rewards appropriate thinking length.
        """
        if not self.with_length:
            return 0.0
        
        # Dynamic max reward length
        if self.schedule_length:
            max_reward_len = (640 - 384) * step / 105 + 384
        else:
            max_reward_len = self.max_reward_len
        
        # Extract think content
        think_content = self._parse_think_content(response)
        if not think_content:
            return self.length_min
        
        # Calculate length-based reward
        word_count = len(think_content.split())
        reward = round(word_count / max_reward_len, 2)
        reward = min(reward, 1.0)
        
        return reward * (self.length_max - self.length_min) + self.length_min
    
    def compute_score(
        self,
        completion: str,
        ground_truth: str,
        step: int = 0
    ) -> Dict[str, float]:
        """
        Compute the full ToolRL reward.
        
        Returns a dictionary with:
        - total: Combined reward score
        - format: Format reward component
        - correctness: Tool call correctness reward
        - length: Length reward (if enabled)
        """
        # Extract assistant content from full response
        response = self._extract_assistant_content(completion)
        
        # Parse ground truth tool calls
        gt_tools = self._parse_tool_calls(ground_truth)
        pd_tools = self._parse_tool_calls(response)
        
        # Compute individual rewards
        format_reward = self.compute_format_reward(response, ground_truth, step)
        correctness_reward = self.compute_tool_call_reward(gt_tools, pd_tools, step)
        length_reward = self.compute_length_reward(response, step)
        
        # Combine rewards
        total = format_reward + correctness_reward + length_reward
        
        return {
            'total': total,
            'format': format_reward,
            'correctness': correctness_reward,
            'length': length_reward
        }
    
    def __call__(
        self,
        completions: List[str],
        ground_truth: Optional[List[str]] = None,
        step: int = 0,
        **kwargs
    ) -> List[float]:
        """
        Main entry point for reward computation.
        Compatible with MS-Swift's reward function interface.
        """
        if not isinstance(completions, list):
            completions = [completions]
        
        # Handle different ground truth parameter names
        if ground_truth is None:
            ground_truth = kwargs.get('ground_truths', None)
        if ground_truth is None:
            ground_truth = kwargs.get('references', None)
        if ground_truth is None:
            ground_truth = kwargs.get('labels', None)
        
        if ground_truth is None:
            ground_truth = [""] * len(completions)
        elif not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        
        # Ensure lists are same length
        while len(ground_truth) < len(completions):
            ground_truth.append(ground_truth[-1] if ground_truth else "")
        
        rewards = []
        for comp, gt in zip(completions, ground_truth):
            try:
                result = self.compute_score(comp, gt, step)
                rewards.append(result['total'])
            except Exception as e:
                print(f"[ToolRL-Reward-ERR] {e}")
                rewards.append(0.0)
        
        return rewards


class ToolRLRewardLlama(ToolRLReward):
    """ToolRL Reward function pre-configured for Llama models."""
    
    def __init__(self, **kwargs):
        kwargs['model_type'] = 'llama'
        super().__init__(**kwargs)


class ToolRLRewardQwen(ToolRLReward):
    """ToolRL Reward function pre-configured for Qwen models."""
    
    def __init__(self, **kwargs):
        kwargs['model_type'] = 'qwen'
        super().__init__(**kwargs)


# For MS-Swift compatibility, we also provide a simple wrapper
def create_toolrl_reward(model_type: str = "llama", **kwargs):
    """Factory function to create ToolRL reward instances."""
    if model_type.lower() == "qwen":
        return ToolRLRewardQwen(**kwargs)
    else:
        return ToolRLRewardLlama(**kwargs)
