#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自定义工具调用对话奖励函数插件
包含ChainDialogueReward奖励函数的实现
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Union
from swift.plugin import ORM, orms

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ChainDialogueReward(ORM):
    """
    专门为工具调用对话设计的奖励函数
    评估模型在工具调用场景中的表现，包括反思、工具调用和最终答案的质量
    """

    def __init__(self, reflect_weight=0.5, call_weight=0.4, final_weight=0.2,
                 tokenizer=None, model=None, ref_model=None):
        self.reflect_weight = reflect_weight
        self.call_weight = call_weight
        self.final_weight = final_weight
        self.tokenizer = tokenizer
        self.model = model
        self.ref_model = ref_model
        self._init_semantic_model()

    def _init_semantic_model(self):
        """初始化语义相似度计算模型"""
        if OPENAI_AVAILABLE:
            try:
                # 这里可以配置你的OpenAI API或其他语义相似度服务
                # 由于是测试环境，我们使用简单的相似度计算
                self.use_semantic_similarity = False
            except Exception:
                self.use_semantic_similarity = False
        else:
            self.use_semantic_similarity = False

    def __call__(self, completions, **kwargs):
        """
        计算工具调用对话的奖励分数

        Args:
            completions: 模型生成的完整对话内容列表
            **kwargs: 其他参数，可能包含ground_truth, messages等

        Returns:
            List[float]: 每个completion的奖励分数
        """
        rewards = []
        ground_truths = self._ensure_list(kwargs.get('ground_truth', []), len(completions))
        messages_list = self._ensure_list(kwargs.get('messages', []), len(completions))

        for i, completion in enumerate(completions):
            ground_truth = ground_truths[i] if i < len(ground_truths) else ''
            messages = messages_list[i] if i < len(messages_list) else []
            reward = self._compute_single_reward(completion, ground_truth, messages)
            rewards.append(reward)

        return rewards

    def _ensure_list(self, data, target_length):
        """确保数据是列表格式"""
        if not isinstance(data, list):
            return [data] * target_length
        return data

    def _compute_single_reward(self, completion, ground_truth, messages):
        """计算单个completion的奖励分数"""
        completion_parts = self.parse_assistant_response(completion)
        ground_truth_parts = self.parse_assistant_response(ground_truth) if ground_truth else {}

        # 计算各部分奖励
        reflect_reward = self.compute_reflect_reward(
            completion_parts.get('reflect', ''),
            ground_truth_parts.get('reflect', ''))

        call_reward = self.compute_call_reward(
            completion_parts.get('call', ''),
            ground_truth_parts.get('call', ''))

        final_reward = self.compute_final_reward(
            completion_parts.get('final', ''),
            ground_truth_parts.get('final', ''))

        return self.compute_weighted_reward(
            completion_parts, ground_truth_parts,
            reflect_reward, call_reward, final_reward)

    def parse_assistant_response(self, response):
        """解析助手回复中的不同部分"""
        parts = {}
        patterns = {
            'reflect': r'<reflect>(.*?)</reflect>',
            'call': r'<call>(.*?)</call>',
            'final': r'<final>(.*?)</final>'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL)
            if match:
                parts[key] = match.group(1).strip()

        return parts

    def compute_reflect_reward(self, completion_reflect, ground_truth_reflect):
        """计算反思部分的奖励"""
        if not completion_reflect:
            return 0.0

        if not ground_truth_reflect:
            # 如果没有ground truth，基于反思内容的质量给分
            return self.evaluate_reflection_quality(completion_reflect)

        return self.compute_semantic_similarity(completion_reflect, ground_truth_reflect)

    def compute_call_reward(self, completion_call, ground_truth_call):
        """计算工具调用部分的奖励"""
        if not completion_call and not ground_truth_call:
            return 1.0
        if not completion_call or not ground_truth_call:
            return 0.0

        # 检查工具调用的格式和内容
        return self.evaluate_tool_call_quality(completion_call, ground_truth_call)

    def compute_final_reward(self, completion_final, ground_truth_final):
        """计算最终答案部分的奖励"""
        if not completion_final:
            return 0.0

        if not ground_truth_final:
            # 基于最终答案的完整性和格式给分
            return self.evaluate_final_answer_quality(completion_final)

        return self.compute_semantic_similarity(completion_final, ground_truth_final)

    def evaluate_reflection_quality(self, reflection):
        """评估反思内容的质量"""
        if not reflection:
            return 0.0

        # 简单的质量评估：长度、关键词等
        score = 0.0

        # 长度奖励（适中的长度）
        length = len(reflection.split())
        if 10 <= length <= 100:
            score += 0.3
        elif length > 5:
            score += 0.1

        # 关键词奖励
        reflection_keywords = ['思考', '分析', '需要', '应该', '可能', '因为', '所以',
                             'think', 'analyze', 'need', 'should', 'might', 'because', 'so']
        keyword_count = sum(1 for keyword in reflection_keywords if keyword in reflection.lower())
        score += min(keyword_count * 0.1, 0.4)

        return min(score, 1.0)

    def evaluate_tool_call_quality(self, completion_call, ground_truth_call):
        """评估工具调用的质量"""
        try:
            # 尝试解析JSON格式的工具调用
            import json
            completion_parsed = json.loads(completion_call)
            ground_truth_parsed = json.loads(ground_truth_call)

            # 检查函数名是否匹配
            if isinstance(completion_parsed, list) and isinstance(ground_truth_parsed, list):
                if len(completion_parsed) > 0 and len(ground_truth_parsed) > 0:
                    comp_name = completion_parsed[0].get('name', '')
                    gt_name = ground_truth_parsed[0].get('name', '')
                    if comp_name == gt_name:
                        return 1.0

            return 0.5  # 格式正确但内容不完全匹配

        except (json.JSONDecodeError, KeyError, IndexError):
            # 如果不是JSON格式，进行简单的文本匹配
            return 1.0 if completion_call.strip() == ground_truth_call.strip() else 0.0

    def evaluate_final_answer_quality(self, final_answer):
        """评估最终答案的质量"""
        if not final_answer:
            return 0.0

        score = 0.0

        # 长度奖励
        length = len(final_answer.split())
        if length >= 10:
            score += 0.4
        elif length >= 5:
            score += 0.2

        # 结构化奖励（包含列表、编号等）
        if any(marker in final_answer for marker in ['1.', '2.', '-', '*', '**']):
            score += 0.3

        # 完整性奖励（包含总结性词汇）
        summary_words = ['总结', '综上', '因此', 'summary', 'conclusion', 'therefore']
        if any(word in final_answer.lower() for word in summary_words):
            score += 0.3

        return min(score, 1.0)

    def compute_weighted_reward(self, completion_parts, ground_truth_parts,
                              reflect_reward, call_reward, final_reward):
        """计算加权总奖励"""
        total_reward = 0.0
        total_weight = 0.0

        # 根据实际存在的部分计算权重
        if 'reflect' in completion_parts or 'reflect' in ground_truth_parts:
            total_reward += self.reflect_weight * reflect_reward
            total_weight += self.reflect_weight

        if 'call' in completion_parts or 'call' in ground_truth_parts:
            total_reward += self.call_weight * call_reward
            total_weight += self.call_weight

        if 'final' in completion_parts or 'final' in ground_truth_parts:
            total_reward += self.final_weight * final_reward
            total_weight += self.final_weight

        # 如果没有任何标签，给予基础分数
        if total_weight == 0:
            return 0.1

        return total_reward / total_weight

    def compute_semantic_similarity(self, text1, text2):
        """计算语义相似度"""
        if not text1 or not text2:
            return 0.0

        if self.use_semantic_similarity:
            # 这里可以集成更复杂的语义相似度计算
            # 目前使用简单的词汇重叠相似度
            return self.simple_similarity(text1, text2)
        else:
            return self.simple_similarity(text1, text2)

    def simple_similarity(self, text1, text2):
        """简单的词汇重叠相似度"""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0


# 注册奖励函数
orms['ChainDialogueReward'] = ChainDialogueReward
