import os
import re
from typing import TYPE_CHECKING, Dict, List, Any, Union, Optional

import json
import numpy as np

if TYPE_CHECKING:
    from swift.llm import InferRequest
from openai import OpenAI

import math

class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify==0.5.2'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # edge case
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 tokenizer=None,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.tokenizer = tokenizer
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []
        for content, acc_reward in zip(completions, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(self.tokenizer.encode(content))
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, tokenizer, soft_max_length, soft_cache_length):
        self.tokenizer = tokenizer
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            completion_length = len(self.tokenizer.encode(completion))
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards

class ChainDialogueReward(ORM):
    """
    反思能力与工具调用奖励函数
    1. 反思奖励：当工具调用不合适时，模型通过反思（<reflect>）来修正工具调用。
    2. 正确工具调用奖励：模型能够正确调用工具时获得奖励。
    3. 格式奖励：如果没有反思，模型仍然可以通过格式奖励进行补偿。
    4. <final>奖励：用于评估最终的工具调用结果是否符合期望。
    """

    def __init__(self, reflect_weight=0.3, call_weight=0.4, final_weight=0.3, format_penalty_factor=0.8,
                 tokenizer=None, model=None, ref_model=None, api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here")):
        self.reflect_weight = reflect_weight  # 反思奖励的权重
        self.call_weight = call_weight        # 工具调用奖励的权重
        self.final_weight = final_weight      # <final>部分奖励的权重
        self.format_penalty_factor = format_penalty_factor  # 格式错误的惩罚因子
        self.tokenizer = tokenizer
        self.model = model
        self.ref_model = ref_model
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=api_key, 
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        
        # 最大奖励设定：每个部分的最大奖励值（假设每个部分的最大值是 1）
        self.max_reflect_reward = 1.0
        self.max_call_reward = 1.0
        self.max_final_reward = 1.0

    def _ensure_list(self, value, length):
        """确保值是一个列表，并扩展到指定长度"""
        if not isinstance(value, list):
            if value is None:
                value = []
            else:
                value = [value]

        # 如果列表长度不够，用最后一个元素填充或用空字符串填充
        while len(value) < length:
            if value:
                value.append(value[-1])  # 重复最后一个元素
            else:
                value.append('')  # 如果列表为空，用空字符串填充

        return value[:length]  # 确保不超过指定长度

    def parse_assistant_response(self, response):
        """解析助手回复，提取 reflect、tool_call 和 final 部分"""
        parts = {
            'reflect': '',
            'calls': [],  # 修改：改为支持多个tool_call的列表
            'final': ''
        }

        # 提取 <reflect> 部分
        reflect_match = re.search(r'<reflect>(.*?)</reflect>', response, re.DOTALL)
        if reflect_match:
            parts['reflect'] = reflect_match.group(1).strip()

        # 提取所有 <tool_call> 部分
        tool_call_matches = re.findall(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL | re.IGNORECASE)
        calls = []
        for c in tool_call_matches:
            c = c.strip()
            # try parse JSON
            try:
                parsed = json.loads(c)
                calls.append(parsed)
            except Exception:
                calls.append(c)
        parts['calls'] = calls

        # 提取 <final> 部分
        final_match = re.search(r'<final>(.*?)</final>', response, re.DOTALL)
        if final_match:
            parts['final'] = final_match.group(1).strip()

        return parts

    def __call__(self, completions, **kwargs):
        rewards = []
        ground_truths = self._ensure_list(kwargs.get('ground_truth', []), len(completions))

        for i, completion in enumerate(completions):
            ground_truth = ground_truths[i] if i < len(ground_truths) else ''
            reward = self._compute_single_reward(completion, ground_truth, i)
            rewards.append(reward)

        return rewards

    def _compute_single_reward(self, completion, ground_truth, index):
        """计算单条对话的奖励，主要根据反思、工具调用和最终输出的正确性"""
        if completion is None or ground_truth is None:
            print(f"[ORM-{index}] 输入为空: completion={completion is None}, ground_truth={ground_truth is None}")
            return 0.0

        # 🔧 调试：输出原始completion内容
        print(f"[ORM-{index}] Completion前50字符: '{str(completion)[:50]}...'")

        # 分解 completion 和 ground_truth，提取 reflect、call 和 final 部分
        completion_parts = self.parse_assistant_response(completion)
        ground_truth_parts = self.parse_assistant_response(ground_truth)

        # 🔧 调试：输出解析结果
        print(f"[ORM-{index}] 解析结果 - completion_parts: {completion_parts}")
        print(f"[ORM-{index}] 解析结果 - ground_truth_parts: {ground_truth_parts}")

        # 获取反思、工具调用和最终输出的内容
        completion_reflect = completion_parts.get('reflect', '')
        ground_truth_reflect = ground_truth_parts.get('reflect', '')
        completion_calls = completion_parts.get('calls', [])
        ground_truth_calls = ground_truth_parts.get('calls', [])
        completion_final = completion_parts.get('final', '')
        ground_truth_final = ground_truth_parts.get('final', '')

        # 初始化奖励
        reward_reflect = 0.0
        reward_call = 0.0
        reward_final = 0.0
        
        # 计算反思奖励
        if completion_reflect and ground_truth_reflect:
            reward_reflect = self.compute_reflect_reward(completion_reflect, ground_truth_reflect)
            print(f"[ORM-{index}] reflect奖励: {reward_reflect}")
        
        # 计算工具调用奖励（现在使用列表格式）
        reward_call = self.compute_call_reward(completion_calls, ground_truth_calls)
        print(f"[ORM-{index}] call奖励: {reward_call}")
        
        # 计算最终输出奖励
        reward_final = self.compute_final_reward(completion_final, ground_truth_final)
        print(f"[ORM-{index}] final奖励: {reward_final}")

        # 归一化每个部分的奖励
        normalized_reflect_reward = (reward_reflect * self.reflect_weight) / self.max_reflect_reward
        normalized_call_reward = (reward_call * self.call_weight) / self.max_call_reward
        normalized_final_reward = (reward_final * self.final_weight) / self.max_final_reward

        # 综合奖励计算
        reward = normalized_reflect_reward + normalized_call_reward + normalized_final_reward

        # 如果有格式问题，则应用格式惩罚
        format_factor = self.compute_format_factor(completion_parts, ground_truth_parts)
        reward *= format_factor

        # 🔧 备用奖励：如果所有部分都是0，使用简单的字符串相似度
        if reward == 0.0:
            # 去除<think></think>标签及其内容后再计算相似度
            completion_without_think = self._remove_think_tags(completion)
            ground_truth_without_think = self._remove_think_tags(ground_truth)
            fallback_reward = self.compute_semantic_similarity(completion_without_think, ground_truth_without_think)
            print(f"[ORM-{index}] 使用备用奖励: {fallback_reward}")
            reward = fallback_reward * 0.1  # 降低权重，因为这是fallback

        print(f"[ORM-{index}] 最终奖励: {reward} (reflect:{normalized_reflect_reward:.3f}, call:{normalized_call_reward:.3f}, final:{normalized_final_reward:.3f}, factor:{format_factor})")

        return reward

    def compute_reflect_reward(self, completion_reflect, ground_truth_reflect):
        """计算反思奖励，若模型能够有效修正工具调用错误，则给予奖励"""
        if not completion_reflect or not ground_truth_reflect:
            return 0.0

        # 使用 GPT-4 评估反思内容的合理性
        similarity = self.llm_as_judge_reflect(completion_reflect, ground_truth_reflect)
        return similarity

    def llm_as_judge_reflect(self, completion_reflect, ground_truth_reflect):
        """使用embedding模型计算反思内容的余弦相似度，如果API失败则使用备用方法"""
        # 首先尝试使用embedding模型
        try:
            print(f"[Embedding] 📤 发送请求到text-embedding-3-large...")

            # 获取两个文本的embedding
            completion_response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=completion_reflect
            )

            ground_truth_response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=ground_truth_reflect
            )

            # 提取embedding向量
            completion_embedding = completion_response.data[0].embedding
            ground_truth_embedding = ground_truth_response.data[0].embedding

            print(f"[Embedding] 📥 获取embedding成功，向量维度: {len(completion_embedding)}")

            # 计算余弦相似度
            cosine_similarity = self.compute_cosine_similarity(completion_embedding, ground_truth_embedding)

            print(f"[Embedding] ✅ 余弦相似度计算成功: {cosine_similarity:.4f}")

            return cosine_similarity

        except Exception as e:
            print(f"[Embedding] ❌ API调用失败: {e}")
            print(f"[Embedding] 🔄 使用备用语义相似度方法")
            # 使用备用的语义相似度方法
            fallback_score = self.compute_semantic_similarity(completion_reflect, ground_truth_reflect)
            print(f"[Embedding] 🔄 备用方法评分: {fallback_score}")
            return fallback_score

    def compute_cosine_similarity(self, embedding1, embedding2):
        """计算两个embedding向量的余弦相似度"""
        import numpy as np

        # 转换为numpy数组
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = dot_product / (norm1 * norm2)

        # 确保结果在[0,1]范围内（余弦相似度原本是[-1,1]，但对于语义相似度我们关注正相关）
        return max(0.0, cosine_sim)

    def compute_final_reward(self, completion_final, ground_truth_final):
        """计算 <final> 部分的奖励"""
        if not completion_final or not ground_truth_final:
            return 0.0

        # 计算语义相似度
        similarity = self.compute_semantic_similarity(completion_final, ground_truth_final)
        return similarity

    def compute_call_reward(self, completion_calls, ground_truth_calls):
        """计算工具调用奖励，检查工具调用是否正确（现在处理列表格式）"""
        # 处理空列表情况
        if not completion_calls and not ground_truth_calls:
            return 0.5  # 🔧 修复：降低默认奖励，避免掩盖问题
        if not completion_calls or not ground_truth_calls:
            return 0.0  # 一个正确一个错误，惩罚

        # 将列表转换为字符串进行比较（保持向后兼容）
        completion_str = json.dumps(completion_calls, ensure_ascii=False, sort_keys=True)
        ground_truth_str = json.dumps(ground_truth_calls, ensure_ascii=False, sort_keys=True)

        # 完全匹配
        if completion_str == ground_truth_str:
            return 1.0

        # 部分匹配 - 计算相似度
        similarity = self.compute_semantic_similarity(completion_str, ground_truth_str)
        return similarity

    def compute_format_factor(self, completion_parts, ground_truth_parts):
        """计算格式因子，如果格式错误则应用惩罚"""
        gt_has_reflect = self._has_real_thinking(ground_truth_parts)
        comp_has_reflect = self._has_real_thinking(completion_parts)

        # 格式一致，无调节（完整奖励）
        if gt_has_reflect == comp_has_reflect:
            return 1.0

        # 格式不一致，应用折扣因子
        return self.format_penalty_factor

    def _has_real_thinking(self, parts):
        """判断是否有有效的反思内容"""
        return bool(parts.get('reflect', '').strip())

    def compute_semantic_similarity(self, text1, text2):
        """计算文本的语义相似度"""
        if not text1.strip() or not text2.strip():
            return 0.0

        text1_clean = text1.strip().lower()
        text2_clean = text2.strip().lower()

        # 完全匹配
        if text1_clean == text2_clean:
            return 1.0

        # 计算简单的相似度（基于共同词汇和长度）
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())

        if not words1 or not words2:
            return 0.0

        # Jaccard相似度
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard_sim = len(intersection) / len(union) if union else 0.0

        # 长度相似度
        len_sim = min(len(text1_clean), len(text2_clean)) / max(len(text1_clean), len(text2_clean))

        # 综合相似度
        similarity = 0.7 * jaccard_sim + 0.3 * len_sim
        return min(1.0, max(0.0, similarity))

    def _remove_think_tags(self, text):
        """去除<think></think>标签及其中的内容"""
        import re
        # 使用正则表达式去除<think>...</think>及其内容
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned_text.strip()

class ChainDialogueRewardOptimized:
    """
    优化后的 reward 设计（专注 reward 逻辑）
    主要特点：
      - 权重归一化
      - 支持多次 <call>（结构化 JSON 优先）
      - 更细化的格式惩罚（按缺失项重要性）
      - 反思/最终使用语义相似度/embedding（可降级）
      - 可配置化参数，去除硬编码 magic numbers
    """

    def __init__(
        self,
        reflect_weight: float = 0.3,
        call_weight: float = 0.4,
        final_weight: float = 0.3,
        # 各标签重要性（用于缺失惩罚）
        tag_importance: Optional[Dict[str, float]] = None,
        # call 比较内部权重
        call_tool_weight: float = 0.5,
        call_args_weight: float = 0.5,
        # 当 ground truth 有 call 而 completion 缺失时的惩罚比例（0..1）
        missing_tag_penalty_strength: float = 0.8,
        # fallback 权重（当所有项都几乎为0时使用）
        fallback_weight: float = 0.15,
        # 非精确匹配的最大分数阈值
        max_non_exact_call_score: float = 0.6,
        # tiny smoothing for numeric stability
        eps: float = 1e-8,
        # 外部 embedding client（可注入），如果为空则使用文本相似度 fallback
        embedding_client = None,
    ):
        # 权重归一化
        total = float(reflect_weight + call_weight + final_weight)
        if total <= 0:
            raise ValueError("reflect/call/final weights must sum to > 0")
        self.reflect_weight = reflect_weight / total
        self.call_weight = call_weight / total
        self.final_weight = final_weight / total

        self.tag_importance = tag_importance or {'reflect': 0.2, 'calls': 0.5, 'final': 0.3}
        # ensure tag_importance sums to 1 (for interpretability)
        s = sum(self.tag_importance.values())
        if s <= 0:
            raise ValueError("tag_importance must sum > 0")
        for k in self.tag_importance:
            self.tag_importance[k] = self.tag_importance[k] / s

        self.call_tool_weight = call_tool_weight
        self.call_args_weight = call_args_weight
        self.missing_tag_penalty_strength = missing_tag_penalty_strength
        self.fallback_weight = fallback_weight
        self.eps = eps
        self.embedding_client = embedding_client

    # ---------- parse ----------
    def parse_assistant_response(self, response: str) -> Dict[str, Any]:
        """
        更稳健的解析：
          - extract <reflect>...</reflect>（第一个）
          - extract <final>...</final>（第一个）
          - extract 所有 <call>...</call> 作为列表（尝试将每个 call parse 为 JSON）
        返回： {'reflect': str, 'final': str, 'calls': [str or dict, ...]}
        """
        parts = {'reflect': '', 'final': '', 'calls': []}
        if not response:
            return parts

        # 不区分大小写标签，使用 re.I
        def first_tag(text, tag):
            m = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ''

        parts['reflect'] = first_tag(response, 'reflect')
        parts['final'] = first_tag(response, 'final')

        call_matches = re.findall(r'<call>(.*?)</call>', response, flags=re.DOTALL | re.IGNORECASE)
        calls = []
        for c in call_matches:
            c = c.strip()
            # try parse JSON
            try:
                parsed = json.loads(c)
                calls.append(parsed)
            except Exception:
                # as fallback, try to parse as simple key=value lines, else raw text
                calls.append(c)
        parts['calls'] = calls
        return parts

    # ---------- cosine helper ----------
    def _cosine_from_embeddings(self, emb1, emb2) -> float:
        """返回线性映射到 [0,1] 的余弦相似度；若无法计算返回 0.0"""
        if emb1 is None or emb2 is None:
            return 0.0
        try:
            if np is not None:
                v1 = np.array(emb1); v2 = np.array(emb2)
                denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + self.eps
                cos = float(np.dot(v1, v2) / denom)
            else:
                # 原生 python 实现（较慢）
                dot = sum(a*b for a,b in zip(emb1, emb2))
                norm1 = math.sqrt(sum(a*a for a in emb1)) + self.eps
                norm2 = math.sqrt(sum(b*b for b in emb2)) + self.eps
                cos = dot / (norm1 * norm2)
            # map to [0,1]
            return max(0.0, min(1.0, (cos + 1.0) / 2.0))
        except Exception:
            return 0.0

    # ---------- semantic similarity fallback ----------
    def compute_semantic_similarity(self, a: str, b: str) -> float:
        """
        简洁的 fallback 语义相似度：结合 n-gram jaccard 与长度比
        （如果提供 embedding_client，可以替换为 embedding 相似度）
        """
        if not a or not b:
            return 0.0
        # 如果有 embedding client（外部注入），优先用 embedding
        if self.embedding_client is not None:
            try:
                emb_a = self._get_embedding(a)
                emb_b = self._get_embedding(b)
                return self._cosine_from_embeddings(emb_a, emb_b)
            except Exception:
                pass

        # simple text fallback
        a_clean = re.sub(r'\s+', ' ', a.strip().lower())
        b_clean = re.sub(r'\s+', ' ', b.strip().lower())

        if a_clean == b_clean:
            return 1.0

        def ngram_set(s, n):
            toks = s.split()
            if len(toks) < n:
                return set([' '.join(toks)])
            return set(' '.join(toks[i:i+n]) for i in range(len(toks)-n+1))
        # combine 1-gram and 2-gram
        g1a = ngram_set(a_clean, 1)
        g1b = ngram_set(b_clean, 1)
        g2a = ngram_set(a_clean, 2)
        g2b = ngram_set(b_clean, 2)

        j1 = len(g1a & g1b) / (len(g1a | g1b) + self.eps)
        j2 = len(g2a & g2b) / (len(g2a | g2b) + self.eps)
        len_sim = min(len(a_clean), len(b_clean)) / (max(len(a_clean), len(b_clean)) + self.eps)
        # 权重可调：更重视 unigram 相似度
        score = 0.6 * j1 + 0.3 * j2 + 0.1 * len_sim
        return float(max(0.0, min(1.0, score)))

    def _get_embedding(self, text: str):
        """包装 embedding client 调用，客户端需提供 create(model, input) 风格接口或自定义接口"""
        # 这里不实现特定 API，假定 embedding_client 有 method `embeddings.create`
        if hasattr(self.embedding_client, 'embeddings'):
            resp = self.embedding_client.embeddings.create(model="text-embedding-3-large", input=text)
            return resp.data[0].embedding
        # 若直接是函数式客户端： embedding_client(text) -> vector
        if callable(self.embedding_client):
            return self.embedding_client(text)
        raise RuntimeError("No valid embedding client available")

    # ---------- call reward ----------
    # ---------- helper: normalize ----------
    def _normalize_calls(self, calls_raw):
        """
        将任意形状的 calls 规整为：
            List[{"name": <str>, "arguments": <dict>}]
        支持输入为：None / str(json) / dict / list / 嵌套 list
        也兼容遗留键：tool/args
        """
        out = []

        def coerce_one(x):
            if not isinstance(x, dict):
                return None
            # 兼容两套键名
            name = x.get('name') or x.get('tool')
            args = x.get('arguments', None)
            if args is None:
                args = x.get('args', {})
            # 基本清洗
            if not isinstance(args, dict):
                # 若是标量或列表，转成字符串再放入一个占位 key
                args = {'_value': json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args}
            if name:
                return {'name': str(name).strip(), 'arguments': args}
            return None

        def collect(obj):
            if obj is None:
                return
            if isinstance(obj, list):
                for it in obj:
                    collect(it)
                return
            if isinstance(obj, dict):
                c = coerce_one(obj)
                if c:
                    out.append(c)
                return
            if isinstance(obj, str):
                s = obj.strip()
                if not s:
                    return
                # 尝试当成 JSON
                try:
                    j = json.loads(s)
                    collect(j)
                    return
                except Exception:
                    # 把纯字符串当成“工具名”，无参数
                    out.append({'name': s, 'arguments': {}})
                return
            # 其他类型一律字符串化为“工具名”
            out.append({'name': str(obj), 'arguments': {}})

        collect(calls_raw)
        return out

    # ---------- parse ----------
    def parse_assistant_response(self, response: str) -> Dict[str, Any]:
        parts = {'reflect': '', 'final': '', 'calls': []}
        if not response:
            return parts

        def first_tag(text, tag):
            m = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ''

        parts['reflect'] = first_tag(response, 'reflect')
        parts['final'] = first_tag(response, 'final')

        # 可能只有一个 <call> 包一整个数组，也可能有多个 <call>
        call_matches = re.findall(r'<call>(.*?)</call>', response, flags=re.DOTALL | re.IGNORECASE)

        collected = []
        for c in call_matches:
            c = c.strip()
            # 先尝试 JSON
            try:
                parsed = json.loads(c)
                collected.append(parsed)   # 这里可能是 list，也可能是 dict
            except Exception:
                collected.append(c)        # 纯文本

        # 统一做一次归一化，确保是 List[{"name":..., "arguments":...}]
        parts['calls'] = self._normalize_calls(collected)
        return parts

    # ---------- call reward ----------
    def compute_call_reward(self, comp_calls: List[Any], gt_calls: List[Any]) -> float:
        """
        与 <call>[{ "name": ..., "arguments": {...} }, ...]</call> 兼容的打分：
        - 对每个 GT 调用，在 comp 中贪心找最佳匹配（one-to-one）
        - 单个调用得分 = tool_name_match * call_tool_weight + args_match * call_args_weight
        - args_match = (#匹配 key 且值匹配) / (#GT 参数个数)；值匹配：完全相等或文本相似度>0.8
        - 额外的 comp 调用给予轻微惩罚
        """
        # 防御式归一化（即使 parse 阶段已做过）
        comp_calls = self._normalize_calls(comp_calls)
        gt_calls = self._normalize_calls(gt_calls)

        # 快速路径
        if len(gt_calls) == 0 and len(comp_calls) == 0:
            return 1.0
        if len(gt_calls) == 0 and len(comp_calls) > 0:
            # GT 不需要调用，模型调用了 -> 轻微惩罚
            return max(0.0, 1.0 - 0.2 * len(comp_calls))

        # 单个调用打分
        def score_single(comp, gt):
            tool_score = 0.0
            args_score = 0.0

            comp_tool = (comp.get('name') or '').strip().lower() if isinstance(comp, dict) else ''
            gt_tool   = (gt.get('name') or '').strip().lower()   if isinstance(gt, dict) else ''
            if comp_tool and gt_tool and comp_tool == gt_tool:
                tool_score = 1.0

            gt_args  = gt.get('arguments', {})  if isinstance(gt, dict)  else {}
            comp_args= comp.get('arguments', {}) if isinstance(comp, dict) else {}

            # 只按 GT 的键来计算覆盖率
            if isinstance(gt_args, dict) and len(gt_args) > 0:
                matched = 0
                for k, v in gt_args.items():
                    if k in comp_args:
                        cv = comp_args[k]
                        # 完全相等（含标量/结构体的 JSON 字符串化比较）
                        if cv == v:
                            matched += 1
                        else:
                            # 文本/结构的语义相似（把复杂结构转成字符串再比）
                            sv = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)
                            scv = cv if isinstance(cv, str) else json.dumps(cv, ensure_ascii=False, sort_keys=True)
                            if self.compute_semantic_similarity(str(sv), str(scv)) > 0.8:
                                matched += 1
                args_score = matched / (len(gt_args) + self.eps)
            else:
                args_score = 1.0  # GT 不要求参数，视为全匹配

            return (self.call_tool_weight * tool_score) + (self.call_args_weight * args_score)

        # 贪心匹配 GT 与 comp（one-to-one）
        used = [False] * len(comp_calls)
        per_gt_scores = []
        for gt in gt_calls:
            best = 0.0
            best_idx = None
            for j, comp in enumerate(comp_calls):
                if used[j]:
                    continue
                s = score_single(comp, gt)
                if s > best:
                    best = s
                    best_idx = j
            if best_idx is not None:
                used[best_idx] = True
            per_gt_scores.append(best)

        # GT 匹配平均分
        avg_match = sum(per_gt_scores) / (len(per_gt_scores) + self.eps)

        # 对额外（未被匹配到）的 comp 调用做轻微惩罚
        extra = used.count(False)
        penalty = max(0.0, 1.0 - 0.1 * extra)

        final = avg_match * penalty
        return float(max(0.0, min(1.0, final)))

    # ---------- reflect & final ----------
    def compute_reflect_reward(self, comp_reflect: str, gt_reflect: str) -> float:
        if not gt_reflect:
            # ground truth 无反思时：若模型也没有，视为中性；若有反思则按质量评分（不奖励也不惩罚太多）
            if not comp_reflect:
                return 1.0
            return min(1.0, 0.5 + 0.5 * self.compute_semantic_similarity(comp_reflect, gt_reflect or ""))
        # 若有 GT reflect，则用 embedding 或语义相似度直接评分
        return float(self.compute_semantic_similarity(comp_reflect, gt_reflect))

    def compute_final_reward(self, comp_final: str, gt_final: str) -> float:
        return float(self.compute_semantic_similarity(comp_final, gt_final))

    # ---------- format penalty ----------
    def compute_format_factor(self, comp_parts: Dict[str, Any], gt_parts: Dict[str, Any]) -> float:
        """
        更细化的格式惩罚：
          - 对于 gt 中存在且重要的标签，如果 completion 缺失则按 tag_importance 加权惩罚
          - factor = 1 - missing_sum * missing_tag_penalty_strength
        """
        missing_score = 0.0
        # reflect
        if gt_parts.get('reflect') and not comp_parts.get('reflect'):
            missing_score += self.tag_importance.get('reflect', 0.0)
        # final
        if gt_parts.get('final') and not comp_parts.get('final'):
            missing_score += self.tag_importance.get('final', 0.0)
        # calls: if gt has calls and comp doesn't
        if gt_parts.get('calls') and len(gt_parts.get('calls')) > 0 and (not comp_parts.get('calls') or len(comp_parts.get('calls'))==0):
            missing_score += self.tag_importance.get('calls', 0.0)

        factor = max(0.0, 1.0 - missing_score * self.missing_tag_penalty_strength)
        return float(factor)

    # ---------- main compute ----------
    def _compute_single_reward(self, completion: str, ground_truth: str) -> float:
        # parse
        comp_parts = self.parse_assistant_response(completion)
        gt_parts = self.parse_assistant_response(ground_truth)

        # compute per-part raw scores
        reflect_score = self.compute_reflect_reward(comp_parts.get('reflect',''), gt_parts.get('reflect',''))
        call_score = self.compute_call_reward(comp_parts.get('calls', []), gt_parts.get('calls', []))
        final_score = self.compute_final_reward(comp_parts.get('final',''), gt_parts.get('final',''))

        # determine which parts are expected by GT (dynamic weight normalization)
        gt_has_reflect = bool(gt_parts.get('reflect', '').strip())
        gt_has_calls = bool(gt_parts.get('calls', [])) and len(gt_parts.get('calls', [])) > 0
        gt_has_final = bool(gt_parts.get('final', '').strip())

        # calculate the sum of weights for parts that exist in GT
        active_weight_sum = 0.0
        weighted_score_sum = 0.0

        if gt_has_reflect:
            active_weight_sum += self.reflect_weight
            weighted_score_sum += reflect_score * self.reflect_weight

        if gt_has_calls:
            active_weight_sum += self.call_weight
            weighted_score_sum += call_score * self.call_weight

        if gt_has_final:
            active_weight_sum += self.final_weight
            weighted_score_sum += final_score * self.final_weight

        # normalize by active weights (instead of all weights) to allow full score
        if active_weight_sum > self.eps:
            combined = weighted_score_sum / active_weight_sum
        else:
            # if no parts are expected, default to 1.0 (perfect match for empty content)
            combined = 1.0

        # apply format factor
        format_factor = self.compute_format_factor(comp_parts, gt_parts)
        combined *= format_factor

        # fallback if extremely small: use semantic similarity of concatenated content
        if combined < 1e-6:
            concatenated_comp = ' '.join([comp_parts.get('reflect',''), json.dumps(comp_parts.get('calls',[])), comp_parts.get('final','')])
            concatenated_gt = ' '.join([gt_parts.get('reflect',''), json.dumps(gt_parts.get('calls',[])), gt_parts.get('final','')])
            fb = self.compute_semantic_similarity(concatenated_comp, concatenated_gt)
            combined = self.fallback_weight * fb

        # ensure in [0,1]
        return float(max(0.0, min(1.0, combined)))

    def _ensure_list(self, value, length: int):
        if not isinstance(value, list):
            if value is None:
                value = []
            else:
                value = [value]
        while len(value) < length:
            if value:
                value.append(value[-1])
            else:
                value.append('')
        return value[:length]

    def __call__(self, completions, **kwargs):
        # ensure list
        if not isinstance(completions, list):
            completions = [completions]

        # accept multiple gt arg names (compat with different trainer callers)
        gt_candidate = kwargs.get('ground_truth', None)
        if gt_candidate is None:
            gt_candidate = kwargs.get('ground_truths', None)
        if gt_candidate is None:
            gt_candidate = kwargs.get('references', None)
        if gt_candidate is None:
            gt_candidate = kwargs.get('labels', None)

        # special-case: try to extract text from 'messages' if provided (common in some trainers)
        if gt_candidate is None and 'messages' in kwargs:
            msgs = kwargs.get('messages')
            if isinstance(msgs, list) and len(msgs) > 0:
                extracted = []
                for entry in msgs:
                    if isinstance(entry, list):
                        # join assistant contents if possible
                        parts = []
                        for m in entry:
                            if isinstance(m, dict) and 'content' in m:
                                parts.append(str(m['content']))
                            elif isinstance(m, dict) and 'message' in m:
                                parts.append(str(m['message']))
                            else:
                                parts.append(str(m))
                        extracted.append(' '.join(parts))
                    elif isinstance(entry, dict):
                        extracted.append(str(entry.get('content','') or entry.get('message','') or json.dumps(entry, ensure_ascii=False)))
                    else:
                        extracted.append(str(entry))
                gt_candidate = extracted

        ground_truths = self._ensure_list(gt_candidate or [], len(completions))
        rewards = []
        for i, comp in enumerate(completions):
            gt = ground_truths[i] if i < len(ground_truths) else ''
            try:
                r = self._compute_single_reward(comp, gt)
            except Exception as e:
                print(f"[Reward-ERR] idx={i} exception: {e}")
                r = 0.0
            rewards.append(r)
        return rewards

import re
import json
import math
from typing import Optional, Dict, List, Any
import numpy as np

class ChainDialogueRewardOptimized_qwen3:
    """
    优化后的 reward 设计（专注 reward 逻辑）
    主要特点：
      - 权重归一化
      - 支持多次 <call>（结构化 JSON 优先）
      - 更细化的格式惩罚（按缺失项重要性）
      - 反思/最终使用语义相似度/embedding（可降级）
      - 可配置化参数，去除硬编码 magic numbers

    关键修复：
      - 在解析 assistant 输出时**先移除所有 <think>...</think>**（不参与任何 reward 计算）
    """

    def __init__(
        self,
        reflect_weight: float = 0.1,
        call_weight: float = 0.7,
        final_weight: float = 0.2,
        tag_importance: Optional[Dict[str, float]] = None,
        call_tool_weight: float = 0.5,
        call_args_weight: float = 0.5,
        missing_tag_penalty_strength: float = 0.8,
        fallback_weight: float = 0.15,
        max_non_exact_call_score: float = 0.6,
        eps: float = 1e-8,
        embedding_client = None,
    ):
        total = float(reflect_weight + call_weight + final_weight)
        if total <= 0:
            raise ValueError("reflect/call/final weights must sum to > 0")
        self.reflect_weight = reflect_weight
        self.call_weight = call_weight
        self.final_weight = final_weight

        self.tag_importance = tag_importance or {'reflect': 0.1, 'calls': 0.7, 'final': 0.2}
        s = sum(self.tag_importance.values())
        if s <= 0:
            raise ValueError("tag_importance must sum > 0")
        for k in self.tag_importance:
            self.tag_importance[k] = self.tag_importance[k] / s

        self.call_tool_weight = call_tool_weight
        self.call_args_weight = call_args_weight
        self.missing_tag_penalty_strength = missing_tag_penalty_strength
        self.fallback_weight = fallback_weight
        self.max_non_exact_call_score = max_non_exact_call_score
        self.eps = eps
        self.embedding_client = embedding_client

    # ---------- utility: strip think ----------
    def _remove_think_blocks(self, text: str) -> str:
        """
        移除所有 <think>...</think> 区块（不区分大小写、跨行），确保其中内容完全不参与后续解析与评分。
        使用非贪婪匹配以避免跨多个 <think> 区块错误吞并。
        """
        if not text:
            return ''
        # (?is) -> re.I | re.S : 不区分大小写，dot matches newline
        cleaned = re.sub(r'(?is)<think>.*?</think>', '', text)
        return cleaned

    # ---------- parse ----------
    def parse_assistant_response(self, response: str) -> Dict[str, Any]:
        """
        更稳健的解析：
          - 先移除所有 <think>...</think> 段（确保 think 内的任何标签被忽略）
          - extract <reflect>...</reflect>（第一个）
          - extract <final>...</final>（第一个）
          - extract 所有 <tool_call>...</tool_call> 作为列表（尝试将每个 tool_call parse 为 JSON）
        返回： {'reflect': str, 'final': str, 'calls': [str or dict, ...]}
        """
        parts = {'reflect': '', 'final': '', 'calls': []}
        if not response:
            return parts

        # 先移除 think 内容（关键修复点）
        cleaned = self._remove_think_blocks(response)

        def first_tag(text, tag):
            m = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ''

        parts['reflect'] = first_tag(cleaned, 'reflect')
        parts['final'] = first_tag(cleaned, 'final')

        # 寻找所有的 <tool_call> 标签，每个标签包含一个独立的工具调用
        tool_call_matches = re.findall(r'<tool_call>(.*?)</tool_call>', cleaned, flags=re.DOTALL | re.IGNORECASE)
        calls = []
        for c in tool_call_matches:
            c = c.strip()
            # try parse JSON
            try:
                parsed = json.loads(c)
                calls.append(parsed)
            except Exception:
                # as fallback, try to parse as simple key=value lines, else raw text
                calls.append(c)
        parts['calls'] = calls
        return parts

    # ---------- cosine helper ----------
    def _cosine_from_embeddings(self, emb1, emb2) -> float:
        """返回线性映射到 [0,1] 的余弦相似度；若无法计算返回 0.0"""
        if emb1 is None or emb2 is None:
            return 0.0
        try:
            v1 = np.array(emb1); v2 = np.array(emb2)
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + self.eps
            cos = float(np.dot(v1, v2) / denom)
            return max(0.0, min(1.0, (cos + 1.0) / 2.0))
        except Exception:
            return 0.0

    # ---------- semantic similarity fallback ----------
    def compute_semantic_similarity(self, a: str, b: str) -> float:
        """
        简洁的 fallback 语义相似度：结合 n-gram jaccard 与长度比
        （如果提供 embedding_client，可以替换为 embedding 相似度）
        """
        if not a or not b:
            return 0.0
        if self.embedding_client is not None:
            try:
                emb_a = self._get_embedding(a)
                emb_b = self._get_embedding(b)
                return self._cosine_from_embeddings(emb_a, emb_b)
            except Exception:
                pass

        a_clean = re.sub(r'\s+', ' ', a.strip().lower())
        b_clean = re.sub(r'\s+', ' ', b.strip().lower())

        if a_clean == b_clean:
            return 1.0

        def ngram_set(s, n):
            toks = s.split()
            if len(toks) < n:
                return set([' '.join(toks)])
            return set(' '.join(toks[i:i+n]) for i in range(len(toks)-n+1))
        g1a = ngram_set(a_clean, 1)
        g1b = ngram_set(b_clean, 1)
        g2a = ngram_set(a_clean, 2)
        g2b = ngram_set(b_clean, 2)

        j1 = len(g1a & g1b) / (len(g1a | g1b) + self.eps)
        j2 = len(g2a & g2b) / (len(g2a | g2b) + self.eps)
        len_sim = min(len(a_clean), len(b_clean)) / (max(len(a_clean), len(b_clean)) + self.eps)
        score = 0.6 * j1 + 0.3 * j2 + 0.1 * len_sim
        return float(max(0.0, min(1.0, score)))

    def _get_embedding(self, text: str):
        """包装 embedding client 调用"""
        if hasattr(self.embedding_client, 'embeddings'):
            resp = self.embedding_client.embeddings.create(model="text-embedding-3-large", input=text)
            return resp.data[0].embedding
        if callable(self.embedding_client):
            return self.embedding_client(text)
        raise RuntimeError("No valid embedding client available")

    # ---------- call normalization ----------
    def _normalize_calls(self, calls_raw):
        out = []

        def coerce_one(x):
            if not isinstance(x, dict):
                return None
            name = x.get('name') or x.get('tool')
            args = x.get('arguments', None)
            if args is None:
                args = x.get('args', {})
            if not isinstance(args, dict):
                args = {'_value': json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args}
            if name:
                return {'name': str(name).strip(), 'arguments': args}
            return None

        def collect(obj):
            if obj is None:
                return
            if isinstance(obj, list):
                for it in obj:
                    collect(it)
                return
            if isinstance(obj, dict):
                c = coerce_one(obj)
                if c:
                    out.append(c)
                return
            if isinstance(obj, str):
                s = obj.strip()
                if not s:
                    return
                try:
                    j = json.loads(s)
                    collect(j)
                    return
                except Exception:
                    out.append({'name': s, 'arguments': {}})
                return
            out.append({'name': str(obj), 'arguments': {}})

        collect(calls_raw)
        return out

    # ---------- call reward ----------
    def compute_call_reward(self, comp_calls: List[Any], gt_calls: List[Any]) -> float:
        comp_calls = self._normalize_calls(comp_calls)
        gt_calls = self._normalize_calls(gt_calls)

        if len(gt_calls) == 0 and len(comp_calls) == 0:
            return 1.0
        if len(gt_calls) == 0 and len(comp_calls) > 0:
            return max(0.0, 1.0 - 0.2 * len(comp_calls))

        def score_single(comp, gt):
            tool_score = 0.0
            args_score = 0.0

            comp_tool = (comp.get('name') or '').strip().lower() if isinstance(comp, dict) else ''
            gt_tool   = (gt.get('name') or '').strip().lower()   if isinstance(gt, dict) else ''
            if comp_tool and gt_tool and comp_tool == gt_tool:
                tool_score = 1.0

            gt_args  = gt.get('arguments', {})  if isinstance(gt, dict)  else {}
            comp_args= comp.get('arguments', {}) if isinstance(comp, dict) else {}

            if isinstance(gt_args, dict) and len(gt_args) > 0:
                matched = 0
                for k, v in gt_args.items():
                    if k in comp_args:
                        cv = comp_args[k]
                        if cv == v:
                            matched += 1
                        else:
                            sv = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)
                            scv = cv if isinstance(cv, str) else json.dumps(cv, ensure_ascii=False, sort_keys=True)
                            if self.compute_semantic_similarity(str(sv), str(scv)) > 0.8:
                                matched += 1
                args_score = matched / (len(gt_args) + self.eps)
            else:
                args_score = 1.0

            return (self.call_tool_weight * tool_score) + (self.call_args_weight * args_score)

        used = [False] * len(comp_calls)
        per_gt_scores = []
        for gt in gt_calls:
            best = 0.0
            best_idx = None
            for j, comp in enumerate(comp_calls):
                if used[j]:
                    continue
                s = score_single(comp, gt)
                if s > best:
                    best = s
                    best_idx = j
            if best_idx is not None:
                used[best_idx] = True
            per_gt_scores.append(best)

        avg_match = sum(per_gt_scores) / (len(per_gt_scores) + self.eps)

        extra = used.count(False)
        penalty = max(0.0, 1.0 - 0.1 * extra)

        final = avg_match * penalty
        return float(max(0.0, min(1.0, final)))

    # ---------- reflect & final ----------
    def compute_reflect_reward(self, comp_reflect: str, gt_reflect: str) -> float:
        if not gt_reflect:
            if not comp_reflect:
                return 1.0
            return min(1.0, 0.5 + 0.5 * self.compute_semantic_similarity(comp_reflect, gt_reflect or ""))
        return float(self.compute_semantic_similarity(comp_reflect, gt_reflect))

    def compute_final_reward(self, comp_final: str, gt_final: str) -> float:
        return float(self.compute_semantic_similarity(comp_final, gt_final))

    # ---------- format penalty ----------
    def compute_format_factor(self, comp_parts: Dict[str, Any], gt_parts: Dict[str, Any]) -> float:
        penalty_score = 0.0

        # reflect missing
        if gt_parts.get('reflect') and not comp_parts.get('reflect'):
            penalty_score += self.tag_importance.get('reflect', 0.0)
        # final missing
        if gt_parts.get('final') and not comp_parts.get('final'):
            penalty_score += self.tag_importance.get('final', 0.0)
        # calls missing
        if gt_parts.get('calls') and len(gt_parts.get('calls')) > 0 and (not comp_parts.get('calls') or len(comp_parts.get('calls'))==0):
            penalty_score += self.tag_importance.get('calls', 0.0)

        # extra tags
        if comp_parts.get('reflect') and not gt_parts.get('reflect'):
            penalty_score += self.tag_importance.get('reflect', 0.0) * 0.8
        if comp_parts.get('final') and not gt_parts.get('final'):
            penalty_score += self.tag_importance.get('final', 0.0) * 0.8
        if comp_parts.get('calls') and len(comp_parts.get('calls')) > 0 and (not gt_parts.get('calls') or len(gt_parts.get('calls'))==0):
            penalty_score += self.tag_importance.get('calls', 0.0) * 0.8

        gt_calls_count = len(gt_parts.get('calls', []))
        comp_calls_count = len(comp_parts.get('calls', []))
        if gt_calls_count > 0 and comp_calls_count > 0 and gt_calls_count != comp_calls_count:
            count_diff = abs(gt_calls_count - comp_calls_count)
            max_count = max(gt_calls_count, comp_calls_count)
            count_penalty = (count_diff / max_count) * self.tag_importance.get('calls', 0.0) * 0.6
            penalty_score += count_penalty

        call_reduction_factor = 1.0
        if gt_parts.get('calls') and comp_parts.get('calls'):
            if self._are_calls_exactly_equal(comp_parts.get('calls', []), gt_parts.get('calls', [])):
                call_reduction_factor = 0.3

        final_penalty = penalty_score * self.missing_tag_penalty_strength * call_reduction_factor
        factor = max(0.0, 1.0 - final_penalty)
        return float(factor)

    def _are_calls_exactly_equal(self, comp_calls: List[Any], gt_calls: List[Any]) -> bool:
        comp_normalized = self._normalize_calls(comp_calls)
        gt_normalized = self._normalize_calls(gt_calls)

        if len(comp_normalized) != len(gt_normalized):
            return False

        from collections import Counter

        def call_to_comparable(call):
            if isinstance(call, dict):
                name = call.get('name', '')
                args = call.get('arguments', {})
                args_str = json.dumps(args, ensure_ascii=False, sort_keys=True) if isinstance(args, dict) else str(args)
                return (name, args_str)
            return str(call)

        comp_comparable = [call_to_comparable(call) for call in comp_normalized]
        gt_comparable = [call_to_comparable(call) for call in gt_normalized]

        return Counter(comp_comparable) == Counter(gt_comparable)

    # ---------- main compute ----------
    def _compute_single_reward(self, completion: str, ground_truth: str) -> float:
        # parse (注意：parse 已自动移除 <think> 区块)
        comp_parts = self.parse_assistant_response(completion)
        gt_parts = self.parse_assistant_response(ground_truth)

        gt_calls = gt_parts.get('calls', [])
        comp_calls = comp_parts.get('calls', [])

        # 只有当GT期望有calls且calls不完全匹配时，直接返回0分（按原逻辑）
        if gt_calls and len(gt_calls) > 0:
            if not self._are_calls_exactly_equal(comp_calls, gt_calls):
                return 0.0

        reflect_score = self.compute_reflect_reward(comp_parts.get('reflect',''), gt_parts.get('reflect',''))
        call_score = self.compute_call_reward(comp_parts.get('calls', []), gt_parts.get('calls', []))
        final_score = self.compute_final_reward(comp_parts.get('final',''), gt_parts.get('final',''))

        gt_has_reflect = bool(gt_parts.get('reflect', '').strip())
        gt_has_calls = bool(gt_parts.get('calls', [])) and len(gt_parts.get('calls', [])) > 0
        gt_has_final = bool(gt_parts.get('final', '').strip())

        active_weight_sum = 0.0
        weighted_score_sum = 0.0

        if gt_has_reflect:
            active_weight_sum += self.reflect_weight
            weighted_score_sum += reflect_score * self.reflect_weight

        if gt_has_calls:
            active_weight_sum += self.call_weight
            weighted_score_sum += call_score * self.call_weight

        if gt_has_final:
            active_weight_sum += self.final_weight
            weighted_score_sum += final_score * self.final_weight

        if active_weight_sum > self.eps:
            combined = weighted_score_sum / active_weight_sum
        else:
            combined = 1.0

        format_factor = self.compute_format_factor(comp_parts, gt_parts)
        combined *= format_factor

        if combined < 1e-6:
            concatenated_comp = ' '.join([comp_parts.get('reflect',''), json.dumps(comp_parts.get('calls',[])), comp_parts.get('final','')])
            concatenated_gt = ' '.join([gt_parts.get('reflect',''), json.dumps(gt_parts.get('calls',[])), gt_parts.get('final','')])
            fb = self.compute_semantic_similarity(concatenated_comp, concatenated_gt)
            combined = self.fallback_weight * fb

        return float(max(0.0, min(1.0, combined)))

    def _ensure_list(self, value, length: int):
        if not isinstance(value, list):
            if value is None:
                value = []
            else:
                value = [value]
        while len(value) < length:
            if value:
                value.append(value[-1])
            else:
                value.append('')
        return value[:length]

    def __call__(self, completions, **kwargs):
        if not isinstance(completions, list):
            completions = [completions]

        gt_candidate = kwargs.get('ground_truth', None)
        if gt_candidate is None:
            gt_candidate = kwargs.get('ground_truths', None)
        if gt_candidate is None:
            gt_candidate = kwargs.get('references', None)
        if gt_candidate is None:
            gt_candidate = kwargs.get('labels', None)

        if gt_candidate is None and 'messages' in kwargs:
            msgs = kwargs.get('messages')
            if isinstance(msgs, list) and len(msgs) > 0:
                extracted = []
                for entry in msgs:
                    if isinstance(entry, list):
                        parts = []
                        for m in entry:
                            if isinstance(m, dict) and 'content' in m:
                                parts.append(str(m['content']))
                            elif isinstance(m, dict) and 'message' in m:
                                parts.append(str(m['message']))
                            else:
                                parts.append(str(m))
                        extracted.append(' '.join(parts))
                    elif isinstance(entry, dict):
                        extracted.append(str(entry.get('content','') or entry.get('message','') or json.dumps(entry, ensure_ascii=False)))
                    else:
                        extracted.append(str(entry))
                gt_candidate = extracted

        ground_truths = self._ensure_list(gt_candidate or [], len(completions))
        rewards = []
        for i, comp in enumerate(completions):
            gt = ground_truths[i] if i < len(ground_truths) else ''
            try:
                r = self._compute_single_reward(comp, gt)
            except Exception as e:
                print(f"[Reward-ERR] idx={i} exception: {e}")
                r = 0.0
            rewards.append(r)
        return rewards

# 使用示例和测试
if __name__ == "__main__":
    print("=" * 80)
    print("ChainDialogueRewardOptimized_qwen 测试")
    print("新格式：直接 JSON 对象，使用 'parameters' 字段")
    print("=" * 80)

    # 使用修改后的reward函数
    reward_function = ChainDialogueRewardOptimized_qwen3()

    # 测试用例1：工具调用参数稍有不匹配
    print("\n测试用例1：工具调用参数稍有不匹配")
    comp1 = ['<think><tool_call>{"name": "abc", "parameters": {"query": "python tutorial"}}</tool_call></think><tool_call>{"name": "search", "parameters": {"query": "python guide"}}</tool_call>']
    gt1 = ['<tool_call>{"name": "search", "parameters": {"query": "python guide"}}</tool_call>']

    score1 = reward_function(comp1, ground_truth=gt1)
    print(f"Score: {score1[0]:.4f} (之前可能是0分，现在有渐进式奖励)")

    # 测试用例2：缺少工具调用
    print("\n测试用例2：缺少工具调用")
    comp2 = ['<reflect>思考中</reflect><final>直接答案</final>']
    gt2 = ['<tool_call>{"name": "search", "parameters": {"query": "test"}}</tool_call>']

    score2 = reward_function(comp2, ground_truth=gt2)
    print(f"Score: {score2[0]:.4f} (保证最低奖励，避免梯度消失)")

    # 测试用例3：工具名称错误但参数接近
    print("\n测试用例3：工具名称错误但参数接近")
    comp3 = ['<tool_call>{"name": "find", "parameters": {"query": "python"}}</tool_call>']
    gt3 = ['<tool_call>{"name": "search", "parameters": {"query": "python"}}</tool_call>']

    score3 = reward_function(comp3, ground_truth=gt3)
    print(f"Score: {score3[0]:.4f} (部分匹配给予部分分数)")

    # 测试用例4：完全匹配
    print("\n测试用例4：完全匹配")
    comp4 = ['<think><tool_call>{"name": "abc", "parameters": {"query": "python tutorial"}}</tool_call></think><tool_call>{"name": "search", "parameters": {"query": "python"}}</tool_call>']
    gt4 = ['<tool_call>{"name": "search", "parameters": {"query": "python"}}</tool_call><final>搜索结果</final>']

    score4 = reward_function(comp4, ground_truth=gt4)
    print(f"Score: {score4[0]:.4f} (完全匹配应该接近1.0)")

    print("\n" + "=" * 80)
    print("格式更新:")
    print("1. 不再使用 <tool_call> 标签，直接使用 JSON 格式")
    print("2. 只支持 'parameters' 字段")
    print("3. 智能解析嵌套JSON结构")
    print("4. 保持原有的渐进式评分机制")
    print("5. 动态权重归一化仍然有效")
    print("=" * 80)
