import os
import re
from typing import TYPE_CHECKING, Dict, List, Any, Union, Optional
from collections import Counter
import json
import numpy as np

if TYPE_CHECKING:
    from swift.llm import InferRequest
from openai import OpenAI

import math

# Import ToolRL reward functions
from swift.plugin.toolrl_reward import ToolRLReward, ToolRLRewardLlama, ToolRLRewardQwen

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

class ChainDialogueRewardOptimized_qwen:
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
        reflect_weight: float = 0.1,
        call_weight: float = 0.7,
        final_weight: float = 0.2,
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
        # 保存原始权重，不提前归一化，让动态权重归一化真正发挥作用
        total = float(reflect_weight + call_weight + final_weight)
        if total <= 0:
            raise ValueError("reflect/call/final weights must sum to > 0")
        self.reflect_weight = reflect_weight
        self.call_weight = call_weight
        self.final_weight = final_weight

        self.tag_importance = tag_importance or {'reflect': 0.1, 'calls': 0.7, 'final': 0.2}
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
          - extract 所有 <tool_call>...</tool_call> 作为列表（尝试将每个 tool_call parse 为 JSON）
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

        # 修改：寻找所有的 <tool_call> 标签，每个标签包含一个独立的工具调用
        tool_call_matches = re.findall(r'<tool_call>(.*?)</tool_call>', response, flags=re.DOTALL | re.IGNORECASE)
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

    def _remove_think_blocks(self, text: str) -> str:
        if not text:
            return ''
        return re.sub(r'(?is)<think>.*?</think>', '', text)

    def _first_balanced_json_obj(self, s: str):
        """从文本中提取首个平衡的 { ... } 并 json.loads；失败返回 None"""
        if not s:
            return None
        s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        n = len(s)
        i = 0
        while i < n:
            if s[i] == '{':
                stack = 1
                in_str = False
                esc = False
                j = i + 1
                while j < n:
                    c = s[j]
                    if in_str:
                        if esc:
                            esc = False
                        elif c == '\\':
                            esc = True
                        elif c == '"':
                            in_str = False
                    else:
                        if c == '"':
                            in_str = True
                        elif c == '{':
                            stack += 1
                        elif c == '}':
                            stack -= 1
                            if stack == 0:
                                seg = s[i:j+1]
                                try:
                                    return json.loads(seg)
                                except Exception:
                                    # 宽松一步：去尾逗号再试
                                    seg2 = re.sub(r',\s*([}\]])', r'\1', seg)
                                    try:
                                        return json.loads(seg2)
                                    except Exception:
                                        return None
                    j += 1
            i += 1
        return None

    # 只接受严格 {"name": str, "arguments": dict}
    def _normalize_calls(self, calls_raw):
        out = []
        def coerce_one(x):
            if not isinstance(x, dict):
                return None  # 不再接受字符串/别名
            name = x.get('name', None)
            args = x.get('arguments', None)
            if isinstance(name, str) and isinstance(args, dict):
                return {'name': name.strip(), 'arguments': args}
            return None
        if isinstance(calls_raw, list):
            for it in calls_raw:
                c = coerce_one(it)
                if c:
                    out.append(c)
        elif isinstance(calls_raw, dict):
            c = coerce_one(calls_raw)
            if c:
                out.append(c)
        # 其它一律忽略
        return out

    def parse_assistant_response(self, response: str):
        """严格 Qwen 解析：剔除<think>，仅提取 <tool_call> 中的严格 JSON 对象"""
        parts = {'reflect': '', 'final': '', 'calls': []}
        if not response:
            return parts
        cleaned = self._remove_think_blocks(response)

        def first_tag(text, tag):
            m = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ''

        parts['reflect'] = first_tag(cleaned, 'reflect')
        parts['final']   = first_tag(cleaned, 'final')

        tool_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', cleaned, flags=re.DOTALL | re.IGNORECASE)
        raw_calls = []
        for blk in tool_blocks:
            obj = self._first_balanced_json_obj(blk)
            # 只接受严格对象；否则忽略（视为无效调用）
            if isinstance(obj, dict) and 'name' in obj and 'arguments' in obj and isinstance(obj['arguments'], dict):
                raw_calls.append(obj)
        parts['calls'] = self._normalize_calls(raw_calls)
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
          - 对于 gt 中存在的标签，如果 completion 缺失则按 tag_importance 加权惩罚
          - 对于 completion 中多余的标签，也施以惩罚
          - 对于 calls 数量不一致的情况施以惩罚
          - 如果 call 部分完全一致，则降低整体惩罚力度
          - factor = 1 - penalty_score * missing_tag_penalty_strength * call_reduction_factor
        """
        penalty_score = 0.0

        # 检查缺失的标签
        # reflect
        if gt_parts.get('reflect') and not comp_parts.get('reflect'):
            penalty_score += self.tag_importance.get('reflect', 0.0)
        # final
        if gt_parts.get('final') and not comp_parts.get('final'):
            penalty_score += self.tag_importance.get('final', 0.0)
        # calls: if gt has calls and comp doesn't
        if gt_parts.get('calls') and len(gt_parts.get('calls')) > 0 and (not comp_parts.get('calls') or len(comp_parts.get('calls'))==0):
            penalty_score += self.tag_importance.get('calls', 0.0)

        # 检查多余的标签（completion有但gt没有的）
        # reflect多余
        if comp_parts.get('reflect') and not gt_parts.get('reflect'):
            penalty_score += self.tag_importance.get('reflect', 0.0) * 0.8  # 多余标签惩罚稍轻
        # final多余
        if comp_parts.get('final') and not gt_parts.get('final'):
            penalty_score += self.tag_importance.get('final', 0.0) * 0.8
        # calls多余
        if comp_parts.get('calls') and len(comp_parts.get('calls')) > 0 and (not gt_parts.get('calls') or len(gt_parts.get('calls'))==0):
            penalty_score += self.tag_importance.get('calls', 0.0) * 0.8

        # 检查calls数量不一致（都有calls但数量不同）
        gt_calls_count = len(gt_parts.get('calls', []))
        comp_calls_count = len(comp_parts.get('calls', []))
        if gt_calls_count > 0 and comp_calls_count > 0 and gt_calls_count != comp_calls_count:
            # 按数量差异程度给予惩罚
            count_diff = abs(gt_calls_count - comp_calls_count)
            max_count = max(gt_calls_count, comp_calls_count)
            count_penalty = (count_diff / max_count) * self.tag_importance.get('calls', 0.0) * 0.6
            penalty_score += count_penalty

        # 检查call部分是否完全一致，如果一致则降低惩罚力度
        call_reduction_factor = 1.0
        if gt_parts.get('calls') and comp_parts.get('calls'):
            # 使用现有的call比较逻辑检查是否完全匹配
            if self._are_calls_exactly_equal(comp_parts.get('calls', []), gt_parts.get('calls', [])):
                call_reduction_factor = 0.3  # 如果call完全一致，将惩罚力度降低到30%

        final_penalty = penalty_score * self.missing_tag_penalty_strength * call_reduction_factor
        factor = max(0.0, 1.0 - final_penalty)
        return float(factor)

    def _are_calls_exactly_equal(self, comp_calls, gt_calls) -> bool:
        comp_normalized = self._normalize_calls(comp_calls)
        gt_normalized   = self._normalize_calls(gt_calls)
        if len(comp_normalized) != len(gt_normalized):
            return False
        def canon(x):
            return json.dumps({'name': x['name'], 'arguments': x['arguments']},
                            ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        return Counter(map(canon, comp_normalized)) == Counter(map(canon, gt_normalized))

    # ---------- main compute ----------
    def _compute_single_reward(self, completion: str, ground_truth: str) -> float:
        # parse
        comp_parts = self.parse_assistant_response(completion)
        gt_parts = self.parse_assistant_response(ground_truth)

        # --- 严格的 call 匹配检查：只有call不匹配时直接返回0 ---
        gt_calls = gt_parts.get('calls', [])
        comp_calls = comp_parts.get('calls', [])

        # 只有当GT期望有calls，但completion的calls与GT不匹配时，才直接返回0分
        # 其他情况（多余calls、缺少calls等）交给格式惩罚机制处理
        if gt_calls and len(gt_calls) > 0:
            # 检查completion是否有精确匹配的calls
            if not self._are_calls_exactly_equal(comp_calls, gt_calls):
                return 0.0  # call不匹配，直接返回0分
        # -------------------------------------------------------------------------

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


class ChainDialogueRewardOptimized_llama:
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
        reflect_weight: float = 0.1,
        call_weight: float = 0.7,
        final_weight: float = 0.2,
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
        # 保存原始权重，不提前归一化，让动态权重归一化真正发挥作用
        total = float(reflect_weight + call_weight + final_weight)
        if total <= 0:
            raise ValueError("reflect/call/final weights must sum to > 0")
        self.reflect_weight = reflect_weight
        self.call_weight = call_weight
        self.final_weight = final_weight

        self.tag_importance = tag_importance or {'reflect': 0.1, 'calls': 0.7, 'final': 0.2}
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
        解析助手回复（新格式）：
          - extract <reflect>...</reflect>（第一个）
          - extract <final>...</final>（第一个）
          - extract 所有 JSON 对象作为工具调用
        返回： {'reflect': str, 'final': str, 'calls': [dict, ...]}
        """
        parts = {'reflect': '', 'final': '', 'calls': []}
        if not response:
            return parts

        def first_tag(text, tag):
            m = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ''

        parts['reflect'] = first_tag(response, 'reflect')
        parts['final'] = first_tag(response, 'final')

        # 新格式：直接提取JSON对象
        calls = self._extract_json_calls(response)
        parts['calls'] = self._normalize_calls(calls)
        return parts

    def _extract_json_calls(self, text: str) -> List[Any]:
        """从文本中提取所有可能的JSON对象，识别工具调用"""
        calls = []

        # 使用栈来匹配括号，寻找完整的JSON块
        i = 0
        while i < len(text):
            if text[i] == '{':
                # 找到JSON开始，现在找到匹配的结束
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

                if stack == 0:  # 找到完整的JSON
                    json_str = text[start:i]
                    try:
                        parsed = json.loads(json_str)
                        # 检查是否是工具调用格式（有name字段）
                        if isinstance(parsed, dict) and 'name' in parsed:
                            calls.append(parsed)
                    except json.JSONDecodeError:
                        pass
            else:
                i += 1

        return calls

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
            List[{"name": <str>, "parameters": <dict>}]
        支持输入为：None / str(json) / dict / list / 嵌套 list
        支持的键名：
        - name/tool (工具名)
        - parameters (参数)
        """
        out = []

        def coerce_one(x):
            if not isinstance(x, dict):
                return None
            # 只支持 name/tool 作为工具名，parameters 作为参数
            name = x.get('name') or x.get('tool')
            args = x.get('parameters', {})  # 只支持 parameters 字段
            # 基本清洗
            if not isinstance(args, dict):
                # 若是标量或列表，转成字符串再放入一个占位 key
                args = {'_value': json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args}
            if name:
                return {'name': str(name).strip(), 'parameters': args}
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
                    out.append({'name': s, 'parameters': {}})
                return
            # 其他类型一律字符串化为“工具名”
            out.append({'name': str(obj), 'parameters': {}})

        collect(calls_raw)
        return out

    # ---------- call reward ----------
    def compute_call_reward(self, comp_calls: List[Any], gt_calls: List[Any]) -> float:
        """
        与 <call>[{ "name": ..., "parameters": {...} }, ...]</call> 兼容的打分：
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

            gt_args  = gt.get('parameters', {})  if isinstance(gt, dict)  else {}
            comp_args= comp.get('parameters', {}) if isinstance(comp, dict) else {}

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
          - 对于 gt 中存在的标签，如果 completion 缺失则按 tag_importance 加权惩罚
          - 对于 completion 中多余的标签，也施以惩罚
          - 对于 calls 数量不一致的情况施以惩罚
          - 如果 call 部分完全一致，则降低整体惩罚力度
          - factor = 1 - penalty_score * missing_tag_penalty_strength * call_reduction_factor
        """
        penalty_score = 0.0

        # 检查缺失的标签
        # reflect
        if gt_parts.get('reflect') and not comp_parts.get('reflect'):
            penalty_score += self.tag_importance.get('reflect', 0.0)
        # final
        if gt_parts.get('final') and not comp_parts.get('final'):
            penalty_score += self.tag_importance.get('final', 0.0)
        # calls: if gt has calls and comp doesn't
        if gt_parts.get('calls') and len(gt_parts.get('calls')) > 0 and (not comp_parts.get('calls') or len(comp_parts.get('calls'))==0):
            penalty_score += self.tag_importance.get('calls', 0.0)

        # 检查多余的标签（completion有但gt没有的）
        # reflect多余
        if comp_parts.get('reflect') and not gt_parts.get('reflect'):
            penalty_score += self.tag_importance.get('reflect', 0.0) * 0.8  # 多余标签惩罚稍轻
        # final多余
        if comp_parts.get('final') and not gt_parts.get('final'):
            penalty_score += self.tag_importance.get('final', 0.0) * 0.8
        # calls多余
        if comp_parts.get('calls') and len(comp_parts.get('calls')) > 0 and (not gt_parts.get('calls') or len(gt_parts.get('calls'))==0):
            penalty_score += self.tag_importance.get('calls', 0.0) * 0.8

        # 检查calls数量不一致（都有calls但数量不同）
        gt_calls_count = len(gt_parts.get('calls', []))
        comp_calls_count = len(comp_parts.get('calls', []))
        if gt_calls_count > 0 and comp_calls_count > 0 and gt_calls_count != comp_calls_count:
            # 按数量差异程度给予惩罚
            count_diff = abs(gt_calls_count - comp_calls_count)
            max_count = max(gt_calls_count, comp_calls_count)
            count_penalty = (count_diff / max_count) * self.tag_importance.get('calls', 0.0) * 0.6
            penalty_score += count_penalty

        # 检查call部分是否完全一致，如果一致则降低惩罚力度
        call_reduction_factor = 1.0
        if gt_parts.get('calls') and comp_parts.get('calls'):
            # 使用现有的call比较逻辑检查是否完全匹配
            if self._are_calls_exactly_equal(comp_parts.get('calls', []), gt_parts.get('calls', [])):
                call_reduction_factor = 0.3  # 如果call完全一致，将惩罚力度降低到30%

        final_penalty = penalty_score * self.missing_tag_penalty_strength * call_reduction_factor
        factor = max(0.0, 1.0 - final_penalty)
        return float(factor)

    def _are_calls_exactly_equal(self, comp_calls: List[Any], gt_calls: List[Any]) -> bool:
        """检查两个call列表是否完全相等（顺序可能不同）"""
        comp_normalized = self._normalize_calls(comp_calls)
        gt_normalized = self._normalize_calls(gt_calls)

        if len(comp_normalized) != len(gt_normalized):
            return False

        # 对每个call进行精确匹配（支持顺序不同）
        

        def call_to_comparable(call):
            if isinstance(call, dict):
                name = call.get('name', '')
                args = call.get('parameters', {})
                # 将parameters转为可比较的字符串（sorted keys确保一致性）
                args_str = json.dumps(args, ensure_ascii=False, sort_keys=True) if isinstance(args, dict) else str(args)
                return (name, args_str)
            return str(call)

        comp_comparable = [call_to_comparable(call) for call in comp_normalized]
        gt_comparable = [call_to_comparable(call) for call in gt_normalized]

        # 使用Counter进行多重集合比较
        return Counter(comp_comparable) == Counter(gt_comparable)

    # ---------- main compute ----------
    def _compute_single_reward(self, completion: str, ground_truth: str) -> float:
        # parse
        comp_parts = self.parse_assistant_response(completion)
        gt_parts = self.parse_assistant_response(ground_truth)

        # --- 严格的 call 匹配检查：只有call不匹配时直接返回0 ---
        gt_calls = gt_parts.get('calls', [])
        comp_calls = comp_parts.get('calls', [])

        # 只有当GT期望有calls，但completion的calls与GT不匹配时，才直接返回0分
        # 其他情况（多余calls、缺少calls等）交给格式惩罚机制处理
        if gt_calls and len(gt_calls) > 0:
            # 检查completion是否有精确匹配的calls
            if not self._are_calls_exactly_equal(comp_calls, gt_calls):
                return 0.0  # call不匹配，直接返回0分
        # -------------------------------------------------------------------------

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

class ChainDialogueRewardOptimized_qwen_v2:
    def __init__(
        self,
        reflect_weight: float = 0.1,
        call_weight: float = 0.7,
        final_weight: float = 0.2,
        # 各标签重要性（用于缺失惩罚）
        tag_importance: Optional[Dict[str, float]] = None,
        # call 比较内部权重 —— 按你的思路：name 0.6 / params 0.4
        call_tool_weight: float = 0.6,
        call_args_weight: float = 0.4,
        # 当 ground truth 有 call 而 completion 缺失时的惩罚比例（0..1）
        missing_tag_penalty_strength: float = 0.8,
        # fallback 权重（当所有项都几乎为0时使用）
        fallback_weight: float = 0.15,
        # 非精确匹配的最大分数阈值（保留参数用于兼容，但不再强制截断）
        max_non_exact_call_score: float = 0.6,
        # tiny smoothing for numeric stability
        eps: float = 1e-8,
        # 外部 embedding client（可注入），如果为空则使用文本相似度 fallback
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
        self.eps = eps
        self.embedding_client = embedding_client
        # 兼容旧参
        self.max_non_exact_call_score = max_non_exact_call_score

    # ---------- parse ----------
    def parse_assistant_response(self, response: str) -> Dict[str, Any]:
        """
        更稳健的解析：
          - extract <reflect>...</reflect>（第一个）
          - extract <final>...</final>（第一个）
          - extract 所有 <tool_call>...</tool_call> 作为列表（尝试将每个 tool_call parse 为 JSON）
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

        # 修改：寻找所有的 <tool_call> 标签，每个标签包含一个独立的工具调用
        tool_call_matches = re.findall(r'<tool_call>(.*?)</tool_call>', response, flags=re.DOTALL | re.IGNORECASE)
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

    def _remove_think_blocks(self, text: str) -> str:
        if not text:
            return ''
        return re.sub(r'(?is)<think>.*?</think>', '', text)

    def _first_balanced_json_obj(self, s: str):
        """从文本中提取首个平衡的 { ... } 并 json.loads；失败返回 None"""
        if not s:
            return None
        s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        n = len(s)
        i = 0
        while i < n:
            if s[i] == '{':
                stack = 1
                in_str = False
                esc = False
                j = i + 1
                while j < n:
                    c = s[j]
                    if in_str:
                        if esc:
                            esc = False
                        elif c == '\\':
                            esc = True
                        elif c == '"':
                            in_str = False
                    else:
                        if c == '"':
                            in_str = True
                        elif c == '{':
                            stack += 1
                        elif c == '}':
                            stack -= 1
                            if stack == 0:
                                seg = s[i:j+1]
                                try:
                                    return json.loads(seg)
                                except Exception:
                                    # 宽松一步：去尾逗号再试
                                    seg2 = re.sub(r',\s*([}\]])', r'\1', seg)
                                    try:
                                        return json.loads(seg2)
                                    except Exception:
                                        return None
                    j += 1
            i += 1
        return None

    # 只接受严格 {"name": str, "arguments": dict}
    def _normalize_calls(self, calls_raw):
        out = []
        def coerce_one(x):
            if not isinstance(x, dict):
                return None  # 不再接受字符串/别名
            name = x.get('name', None)
            args = x.get('arguments', None)
            if isinstance(name, str) and isinstance(args, dict):
                return {'name': name.strip(), 'arguments': args}
            return None
        if isinstance(calls_raw, list):
            for it in calls_raw:
                c = coerce_one(it)
                if c:
                    out.append(c)
        elif isinstance(calls_raw, dict):
            c = coerce_one(calls_raw)
            if c:
                out.append(c)
        # 其它一律忽略
        return out

    def parse_assistant_response(self, response: str):
        """严格 Qwen 解析：剔除<think>，仅提取 <tool_call> 中的严格 JSON 对象"""
        parts = {'reflect': '', 'final': '', 'calls': []}
        if not response:
            return parts
        cleaned = self._remove_think_blocks(response)

        def first_tag(text, tag):
            m = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ''

        parts['reflect'] = first_tag(cleaned, 'reflect')
        parts['final']   = first_tag(cleaned, 'final')

        tool_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', cleaned, flags=re.DOTALL | re.IGNORECASE)
        raw_calls = []
        for blk in tool_blocks:
            obj = self._first_balanced_json_obj(blk)
            # 只接受严格对象；否则忽略（视为无效调用）
            if isinstance(obj, dict) and 'name' in obj and 'arguments' in obj and isinstance(obj['arguments'], dict):
                raw_calls.append(obj)
        parts['calls'] = self._normalize_calls(raw_calls)
        return parts

    # ---------- call reward ----------
    def compute_call_reward(self, comp_calls: List[Any], gt_calls: List[Any]) -> float:
        """
        按 GT 的每个调用在 comp 中做贪心一一匹配：
          单个调用得分 = 0 或 1 的 tool_name_match * 0.6 + 参数匹配率 * 0.4
          参数匹配率 = (#键存在且值匹配或高相似) / (#GT 参数键)
        额外的 completion 调用有轻度惩罚；空空对齐为 1.0。
        """
        comp_calls = self._normalize_calls(comp_calls)
        gt_calls = self._normalize_calls(gt_calls)

        if len(gt_calls) == 0 and len(comp_calls) == 0:
            return 1.0
        if len(gt_calls) == 0 and len(comp_calls) > 0:
            return max(0.0, 1.0 - 0.2 * len(comp_calls))

        def score_single(comp, gt):
            # tool name：完全一致计 1，否则 0
            comp_tool = (comp.get('name') or '').strip().lower() if isinstance(comp, dict) else ''
            gt_tool   = (gt.get('name') or '').strip().lower()   if isinstance(gt, dict) else ''
            tool_score = 1.0 if comp_tool and gt_tool and comp_tool == gt_tool else 0.0

            # params：只按 GT 键覆盖率计分；值相等或相似度>0.8 视为匹配
            gt_args   = gt.get('arguments', {})   if isinstance(gt, dict) else {}
            comp_args = comp.get('arguments', {}) if isinstance(comp, dict) else {}
            if isinstance(gt_args, dict) and len(gt_args) > 0:
                matched = 0
                for k, v in gt_args.items():
                    if k in comp_args:
                        cv = comp_args[k]
                        if cv == v:
                            matched += 1
                        else:
                            sv  = v  if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)
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

        # 未匹配到的多余 completion 调用做轻惩罚
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
        更细化的格式惩罚（支持部分命中时减弱惩罚）：
          - 缺失/多余标签与数量差异
          - 若 call 完全一致：惩罚*0.3
          - 若 call 部分命中：按 call_score 线性降低惩罚（惩罚 * (1 - 0.5*call_score)）
        """
        penalty_score = 0.0

        if gt_parts.get('reflect') and not comp_parts.get('reflect'):
            penalty_score += self.tag_importance.get('reflect', 0.0)
        if gt_parts.get('final') and not comp_parts.get('final'):
            penalty_score += self.tag_importance.get('final', 0.0)
        if gt_parts.get('calls') and len(gt_parts.get('calls')) > 0 and (not comp_parts.get('calls') or len(comp_parts.get('calls'))==0):
            penalty_score += self.tag_importance.get('calls', 0.0)

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

        # 按 call 一致度动态折减惩罚
        call_reduction_factor = 1.0
        if gt_parts.get('calls') and comp_parts.get('calls'):
            if self._are_calls_exactly_equal(comp_parts.get('calls', []), gt_parts.get('calls', [])):
                call_reduction_factor = 0.3
            else:
                # 部分命中：惩罚 * (1 - 0.5 * call_score)
                try:
                    partial_call_score = self.compute_call_reward(comp_parts.get('calls', []), gt_parts.get('calls', []))
                except Exception:
                    partial_call_score = 0.0
                call_reduction_factor = max(0.0, 1.0 - 0.5 * partial_call_score)

        final_penalty = penalty_score * self.missing_tag_penalty_strength * call_reduction_factor
        factor = max(0.0, 1.0 - final_penalty)
        return float(factor)

    def _are_calls_exactly_equal(self, comp_calls, gt_calls) -> bool:
        comp_normalized = self._normalize_calls(comp_calls)
        gt_normalized   = self._normalize_calls(gt_calls)
        if len(comp_normalized) != len(gt_normalized):
            return False
        def canon(x):
            return json.dumps({'name': x['name'], 'arguments': x['arguments']},
                            ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        return Counter(map(canon, comp_normalized)) == Counter(map(canon, gt_normalized))

    # ---------- main compute ----------
    def _compute_single_reward(self, completion: str, ground_truth: str) -> float:
        comp_parts = self.parse_assistant_response(completion)
        gt_parts   = self.parse_assistant_response(ground_truth)

        # ✅ 移除“call 不精确就直接 0 分”的硬拦截，改为依赖 compute_call_reward 的部分得分
        # （原逻辑在此处直接 return 0.0，现已删除）

        reflect_score = self.compute_reflect_reward(comp_parts.get('reflect',''), gt_parts.get('reflect',''))
        call_score    = self.compute_call_reward(comp_parts.get('calls', []), gt_parts.get('calls', []))
        final_score   = self.compute_final_reward(comp_parts.get('final',''), gt_parts.get('final',''))

        gt_has_reflect = bool(gt_parts.get('reflect', '').strip())
        gt_has_calls   = bool(gt_parts.get('calls', [])) and len(gt_parts.get('calls', [])) > 0
        gt_has_final   = bool(gt_parts.get('final', '').strip())

        active_weight_sum   = 0.0
        weighted_score_sum  = 0.0
        if gt_has_reflect:
            active_weight_sum  += self.reflect_weight
            weighted_score_sum += reflect_score * self.reflect_weight
        if gt_has_calls:
            active_weight_sum  += self.call_weight
            weighted_score_sum += call_score * self.call_weight
        if gt_has_final:
            active_weight_sum  += self.final_weight
            weighted_score_sum += final_score * self.final_weight

        combined = (weighted_score_sum / active_weight_sum) if active_weight_sum > self.eps else 1.0

        format_factor = self.compute_format_factor(comp_parts, gt_parts)
        combined *= format_factor

        if combined < 1e-6:
            concatenated_comp = ' '.join([comp_parts.get('reflect',''), json.dumps(comp_parts.get('calls',[])), comp_parts.get('final','')])
            concatenated_gt   = ' '.join([gt_parts.get('reflect',''), json.dumps(gt_parts.get('calls',[])), gt_parts.get('final','')])
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


class simple_reward_qwen:
    """
    简化的 Qwen reward：
    - <tool_call> 内部格式严格为 {"name": str, "arguments": dict}
    - 工具调用用多重集合进行严格比较（完全一致=1，否则=0）
    - <reflect>/<final> 用 n-gram Jaccard 相似度
    - 仅对 GT 中出现的标签进行动态权重归一化
    """

    def __init__(
        self,
        reflect_weight: float = 0.1,
        call_weight: float = 0.7,
        final_weight: float = 0.2,
        eps: float = 1e-8,
        embedding_client=None,
    ):
        self.reflect_weight = reflect_weight
        self.call_weight = call_weight
        self.final_weight = final_weight
        self.eps = eps
        self.embedding_client = embedding_client

    # ---------- 解析 ----------
    def _remove_think_blocks(self, text: str) -> str:
        if not text:
            return ''
        return re.sub(r'(?is)<think>.*?</think>', '', text)

    def _first_tag_content(self, text: str, tag: str) -> str:
        m = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ''

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """严格格式：每个 <tool_call> 块内必须是合法 JSON，对象含 name+arguments(dict)"""
        calls = []
        blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, flags=re.DOTALL | re.IGNORECASE)
        for blk in blocks:
            blk = blk.strip()
            obj = json.loads(blk)
            if not isinstance(obj, dict):
                continue
            name = obj.get('name', None)
            args = obj.get('arguments', None)
            if isinstance(name, str) and isinstance(args, dict):
                calls.append({'name': name.strip(), 'arguments': args})
        return calls

    def parse_assistant_response(self, response: str) -> Dict[str, Any]:
        parts = {'reflect': '', 'final': '', 'calls': []}
        if not response:
            return parts
        cleaned = self._remove_think_blocks(response)
        parts['reflect'] = self._first_tag_content(cleaned, 'reflect')
        parts['final']   = self._first_tag_content(cleaned, 'final')
        parts['calls']   = self._parse_tool_calls(cleaned)
        return parts

    # ---------- 工具调用多重集合严格比较 ----------
    def _are_calls_exactly_equal(self, comp_calls: List[Dict[str, Any]], gt_calls: List[Dict[str, Any]]) -> bool:
        def canon(call: Dict[str, Any]) -> str:
            return json.dumps(
                {'name': call.get('name', ''), 'arguments': call.get('arguments', {})},
                ensure_ascii=False, sort_keys=True
            )
        comp_multiset = Counter([canon(c) for c in comp_calls])
        gt_multiset   = Counter([canon(c) for c in gt_calls])
        return comp_multiset == gt_multiset

    def compute_semantic_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a_clean = re.sub(r'\s+', ' ', a.strip().lower())
        b_clean = re.sub(r'\s+', ' ', b.strip().lower())
        if a_clean == b_clean:
            return 1.0

        def ngram_set(s, n):
            toks = s.split()
            if len(toks) < n:
                return {' '.join(toks)} if toks else set()
            return {' '.join(toks[i:i+n]) for i in range(len(toks)-n+1)}

        # n-gram jaccard相似度计算
        g1a, g1b = ngram_set(a_clean, 1), ngram_set(b_clean, 1)
        g2a, g2b = ngram_set(a_clean, 2), ngram_set(b_clean, 2)
        j1 = len(g1a & g1b) / (len(g1a | g1b) + self.eps)
        j2 = len(g2a & g2b) / (len(g2a | g2b) + self.eps)
        len_sim = min(len(a_clean), len(b_clean)) / (max(len(a_clean), len(b_clean)) + self.eps)
        return float(max(0.0, min(1.0, 0.6*j1 + 0.3*j2 + 0.1*len_sim)))

    def _compute_single_reward(self, completion: str, ground_truth: str) -> float:
        comp = self.parse_assistant_response(completion)
        gt   = self.parse_assistant_response(ground_truth)

        reflect_score = 0.0
        call_score    = 0.0
        final_score   = 0.0

        if gt['reflect']:
            reflect_score = self.compute_semantic_similarity(comp['reflect'], gt['reflect'])

        if gt['calls']:
            call_score = 1.0 if self._are_calls_exactly_equal(comp['calls'], gt['calls']) else 0.0

        if gt['final']:
            final_score = self.compute_semantic_similarity(comp['final'], gt['final'])

        # 仅启用 GT 中出现的权重
        active_w = 0.0
        total = 0.0
        if gt['reflect']:
            active_w += self.reflect_weight
            total += reflect_score * self.reflect_weight
        if gt['calls']:
            active_w += self.call_weight
            total += call_score * self.call_weight
        if gt['final']:
            active_w += self.final_weight
            total += final_score * self.final_weight

        if active_w > self.eps:
            return float(max(0.0, min(1.0, total / active_w)))
        return 1.0  # GT 完全空的情况

    def _ensure_list(self, v, n):
        if not isinstance(v, list):
            v = [v] if v is not None else []
        while len(v) < n:
            v.append(v[-1] if v else '')
        return v[:n]

    def __call__(self, completions, **kwargs):
        if not isinstance(completions, list):
            completions = [completions]

        gt_candidate = None
        for key in ['ground_truths', 'ground_truth', 'gt', 'references', 'targets']:
            if key in kwargs:
                gt_candidate = kwargs[key]
                break

        if isinstance(gt_candidate, str):
            gts = [gt_candidate]
        elif isinstance(gt_candidate, list):
            gts = [str(x) for x in gt_candidate]
        else:
            gts = []

        gts = self._ensure_list(gts, len(completions))

        rewards = []
        for i, comp in enumerate(completions):
            try:
                rewards.append(self._compute_single_reward(comp, gts[i]))
            except Exception as e:
                print(f"[Reward-ERR] idx={i} exception: {e}")
                rewards.append(0.0)
        return rewards

class simple_reward_llama:
    """
    简化版llama reward计算，计算方案同 qwen：
    - <tool_call>部分：llama模型没有<tool_call>标签，而是直接用 list 进行工具调用
    - <reflect>和<final>部分：与 simple_reward_qwen 对齐：去除<think>块 + 语义相似度（n-gram Jaccard + 长度）
    - 最终结果用激活标签权重归一化（保持原始权重配置）
    """

    def __init__(
        self,
        reflect_weight: float = 0.1,
        call_weight: float = 0.7,
        final_weight: float = 0.2,
        eps: float = 1e-8,
        embedding_client=None,
    ):
        self.reflect_weight = reflect_weight
        self.call_weight = call_weight
        self.final_weight = final_weight
        self.eps = eps
        self.embedding_client = embedding_client

    def _normalize_calls(self, calls_raw):
        out = []
        def coerce_one(x):
            if isinstance(x, dict):
                name = x.get('name')
                args = x.get('parameters', {})
                if not isinstance(args, dict):
                    args = {'_value': json.dumps(args, ensure_ascii=False)}
                if name:
                    return {'name': str(name).strip(), 'parameters': args}
            elif isinstance(x, str):
                return {'name': x.strip(), 'parameters': {}}
            return None

        for call in calls_raw or []:
            processed = coerce_one(call)
            if processed:
                out.append(processed)
        return out

    def _are_calls_exactly_equal(self, comp_calls, gt_calls):
        comp_normalized = self._normalize_calls(comp_calls)
        gt_normalized = self._normalize_calls(gt_calls)
        if len(comp_normalized) != len(gt_normalized):
            return False

        def call_to_comparable(call):
            if isinstance(call, dict):
                name = call.get('name', '')
                args = call.get('parameters', {})
                args_str = json.dumps(args, ensure_ascii=False, sort_keys=True) if isinstance(args, dict) else str(args)
                return (name, args_str)
            return str(call)

        return Counter([call_to_comparable(c) for c in comp_normalized]) == \
               Counter([call_to_comparable(c) for c in gt_normalized])

    def _remove_think_blocks(self, text: str) -> str:
        if not text:
            return ''
        return re.sub(r'(?is)<think>.*?</think>', '', text)

    def _extract_first_json_segment(self, text: str):
        if not text:
            return None
        trimmed = text.strip()
        try:
            return json.loads(trimmed)
        except Exception:
            pass

        opens = {'[': ']', '{': '}'}
        n = len(text)
        i = 0
        while i < n:
            ch = text[i]
            if ch in opens:
                stack = [opens[ch]]
                in_string = False
                escape = False
                j = i + 1
                while j < n:
                    c = text[j]
                    if in_string:
                        if escape:
                            escape = False
                        elif c == '\\':
                            escape = True
                        elif c == '"':
                            in_string = False
                    else:
                        if c == '"':
                            in_string = True
                        elif c in opens:
                            stack.append(opens[c])
                        elif c in (']', '}'):
                            if not stack or c != stack[-1]:
                                break
                            stack.pop()
                            if not stack:
                                segment = text[i:j+1]
                                try:
                                    return json.loads(segment)
                                except Exception:
                                    break
                    j += 1
            i += 1
        return None

    def _extract_calls_from_text(self, text: str):
        obj = self._extract_first_json_segment(text)
        if obj is None:
            return []
        if isinstance(obj, list):
            return self._normalize_calls(obj)
        if isinstance(obj, dict):
            for key in ('tool_calls', 'calls', 'tools', 'arguments', 'toolCalls'):
                v = obj.get(key)
                if isinstance(v, list):
                    return self._normalize_calls(v)
        # 再兜底尝试一次
        s = json.dumps(obj, ensure_ascii=False)
        arr = self._extract_first_json_segment(s)
        if isinstance(arr, list):
            return self._normalize_calls(arr)
        return []

    def parse_assistant_response(self, response: str):
        parts = {'reflect': '', 'final': '', 'calls': []}
        if not response:
            return parts

        cleaned = self._remove_think_blocks(response)

        def first_tag(text, tag):
            m = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ''

        parts['reflect'] = first_tag(cleaned, 'reflect')
        parts['final'] = first_tag(cleaned, 'final')
        parts['calls'] = self._extract_calls_from_text(cleaned)
        return parts

    def compute_semantic_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a_clean = re.sub(r'\s+', ' ', a.strip().lower())
        b_clean = re.sub(r'\s+', ' ', b.strip().lower())
        if not a_clean or not b_clean:
            return 0.0
        if a_clean == b_clean:
            return 1.0

        def ngram_set(s, n):
            toks = s.split()
            if len(toks) < n:
                return {' '.join(toks)} if toks else set()
            return {' '.join(toks[i:i+n]) for i in range(len(toks) - n + 1)}

        g1a, g1b = ngram_set(a_clean, 1), ngram_set(b_clean, 1)
        g2a, g2b = ngram_set(a_clean, 2), ngram_set(b_clean, 2)
        j1 = len(g1a & g1b) / (len(g1a | g1b) + self.eps) if (g1a or g1b) else 0.0
        j2 = len(g2a & g2b) / (len(g2a | g2b) + self.eps) if (g2a or g2b) else 0.0
        len_sim = min(len(a_clean), len(b_clean)) / (max(len(a_clean), len(b_clean)) + self.eps)
        return float(max(0.0, min(1.0, 0.6*j1 + 0.3*j2 + 0.1*len_sim)))

    def _compute_single_reward(self, completion: str, ground_truth: str) -> float:
        comp_parts = self.parse_assistant_response(completion)
        gt_parts = self.parse_assistant_response(ground_truth)

        reflect_score = 0.0
        call_score = 0.0
        final_score = 0.0

        comp_reflect = comp_parts.get('reflect', '').strip()
        gt_reflect = gt_parts.get('reflect', '').strip()
        if gt_reflect:
            reflect_score = self.compute_semantic_similarity(comp_reflect, gt_reflect)

        comp_calls = comp_parts.get('calls', [])
        gt_calls = gt_parts.get('calls', [])
        if gt_calls:
            call_score = 1.0 if self._are_calls_exactly_equal(comp_calls, gt_calls) else 0.0

        comp_final = comp_parts.get('final', '').strip()
        gt_final = gt_parts.get('final', '').strip()
        if gt_final:
            final_score = self.compute_semantic_similarity(comp_final, gt_final)

        active_weight_sum = 0.0
        weighted_score_sum = 0.0
        if gt_reflect:
            active_weight_sum += self.reflect_weight
            weighted_score_sum += reflect_score * self.reflect_weight
        if gt_calls:
            active_weight_sum += self.call_weight
            weighted_score_sum += call_score * self.call_weight
        if gt_final:
            active_weight_sum += self.final_weight
            weighted_score_sum += final_score * self.final_weight

        if active_weight_sum > self.eps:
            final_reward = weighted_score_sum / active_weight_sum
        else:
            final_reward = 1.0
        return float(max(0.0, min(1.0, final_reward)))

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

        gt_candidate = None
        for key in ['ground_truths', 'ground_truth', 'gt', 'references', 'targets']:
            if key in kwargs:
                gt_candidate = kwargs[key]
                break

        if gt_candidate is not None:
            if isinstance(gt_candidate, str):
                gt_candidate = [gt_candidate]
            elif isinstance(gt_candidate, list):
                extracted = []
                for entry in gt_candidate:
                    if isinstance(entry, str):
                        extracted.append(entry)
                    elif isinstance(entry, list):
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
                        extracted.append(
                            str(entry.get('content', '') or
                                entry.get('message', '') or
                                json.dumps(entry, ensure_ascii=False))
                        )
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

orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'ChainDialogueRewardOptimized_qwen': ChainDialogueRewardOptimized_qwen,
    'ChainDialogueRewardOptimized_qwen3': ChainDialogueRewardOptimized_qwen3,
    'ChainDialogueRewardOptimized_llama': ChainDialogueRewardOptimized_llama,
    'simple_reward_llama':simple_reward_llama,
    'simple_reward_qwen':simple_reward_qwen,
    'ChainDialogueRewardOptimized_qwen_v2':ChainDialogueRewardOptimized_qwen_v2,
    # ToolRL reward functions (for comparison experiments)
    'ToolRLReward': ToolRLReward,
    'ToolRLRewardLlama': ToolRLRewardLlama,
    'ToolRLRewardQwen': ToolRLRewardQwen,
}
