import json
import re
import os
from vllm import LLM, SamplingParams
from typing import List, Dict, Any
import torch
from openai import OpenAI
import time


def detect_model_type(model_name: str) -> str:
    """检测模型类型"""
    model_name_lower = model_name.lower()

    # 优先检查更具体的模型标识
    if "llama-3" in model_name_lower or "llama3" in model_name_lower:
        return "llama"
    elif "llama" in model_name_lower:
        return "llama"
    elif "qwen" in model_name_lower:
        return "qwen"
    else:
        return "qwen"

def extract_tool_calls_universal(text: str, model_type: str = "qwen") -> tuple[List[Dict], List[str]]:
    """根据模型类型提取工具调用"""
    tool_calls = []
    format_errors = []

    if model_type == "qwen" or model_type != "llama":
        # Qwen系列：只支持<tool_call>标签格式，参数字段为arguments
        tool_call_pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
        tool_matches = re.findall(tool_call_pattern, text, re.DOTALL)

        for match in tool_matches:
            try:
                tool_call = json.loads(match)
                if 'name' in tool_call and 'arguments' in tool_call:
                    tool_calls.append(tool_call)
                else:
                    format_errors.append(f"Tool call missing required fields: {match[:100]}...")
            except json.JSONDecodeError as e:
                format_errors.append(f"Invalid JSON in tool call: {match[:100]}...")
                continue

        # 严格检查格式问题
        # 1. 检查是否有不完整的工具调用
        incomplete_pattern = r'<tool_call>\s*\{[^}]*$'
        if re.search(incomplete_pattern, text, re.DOTALL):
            format_errors.append("Found incomplete tool calls (missing closing tag or brace)")

        # 2. 检查是否有裸露的JSON（没有标签包围的工具调用）
        bare_json_pattern = r'(?<!<tool_call>)\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}(?!\s*</tool_call>)'
        if re.search(bare_json_pattern, text):
            format_errors.append("Found bare JSON tool calls without proper tags")

    elif model_type == "llama":
        # 先找到所有可能的JSON数组结构
        bracket_start = 0
        while bracket_start < len(text):
            start_pos = text.find('[', bracket_start)
            if start_pos == -1:
                break

            # 找到匹配的右括号
            bracket_count = 1
            end_pos = start_pos + 1
            while end_pos < len(text) and bracket_count > 0:
                if text[end_pos] == '[':
                    bracket_count += 1
                elif text[end_pos] == ']':
                    bracket_count -= 1
                end_pos += 1

            if bracket_count == 0:  # 找到完整的[]对
                json_candidate = text[start_pos:end_pos]

                try:
                    parsed = json.loads(json_candidate)
                    if isinstance(parsed, list):
                        # 验证是否是工具调用数组
                        is_tool_call_array = True
                        for item in parsed:
                            if not (isinstance(item, dict) and 'name' in item and 'parameters' in item):
                                is_tool_call_array = False
                                break

                        if is_tool_call_array and len(parsed) > 0:
                            tool_calls.extend(parsed)
                        elif len(parsed) > 0:  # 有内容但不是工具调用格式
                            format_errors.append(f"Invalid tool call array structure: {json_candidate[:100]}...")

                except json.JSONDecodeError:
                    # 检查是否看起来像工具调用但格式错误
                    if '"name"' in json_candidate and '"parameters"' in json_candidate:
                        format_errors.append(f"Invalid JSON array format: {json_candidate[:100]}...")

            bracket_start = start_pos + 1

    return tool_calls, format_errors

def normalize_tool_calls(tool_calls: List[Dict], model_type: str = "qwen") -> List[Dict]:
    """根据模型类型标准化tool call格式"""
    normalized = []

    # 根据模型类型确定参数字段名
    param_field = "arguments" if model_type == "qwen" else "parameters"

    for call in tool_calls:
        normalized_call = {
            "name": call.get("name", ""),
            param_field: call.get(param_field, {})
        }

        # 对参数按键排序，确保比较一致性
        if isinstance(normalized_call[param_field], dict):
            normalized_call[param_field] = dict(sorted(normalized_call[param_field].items()))
        normalized.append(normalized_call)

    # 修改排序逻辑：同时考虑工具名称和参数，确保完全一致的排序
    def sort_key(call):
        name = call["name"]
        params = call[param_field]
        # 将参数转换为字符串用于排序，确保相同参数的调用排序一致
        params_str = json.dumps(params, sort_keys=True) if params else ""
        return (name, params_str)

    return sorted(normalized, key=sort_key)

def compare_tool_calls_as_sets(completion: str, ground_truth: str, model_type: str = "qwen") -> tuple[bool, List[str]]:
    """将工具调用作为集合进行比较，忽略顺序"""
    completion_calls, comp_errors = extract_tool_calls_universal(completion, model_type)
    gt_calls, gt_errors = extract_tool_calls_universal(ground_truth, model_type)

    errors = []

    # 如果completion有任何格式错误，直接判为不正确
    if comp_errors:
        errors.extend([f"Completion format error: {err}" for err in comp_errors])
        return False, errors

    # 检查是否都没有工具调用的情况（保持原有逻辑）
    if not gt_calls and not completion_calls:
        # 检查ground truth中是否有工具调用相关的文本但解析失败
        if ('"name"' in ground_truth and '"parameters"' in ground_truth) or ('[' in ground_truth and ']' in ground_truth):
            errors.append("Ground truth appears to contain tool calls but extraction failed")
            return False, errors
        # 检查completion中是否有工具调用相关的文本但解析失败
        if ('"name"' in completion and '"parameters"' in completion) or ('[' in completion and ']' in completion):
            errors.append("Completion appears to contain tool calls but extraction failed")
            return False, errors
        # 都没有工具调用相关内容，可能是纯文本对话
        return True, []

    # 标准化格式
    completion_calls_norm = normalize_tool_calls(completion_calls, model_type)
    gt_calls_norm = normalize_tool_calls(gt_calls, model_type)

    # 检查数量是否匹配
    if len(completion_calls_norm) != len(gt_calls_norm):
        errors.append(f"Tool call count mismatch: completion has {len(completion_calls_norm)}, ground truth has {len(gt_calls_norm)}")
        return False, errors

    # 将标准化后的工具调用转换为可比较的元组集合
    def call_to_tuple(call, param_field):
        name = call["name"]
        params = call[param_field]
        # 将参数字典转换为排序后的元组，便于集合比较
        if isinstance(params, dict):
            params_tuple = tuple(sorted(params.items()))
        else:
            params_tuple = (params,)
        return (name, params_tuple)

    param_field = "arguments" if model_type == "qwen" else "parameters"

    completion_set = set(call_to_tuple(call, param_field) for call in completion_calls_norm)
    gt_set = set(call_to_tuple(call, param_field) for call in gt_calls_norm)

    # 比较集合是否相等（忽略顺序）
    if completion_set == gt_set:
        return True, []
    else:
        # 找出差异
        missing_in_completion = gt_set - completion_set
        extra_in_completion = completion_set - gt_set

        if missing_in_completion:
            errors.append(f"Missing tool calls in completion: {list(missing_in_completion)}")
        if extra_in_completion:
            errors.append(f"Extra tool calls in completion: {list(extra_in_completion)}")

        return False, errors

def calculate_tool_call_score_with_sets(completion: str, ground_truth: str, model_type: str = "qwen") -> tuple[float, List[str]]:
    """使用集合方式计算工具调用匹配分数"""
    completion_calls, comp_errors = extract_tool_calls_universal(completion, model_type)
    gt_calls, gt_errors = extract_tool_calls_universal(ground_truth, model_type)

    errors = []

    # 如果有任何格式错误，分数为0
    if comp_errors:
        errors.extend([f"Completion format error: {err}" for err in comp_errors])
        return 0.0, errors

    # 数量检查逻辑保持不变
    if gt_calls and not completion_calls:
        return 0.0, ["Ground truth has tool calls but completion doesn't"]

    if completion_calls and not gt_calls:
        return 0.0, ["Completion has tool calls but ground truth doesn't"]

    if not gt_calls and not completion_calls:
        # 检查是否应该有工具调用但提取失败
        if ('"name"' in ground_truth and '"parameters"' in ground_truth) or ('[' in ground_truth and ']' in ground_truth):
            errors.append("Ground truth appears to contain tool calls but extraction failed")
            return 0.0, errors
        if ('"name"' in completion and '"parameters"' in completion) or ('[' in completion and ']' in completion):
            errors.append("Completion appears to contain tool calls but extraction failed")
            return 0.0, errors
        return 1.0, []

    # 标准化并转换为集合
    completion_calls_norm = normalize_tool_calls(completion_calls, model_type)
    gt_calls_norm = normalize_tool_calls(gt_calls, model_type)

    def call_to_tuple(call, param_field):
        name = call["name"]
        params = call[param_field]
        if isinstance(params, dict):
            params_tuple = tuple(sorted(params.items()))
        else:
            params_tuple = (params,)
        return (name, params_tuple)

    param_field = "arguments" if model_type == "qwen" else "parameters"

    completion_set = set(call_to_tuple(call, param_field) for call in completion_calls_norm)
    gt_set = set(call_to_tuple(call, param_field) for call in gt_calls_norm)

    # 完全匹配（集合相等）
    if completion_set == gt_set:
        return 1.0, []
    else:
        # 计算部分匹配分数：交集大小 / ground truth大小
        intersection = completion_set & gt_set
        partial_score = len(intersection) / len(gt_set) if gt_set else 0.0

        errors.append(f"Partial match: {len(intersection)}/{len(gt_set)} calls matched (ignoring order)")

        # 添加详细的差异信息
        missing = gt_set - completion_set
        extra = completion_set - gt_set
        if missing:
            errors.append(f"Missing: {list(missing)}")
        if extra:
            errors.append(f"Extra: {list(extra)}")

        return partial_score, errors

def validate_tool_call_format(tool_call_text: str, model_type: str) -> tuple[bool, List[str]]:
    """严格验证工具调用格式"""
    try:
        calls, errors = extract_tool_calls_universal(tool_call_text, model_type)

        # 有任何格式错误就认为无效
        if errors:
            return False, errors

        # 没有工具调用但也没有错误，可能是纯文本
        if not calls:
            return True, ["No tool calls found - may be a text response"]

        return True, []

    except Exception as e:
        return False, [f"Format validation error: {str(e)}"]

def build_prompt(messages: List[Dict]) -> str:
    """备用方法，构建prompt"""
    prompt = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"

    prompt += "Assistant: "
    return prompt

def apply_chat_template_safe(llm, messages: List[Dict]) -> str:
    """安全地应用chat template"""
    try:
        # 尝试使用模型自带的chat template
        tokenizer = llm.get_tokenizer()
        if hasattr(tokenizer, 'apply_chat_template'):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    except Exception as e:
        print(f"Chat template应用失败，回退到手动拼接: {e}")

    # 回退到原来的方法
    return build_prompt(messages)

def load_test_data(testfile: str) -> List[Dict]:
    """加载测试数据"""
    test_data = []
    with open(testfile, "r", encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                data['line_num'] = line_num
                test_data.append(data)
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行JSON解析错误: {e}")
                continue
    return test_data

def get_api_response_batch(client: OpenAI, batch_messages: List[List[Dict]], max_tokens: int = 1024, temperature: float = 0.0, model_name: str = "gpt-3.5-turbo") -> List[str]:
    """批量API调用（同步版本）"""
    results = []

    for i, messages in enumerate(batch_messages):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            result = response.choices[0].message.content
        except Exception as e:
            result = f"ERROR: API call failed - {str(e)}"

        results.append(result)

        # 打印进度和简单的速率控制
        if (i + 1) % 10 == 0:
            print(f"  API调用进度: {i + 1}/{len(batch_messages)}")

        # 简单的速率控制，避免触发限制
        if i < len(batch_messages) - 1:
            time.sleep(0.05)  # 50ms延迟

    return results

def remove_think_tags(text: str) -> str:
    """去除文本中的<think></think>标签及其内容"""
    if not text:
        return text
    
    # 使用正则表达式去除<think></think>标签及其内容
    # re.DOTALL 确保.可以匹配换行符
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 清理多余的空白字符
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)  # 去除多余空行
    cleaned_text = cleaned_text.strip()  # 去除首尾空白
    
    return cleaned_text

def main(model_name, batch_size=32, gpu_memory_utilization=0.9, tensor_parallel_size=None,
         use_api=False, api_key="", base_url="", api_model_name="gpt-3.5-turbo", n_attempts=3):
    """
    主测试函数 - 支持Pass@N评估
    
    参数:
        model_name: 模型名称或路径
        batch_size: 批处理大小
        gpu_memory_utilization: GPU内存使用率（仅VLLM模式）
        tensor_parallel_size: 张量并行大小（仅VLLM模式）
        use_api: 是否使用API模式
        api_key: API密钥
        base_url: API基础URL
        api_model_name: API调用时使用的模型名称
        n_attempts: 每道题测试的次数（Pass@N中的N）
    """
    # 检测模型类型
    model_type = detect_model_type(model_name)
    testfile = ""
    if model_type == "qwen":
        testfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_qwen_1000.jsonl")
    else:
        testfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_llama_1000.jsonl")

    param_field = "arguments" if model_type == "qwen" else "parameters"

    print(f"模式: {'API' if use_api else 'VLLM本地'}")
    print(f"检测到模型类型: {model_type}")
    print(f"参数字段名: {param_field}")
    print(f"Pass@N评估: N = {n_attempts}")
    
    if use_api:
        # API模式
        print("API Base URL: [configured]")
        print(f"API Model Name: {api_model_name}")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        llm = None  # API模式不需要本地LLM

    else:
        # VLLM本地模式
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()
            print(f"自动检测到 {tensor_parallel_size} 个GPU")
    
        print(f"初始化模型: {model_name}")
        print(f"GPU内存使用率: {gpu_memory_utilization}")
        print(f"并行GPU数量: {tensor_parallel_size}")
        print(f"批量推理大小: {batch_size}")

        llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=20000,
            trust_remote_code=True
        )

        # 设置采样参数 - 增加随机性以获得多样化的回答
        sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=0.7,  # 增加温度以获得不同回答
            top_p=0.9,        # 添加top_p采样
            repetition_penalty=1.1
        )
        client = None

    # 获取testfile的目录路径
    test_dir = os.path.dirname(testfile)
    
    # 从模型路径提取文件名作为结果文件名
    model_filename = os.path.basename(model_name.rstrip('/'))
    mode_suffix = "_api" if use_api else "_vllm"
    result_file = os.path.join(test_dir, f"{model_filename}{mode_suffix}_pass_at_{n_attempts}_results.jsonl")
    summary_file = os.path.join(test_dir, f"{model_filename}{mode_suffix}_pass_at_{n_attempts}_summary.json")
    
    print(f"Test directory: {test_dir}")
    print(f"Result file will be saved to: {result_file}")
    print(f"Summary file will be saved to: {summary_file}")
    
    # 检查目录权限
    if not os.path.exists(test_dir):
        print(f"ERROR: Directory {test_dir} does not exist!")
        return
    if not os.access(test_dir, os.W_OK):
        print(f"ERROR: No write permission for directory {test_dir}")
        return

    # 加载所有测试数据
    print("加载测试数据...")
    test_data = load_test_data(testfile)
    total_samples = len(test_data)
    print(f"共加载 {total_samples} 条测试数据")

    total_questions = 0
    passed_questions = 0  # Pass@N中通过的问题数
    all_attempts_sum = 0.0  # 所有尝试的总分数
    all_attempts_count = 0  # 所有尝试的总次数
    
    with open(result_file, "w", encoding='utf-8') as outfile:
        # 对每道题进行N次测试
        for question_idx, item in enumerate(test_data):
            line_num = item['line_num']
            messages = item["messages"]
            
            # 处理ground_truth
            gt_obj = item.get("groundtruth", item.get("ground_truth", ""))
            if isinstance(gt_obj, dict):
                ground_truth = gt_obj.get("content", "")
            else:
                ground_truth = str(gt_obj)
            
            print(f"测试题目 {question_idx + 1}/{total_samples} (line {line_num})")
            
            # 为当前问题进行N次尝试
            attempts_results = []
            question_passed = False
            
            # 准备N次相同的输入
            if use_api:
                batch_messages = [messages] * n_attempts
                
                print(f"  开始{n_attempts}次API调用...")
                api_results = get_api_response_batch(client, batch_messages,
                                                  max_tokens=1024, temperature=0.7,  # API也使用随机采样
                                                  model_name=api_model_name)
                
                # 构造输出格式
                class APIOutput:
                    def __init__(self, text):
                        self.text = text
                
                class APIOutputWrapper:
                    def __init__(self, text):
                        self.outputs = [APIOutput(text)]

                outputs = [APIOutputWrapper(result) for result in api_results]
                
            else:
                # VLLM模式：准备N个相同的prompt
                prompts = []
                base_prompt = apply_chat_template_safe(llm, messages)
                prompts = [base_prompt] * n_attempts
                
                print(f"  开始{n_attempts}次VLLM推理...")
                outputs = llm.generate(prompts, sampling_params)
            
            # 评估N次尝试的结果
            for attempt_idx in range(n_attempts):
                try:
                    output = outputs[attempt_idx]
                    completion = ""
                    
                    if output.outputs:
                        raw_completion = output.outputs[0].text
                        # *** 关键修改：去除<think></think>标签内容 ***
                        completion = remove_think_tags(raw_completion)
                        
                        # 如果去除think标签后内容为空，记录警告
                        if not completion and raw_completion:
                            completion = "ERROR: Content empty after removing think tags"
                    else:
                        raw_completion = "ERROR: No output generated"
                        completion = raw_completion

                    # 评估这次尝试
                    score = 0.0
                    is_correct = False
                    error_message = ""
                    
                    try:
                        format_valid, format_errors = validate_tool_call_format(completion, model_type)
                        score, eval_errors = calculate_tool_call_score_with_sets(completion, ground_truth, model_type)
                        is_correct, match_errors = compare_tool_calls_as_sets(completion, ground_truth, model_type)

                        all_errors = format_errors + eval_errors + match_errors
                        if all_errors:
                            error_message = "; ".join(all_errors)

                    except Exception as e:
                        score = 0.0
                        is_correct = False
                        error_message = f"Evaluation error: {str(e)}"
                    
                    # 记录这次尝试的结果
                    attempt_result = {
                        "attempt": attempt_idx + 1,
                        "score": score,
                        "is_correct": is_correct,
                        "completion": completion,
                        "error_message": error_message if error_message else None
                    }
                    attempts_results.append(attempt_result)
                    
                    # 更新统计
                    all_attempts_sum += score
                    all_attempts_count += 1
                    
                    # 如果这次尝试成功，标记问题为通过
                    if is_correct:
                        question_passed = True
                    
                    print(f"    尝试 {attempt_idx + 1}: {'✓' if is_correct else '✗'} {score:.3f}")
                    
                except Exception as e:
                    attempt_result = {
                        "attempt": attempt_idx + 1,
                        "score": 0.0,
                        "is_correct": False,
                        "completion": f"ERROR: {str(e)}",
                        "error_message": f"Processing error: {str(e)}"
                    }
                    attempts_results.append(attempt_result)
                    all_attempts_count += 1
                    print(f"    尝试 {attempt_idx + 1}: ERROR - {str(e)}")
            
            # 统计这道题的结果
            total_questions += 1
            if question_passed:
                passed_questions += 1
            
            # 保存这道题的完整结果
            question_result = {
                "line_num": line_num,
                "question_idx": question_idx + 1,
                "n_attempts": n_attempts,
                "question_passed": question_passed,
                "attempts": attempts_results,
                "ground_truth": ground_truth,
                "messages": messages,
                "model_name": model_name,
                "model_type": model_type,
                "inference_mode": "api" if use_api else "vllm"
            }
            
            outfile.write(json.dumps(question_result, ensure_ascii=False) + '\n')
            outfile.flush()
            
            # 计算当前的Pass@N
            current_pass_at_n = passed_questions / total_questions
            current_avg_score = all_attempts_sum / all_attempts_count if all_attempts_count > 0 else 0.0
            
            print(f"  问题结果: {'PASS' if question_passed else 'FAIL'}")
            print(f"  当前Pass@{n_attempts}: {current_pass_at_n:.4f} ({passed_questions}/{total_questions})")
            print(f"  当前平均分数: {current_avg_score:.4f}")
            print()

    # 计算最终结果
    final_pass_at_n = passed_questions / total_questions if total_questions > 0 else 0.0
    final_avg_score = all_attempts_sum / all_attempts_count if all_attempts_count > 0 else 0.0
    
    # 保存汇总结果
    summary_data = {
        "model_name": model_name,
        "model_type": model_type,
        "inference_mode": "api" if use_api else "vllm",
        "n_attempts": n_attempts,
        "total_questions": total_questions,
        "passed_questions": passed_questions,
        f"pass_at_{n_attempts}": final_pass_at_n,
        "average_score_all_attempts": final_avg_score,
        "total_attempts": all_attempts_count,
        "batch_size": batch_size,
        "result_file": result_file,
        "processing_status": "completed" if total_questions > 0 else "failed"
    }
        
    if not use_api:
        summary_data.update({
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
        })
    else:
        summary_data.update({
            "api_model_name": api_model_name,
        })

    with open(summary_file, "w", encoding='utf-8') as summary_f:
        json.dump(summary_data, summary_f, ensure_ascii=False, indent=2)
        
    print(f"\n=== Final Results ===")
    print(f"Inference Mode: {'API' if use_api else 'VLLM'}")
    print(f"Model Type: {model_type}")
    print(f"Total questions: {total_questions}")
    print(f"Passed questions: {passed_questions}")
    print(f"Pass@{n_attempts}: {final_pass_at_n:.4f} ({final_pass_at_n*100:.2f}%)")
    print(f"Average score (all attempts): {final_avg_score:.4f}")
    print(f"Total attempts: {all_attempts_count}")
    print(f"Detailed results saved to: {result_file}")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tool-Reflection-Bench Evaluation")

    # Model specification (mutually exclusive: local vs API)
    parser.add_argument("--model", type=str, default=None,
                        help="Path to local HuggingFace model directory (vLLM mode)")
    parser.add_argument("--api", type=str, default=None,
                        help="API model name, e.g. gpt-4o (API mode)")

    # vLLM options
    parser.add_argument("--tp", type=int, default=None,
                        help="Tensor parallel size for vLLM (default: auto-detect GPU count)")
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size for vLLM inference (default: 64)")
    parser.add_argument("--gpu-mem", type=float, default=0.9,
                        help="GPU memory utilization for vLLM (default: 0.9)")

    # API options
    parser.add_argument("--api-key", type=str,
                        default=os.getenv("OPENAI_API_KEY", ""),
                        help="API key (default: $OPENAI_API_KEY)")
    parser.add_argument("--base-url", type=str,
                        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                        help="API base URL (default: $OPENAI_BASE_URL)")

    # Evaluation options
    parser.add_argument("--n", type=int, default=1,
                        help="Number of attempts per question for Pass@N (default: 1)")

    args = parser.parse_args()

    if args.model is None and args.api is None:
        parser.error("Specify either --model <path> (vLLM) or --api <model_name> (API)")

    use_api = args.api is not None
    model_name = args.api if use_api else args.model

    main(
        model_name=model_name,
        batch_size=args.batch,
        gpu_memory_utilization=args.gpu_mem,
        tensor_parallel_size=args.tp,
        use_api=use_api,
        api_key=args.api_key,
        base_url=args.base_url,
        api_model_name=args.api if use_api else "",
        n_attempts=args.n,
    )
