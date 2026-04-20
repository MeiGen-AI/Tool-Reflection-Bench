import os
import json
import re
import time
import argparse

DUMP_COMPLETIONS_PATH = ""

def _append_jsonl(path, obj):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def _dump_completions_to_jsonl(completions, rewards, **kwargs):
    ts = int(time.time())
    gt = kwargs.get('ground_truth')
    for i, comp in enumerate(completions):
        rec = {
            'ts': ts,
            'rank': _get_rank(),
            'completion': comp,
            'reward': float(rewards[i]) if isinstance(rewards, (list, tuple)) and i < len(rewards) else None,
        }
        # ground_truth 可能是 list 或 str
        if isinstance(gt, list):
            rec['ground_truth'] = gt[i] if i < len(gt) else None
        elif isinstance(gt, str):
            rec['ground_truth'] = gt
        _append_jsonl(DUMP_COMPLETIONS_PATH, rec)

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
BASE_CACHE = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
HF_HOME = os.path.join(BASE_CACHE, RUN_ID)
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

try:
    from datasets import disable_caching
    disable_caching()
    print("[INFO] datasets caching disabled")
except Exception as e:
    print("[WARN] cannot disable datasets caching:", e)

def _get_world_size():
    ws = os.environ.get('WORLD_SIZE')
    return int(ws) if ws and ws.isdigit() else 1

def _get_rank():
    rk = os.environ.get('RANK')
    return int(rk) if rk and rk.isdigit() else 0

def is_main_process():
    return _get_rank() == 0

def log0(msg: str):
    if is_main_process():
        print(msg)

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(BASE_CACHE, "outputs", RUN_ID),
        help=" "
    )
    p.add_argument(
        "--model_dir",
        type=str,
        default="",
        required=True,
        help=" "
    )
    p.add_argument(
        "--train_data",
        type=str,
        default="",
        required=True,
        help="Path to training data jsonl file"
    )
    return p.parse_args()

# ---------------- utils: trim long contexts to avoid negative max_tokens ----------------
# def convert_chain_dialogue_format(input_file, output_file):
#     """转换链式对话数据格式为GRPO训练格式

#     Args:
#         input_file: 原始数据文件路径
#         output_file: 转换后的输出文件路径

#     Returns:
#         int: 生成的训练样本数量
#     """
#     # 检查输入文件是否存在
#     if not os.path.exists(input_file):
#         raise FileNotFoundError(f"输入文件不存在: {input_file}")

#     # 检查输出文件是否已存在
#     if os.path.exists(output_file):
#         print(f'[DataConvert] 输出文件已存在: {output_file}')
#         # 计算现有文件的样本数量
#         with open(output_file, 'r', encoding='utf-8') as f:
#             existing_samples = sum(1 for line in f if line.strip())
#         print(f'[DataConvert] 现有训练样本: {existing_samples} 个')
#         return existing_samples

#     # 读取原始数据
#     with open(input_file, 'r', encoding='utf-8') as f:
#         original_data = [json.loads(line.strip()) for line in f if line.strip()]

#     print(f'[DataConvert] 原始数据: {len(original_data)} 行')

#     # 处理数据：为每个有效assistant轮次生成训练样本
#     training_samples = []
#     error_patterns = ['failed', 'could not be completed', 'error', 'invalid']

#     def clean_spaces_in_ground_truth(text):
#         """只清理多余空格，保留换行符和所有标签"""
#         # 只标准化连续的空格，保留换行符
#         text = re.sub(r'[ \t]+', ' ', text.strip())
#         # 清理函数参数中的空格
#         text = re.sub(r"=\s*'([^']*?)\s+([^']*?)'", r"='\1\2'", text)
#         text = re.sub(r'=\s*"([^"]*?)\s+([^"]*?)"', r'="\1\2"', text)
#         return text

#     for data_idx, data in enumerate(original_data):
#         messages = data.get('messages', [])

#         # 🔧 新增：预先标记包含<reflect>标签的assistant回复前一条assistant数据为错误数据
#         error_assistant_indices = set()

#         # 第一遍扫描：找到包含<reflect>标签的assistant回复，标记其前一条assistant数据
#         for i, message in enumerate(messages):
#             if message.get('role') != 'assistant':
#                 continue

#             content = message.get('content', '')
#             if '<reflect>' in content:
#                 # 找到前一条assistant消息的索引
#                 for j in range(i - 1, -1, -1):
#                     if messages[j].get('role') == 'assistant':
#                         error_assistant_indices.add(j)
#                         print(f'[DataConvert] 标记错误assistant数据: 对话{data_idx+1}, assistant{j} (因为assistant{i}包含<reflect>标签)')
#                         break

#         for i, message in enumerate(messages):
#             if message.get('role') != 'assistant':
#                 continue

#             # 跳过被标记为错误的assistant数据
#             if i in error_assistant_indices:
#                 print(f'[DataConvert] 跳过错误assistant数据: 对话{data_idx+1}, assistant{i}')
#                 continue

#             # 检查下一个是否是失败的tool response
#             if i + 1 < len(messages) and messages[i + 1].get('role') == 'tool':
#                 tool_response = messages[i + 1].get('content', '').lower()
#                 is_failed = any(pattern in tool_response for pattern in error_patterns)
#                 if is_failed:
#                     print(f'[DataConvert] 跳过失败的tool call轮次: 对话{data_idx+1}, assistant{i}')
#                     continue  # 跳过失败的tool call

#             # 🔧 关键修复：构建prompt（不包含当前要训练的assistant回复）
#             training_context = []

#             # 只包含第i个assistant回复之前的所有消息作为prompt
#             for j in range(i):  # 注意：range(i) 不包含第i个消息
#                 msg = messages[j]
#                 role = msg.get('role')
#                 content = msg.get('content', '')

#                 if role == 'assistant':
#                     # 检查这个assistant回复是否应该被跳过（错误的tool call）
#                     if j + 1 < len(messages) and messages[j + 1].get('role') == 'tool':
#                         tool_resp = messages[j + 1].get('content', '').lower()
#                         if any(pattern in tool_resp for pattern in error_patterns):
#                             print(f'[DataConvert] 在构建上下文时跳过失败tool call的assistant回复: 对话{data_idx+1}, assistant{j}')
#                             continue  # 跳过这个错误的assistant回复

#                     # 保留assistant回答的所有内容，包括所有标签
#                     # 注意：即使是被标记为错误的assistant数据，也要保留在上下文中
#                     training_context.append({
#                         'role': role,
#                         'content': content
#                     })

#                 elif role == 'tool':
#                     training_context.append({
#                         'role': "user",
#                         'content': f"Tool response: {content}"
#                     })

#                 elif role in ['system', 'user']:
#                     # 直接保留system和user消息
#                     training_context.append(msg)

#             # 生成ground_truth：使用完整的assistant回答作为ground_truth
#             content = message.get('content', '')

#             # 保留完整的assistant回答，包括所有标签（包括<think>标签）
#             ground_truth = content

#             # 清理多余的空格，但保留所有标签和内容结构
#             ground_truth = clean_spaces_in_ground_truth(ground_truth)

#             training_sample = {
#                 'messages': training_context,
#                 'ground_truth': ground_truth
#             }
#             training_samples.append(training_sample)

#     print(f'[DataConvert] 生成训练样本: {len(training_samples)} 个')

#     # 确保输出目录存在
#     output_dir = os.path.dirname(output_file)
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # 写入输出文件
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for sample in training_samples:
#             f.write(json.dumps(sample, ensure_ascii=False) + '\n')

#     print(f'[DataConvert] 数据格式转换完成！输出文件: {output_file}')
#     return len(training_samples)

# =============== 训练参数 ===============
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def test_chain_dialogue_reward():
    from swift.plugin.orm import ChainDialogueReward
    reward_func = ChainDialogueReward()
    test_completions = [
        "<reflect>用户想要查询天气</reflect><call>get_weather()</call><final>今天天气不错</final>",
        "<call>get_rate()</call><final>100美元=720人民币</final>"
    ]
    test_ground_truths = [
        "<reflect>用户想要查询天气</reflect><call>get_weather()</call><final>今天天气不错</final>",
        "<call>get_rate()</call><final>100美元=720人民币</final>"
    ]
    log0("=== 测试ChainDialogueReward ===")
    rewards = reward_func(test_completions, ground_truth=test_ground_truths)
    log0(f"测试奖励: {rewards}")
    log0("=== 测试完成 ===\n")


def configure_vllm_env():
    """
    让训练端保持标准 DDP（每个进程能看到全部可见 GPU，由 torchrun 映射 local_rank）
    仅将 vLLM 绑定到当前 local_rank 对应的那张卡；TP=1。
    """
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

    # 不再改写 CUDA_VISIBLE_DEVICES！
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    # vLLM 仅用一张卡
    os.environ['VLLM_TENSOR_PARALLEL_SIZE'] = '1'

    # 将 vLLM 绑定到当前进程对应的 GPU（兼容自定义的 CUDA_VISIBLE_DEVICES 列表）
    vis = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if vis:
        ids = [x.strip() for x in vis.split(',') if x.strip() != '']
        vllm_gpu = ids[local_rank] if 0 <= local_rank < len(ids) else ids[0]
    else:
        # 没设 CUDA_VISIBLE_DEVICES 时，系统 GPU 编号就是 0..N-1
        vllm_gpu = str(local_rank)

    os.environ['VLLM_GPU_IDS'] = str(vllm_gpu)

    # 仅打印，不改动 WORLD_SIZE/NPROC
    log0(f"[Dist] local_rank={local_rank}, CUDA_VISIBLE_DEVICES={vis or '(all)'}; "
         f"vLLM TP={os.environ['VLLM_TENSOR_PARALLEL_SIZE']}, VLLM_GPU_IDS={os.environ['VLLM_GPU_IDS']}")

def test_llm_vllm(model_dir, output_dir):
    from swift.llm import rlhf_main, RLHFArguments
    from swift.plugin.orm import orms, ChainDialogueRewardOptimized, ChainDialogueRewardOptimized_qwen, ChainDialogueRewardOptimized_qwen3, ChainDialogueRewardOptimized_llama, ChainDialogueRewardOptimized_qwen_v2
    # class ChainDialogueRewardWithDump(ChainDialogueRewardOptimized):
    #     def __call__(self, completions, **kwargs):
    #         rewards = super().__call__(completions, **kwargs)
    #         try:
    #             if is_main_process() and DUMP_COMPLETIONS_PATH:
    #                 _dump_completions_to_jsonl(completions, rewards, **kwargs)
    #         except Exception as e:
    #             log0(f"[Dump] 记录completions失败: {e}")
    #         return rewards
    # orms['ChainDialogueRewardWithDump'] = ChainDialogueRewardWithDump
    # —— 在创建 vLLM 引擎之前，尽量清掉前面加载造成的显存碎片 —— #
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # if is_main_process():
    #     os.makedirs(output_dir, exist_ok=True)
    #     print(f"[Run] 模型将保存到：{output_dir}")
    #     if 'ChainDialogueReward' in orms:
    #         print("✅ ChainDialogueReward已正确注册")
    #         print(f"   注册的类: {orms['ChainDialogueReward']}")
    #     else:
    #         print("❌ ChainDialogueReward未找到，可用的reward函数:")
    #         for k in orms.keys():
    #             print(f"   - {k}")

    result = rlhf_main(
        RLHFArguments(
            rlhf_type='grpo',
            model=model_dir,
            train_type='lora',
            model_type='qwen2_5',
            reward_funcs=['ChainDialogueRewardOptimized_qwen'],
            dataset=[args.train_data]  # Set via --train_data argument,
            # loss_scale="last_round_with_ignore_think",
            # agent_template="custom",
            log_completions=True,
            warmup_ratio=0.05,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_steps=50,
            num_train_epochs=1,
            # max_steps=1000,
            # 🔧 修复数值不稳定问题
            learning_rate=1e-5,        
            max_grad_norm=0.5,         # 更严格的梯度裁剪

            # 🔧 添加评估设置，显示train/eval loss
            eval_strategy='steps',     
            eval_steps=50,            
            do_eval=True,              
            logging_steps=5,           
            split_dataset_ratio=0.01,   # 从训练集中分出10%作为验证集
            # resume_from_checkpoint="",  # Set checkpoint path if resuming
            
            # 生成侧先保守一些，避免 KV cache 撑爆
            max_completion_length=1024, 
            max_model_len=25000,
            num_generations=4,
            generation_batch_size = 16,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1, 

            use_vllm=True,
            vllm_gpu_memory_utilization=0.5,
            vllm_enforce_eager=True,
            # 如果你们的 vLLM 版本支持，可尝试进一步省显存（否则删掉这两行）：
            # vllm_kv_cache_dtype="fp8",         # 或 "fp8_e5m2"
            # vllm_kv_cache_block_size=16,

            # DAPO
            epsilon_high=0.28,
            loss_type="bnpo",
            dynamic_sample=True,
            max_resample_times=3,

            # overlong_filter=True,

            # 🔧 基于GSPO论文的推荐参数 - 彻底解决数值不稳定
            beta=0.05,       # GSPO推荐：零KL正则化，避免KL散度爆炸
            epsilon=0.2,   # GSPO推荐：更小clip范围，防止重要性比率过大
            importance_sampling_level='sequence',  # GSPO：序列级采样，低方差
            # importance_sampling_level='token',

            bf16=True,
            gradient_checkpointing=True,
            offload_optimizer=True,
            output_dir=output_dir,
        )
    )

    if is_main_process():
        print("=== 训练完成，结果信息 ===")
        print(f"模型保存路径: {result.get('last_model_checkpoint', 'Unknown')}")
        print(f"训练日志路径: {result.get('log_history', 'Unknown')}")
        if 'trainer_state' in result:
            print(f"训练步数: {result['trainer_state'].get('global_step', 'Unknown')}")
            print(f"损失值: {result['trainer_state'].get('log_history', [])[-1:] if result['trainer_state'].get('log_history') else 'Unknown'}")
        print("=" * 50)


if __name__ == '__main__':
    args = parse_cli()
    # DUMP_COMPLETIONS_PATH = os.path.join(
    #     f"grpo_completions_{RUN_ID}.jsonl"
    # )
    # if is_main_process():
    #     print(f"[Dump] 采样completions将保存到: {DUMP_COMPLETIONS_PATH}")
    # if is_main_process():
    #     print("=== 开始数据格式转换 ===")
    # convert_chain_dialogue_format(input_file, output_file)
    # if is_main_process():
    #     print("=== 数据格式转换完成 ===\n")

    # if is_main_process():
    #     test_chain_dialogue_reward()

    configure_vllm_env()

    # 启动 GRPO 训练
    test_llm_vllm(model_dir=args.model_dir, output_dir=args.output_dir)
