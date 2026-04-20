import argparse, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def log(*a): print(*a, file=sys.stderr)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct"  # Set your base model path)
    p.add_argument("--adapter_ckpt", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--dtype", default="bf16", choices=["bf16","fp16","fp32"])
    p.add_argument("--device_map", default="auto", choices=["auto","cpu"])
    p.add_argument("--max_shard_size", default="2GB")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--print_path_only", action="store_true")
    args = p.parse_args()

    if not args.print_path_only:
        log(f"[merge] base_model={args.base_model}")
        log(f"[merge] adapter_ckpt={args.adapter_ckpt}")
        log(f"[merge] save_dir={args.save_dir}")
        log(f"[merge] dtype={args.dtype}, device_map={args.device_map}")

    os.makedirs(args.save_dir, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=None if args.device_map=="cpu" else "auto",
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, args.adapter_ckpt).merge_and_unload()
    model.save_pretrained(args.save_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tok.save_pretrained(args.save_dir)

    if args.print_path_only:
        # 标准输出只给路径一行，便于 shell 捕获
        print(args.save_dir)
    else:
        log(f"[merge] Merged model saved to: {args.save_dir}")

if __name__ == "__main__":
    main()

