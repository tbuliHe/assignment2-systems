import time
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from cs336_basics.model import BasicsTransformerLM

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_model_config():
    """返回用于基准测试的模型配置"""
    # 这里使用一个中等大小的配置进行测试，你可以根据显存大小调整
    return {
        "vocab_size": 10000,
        "context_length": 512,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
        "rope_theta": 10000.0,
    }

def benchmark():
    # 1. 设备检查
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == 'cpu':
        logger.warning("Warning: Benchmarking on CPU. NVTX and sync/async timing will not be accurate for GPU workloads.")

    # 2. 模型初始化
    config = get_model_config()
    model = BasicsTransformerLM(**config).to(device)
    model.train()

    # 3. 自动计算参数量与 FLOPs (6ND 公式)
    # N: 总参数量
    num_params = sum(p.numel() for p in model.parameters())
    
    # 设定 Batch Size
    batch_size = 8
    seq_len = config["context_length"]
    
    # D: 单步训练处理的 Token 数量 (Batch Size * Sequence Length)
    tokens_per_step = batch_size * seq_len
    
    # 6ND FLOPs 估算: 每次迭代 Forward + Backward 约为 6 * N * D
    flops_per_step = 6 * num_params * tokens_per_step
    
    logger.info(f"Model Parameters (N): {num_params:,}")
    logger.info(f"Tokens per step (D): {tokens_per_step:,}")
    logger.info(f"Estimated FLOPs per step (6ND): {flops_per_step:,.0f}")

    # 准备 Dummy 数据
    x = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
    y = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # 4. Warmup 预热
    # 防止 CUDA 初始化、allocator 分配等开销影响计时
    warmup_steps = 5
    logger.info(f"Starting warmup ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.view(-1, config["vocab_size"]), y.view(-1))
        loss.backward()
        optimizer.step()

    # 5. 精确计时 Benchmark
    steps = 20
    logger.info(f"Starting benchmark ({steps} steps)...")

    # 关键：计时前同步 CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()

    for i in range(steps):
        optimizer.zero_grad()
        
        # --- Forward ---
        if device.type == "cuda":
            torch.cuda.nvtx.range_push("Forward")
        
        logits = model(x)
        loss = loss_fn(logits.view(-1, config["vocab_size"]), y.view(-1))
        
        if device.type == "cuda":
            torch.cuda.nvtx.range_pop()

        # --- Backward ---
        if device.type == "cuda":
            torch.cuda.nvtx.range_push("Backward")
        
        loss.backward()
        
        if device.type == "cuda":
            torch.cuda.nvtx.range_pop()

        # --- Optimizer ---
        if device.type == "cuda":
            torch.cuda.nvtx.range_push("Optimizer")
        
        optimizer.step()
        
        if device.type == "cuda":
            torch.cuda.nvtx.range_pop()

    # 关键：计时结束前同步 CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()

    # 6. 结果计算
    total_time = end_time - start_time
    avg_time = total_time / steps
    
    # TFLOPS = Total FLOPs / Seconds / 10^12
    achieved_tflops = (flops_per_step / avg_time) / 1e12
    tokens_per_sec = tokens_per_step / avg_time

    logger.info("Benchmark Results:")
    logger.info(f"  Avg Time per Step: {avg_time * 1000:.2f} ms")
    logger.info(f"  Throughput: {tokens_per_sec:,.2f} tokens/s")
    logger.info(f"  Achieved Performance: {achieved_tflops:.4f} TFLOPS")

if __name__ == "__main__":
    benchmark()