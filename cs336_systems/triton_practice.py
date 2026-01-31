import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,          # 指向输入向量 a 的指针
    y_ptr,          # 指向输入向量 b 的指针
    output_ptr,     # 指向输出向量 c 的指针
    n_elements,     # 向量中的元素总数
    BLOCK_SIZE: tl.constexpr,  # 每个 program 处理的元素数（编译期常量）
):
    # 当前 program 的 ID（类似 CUDA 的 blockIdx.x）
    pid = tl.program_id(axis=0)

    # 计算当前 program 负责的数据范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 创建 mask，防止越界访问
    mask = offsets < n_elements

    # 从全局显存加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 执行向量加法
    output = x + y

    # 将结果写回全局显存
    tl.store(output_ptr + offsets, output, mask=mask)


# ------------------------
# 在 Python / CPU 侧调用
# ------------------------

size = 98432

# 创建输入数据（CUDA Tensor）
a = torch.randn((size,), device="cuda")
b = torch.randn((size,), device="cuda")
output = torch.empty_like(a)

# 定义执行配置
BLOCK_SIZE = 1024
grid = (triton.cdiv(size, BLOCK_SIZE),)

# 启动 Triton kernel
add_kernel[grid](
    a,
    b,
    output,
    size,
    BLOCK_SIZE=BLOCK_SIZE,
)

# 验证结果
print(torch.allclose(output, a + b))
