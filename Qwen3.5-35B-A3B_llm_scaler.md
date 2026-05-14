# Qwen3.5-35B-A3B 在 PTL 用llm-scaler 容器启动指南

## 环境说明

- 镜像：`intel/llm-scaler-vllm:0.14.0-b8.2.1`
- 模型：`Qwen3.5-35B-A3B`（MoE 架构，约 35B 参数）
- 量化：FP8 / SYM_INT4（均已验证可用）
- 设备：PTL

---

## 前置：拉取 Docker 镜像（宿主机）

在宿主机上执行以下命令拉取镜像（约 20~30GB，需确保磁盘空间充足）：

```bash
docker pull intel/llm-scaler-vllm:0.14.0-b8.2.1
```

---

## 步骤一：启动 Docker 容器

```bash
docker run -t -d --rm \
  --shm-size 32g \
  --net=host \
  --ipc=host \
  --privileged \
  -e no_proxy="localhost,127.0.0.1" \
  -e NO_PROXY="localhost,127.0.0.1" \
  --cap-add=SYS_PTRACE \
  --cap-add=SYS_ADMIN \
  --security-opt seccomp=unconfined \
  -v /dev/dri/by-path:/dev/dri/by-path \
  --name=vllm-test \
  --device /dev/dri:/dev/dri \
  -v /home/intel/vllm/models:/workspace/vllm/models \
  --entrypoint= \
  intel/llm-scaler-vllm:0.14.0-b8.2.1 \
  /bin/bash
```

---

## 步骤二：宿主机增加 Swap 空间（必须）

> **原因**：模型加载过程中 CPU 内存峰值约 58GB，若物理内存不足 64GB 或有其他进程占用，会触发 Linux OOM Killer 杀死 vLLM 进程。增加 Swap 可避免此问题。

在**宿主机**上执行：

```bash
sudo rm -f /swapfile
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 写入 fstab 使重启后自动生效
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 验证
cat /etc/fstab
free -h
```

---

## 步骤三：修复 vLLM 源码（容器内）

> **原因**：`Qwen2VLImageProcessor` 对象在新版 transformers 中不再有 `min_pixels` / `max_pixels` 属性，vLLM 直接访问会抛出 `AttributeError`，需改为通过 `.size` 字典读取。

进入容器：

```bash
docker exec -it vllm-test bash
```

编辑文件 `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2_vl.py`，共需修改两处：

### 修改一：`_get_vision_info` 方法（约第 874 行）

找到 `smart_resize` 调用中的以下两行：

```python
# 原始代码（会报错）
                min_pixels=image_processor.min_pixels,
                max_pixels=image_processor.max_pixels,
```

替换为：

```python
                min_pixels=image_processor.size["shortest_edge"],
                max_pixels=image_processor.size["longest_edge"],
```

### 修改二：`get_image_size_with_most_features` 方法（约第 944 行）

找到以下这行：

```python
# 原始代码（会报错）
        max_pixels = image_processor.max_pixels or image_processor.size["longest_edge"]
```

替换为：

```python
        max_pixels = image_processor.size["longest_edge"]
```

### 删除编译缓存

修改完成后执行，确保 Python 使用修改后的源码而非旧的 `.pyc` 缓存：

```bash
rm -f /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/__pycache__/qwen2_vl.cpython-312.pyc
```

---

## 步骤四：启动 vLLM API Server（容器内）

### 方案 A：FP8 量化（推荐）

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn python -m vllm.entrypoints.openai.api_server \
  --model /workspace/vllm/models/Qwen3.5-35B-A3B \
  --max-model-len 1200 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --quantization fp8 \
  --block-size 64 \
  --trust-remote-code \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 2
```

### 方案 B：SYM_INT4 量化

> **注意**：`sym_int4` 仅支持 `float16`，必须显式指定 `--dtype float16`，否则默认的 `bfloat16` 会导致启动失败。

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn python -m vllm.entrypoints.openai.api_server \
  --model /workspace/vllm/models/Qwen3.5-35B-A3B \
  --max-model-len 1200 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --quantization sym_int4 \
  --dtype float16 \
  --block-size 64 \
  --trust-remote-code \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 2
```

模型加载约需 **2~3 分钟**（需加载 14 个 safetensors 分片）。

---

## 步骤五：验证服务（宿主机另开终端）

```bash
curl http://localhost:8000/v1/models
```

成功响应示例：

```json
{
  "object": "list",
  "data": [
    {
      "id": "/workspace/vllm/models/Qwen3.5-35B-A3B",
      "object": "model",
      "created": 1778681535,
      "owned_by": "vllm",
      "max_model_len": 1200
    }
  ]
}
```

---

## 步骤六：运行 Benchmark（容器内）

### 前提

确保 vLLM API Server 已正常启动（步骤五验证通过），且在**同一容器内**的另一个终端中执行以下命令。

### 命令

```bash
no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1" \
vllm bench serve \
    --model /workspace/vllm/models/Qwen3.5-35B-A3B \
    --served-model-name /workspace/vllm/models/Qwen3.5-35B-A3B \
    --dataset-name random \
    --random-input-len 200 \
    --random-output-len 200 \
    --ignore-eos \
    --num-prompts 10 \
    --request-rate inf \
    --port 8000
```



### 参数说明

| 参数 | 值 | 说明 |
|------|----|------|
| `--model` | 模型路径 | 与 server 启动时一致（用于加载 tokenizer） |
| `--served-model-name` | 模型路径 | API 请求中的模型名，必须与 server 完全一致 |
| `--dataset-name` | `random` | 使用随机生成的数据集 |
| `--random-input-len` | `200` | 每个请求的输入 token 数 |
| `--random-output-len` | `200` | 每个请求的输出 token 数 |
| `--ignore-eos` | - | 忽略 EOS token，强制生成到指定长度 |
| `--num-prompts` | `10` | 发送的请求总数 |
| `--request-rate` | `inf` | 以最大速率发送所有请求（不限速） |
| `--port` | `8000` | 服务端口，与 server 一致 |

> **关于 input/output 长度限制**：本模型 `--max-model-len 1200`
> 因此 `--random-input-len` + `--random-output-len` 之和不能超过 1200，例如取 200（合计 400）。

### Benchmark 结果示例

以下为在 Intel NucBox EVO T2S（Intel Arc GPU，SYM_INT4 量化）上的实测结果：

```
============ Serving Benchmark Result ============
Successful requests:                     10
Failed requests:                         0
Benchmark duration (s):                  53.62
Total input tokens:                      2000
Total generated tokens:                  2000
Request throughput (req/s):              0.19
Output token throughput (tok/s):         37.30
Peak output token throughput (tok/s):    40.00
Peak concurrent requests:                10.00
Total token throughput (tok/s):          74.60
---------------Time to First Token----------------
Mean TTFT (ms):                          21839.30
Median TTFT (ms):                        21810.31
P99 TTFT (ms):                           43310.12
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          52.41
Median TPOT (ms):                        52.49
P99 TPOT (ms):                           52.73
---------------Inter-token Latency----------------
Mean ITL (ms):                           52.41
Median ITL (ms):                         50.93
P99 ITL (ms):                            59.51
==================================================
```

### 指标说明

| 指标 | 含义 |
|------|------|
| TTFT (Time to First Token) | 从发送请求到收到第一个 token 的时间，反映首包延迟 |
| TPOT (Time per Output Token) | 生成每个输出 token 的平均时间（不含第一个） |
| ITL (Inter-token Latency) | 相邻两个输出 token 之间的时间间隔 |
| Output token throughput | 每秒生成的输出 token 数 |
| Total token throughput | 每秒处理的总 token 数（输入 + 输出） |

---

## 常见问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `AttributeError: 'Qwen2VLImageProcessor' object has no attribute 'max_pixels'` | transformers 新版 API 变更 | 执行步骤三的源码修复 |
| `RuntimeError: Engine core initialization failed` | EngineCore 进程被 OOM Killer 杀死 | 执行步骤二增加 Swap |
| 修改源码后报错行号不变 | Python 使用了旧的 `.pyc` 缓存 | 删除对应的 `.pyc` 文件 |
| `torch.bfloat16 is not supported for quantization method sym_int4` | sym_int4 不支持 bfloat16 | 命令中加上 `--dtype float16` |
| 端口 8088 无法连接 | vLLM 默认端口为 8000 | 使用 `http://localhost:8000` |
| Benchmark 报 `Error: Forbidden` (HTTP 403) | 容器内代理拦截了发往 `127.0.0.1` 的请求 | benchmark 命令前加 `no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"`，或在 `docker run` 时加 `-e no_proxy="localhost,127.0.0.1" -e NO_PROXY="localhost,127.0.0.1"` |
| Benchmark 报 `maximum context length is 688 tokens` | input+output 超过模型实际可用上下文 | 将 `--random-input-len` 和 `--random-output-len` 各设为 200 以内 |
