# LLM Inference Lab

在百度云一台 8 × A800-SXM4-80GB 单节点上部署并评测 2026 年的开源大模型的学习记录。

**站点：https://haoxiangntu.github.io/llm-inference-lab/**

| 页面 | 内容 |
|---|---|
| [原理实验台](https://haoxiangntu.github.io/llm-inference-lab/docs/lab.html) | 六个可交互实验：prefill 与 decode、KV cache 分页 3D、多卡并行、训练与推理内存账 |
| [指标解读](https://haoxiangntu.github.io/llm-inference-lab/docs/metrics.html) | vLLM 是什么，TTFT / TPOT / 吞吐 / P99，各质量测试的判分方式 |
| [评测报告](https://haoxiangntu.github.io/llm-inference-lab/docs/benchmark.html) | GLM-5.3-Flash、Qwen3.6-35B-A3B、Qwen3.8-27B 的性能矩阵与质量成绩 |
| [部署实录](https://haoxiangntu.github.io/llm-inference-lab/docs/deployment.html) | 机器分析、模型选型、两条路线、chroot 代替 Docker、CUDA 前向兼容、各种坑 |

## 目录

```
docs/      站点页面（静态 HTML）
bench/     评测套件：quality.py（质量）、perf.sh（性能矩阵）、report.py（汇总）、run_model.sh
deploy/    部署脚本：pull_image.py（不用 Docker 拉镜像）、glm53_chroot.sh、glm53_build.sh、serve_*.sh、verify_model.py
results/   三个模型的原始结果 JSON 与 report.md
```

## 复现评测

```bash
# 服务端已有一个 OpenAI 兼容接口，例如 http://localhost:8010
bash bench/perf.sh http://localhost:8010 <served_model_name> <tokenizer_path> results/<tag>/perf
python3 bench/quality.py --base-url http://localhost:8010/v1 --model <served_model_name> --out results/<tag>/quality.json [--vision]
python3 bench/report.py results
```

数据集（GSM8K 测试集、HumanEval、C-Eval 验证集）从 ModelScope 获取，放在 `bench/data/`，脚本里有加载方式。

## 关键结论

- 同样 2 张卡，MoE 激活 3B 的 Qwen3.6-35B-A3B 单流 173 tok/s，稠密 27B 的 Qwen3.8-27B 49 tok/s：decode 速度由每步要从显存读的激活参数字节数决定。
- GLM-5.3-Flash 官方 FP8 权重在 Ampere 上可用 Marlin 以 W8A16 执行，稀疏注意力用社区的 Triton 内核；8 卡 TP4 × PP2，单流 50 tok/s，1M 上下文，GSM8K 98.5%。
- 驱动 535 跑 CUDA 12.9 / 13 程序需要 NVIDIA 前向兼容库，否则新工具链的 PTX 无法被驱动 JIT。
- ModelScope 命令行工具会在分片超时后仍返回成功，权重必须按远端清单逐文件校验。
- 工具调用解析器必须和模型输出格式匹配：Qwen3.x 用 `qwen3_xml`，hermes 会得 0 分。

公开脚本中的服务器地址已替换为占位符。
