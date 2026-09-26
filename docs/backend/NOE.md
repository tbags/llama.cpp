# Cix Zhouyi NPU Backend (NOE)

This document describes how to use and compile models for the Cix Zhouyi X2 NPU backend (`GGML_NOE`) on Cix P1 platforms such as the Orange Pi 6 Plus.

---

## 1. Overview and Architecture

The Cix P1 SoC integrates a 3-core Arm China Zhouyi X2 NPU (`X2_1204MP3`) providing up to 28.8 TOPs of INT8 compute with unified memory architecture (UMA) sharing physical RAM with the CPU.

```
llama.cpp / GGML
       |
       v
  ggml_backend (NOE)
       |
       +--- Device selection (--device NOE)
       +--- Host buffer mapping (UMA zero-copy)
       +--- Kernel / Subgraph execution
                |
                v
            NOE UMD (libnoe.so)
                |
       +--------+--------+
       |                 |
  .cix binary          Job
  (Compass NN)      (Execution)
       |                 |
       v                 v
   Zhouyi NPU hardware execution (X2_1204MP3)
```

The NOE backend supports two execution models:
1. **Fused Subgraph Offload (`qwen_mlp.cix`)**:
   Fuses the SwiGLU MLP block (`gate_proj`, `up_proj`, `SiLU`, and `down_proj`) into a single NPU graph. Intermediate activations (18944-dim in Qwen2.5-7B) stay entirely inside NPU SRAM/TPC, avoiding DDR bandwidth bottlenecks and reducing kernel launch overhead to a single sync call (~9.1 ms per layer).
2. **Dynamic Weight MatMul (`dynamic_matmul.cix`)**:
   General projection kernel where the activation tensor and weight matrix are both dynamic inputs to the graph. Allows running arbitrary layers without baking weights into the `.cix` binary.

---

## 2. Compilation with cixbuild (Compass Toolchain)

`cixbuild` (AIPUBuilder/GBuilder) compiles ONNX computational graphs into target-specific `.cix` binaries for the `X2_1204MP3` NPU core.

> [!NOTE]
> The `cixbuild` compiler toolchain runs on **x86_64 Linux** development hosts. The generated `.cix` binary is copied to the target board for runtime execution by `libnoe.so`.

### Prerequisites on x86 Host

- Cix NOE SDK release environment (e.g. `cix_noe_sdk_25_q3_release`)
- Python 3.10 with PyTorch, ONNX, and AIPUBuilder
- Simulator libraries (`libaipu_simulator_x2.so`) added to `LD_LIBRARY_PATH`

Example environment activation:
```bash
source /path/to/cix_noe_sdk_25_q3_release/env/bin/activate
export LD_LIBRARY_PATH=/path/to/compat_lib:/path/to/cix_noe_sdk_25_q3_release/env/lib/python3.10/site-packages/AIPUBuilder/simulator-lib:$LD_LIBRARY_PATH
```

---

### Pattern A: Compiling Fused SwiGLU MLP Subgraph

#### Step 1: Export ONNX Graph
Run this Python script on the x86 host to construct and export the fused MLP block:

```python
import torch
import torch.nn as nn
import numpy as np

class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size=3584, intermediate_size=18944):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

mlp = Qwen2MLP().eval()
x = torch.randn(1, 3584)
torch.onnx.export(
    mlp, x, "qwen_mlp.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=14
)

# Generate calibration sample
calib = np.random.randn(5, 3584).astype(np.float32)
np.save("calib_mlp.npy", calib)
```

#### Step 2: Create cixbuild Configuration (`qwen_mlp.cfg`)
```ini
[Common]
mode = build

[Parser]
model_type = onnx
model_name = qwen_mlp
input_model = ./qwen_mlp.onnx
input = input
input_shape = [1, 3584]
output = output
output_dir = ./out_mlp

[Optimizer]
calibration_data = ./calib_mlp.npy
calibration_batch_size = 1
dataset = NumpyDataset
weight_bits = 8
activation_bits = 8
bias_bits = 32
quantize_method_for_weight = per_channel_symmetric_restricted_range
quantize_method_for_activation = per_tensor_asymmetric
cast_dtypes_for_lib = true
output_dir = ./out_mlp

[GBuilder]
target = X2_1204MP3
outputs = qwen_mlp.cix
tiling = fps
```

#### Step 3: Run cixbuild
Run with CPU execution provider to avoid host GPU memory limits:
```bash
CUDA_VISIBLE_DEVICES="" cixbuild qwen_mlp.cfg
```
Output binary: `qwen_mlp.cix` (~195 MB with packed INT8 layer weights).

---

### Pattern B: Compiling Dynamic Weight MatMul

#### Step 1: Export ONNX Graph
```python
import torch
import torch.nn as nn
import numpy as np

class DynamicMatMul(nn.Module):
    def forward(self, x, w):
        return torch.matmul(x, w.t())

m = DynamicMatMul().eval()
x = torch.randn(1, 3584)
w = torch.randn(18944, 3584)
torch.onnx.export(
    m, (x, w), "dynamic_matmul.onnx",
    input_names=["input", "weight"], output_names=["output"],
    opset_version=14
)

# Calibration dictionary for multi-input dataset
calib = {
    "input": np.random.randn(2, 1, 3584).astype(np.float32),
    "weight": np.random.randn(2, 18944, 3584).astype(np.float32)
}
np.save("calib_dynamic.npy", calib)
```

#### Step 2: Create cixbuild Configuration (`dynamic_matmul.cfg`)
```ini
[Common]
mode = build

[Parser]
model_type = onnx
model_name = dynamic_matmul
input_model = ./dynamic_matmul.onnx
input = input,weight
input_shape = [1, 3584],[18944, 3584]
output = output
output_dir = ./out_dynamic

[Optimizer]
calibration_data = ./calib_dynamic.npy
calibration_batch_size = 1
dataset = NumpyMultiInputDataset
weight_bits = 8
activation_bits = 8
bias_bits = 32
cast_dtypes_for_lib = true
output_dir = ./out_dynamic

[GBuilder]
target = X2_1204MP3
outputs = dynamic_matmul.cix
tiling = fps
```

#### Step 3: Run cixbuild
```bash
CUDA_VISIBLE_DEVICES="" cixbuild dynamic_matmul.cfg
```
Output binary: `dynamic_matmul.cix` (~382 KB without embedded weights).

---

## 3. Configuring and Running on the Target Board

### Step 1: Transfer Compiled Binaries
Copy the `.cix` files to the target board:
```bash
mkdir -p /path/to/models
scp user@x86-host:/path/to/dynamic_matmul.cix /path/to/models/
scp user@x86-host:/path/to/qwen_mlp.cix /path/to/models/
```

### Step 2: Build llama.cpp with NOE Backend
On the Orange Pi 6 Plus:
```bash
cmake -B build -DGGML_NOE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

### Step 3: Specifying the Model Kernel Directory
Paths can be configured dynamically without hardcoded directories:

1. **Via CLI Flag**:
   ```bash
   ./build/bin/llama-cli -m <model.gguf> --device NOE --noe-model-dir /path/to/models
   ```

2. **Via Environment Variable**:
   ```bash
   export GGML_NOE_MODEL_DIR=/path/to/models
   ./build/bin/llama-cli -m <model.gguf> --device NOE
   ```

3. **Default Fallback Paths**:
   If neither flag nor environment variable is set, the backend checks:
   - `./models`
   - `/usr/share/cix/models`

---

## 4. Verification and Telemetry

### Enable Live Telemetry
Set `GGML_NOE_ENABLE=1` and `GGML_NOE_LOG=1` to see real-time layer execution:

```bash
GGML_NOE_ENABLE=1 GGML_NOE_LOG=1 ./build/bin/llama-cli \
  -m ~/.ollama/models/blobs/<model_blob> \
  -p "Hello" \
  -n 1 \
  --device NOE \
  -st
```

Sample output:
```text
[GGML-NOE] Executed 'ffn_gate-0' on Zhouyi NPU
[GGML-NOE] Executed 'ffn_up-0' on Zhouyi NPU
[GGML-NOE] Executed 'ffn_gate-1' on Zhouyi NPU
[GGML-NOE] Executed 'ffn_up-1' on Zhouyi NPU
...
[GGML-NOE] Executed 'ffn_gate-27' on Zhouyi NPU
[GGML-NOE] Executed 'ffn_up-27' on Zhouyi NPU
```

### Direct Kernel Latency Test
A standalone C++ test runner can be compiled directly against `libnoe.so` on the board to measure execution time on the physical hardware:

```bash
g++ -O3 -I/usr/share/cix/include/npu test_runner.cpp -L/usr/share/cix/lib -Wl,-rpath,/usr/share/cix/lib -lnoe -o test_runner
./test_runner /path/to/models/qwen_mlp.cix 20
```

Observed latencies on Orange Pi 6 Plus (`X2_1204MP3`, 3 cores):
- **Fused SwiGLU MLP**: ~9.1 ms
- **Dynamic Weight MatMul**: ~20.0 ms
