---
description: Run tests on remote ssh server
agent: build
subtask: true
---

Run tests on a remote SSH server using script: `~/.config/opencode/commands/remote-run.sh`.

$1 is the remote server name
$2 is the task to run
$3 is the other arguments to pass to the script

Examples:

* Upload code to server `h20`:
```bash
bash ~/.config/opencode/commands/remote-run.sh -s h20 -t upload --local "/localdisk/kurt/workspace/code/xpu/llama.cpp" --remote "/ssh/kurt/"
```

* Build on server `h20`:
```bash
bash ~/.config/opencode/commands/remote-run.sh -s h20 -t build
```

* Run tests on server `h20`: IMPORTANT: always add `-e CUDA_VISIBLE_DEVICES="0"` to avoid using all GPUs:
```bash
bash ~/.config/opencode/commands/remote-run.sh -s h20 -t run -e CUDA_VISIBLE_DEVICES="0"
```

* Upload code to server `b60`:
```bash
bash ~/.config/opencode/commands/remote-run.sh -s b60 -t upload --local "/localdisk/kurt/workspace/code/xpu/llama.cpp" --remote "/intel/kurt/"
```

* Build on server `b60`:
```bash
bash ~/.config/opencode/commands/remote-run.sh -s b60 -t build
```

* Run tests on server `b60`: IMPORTANT: always add `-e ZES_ENABLE_SYSMAN=1` to support to get free memory of GPU by `sycl::aspect::ext_intel_free_memory`.
```bash
bash ~/.config/opencode/commands/remote-run.sh -s b60 -t run -e ZES_ENABLE_SYSMAN=1
```

* Run custom command on server `b60`: assume $3 is `--command "cat /intel/kurt/llama.cpp/logs/b60_build_20260202_133422.log | tail -100"`
```bash
bash ~/.config/opencode/commands/remote-run.sh -s b60 -t custom --command "cat /intel/kurt/llama.cpp/logs/b60_build_20260202_133422.log | tail -100"
```
