---
name: execute-remote-ssh
description: "execute commands on remote GPU servers via SSH with file transfer and result collection"
---

# execute-remote-ssh

Executes commands on remote GPU servers via SSH with file transfer and result collection.

## Description

This skill manages secure SSH connections to remote GPU servers, executes build and test commands, transfers files, and collects execution results. Essential for Phase 6 validation where CUDA and SYCL kernels run on different servers.

## Usage

```bash
python execute_remote_ssh.py <action> [options]
```

### Actions

- `connect`: Test SSH connection
- `execute`: Run command on remote server
- `upload`: Upload files to remote server
- `download`: Download files from remote server
- `build`: Build project on remote server
- `run-tests`: Execute test suite
- `full-workflow`: Complete build-test-download cycle

### Arguments

- `--host`: Remote server hostname or IP
- `--user`: SSH username
- `--key`: Path to SSH private key
- `--password`: SSH password (use key authentication when possible)
- `--port`: SSH port (default: 22)
- `--command`: Command to execute
- `--source`: Source file/directory for upload
- `--destination`: Destination path on remote server
- `--working-dir`: Remote working directory

### Examples

```bash
# Test connection
python execute_remote_ssh.py connect \
    --host gpu-server.example.com \
    --user username \
    --key ~/.ssh/id_rsa

# Execute command
python execute_remote_ssh.py execute \
    --host gpu-server.example.com \
    --user username \
    --key ~/.ssh/id_rsa \
    --command "nvidia-smi"

# Upload project
python execute_remote_ssh.py upload \
    --host gpu-server.example.com \
    --user username \
    --key ~/.ssh/id_rsa \
    --source ./cuda-project \
    --destination ~/remote-workspace/cuda-project

# Run tests
python execute_remote_ssh.py run-tests \
    --host gpu-server.example.com \
    --user username \
    --key ~/.ssh/id_rsa \
    --working-dir ~/remote-workspace/cuda-project/build \
    --command "./cuda_tests --gtest_output=json:results.json"

# Download results
python execute_remote_ssh.py download \
    --host gpu-server.example.com \
    --user username \
    --key ~/.ssh/id_rsa \
    --source ~/remote-workspace/cuda-project/build/cuda_outputs \
    --destination ./local_results
```

## Configuration File Format

```json
{
  "nvidia_server": {
    "host": "gpu1.example.com",
    "user": "username",
    "key_file": "~/.ssh/id_rsa",
    "remote_workspace": "/home/username/cuda-project",
    "local_project_dir": "./cuda",
    "local_results_dir": "./cuda_results",
    "cuda_arch": "80,86",
    "gpu_id": 0
  },
  "intel_server": {
    "host": "gpu2.example.com",
    "user": "username",
    "key_file": "~/.ssh/id_rsa",
    "remote_workspace": "/home/username/sycl-project",
    "local_project_dir": "./sycl",
    "local_results_dir": "./sycl_results",
    "oneapi_path": "/opt/intel/oneapi",
    "sycl_backend": "level_zero"
  }
}
```

## Error Handling

- Connection timeouts with retry logic
- Partial upload/download recovery
- Build failure diagnostics
- Test failure reporting
- Automatic cleanup on error

## Security Best Practices

- Use SSH key authentication (not passwords)
- Verify host keys
- Use secure file permissions (600 for keys)
- Clean up temporary files
- Log all operations

## Related Skills

- `generate-cmake`: Creates build files for remote execution
- `create-cuda-tests`: Generates tests to run remotely
- `compare-numerical-results`: Processes downloaded results

## Skill Metadata

```yaml
name: execute-remote-ssh
version: 1.0.0
category: execution
dependencies:
  - python >= 3.8
  - paramiko >= 2.11.0
inputs:
  - action: string
  - configuration: dict
outputs:
  - execution_result: dict
execution_time: 30-600 seconds (depends on action)
```
