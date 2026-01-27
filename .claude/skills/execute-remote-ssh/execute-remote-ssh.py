#!/usr/bin/env python3
"""
Skill: execute-remote-ssh
Execute commands on remote GPU servers via SSH.
"""

import paramiko
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

class RemoteExecutor:
    """Manages SSH connections and remote execution."""

    def __init__(self, host: str, user: str, 
                 key_file: Optional[str] = None,
                 password: Optional[str] = None,
                 port: int = 22):
        """Initialize SSH connection parameters."""
        self.host = host
        self.user = user
        self.key_file = key_file
        self.password = password
        self.port = port
        self.client = None

    def connect(self) -> Dict[str, Any]:
        """Establish SSH connection."""
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect with key or password
            if self.key_file:
                self.client.connect(
                    self.host,
                    port=self.port,
                    username=self.user,
                    key_filename=self.key_file,
                    timeout=30
                )
            elif self.password:
                self.client.connect(
                    self.host,
                    port=self.port,
                    username=self.user,
                    password=self.password,
                    timeout=30
                )
            else:
                raise ValueError("Either key_file or password required")

            return {
                "status": "success",
                "message": f"Connected to {self.user}@{self.host}"
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def execute_command(self, command: str, 
                       working_dir: Optional[str] = None,
                       timeout: int = 300) -> Dict[str, Any]:
        """Execute command on remote server."""
        if not self.client:
            conn_result = self.connect()
            if conn_result["status"] != "success":
                return conn_result

        try:
            # Prepend cd command if working_dir specified
            if working_dir:
                command = f"cd {working_dir} && {command}"

            stdin, stdout, stderr = self.client.exec_command(
                command, 
                timeout=timeout
            )

            exit_status = stdout.channel.recv_exit_status()

            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')

            return {
                "status": "success" if exit_status == 0 else "error",
                "exit_code": exit_status,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "command": command
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "command": command
            }

    def upload_file(self, local_path: str, 
                   remote_path: str) -> Dict[str, Any]:
        """Upload file to remote server."""
        if not self.client:
            conn_result = self.connect()
            if conn_result["status"] != "success":
                return conn_result

        try:
            sftp = self.client.open_sftp()

            # Create remote directory if needed
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                try:
                    sftp.stat(remote_dir)
                except FileNotFoundError:
                    self._create_remote_dir(sftp, remote_dir)

            sftp.put(local_path, remote_path)
            sftp.close()

            return {
                "status": "success",
                "local_path": local_path,
                "remote_path": remote_path
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def upload_directory(self, local_dir: str, 
                        remote_dir: str) -> Dict[str, Any]:
        """Upload directory recursively."""
        if not self.client:
            conn_result = self.connect()
            if conn_result["status"] != "success":
                return conn_result

        try:
            sftp = self.client.open_sftp()

            uploaded_files = []

            for root, dirs, files in os.walk(local_dir):
                # Calculate relative path
                rel_path = os.path.relpath(root, local_dir)
                remote_root = os.path.join(remote_dir, rel_path)

                # Create remote directory
                self._create_remote_dir(sftp, remote_root)

                # Upload files
                for file in files:
                    local_file = os.path.join(root, file)
                    remote_file = os.path.join(remote_root, file)

                    sftp.put(local_file, remote_file)
                    uploaded_files.append(remote_file)

            sftp.close()

            return {
                "status": "success",
                "uploaded_files": len(uploaded_files),
                "files": uploaded_files
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def download_file(self, remote_path: str, 
                     local_path: str) -> Dict[str, Any]:
        """Download file from remote server."""
        if not self.client:
            conn_result = self.connect()
            if conn_result["status"] != "success":
                return conn_result

        try:
            sftp = self.client.open_sftp()

            # Create local directory if needed
            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)

            sftp.get(remote_path, local_path)
            sftp.close()

            return {
                "status": "success",
                "remote_path": remote_path,
                "local_path": local_path
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def download_directory(self, remote_dir: str, 
                          local_dir: str) -> Dict[str, Any]:
        """Download directory recursively."""
        if not self.client:
            conn_result = self.connect()
            if conn_result["status"] != "success":
                return conn_result

        try:
            sftp = self.client.open_sftp()

            downloaded_files = []
            self._download_recursive(sftp, remote_dir, local_dir, 
                                   downloaded_files)

            sftp.close()

            return {
                "status": "success",
                "downloaded_files": len(downloaded_files),
                "files": downloaded_files
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _create_remote_dir(self, sftp, remote_dir: str):
        """Create remote directory recursively."""
        dirs = remote_dir.split('/')
        current = ''

        for d in dirs:
            if not d:
                current = '/'
                continue

            current = os.path.join(current, d)

            try:
                sftp.stat(current)
            except FileNotFoundError:
                sftp.mkdir(current)

    def _download_recursive(self, sftp, remote_dir: str, 
                           local_dir: str, downloaded: List):
        """Helper for recursive directory download."""
        os.makedirs(local_dir, exist_ok=True)

        for item in sftp.listdir_attr(remote_dir):
            remote_path = os.path.join(remote_dir, item.filename)
            local_path = os.path.join(local_dir, item.filename)

            if item.st_mode & 0o040000:  # Directory
                self._download_recursive(sftp, remote_path, 
                                       local_path, downloaded)
            else:  # File
                sftp.get(remote_path, local_path)
                downloaded.append(local_path)

    def close(self):
        """Close SSH connection."""
        if self.client:
            self.client.close()


# High-level workflow functions

def cuda_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute CUDA build-test-download workflow."""
    executor = RemoteExecutor(
        host=config["host"],
        port=config.get("port", 22),
        user=config["user"],
        key_file=config.get("key_file"),
        password=config.get("password")
    )

    results = {
        "stage": "initialization",
        "steps": []
    }

    try:
        # 1. Upload project
        results["stage"] = "upload"
        upload_result = executor.upload_directory(
            config["local_project_dir"],
            config["remote_workspace"]
        )
        results["steps"].append(upload_result)

        if upload_result["status"] != "success":
            return results

        # 2. Configure build
        results["stage"] = "configure"
        configure_cmd = f"""
            mkdir -p {config["remote_workspace"]}/build && \
            cd {config["remote_workspace"]}/build && \
            cmake .. \
                -DCMAKE_CUDA_ARCHITECTURES="{config.get('cuda_arch', '80')}" \
                -DCMAKE_BUILD_TYPE=Release
        """
        configure_result = executor.execute_command(configure_cmd)
        results["steps"].append(configure_result)

        if configure_result["status"] != "success":
            return results

        # 3. Build
        results["stage"] = "build"
        build_cmd = f"""
            cd {config["remote_workspace"]}/build && \
            make -j$(nproc)
        """
        build_result = executor.execute_command(build_cmd, timeout=600)
        results["steps"].append(build_result)

        if build_result["status"] != "success":
            return results

        # 4. Run tests
        results["stage"] = "test"
        test_cmd = f"""
            cd {config["remote_workspace"]}/build && \
            CUDA_VISIBLE_DEVICES={config.get('gpu_id', 0)} \
            ./cuda_tests --gtest_output=json:test_results.json
        """
        test_result = executor.execute_command(test_cmd, timeout=600)
        results["steps"].append(test_result)

        # 5. Download results (even if tests failed)
        results["stage"] = "download"
        download_result = executor.download_directory(
            f"{config['remote_workspace']}/build/cuda_outputs",
            config.get("local_results_dir", "./cuda_results")
        )
        results["steps"].append(download_result)

        results["status"] = "success"
        return results

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        return results

    finally:
        executor.close()


def sycl_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute SYCL build-test-download workflow."""
    executor = RemoteExecutor(
        host=config["host"],
        port=config.get("port", 22),
        user=config["user"],
        key_file=config.get("key_file"),
        password=config.get("password")
    )

    results = {
        "stage": "initialization",
        "steps": []
    }

    try:
        # 1. Upload project
        results["stage"] = "upload"
        upload_result = executor.upload_directory(
            config["local_project_dir"],
            config["remote_workspace"]
        )
        results["steps"].append(upload_result)

        if upload_result["status"] != "success":
            return results

        # 2. Set up oneAPI environment and configure
        results["stage"] = "configure"
        configure_cmd = f"""
            source {config.get('oneapi_path', '/opt/intel/oneapi')}/setvars.sh && \
            mkdir -p {config["remote_workspace"]}/build && \
            cd {config["remote_workspace"]}/build && \
            CXX=icpx cmake .. \
                -DCMAKE_BUILD_TYPE=Release \
                -DSYCL_BACKEND={config.get('sycl_backend', 'level_zero')}
        """
        configure_result = executor.execute_command(configure_cmd)
        results["steps"].append(configure_result)

        if configure_result["status"] != "success":
            return results

        # 3. Build
        results["stage"] = "build"
        build_cmd = f"""
            source {config.get('oneapi_path', '/opt/intel/oneapi')}/setvars.sh && \
            cd {config["remote_workspace"]}/build && \
            make -j$(nproc)
        """
        build_result = executor.execute_command(build_cmd, timeout=600)
        results["steps"].append(build_result)

        if build_result["status"] != "success":
            return results

        # 4. Run tests with device selection
        results["stage"] = "test"
        test_cmd = f"""
            source {config.get('oneapi_path', '/opt/intel/oneapi')}/setvars.sh && \
            cd {config["remote_workspace"]}/build && \
            ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
            ./sycl_tests --gtest_output=json:test_results.json
        """
        test_result = executor.execute_command(test_cmd, timeout=600)
        results["steps"].append(test_result)

        # 5. Download results
        results["stage"] = "download"
        download_result = executor.download_directory(
            f"{config['remote_workspace']}/build/sycl_outputs",
            config.get("local_results_dir", "./sycl_results")
        )
        results["steps"].append(download_result)

        results["status"] = "success"
        return results

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        return results

    finally:
        executor.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Execute commands on remote GPU servers")
    parser.add_argument("action", choices=["connect", "execute", "upload",
                                          "download", "cuda-workflow",
                                          "sycl-workflow"])
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument("--host", help="Remote host")
    parser.add_argument("--user", help="SSH username")
    parser.add_argument("--key", help="Path to SSH private key file")
    parser.add_argument("--password", help="SSH password")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--command", help="Command to execute on remote server")
    parser.add_argument("--source", help="Local file/directory to upload")
    parser.add_argument("--destination", help="Remote file/directory to download to/upload to")
    parser.add_argument("--working-dir", help="Remote working directory for command execution")

    args = parser.parse_args()

    # Load config and execute action
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "host": args.host,
            "user": args.user,
            "key_file": args.key,
            "password": args.password,
            "port": args.port
        }

    # route action to corresponding function
    if args.action in ["connect", "execute", "upload", "download"]:
        executor = RemoteExecutor(
            host=config["host"],
            user=config["user"],
            key_file=config.get("key_file"),
            password=config.get("password"),
            port=config.get("port", 22)
        )

    if args.action == "connect":
        result = executor.connect()
        print(json.dumps(result, indent=2))

    elif args.action == "execute":
        if not args.command:
            print("Error: --command required for execute action", file=sys.stderr)
            sys.exit(1)
        result = executor.execute_command(
            args.command,
            working_dir=args.working_dir
        )
        print(json.dumps(result, indent=2))

    elif args.action == "upload":
        if not args.source or not args.destination:
            print("Error: --source and --destination required for upload action", file=sys.stderr)
            sys.exit(1)

        if os.path.isdir(args.source):
            result = executor.upload_directory(args.source, args.destination)
        else:
            result = executor.upload_file(args.source, args.destination)
        print(json.dumps(result, indent=2))

    elif args.action == "download":
        if not args.source or not args.destination:
            print("Error: --source and --destination required for download action", file=sys.stderr)
            sys.exit(1)

        # Check if source is a directory on remote server
        sftp = executor.client.open_sftp()
        try:
            attr = sftp.stat(args.source)
            if attr.st_mode & 0o040000:  # Directory
                result = executor.download_directory(args.source, args.destination)
            else:
                result = executor.download_file(args.source, args.destination)
        except FileNotFoundError:
            print(f"Error: Remote path {args.source} does not exist", file=sys.stderr)
            sys.exit(1)
        finally:
            sftp.close()

        print(json.dumps(result, indent=2))

    elif args.action == "cuda-workflow":
        workflow_result = cuda_workflow(config['gpu_server'])
        print(json.dumps(workflow_result, indent=2))

    elif args.action == "sycl-workflow":
        workflow_result = sycl_workflow(config['intel_gpu_server'])
        print(json.dumps(workflow_result, indent=2))

    if args.action in ["connect", "execute", "upload", "download"]:
        executor.close()
