#!/usr/bin/env python3
"""
Remote Test Runner for CUDA and SYCL Validation

This script connects to NVIDIA and Intel GPU servers via SSH, builds and executes
test suites, collects outputs and benchmark data, and performs numerical comparisons.

Usage:
    python remote_runner.py --config config.json
    python remote_runner.py --nvidia-server user@nvidia-host --intel-server user@intel-host
"""

import argparse
import json
import logging
import os
import paramiko
import scp
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import numpy as np
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for a remote server"""
    host: str
    user: str
    password: Optional[str] = None
    key_file: Optional[str] = None
    port: int = 22
    work_dir: str = "~/sycl_migration_tests"


@dataclass
class TestResults:
    """Test execution results"""
    server_name: str
    build_success: bool
    test_success: bool
    execution_time: float
    output_files: List[str]
    benchmark_data: Dict
    error_logs: List[str]


class RemoteTestRunner:
    """Main class for running remote tests"""

    def __init__(self, config_file: Optional[str] = None):
        self.setup_logging()
        self.config = self.load_config(config_file) if config_file else None
        self.results = {}

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"remote_runner_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise

    def create_ssh_client(self, server: ServerConfig) -> paramiko.SSHClient:
        """Create and configure SSH client"""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            if server.key_file:
                client.connect(
                    hostname=server.host,
                    port=server.port,
                    username=server.user,
                    key_filename=server.key_file,
                    timeout=30
                )
            else:
                client.connect(
                    hostname=server.host,
                    port=server.port,
                    username=server.user,
                    password=server.password,
                    timeout=30
                )
            self.logger.info(f"Connected to {server.user}@{server.host}")
            return client
        except Exception as e:
            self.logger.error(f"Failed to connect to {server.host}: {e}")
            raise

    def setup_remote_environment(self, client: paramiko.SSHClient, server: ServerConfig) -> bool:
        """Setup remote environment and directories"""
        try:
            # Create work directory
            commands = [
                f"mkdir -p {server.work_dir}",
                f"mkdir -p {server.work_dir}/cuda",
                f"mkdir -p {server.work_dir}/sycl",
                f"mkdir -p {server.work_dir}/results",
                f"mkdir -p {server.work_dir}/logs",
            ]

            for cmd in commands:
                stdin, stdout, stderr = client.exec_command(cmd)
                exit_status = stdout.channel.recv_exit_status()
                if exit_status != 0:
                    self.logger.error(f"Failed to execute '{cmd}': {stderr.read().decode()}")
                    return False

            self.logger.info(f"Remote environment setup on {server.host}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to setup remote environment: {e}")
            return False

    def transfer_code_to_remote(self, client: paramiko.SSHClient, local_path: str,
                               remote_path: str, server: ServerConfig) -> bool:
        """Transfer code files to remote server"""
        try:
            with scp.SCPClient(client.get_transport()) as scp_client:
                # Transfer directory recursively
                if Path(local_path).is_dir():
                    scp_client.put(local_path, remote_path, recursive=True)
                else:
                    scp_client.put(local_path, remote_path)

            self.logger.info(f"Transferred {local_path} to {server.host}:{remote_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to transfer files to {server.host}: {e}")
            return False

    def build_remote_project(self, client: paramiko.SSHClient, project_type: str,
                           work_dir: str, server_name: str) -> Tuple[bool, List[str]]:
        """Build project on remote server"""
        logs = []
        try:
            if project_type == "cuda":
                build_cmd = f"cd {work_dir}/cuda && mkdir -p build && cd build && cmake .. && make -j$(nproc)"
            else:  # sycl
                build_cmd = f"cd {work_dir}/sycl && mkdir -p build && cd build && cmake .. && make -j$(nproc)"

            self.logger.info(f"Building {project_type} on {server_name}")

            stdin, stdout, stderr = client.exec_command(build_cmd)
            exit_status = stdout.channel.recv_exit_status()

            output = stdout.read().decode()
            error = stderr.read().decode()

            logs.extend([
                f"Build command: {build_cmd}",
                f"Exit status: {exit_status}",
                f"Output:\n{output}",
                f"Error:\n{error}"
            ])

            if exit_status == 0:
                self.logger.info(f"Successfully built {project_type} on {server_name}")
                return True, logs
            else:
                self.logger.error(f"Failed to build {project_type} on {server_name}")
                return False, logs

        except Exception as e:
            self.logger.error(f"Build failed: {e}")
            logs.append(f"Exception: {str(e)}")
            return False, logs

    def run_remote_tests(self, client: paramiko.SSHClient, project_type: str,
                        work_dir: str, server_name: str) -> Tuple[bool, Dict, List[str]]:
        """Run tests on remote server and collect results"""
        start_time = time.time()
        test_logs = []
        benchmark_data = {}

        try:
            if project_type == "cuda":
                test_executable = f"{work_dir}/cuda/build/tests/test_runner"
                results_dir = f"{work_dir}/results/cuda"
            else:  # sycl
                test_executable = f"{work_dir}/sycl/build/tests/test_runner"
                results_dir = f"{work_dir}/results/sycl"

            # Create results directory
            client.exec_command(f"mkdir -p {results_dir}")

            # Run tests with environment setup
            env_setup = ""
            if project_type == "cuda":
                env_setup = "source ~/.bashrc && export CUDA_VISIBLE_DEVICES=0"
            else:  # sycl
                env_setup = "source ~/.bashrc && export ONEAPI_DEVICE_SELECTOR=level_zero:gpu"

            test_cmd = f"{env_setup} && cd {results_dir} && {test_executable} --gtest_output=json:test_results.json --benchmark_out=benchmark.json"

            self.logger.info(f"Running {project_type} tests on {server_name}")

            stdin, stdout, stderr = client.exec_command(test_cmd, timeout=300)
            exit_status = stdout.channel.recv_exit_status()

            output = stdout.read().decode()
            error = stderr.read().decode()

            execution_time = time.time() - start_time

            test_logs.extend([
                f"Test command: {test_cmd}",
                f"Exit status: {exit_status}",
                f"Execution time: {execution_time:.2f} seconds",
                f"Output:\n{output}",
                f"Error:\n{error}"
            ])

            # Collect benchmark data if available
            try:
                bench_cmd = f"cat {results_dir}/benchmark.json"
                stdin, stdout, stderr = client.exec_command(bench_cmd)
                benchmark_output = stdout.read().decode()
                if benchmark_output:
                    benchmark_data = json.loads(benchmark_output)
            except:
                self.logger.warning(f"Could not parse benchmark data from {server_name}")

            if exit_status == 0:
                self.logger.info(f"Tests completed successfully on {server_name}")
                return True, benchmark_data, test_logs
            else:
                self.logger.error(f"Tests failed on {server_name}")
                return False, benchmark_data, test_logs

        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            test_logs.append(f"Exception: {str(e)}")
            return False, benchmark_data, test_logs

    def collect_results_from_remote(self, client: paramiko.SSHClient, project_type: str,
                                  work_dir: str, local_results_dir: str, server_name: str) -> bool:
        """Collect result files from remote server"""
        try:
            remote_results_dir = f"{work_dir}/results/{project_type}"

            # Create local results directory
            Path(local_results_dir).mkdir(parents=True, exist_ok=True)

            with scp.SCPClient(client.get_transport()) as scp_client:
                # Get all result files
                client.exec_command(f"find {remote_results_dir} -type f -name '*.json' -o -name '*.csv' -o -name '*.log'")
                stdin, stdout, stderr = client.exec_command(f"find {remote_results_dir} -type f")

                files = stdout.read().decode().strip().split('\n')
                for remote_file in files:
                    if remote_file.strip():
                        local_file = Path(local_results_dir) / Path(remote_file).name
                        scp_client.get(remote_file, str(local_file))

            self.logger.info(f"Collected results from {server_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to collect results from {server_name}: {e}")
            return False

    def run_complete_test_suite(self) -> Dict[str, TestResults]:
        """Run complete test suite on both servers"""
        if not self.config:
            raise ValueError("Configuration not loaded")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results") / f"test_run_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Parse server configurations
        nvidia_config = ServerConfig(**self.config['nvidia_server'])
        intel_config = ServerConfig(**self.config['intel_server'])

        servers = {
            'nvidia': nvidia_config,
            'intel': intel_config
        }

        project_types = ['cuda', 'sycl']

        for server_name, server_config in servers.items():
            self.logger.info(f"Processing {server_name} server: {server_config.host}")

            client = None
            try:
                client = self.create_ssh_client(server_config)

                # Setup environment
                if not self.setup_remote_environment(client, server_config):
                    continue

                # Transfer code
                local_code_dir = "."
                remote_code_dir = f"{server_config.work_dir}"

                if not self.transfer_code_to_remote(client, local_code_dir, remote_code_dir, server_config):
                    continue

                # Test each project type
                for project_type in project_types:
                    self.logger.info(f"Building and testing {project_type} on {server_name}")

                    # Build
                    build_success, build_logs = self.build_remote_project(
                        client, project_type, server_config.work_dir, server_name
                    )

                    # Run tests if build successful
                    test_success, benchmark_data, test_logs = False, {}, []
                    if build_success:
                        test_success, benchmark_data, test_logs = self.run_remote_tests(
                            client, project_type, server_config.work_dir, server_name
                        )

                    # Collect results
                    local_results = str(results_dir / f"{server_name}_{project_type}")
                    self.collect_results_from_remote(
                        client, project_type, server_config.work_dir, local_results, server_name
                    )

                    # Store results
                    result_key = f"{server_name}_{project_type}"
                    self.results[result_key] = TestResults(
                        server_name=server_name,
                        build_success=build_success,
                        test_success=test_success,
                        execution_time=benchmark_data.get('execution_time', 0) if benchmark_data else 0,
                        output_files=[],
                        benchmark_data=benchmark_data,
                        error_logs=build_logs + test_logs
                    )

            except Exception as e:
                self.logger.error(f"Error processing {server_name}: {e}")
            finally:
                if client:
                    client.close()

        return self.results

    def save_results_summary(self, results_dir: str = "results"):
        """Save summary of all test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = Path(results_dir) / f"summary_{timestamp}.json"

        summary = {
            "timestamp": timestamp,
            "results": {}
        }

        for key, result in self.results.items():
            summary["results"][key] = {
                "server_name": result.server_name,
                "build_success": result.build_success,
                "test_success": result.test_success,
                "execution_time": result.execution_time,
                "benchmark_data": result.benchmark_data,
                "error_count": len(result.error_logs)
            }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Results summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Remote Test Runner for CUDA and SYCL")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--nvidia-server", type=str, help="NVIDIA server SSH config")
    parser.add_argument("--intel-server", type=str, help="Intel server SSH config")

    args = parser.parse_args()

    runner = RemoteTestRunner(args.config)

    try:
        results = runner.run_complete_test_suite()
        runner.save_results_summary()

        # Print summary
        print("\n=== Test Execution Summary ===")
        for key, result in results.items():
            status = "PASS" if result.build_success and result.test_success else "FAIL"
            print(f"{key}: {status} (Build: {result.build_success}, Tests: {result.test_success})")
            if result.execution_time > 0:
                print(f"  Execution time: {result.execution_time:.2f} seconds")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()