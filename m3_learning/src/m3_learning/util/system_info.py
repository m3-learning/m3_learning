"""Utility to record basic CPU/GPU information for benchmark provenance.

This module was referenced by the 2023 Rapid Fitting benchmark notebooks but
was missing from the packaged SDK; it is reimplemented here using only the
standard library and (optionally) PyTorch, so it works on CPU-only machines,
Google Colab, and multi-GPU servers alike.
"""

import datetime
import os
import platform

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dep of the SDK
    torch = None


class SystemInfo:
    """Collects and saves system (CPU + GPU) information."""

    def get_system_info(self):
        """Gathers CPU and GPU information.

        Returns:
            tuple: (cpu_info dict, gpu_info list of dicts -- empty if no CUDA GPU)
        """

        cpu_info = {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
        }

        gpu_info = []

        if torch is not None:
            cpu_info["torch_version"] = str(torch.__version__)

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info.append(
                        {
                            "name": props.name,
                            "total_memory_GB": round(props.total_memory / 1024**3, 2),
                            "compute_capability": f"{props.major}.{props.minor}",
                        }
                    )

        return cpu_info, gpu_info

    def save_to_file(self, folder, filename, cpu_info, gpu_info):
        """Saves the system information to a text file.

        Args:
            folder (str): folder where the file is written (created if missing)
            filename (str): name of the output text file
            cpu_info (dict): CPU information from `get_system_info`
            gpu_info (list): GPU information from `get_system_info`

        Returns:
            str: path of the written file
        """

        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)

        with open(path, "w") as f:
            f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")

            f.write("CPU info:\n")
            for key, value in cpu_info.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nGPU info:\n")
            if gpu_info:
                for i, gpu in enumerate(gpu_info):
                    f.write(f"  GPU {i}:\n")
                    for key, value in gpu.items():
                        f.write(f"    {key}: {value}\n")
            else:
                f.write("  No CUDA GPU detected\n")

        return path
