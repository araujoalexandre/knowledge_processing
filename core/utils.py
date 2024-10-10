import os
import logging 
import argparse
import subprocess
from typing import Optional, Union, List, Tuple
import torch.distributed as dist


class MessageBuilder:
    """A class for building formatted messages."""

    def __init__(self) -> None:
        self.msg: List[str] = []

    def add(
            self,
            name: str,
            values: Union[list, int, float],
            align: Optional[str] = ">",
            width: Optional[int] = 0,
            format: Optional[str] = None
    ) -> None:
        """
        Add a formatted metric to the message.
        
        Args:
            name: Name of the metric.
            values: Value(s) of the metric.
            align: Alignment of the value(s) (default: right-aligned).
            width: Width of the field for the value(s).
            format: Format specifier for the value(s).
        """
        if name:
            metric_str = "{}: ".format(name)
        else:
            metric_str = ""
        values_str = []
        if type(values) != list:
            values = [values]
        for value in values:
            if format:
                values_str.append("{value:{align}{width}{format}}".format(
                    value=value, align=align, width=width, format=format))
            else:
                values_str.append("{value:{align}{width}}".format(
                    value=value, align=align, width=width))
        metric_str += '/'.join(values_str)
        self.msg.append(metric_str)

    def get_message(self) -> str:
        """Combine all added metrics into a single message and clear the buffer."""
        message = " | ".join(self.msg)
        self.clear()
        return message

    def clear(self) -> None:
        """Clear the message buffer."""
        self.msg = []


def setup_logging(args: argparse.Namespace) -> logging.Logger:
    """
    Set up logging based on the provided command-line arguments.
    
    Args:
        args: ArgsParse object containing logging settings.
    
    Returns:
        A configured logger instance.
    """
    os.makedirs('./slurm_outputs', exist_ok=True)
    format_ = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
    filename = './slurm_outputs/log_{}.logs'.format(args.mode)
    logging.basicConfig(filename=filename, level=20, format=format_, datefmt='%H:%M:%S')
    return logging.getLogger('INFO')


def get_env(ngpus: int) -> Tuple[int, int, int, int, bool, int]:
    """
    Get environment variables for distributed training setup.
    
    Args:
        ngpus: Number of GPUs per node.
    
    Returns:
        Tuple containing rank, local_rank, num_nodes, num_tasks, is_master, and world_size.
    """
    def _local_world_rank() -> int:
        local_rank = os.environ.get('LOCAL_WORLD_SIZE', False)
        if local_rank: return int(local_rank)
        lws = os.environ["SLURM_STEP_TASKS_PER_NODE"]
        lws = lws.split("(")[0]
        return int(lws)

    get = lambda key: os.environ.get(key, False)

    rank: int = int(get('RANK') or get('SLURM_PROCID'))
    local_rank: int = int(get('LOCAL_RANK') or get('SLURM_LOCALID'))
    num_nodes: int = _local_world_rank() 
    num_tasks: int = int(get('WORLD_SIZE') or get('SLURM_STEP_NUM_TASKS'))
    is_master: bool = bool(local_rank == 0)
    world_size: int = num_nodes * ngpus
    assert world_size == num_tasks
    return rank, local_rank, num_nodes, num_tasks, is_master, world_size 


def setup_distributed_training(world_size: int, rank: int) -> None:
    """
    Set up distributed training environment.
    
    Args:
        world_size: Total number of processes.
        rank: Rank of the current process.
    """
    # make sure http proxy are unset, in order for the nodes to communicate
    for var in ['http_proxy', 'https_proxy']:
        if var in os.environ:
            del os.environ[var]
        if var.upper() in os.environ:
            del os.environ[var.upper()]
    # get distributed url 
    cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
    stdout = subprocess.check_output(cmd.split())
    host_name = stdout.decode().splitlines()[0]
    dist_url = f'tcp://{host_name}:9000'
    # setup dist.init_process_group
    dist.init_process_group(backend='nccl', init_method=dist_url,
    world_size=world_size, rank=rank)