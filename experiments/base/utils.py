import torch
 
"""  misc methods """
def monitor_gpu_memory(device, logger):
    """
    监控指定GPU设备的显存使用情况和显存保留情况，并使用logger.info()打印出来。

    参数:
        device (torch.device): 要监控的GPU设备。
    """
    # 记录显存使用情况
    allocated_memory = torch.cuda.memory_allocated(device)

    # 记录显存保留情况
    reserved_memory = torch.cuda.memory_reserved(device)
    max_reserved_memory = torch.cuda.max_memory_reserved(device)
    
    # 使用logger.info()直接打印显存使用情况和显存保留情况
    logger.info(f"Device: {device}, Allocated Memory: {allocated_memory / 1024**3:.2f} GB, Max Reserved Memory: {max_reserved_memory / 1024**3:.2f} GB, Reserved Memory: {reserved_memory / 1024**3:.2f} GB")

    logger.info(f'memory summary: {torch.cuda.memory_summary(device)}')