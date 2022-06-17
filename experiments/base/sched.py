from timm.scheduler import create_scheduler


def get_scheduler(args, optimizer):
    return create_scheduler(args, optimizer)
