from timm.scheduler import create_scheduler


def get_schedule(args, optimizer):
    return create_scheduler(args, optimizer)
