import craftutils.utils as u


def data_select(directory: str, pipe_to: str = None, **params):
    u.system_command("dataselect", [directory], **params, pipe_to=pipe_to)
