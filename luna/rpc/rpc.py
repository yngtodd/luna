from torch.distributed import rpc


def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    return rpc.rpc_sync(
        rref.owner(),
        _call_method,
        args=[method, rref] + list(args),
        kwargs=kwargs
    )


def _parameter_rrefs(module):
    r"""
    Create one RRef for each parameter in the given local module, and return a
    list of RRefs.
    """
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs
