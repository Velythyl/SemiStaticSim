from functools import wraps
from collections.abc import Iterable


def monadic(monad_argnums=None, monad_argnames=None):
    if monad_argnums and monad_argnames:
        raise ValueError("Specify either monad_argnums or monad_argnames, not both.")

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from inspect import signature
            sig = signature(func)
            param_names = list(sig.parameters.keys())

            new_args = list(args)

            # Default case: All arguments except 'self' (for instance methods)
            if monad_argnums is None and monad_argnames is None:
                indices_to_transform = [i for i in range(len(args)) if param_names[i] != "self"]
            elif monad_argnums is not None:
                indices_to_transform = monad_argnums
            else:
                indices_to_transform = [param_names.index(name) for name in monad_argnames if name in param_names]

            for i in indices_to_transform:
                if i < len(new_args) and not isinstance(new_args[i], Iterable):
                    new_args[i] = [new_args[i]]

            for key in kwargs:
                if monad_argnames and key in monad_argnames and not isinstance(kwargs[key], Iterable):
                    kwargs[key] = [kwargs[key]]

            return func(*new_args, **kwargs)

        return wrapper

    return decorator

if __name__ == "__main__":
    