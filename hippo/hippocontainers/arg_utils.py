import functools


def apply_to_args(transformations):
    """
    Decorator that applies a function to specific named arguments before passing them
    to the decorated function.

    :param transformations: A dictionary mapping argument names to functions.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Modify named arguments based on transformations
            for arg_name, transform in transformations.items():
                if arg_name in kwargs:
                    kwargs[arg_name] = transform(kwargs[arg_name])

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Example usage:
@apply_to_args({"x": lambda v: v * 2, "y": str.upper})
def process(x, y, z):
    return f"x: {x}, y: {y}, z: {z}"


print(process(x=3, y="hello", z=True))  # Outputs: x: 6, y: HELLO, z: True


# Constructed decorator that always doubles 'x'
def double_x(func):
    return apply_to_args({"x": lambda v: v * 2})(func)


@double_x
def compute(x, y):
    return f"x: {x}, y: {y}"