import functools
import time


def retry_on_fail(retries=2, delay=0.5):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries:
                        raise
                    print(
                        f"Retrying {func.__name__} in {delay}s... ({i + 1}/{retries})"
                    )
                    time.sleep(delay)
            return None

        return wrapper

    return decorator
