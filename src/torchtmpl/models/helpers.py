from collections.abc import Iterable


def chunks(data: Iterable, sizes: list[int]):
    curr = 0
    for size in sizes:
        chunk = data[curr : curr + size]
        curr += size
        yield chunk
