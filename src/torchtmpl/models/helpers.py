from typing import Iterable, List

def chunks(data: Iterable, sizes: List[int]):
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk