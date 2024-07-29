import multiprocessing
import time
from queue import Empty


class SingleShotQueue:
    def __init__(self):
        self._queue = multiprocessing.Queue(maxsize=1)
        self._lock = multiprocessing.Lock()

    def put(self, item):
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put(item)

    def get(self):
        return self._queue.get()


def producer(q: SingleShotQueue):
    for i in range(15):
        q.put(i)
        print(f"Put item: {i}")
        if i % 3 == 0:
            time.sleep(1)
    q.put("done")


def consumer(q: SingleShotQueue):
    while True:
        item = q.get()
        print(f"Got item: {item}")
        time.sleep(0.5)
        if item == "done":
            break


if __name__ == "__main__":
    queue = SingleShotQueue()

    producer_process = multiprocessing.Process(target=producer, args=(queue,))
    consumer_process = multiprocessing.Process(target=consumer, args=(queue,))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
