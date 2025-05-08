import time
from multiprocessing import Process

def worker_process(index):
    print(f"Worker process {index} started")
    for i in range(1, 4):
        print(f"Worker process {index}: {i}")
        time.sleep(1)
    print(f"Worker process {index} finished")

if __name__ == "__main__":
    print("main process here")
    processes = []
    for i in range(1, 4):
        p = Process(target=worker_process, args=(i,))
        processes.append(p)
        p.start()

    for i in range(1, 4):
        print(f"Main process: {i}")
        time.sleep(1)

    for p in processes:
        p.join()