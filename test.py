import threading
import subprocess
import random

def run_main_with_seed(seed):
    random.seed(seed)
    subprocess.run(['python3', 'main.py'])

threads = []
for i in range(5):
    seed = random.randint(0, 10000)
    thread = threading.Thread(target=run_main_with_seed, args=(seed,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()