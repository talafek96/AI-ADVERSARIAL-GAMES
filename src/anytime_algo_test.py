import threading as th
import time


def do_stuff(secs: float):
    secs = float(secs)
    time.sleep(secs)
    print(f'Did stuff for {secs:.2} seconds!')



class AnytimeAlgorithm(th.Thread):
    """
    Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition.
    """

    def __init__(self,  *args, **kwargs):
        super(AnytimeAlgorithm, self).__init__(*args, **kwargs)
        self._stop_event = th.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

class MyTask(AnytimeAlgorithm):
    def __init__(self, *args, **kwargs):
        super(MyTask, self).__init__(*args, **kwargs)
        self.print_lock = th.Lock()
        self.num_iters = 0

    def run(self):
        seconds = float(self._kwargs['seconds'])
        self.num_iters = 0

        while not self.stopped():
            time.sleep(seconds)
            with self.print_lock:
                if self.stopped():
                    print('Was about to tell you something but I was stopped!')
                    return
                print(f'Did stuff for {seconds:.2} seconds!')
                self.num_iters += 1

def main():
    secs = 3
    print('Hi main!')

    t = MyTask(kwargs={'seconds': secs})
    t.start()

    print(f'Main has started do_stuff for {secs} seconds!')
    time.sleep(2 * secs - 0.2)

    with t.print_lock:
        t.stop()
        print('main has stopped MyTask!')
    print(f'MyTask has done a total of {t.num_iters} iterations')
    time.sleep(0.5)
    assert not t.is_alive()
    print(f'MyTask has been verified dead and done a total of {t.num_iters} iterations')

if __name__ == '__main__':
    main()
