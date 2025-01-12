import pandas as pd
from threading import Timer
from typing import Dict

class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

class MultiTimer(object):
    def __init__(self, freqs=[]) -> None:
        self.timers: Dict[str, Timer] = {}
        self.finished = {}
        for freq in freqs:
            self.register(freq)
    
    def register(self, freq: str):
        if freq in self.timers:
            return
        interval = pd.to_timedelta(freq).total_seconds()
        self.timers[freq] = RepeatTimer(interval, self.onTime, args=[freq])
        self.finished[freq] = False
        self.timers[freq].start()
    
    def cancel(self, freq: str=None):
        if freq is not None:
            freqs = [freq]
        else:
            freqs = list(self.timers.keys())
        for freq in freqs:
            self.timers[freq].cancel()
            del self.timers[freq]
            del self.finished[freq]

    def onTime(self, freq: str):
        self.finished[freq] = True
    
    def check(self, freq: str):
        if self.finished[freq]:
            self.finished[freq] = False
            return True
        return False

# MultiTimer(['5s', '1h'])