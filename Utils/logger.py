import logging
from Config import config
from discord import SyncWebhook

class Logger:
    def __init__(self, name, discord: bool = config.DCLOGGERACTIVATE) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(config.LOGGERLEVEL)
        
        # console handler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(config.CONSOLELOGGERLEVEL)
        self.ch.setFormatter(logging.Formatter(config.CONSOLELOGGERFORMAT,
                                               config.CONSOLELOGGERDATEFORMAT))
        self._logger.addHandler(self.ch)
        
        if discord:
            # discord handler
            self.dch = DiscordHandler(config.DCLOGGERURL)
            self.dch.setLevel(config.DCLOGGERLEVEL)
            self.dch.setFormatter(logging.Formatter(config.DCLOGGERFORMAT,
                                                config.DCLOGGERDATEFORMAT))
            self._logger.addHandler(self.dch)
    
    def __getattr__(self, name):
        # Check if the attribute exists in self.a
        if hasattr(self._logger, name):
            # Get the attribute from self.a and return it as a bound method
            return getattr(self._logger, name).__get__(self._logger, self._logger.__class__)

        # If the attribute doesn't exist in self.a, raise AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class DiscordHandler(logging.Handler):
    def __init__(self, url):
        logging.Handler.__init__(self)
        self.url = url
        self.webhook = SyncWebhook.from_url(self.url)
    
    def emit(self, record):
        msg = self.format(record)
        text = f"```\n{msg}\n```"
        self.webhook.send(content=text)