import logging

from ruamel.yaml import YAML


class YParams:
    """Yaml file parser."""

    def __init__(self, yaml_filename, config_name, print_params=False) -> None:
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        self.params = {}

        if print_params:
            print("------------------ Configuration ------------------")

        with open(yaml_filename, encoding="utf-8") as _file:
            for key, val in YAML().load(_file)[config_name].items():
                if print_params:
                    print(key, val)
                if val == "None":
                    val = None

                self.params[key] = val
                self.__setattr__(key, val)

        if print_params:
            print("---------------------------------------------------")

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val) -> None:
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key) -> bool:
        return key in self.params

    def update_params(self, config) -> None:
        for key, val in config.items():
            self.params[key] = val
            self.__setattr__(key, val)

    def log(self) -> None:
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: " + str(self._yaml_filename))
        logging.info("Configuration name: " + str(self._config_name))
        for key, val in self.params.items():
            logging.info(str(key) + " " + str(val))
        logging.info("---------------------------------------------------")
