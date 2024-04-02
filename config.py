import yaml

class TrainConfig:
    def __init__(self):
        with open('config.yaml', 'r') as file:
            config_dict = yaml.safe_load(file)
        self.config_dict = config_dict

    def __getattr__(self, attr):
        try:
            return self.config_dict[attr]
        except KeyError:
            raise AttributeError(f"Attribute '{attr}' not found in the configuration")