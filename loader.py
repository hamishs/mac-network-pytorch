import yaml


class Loader:
    def __init__(self, input_obj):
        if isinstance(input_obj, dict):
            input_dict = input_obj
        else:
            input_dict = input_obj.__dict__
        custom_params = {k:v for k,v in input_dict.items() if v is not None}

        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)[input_dict['config']]
        
        combined_dict = {**config, **custom_params}
        self._generate_loader(combined_dict)

    def _generate_loader(self, combined_dict):
        for key, val in combined_dict.items():
            if isinstance(val, (int, str, list, dict, float, tuple)):
                setattr(self, key, val)

