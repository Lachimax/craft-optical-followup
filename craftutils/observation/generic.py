import os

import craftutils.params as p

import craftutils.utils as u


class Generic:
    def __init__(
            self,
            name: str = None,
            param_path: str = None,
            data_path: str = None,
            **kwargs
    ):
        self.name = name
        self.param_path = param_path

        self.data_path = None
        self.data_path_relative = None
        if data_path is not None:
            self.data_path = os.path.join(p.data_dir, data_path)
            self.data_path_relative = data_path
        if data_path is not None:
            os.makedirs(self.data_path, exist_ok=True)

        self.output_file = None  # This will be set during the load_output_file call

        self.param_file = kwargs

    def update_output_file(self):
        p.update_output_file(self)

    def load_output_file(self, **kwargs):
        return p.load_output_file(self)

    def _output_dict(self):
        return {}

    def _updateable(self):
        return {"name": self.name}

    def update_param_file(self, param: str):
        p_dict = self._updateable()
        if param not in p_dict:
            raise ValueError(f"Either {param} is not a valid parameter, or it has not been configured.")
        if self.param_path is None:
            raise ValueError("param_path has not been set.")
        else:
            params = p.load_params(self.param_path)
        params[param] = p_dict[param]
        p.save_params(file=self.param_path, dictionary=params)
