from os.path import join, abspath, dirname, pardir
from pathlib import Path


class ConfigDirectory:
    def __init__(self):
        self.parent_dir = abspath(join(dirname(__file__), pardir, pardir, pardir, pardir))
        self.module_dir = abspath(join(dirname(__file__), pardir))

        self.dataset_dir = join(self.parent_dir, "dataset")

        self.save_dir = join(self.module_dir, "saved")

        self.plot_dir = join(self.save_dir, "plot")

        self.models_dir = join(self.save_dir, "models")
        self.logs_dir = join(self.save_dir, "logs")

        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.plot_dir).mkdir(parents=True, exist_ok=True)

    def get_model_dir(self):
        return self.models_dir

    def get_plot_dir(self):
        return self.plot_dir

    def get_dataset_dir(self):
        return self.dataset_dir

    def get_logs_dir(self):
        return self.logs_dir
