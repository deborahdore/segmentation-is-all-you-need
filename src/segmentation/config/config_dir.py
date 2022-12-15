from os.path import join, abspath, dirname, pardir
from pathlib import Path


class ConfigDirectory:
    def __init__(self):
        self.base_dir = abspath(join(dirname(__file__), pardir, pardir, pardir))
        self.module_dir = abspath(join(dirname(__file__), pardir))

        self.dataset_dir = join(self.base_dir, "dataset")
        self.plot_dir = join(self.module_dir, "saved", "plot")
        self.report_dir = join(self.module_dir, "saved", "report")

        Path(self.plot_dir).mkdir(parents=True, exist_ok=True)
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)

    def get_dataset_dir(self):
        return self.dataset_dir

    def get_plot_dir(self):
        return self.plot_dir

    def get_report_dir(self):
        return self.report_dir
