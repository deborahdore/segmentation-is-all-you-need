from os.path import join, abspath, dirname, pardir


# > This class is used to create a directory structure for storing configuration files
class ConfigDirectory:
    def __init__(self):
        self.parent_dir = abspath(join(dirname(__file__), pardir, pardir, pardir, pardir))
        self.module_dir = abspath(join(dirname(__file__), pardir))

        self.video_dir = join(self.module_dir, "videos")
        self.xml_dir = join(self.module_dir, "config", "xml")
        self.model_dir = join(self.module_dir, "saved", "model")

    def get_model_dir(self):
        return self.model_dir

    def get_xml_dir(self):
        return self.xml_dir

    def get_video_dir(self):
        return self.video_dir
