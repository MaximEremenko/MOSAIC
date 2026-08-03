# readers/lammps_file_reader.py

from core.config.contracts.base_interfaces import IFileReader


class LammpsFileReader(IFileReader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read(self) -> str:
        with open(self.file_path, "r") as file:
            content = file.read()
        return content
