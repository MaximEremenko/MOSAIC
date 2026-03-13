from __future__ import annotations

from typing import Protocol

from core.factories.configuration_processor_factory import ConfigurationProcessorFactoryProvider
from core.interfaces.base_interfaces import IConfigurationProcessorFactory


class ConfigurationProcessorRegistry(Protocol):
    def get_factory(self, file_type: str) -> IConfigurationProcessorFactory:
        ...


class DefaultConfigurationProcessorRegistry:
    def get_factory(self, file_type: str) -> IConfigurationProcessorFactory:
        return ConfigurationProcessorFactoryProvider.get_factory(file_type)
