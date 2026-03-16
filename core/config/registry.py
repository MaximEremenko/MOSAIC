from __future__ import annotations

from typing import Protocol

from core.config.contracts.base_interfaces import (
    IConfigurationProcessorFactory,
)
from core.config.factories.configuration_factory import (
    ConfigurationProcessorFactoryProvider,
)


class ConfigurationProcessorRegistry(Protocol):
    def get_factory(self, file_type: str) -> IConfigurationProcessorFactory:
        ...


class DefaultConfigurationProcessorRegistry:
    def get_factory(self, file_type: str) -> IConfigurationProcessorFactory:
        return ConfigurationProcessorFactoryProvider.get_factory(file_type)
