# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from heretic.config import Settings


class IsolatedSettings(Settings):
    """
    Settings built from constructor arguments only.

    The real Settings class also reads the command line, environment
    variables, and config.toml from the working directory, none of which
    should leak into unit tests. Validation behaves exactly as in Settings.
    """

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)
