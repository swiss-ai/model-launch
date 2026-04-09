import argparse
import inspect
import os
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, cast

import keyring
import questionary
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from questionary import Question

GetValueFn = Callable[[str], str | None]
OptionsDict = dict[str, tuple[str, str]]
ValidatorFn = Callable[[str], bool]

_KEYRING_SERVICE = "swiss_ai_model_launch"
_KEYRING_PLACEHOLDER = "__keyring__"


class _Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError  # pragma: no cover

    async def aconfigure(
        self,
        get_value: GetValueFn | None = None,
        args: argparse.Namespace | None = None,
        non_interactive: bool = False,
    ) -> None:
        raise NotImplementedError  # pragma: no cover

    def get_value(self, name: str) -> str | None:
        raise NotImplementedError  # pragma: no cover

    def get_non_none_value(self, name: str) -> str:
        value = self.get_value(name)
        if value is None:
            raise ValueError(f"Configuration '{name}' is not set.")
        return value

    def set_value(self, name: str, value: str) -> None:
        raise NotImplementedError  # pragma: no cover


class _ResolvableConfiguration(_Configuration):
    value: str | None = None
    prompt: str | None = Field(default=None, exclude=True)
    env_var: str | None = Field(default=None, exclude=True)
    expose_as_arg: bool = Field(default=True, exclude=True)

    def _get_question(self) -> Question:
        raise NotImplementedError  # pragma: no cover

    def _on_answer(self) -> None:
        pass

    def _try_resolve_without_prompt(self, args: argparse.Namespace | None) -> str | None:
        if self.expose_as_arg and args is not None:
            arg_value = getattr(args, self.name, None)
            if arg_value is not None:
                return str(arg_value)
        if self.env_var is not None:
            env_value = os.environ.get(self.env_var)
            if env_value is not None:
                return env_value
        return None

    async def aconfigure(
        self,
        get_value: GetValueFn | None = None,
        args: argparse.Namespace | None = None,
        non_interactive: bool = False,
    ) -> None:
        resolved = self._try_resolve_without_prompt(args)
        if resolved is not None:
            self.value = resolved
            self._on_answer()
            return
        if non_interactive:
            raise ValueError(f"Missing required argument --{self.name.replace('_', '-')} (non-interactive mode)")
        self.value = await self._get_question().ask_async()
        self._on_answer()

    def get_value(self, name: str) -> str | None:
        if self.name != name:
            raise KeyError(f"Configuration '{name}' not found.")
        return self.value

    def set_value(self, name: str, value: str) -> None:
        if self.name != name:
            raise KeyError(f"Configuration '{name}' not found.")
        self.value = value


class TextConfiguration(_ResolvableConfiguration):
    type: Literal["text"] = "text"
    default: str | None = None
    default_factory: Callable[[], Awaitable[str | None]] | Callable[[GetValueFn], Awaitable[str | None]] | None = Field(
        default=None, exclude=True
    )
    validator: ValidatorFn | Callable[[str, GetValueFn], bool] | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _check_default_source(self) -> "TextConfiguration":
        if self.default is not None and self.default_factory is not None:
            raise ValueError("Provide only one of `default` or `default_factory`.")
        return self

    async def _resolve_default(self, get_value: GetValueFn | None) -> str | None:
        if self.default_factory is None:
            return self.default
        if bool(inspect.signature(self.default_factory).parameters):
            if get_value is None:
                raise RuntimeError(
                    f"TextConfiguration '{self.name}': `default_factory` requires "
                    "context but no get_value was provided."
                )
            return await cast(Callable[[GetValueFn], Awaitable[str | None]], self.default_factory)(get_value)
        return await cast(Callable[[], Awaitable[str | None]], self.default_factory)()

    def _resolve_validator(self, get_value: GetValueFn | None) -> ValidatorFn | None:
        if self.validator is None:
            return None
        if len(inspect.signature(self.validator).parameters) > 1:
            if get_value is None:
                raise RuntimeError(
                    f"TextConfiguration '{self.name}': `validator` requires context but no get_value was provided."
                )
            validator = cast(Callable[[str, GetValueFn], bool], self.validator)
            return lambda value: validator(value, get_value)
        return cast(ValidatorFn, self.validator)

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        if not self.expose_as_arg:
            return
        parser.add_argument(
            f"--{self.name.replace('_', '-')}",
            dest=self.name,
            default=None,
            metavar=self.name.upper(),
            help=self.prompt or self.name,
        )

    async def aconfigure(
        self,
        get_value: GetValueFn | None = None,
        args: argparse.Namespace | None = None,
        non_interactive: bool = False,
    ) -> None:
        resolved = self._try_resolve_without_prompt(args)
        if resolved is not None:
            validator = self._resolve_validator(get_value)
            if validator is not None and not validator(resolved):
                raise ValueError(f"Invalid value for --{self.name.replace('_', '-')}: {resolved!r}")
            self.value = resolved
            self._on_answer()
            return
        if non_interactive:
            raise ValueError(f"Missing required argument --{self.name.replace('_', '-')} (non-interactive mode)")
        self.value = await questionary.text(
            self.prompt or self.name,
            default=await self._resolve_default(get_value) or "",
            validate=self._resolve_validator(get_value),
        ).ask_async()
        self._on_answer()


class PasswordConfiguration(_ResolvableConfiguration):
    type: Literal["password"] = "password"

    @field_serializer("value")
    def serialize_value(self, value: str | None) -> str | None:
        return _KEYRING_PLACEHOLDER if value is not None else value

    @model_validator(mode="after")
    def load_from_keyring(self) -> "PasswordConfiguration":
        if self.value == _KEYRING_PLACEHOLDER:
            self.value = keyring.get_password(_KEYRING_SERVICE, self.name)
        return self

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        if not self.expose_as_arg:
            return
        parser.add_argument(
            f"--{self.name.replace('_', '-')}",
            dest=self.name,
            default=None,
            metavar="PASSWORD",
            help=self.prompt or self.name,
        )

    def _get_question(self) -> Question:
        return questionary.password(self.prompt or self.name)

    def _on_answer(self) -> None:
        if self.value is not None:
            keyring.set_password(_KEYRING_SERVICE, self.name, self.value)


class OptionsConfiguration(_ResolvableConfiguration):
    type: Literal["options"] = "options"
    options: OptionsDict | None = None
    options_factory: Callable[[], Awaitable[OptionsDict]] | Callable[[GetValueFn], Awaitable[OptionsDict]] | None = (
        Field(default=None, exclude=True)
    )

    @model_validator(mode="after")
    def _check_options_source(self) -> "OptionsConfiguration":
        if self.options is None and self.options_factory is None:
            raise ValueError("Either `options` or `options_factory` must be provided.")
        if self.options is not None and self.options_factory is not None:
            raise ValueError("Provide only one of `options` or `options_factory`.")
        return self

    def _build_question(self, options: OptionsDict) -> Question:
        return questionary.select(
            self.prompt or self.name,
            choices=[
                questionary.Choice(title=title, value=value, description=description)
                for value, (title, description) in options.items()
            ],
        )

    def _factory_wants_context(self) -> bool:
        if self.options_factory is None:
            raise RuntimeError(
                f"OptionsConfiguration '{self.name}': `_factory_wants_context` called but `options_factory` is None."
            )
        return bool(inspect.signature(self.options_factory).parameters)

    async def _resolve_options(self, get_value: GetValueFn | None = None) -> OptionsDict:
        if self.options is not None:
            return self.options
        if self._factory_wants_context():
            if get_value is None:
                raise RuntimeError(
                    f"OptionsConfiguration '{self.name}': `options_factory` requires "
                    "context but no get_value was provided."
                )
            return await cast(Callable[[GetValueFn], Awaitable[OptionsDict]], self.options_factory)(get_value)
        return await cast(Callable[[], Awaitable[OptionsDict]], self.options_factory)()

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        if not self.expose_as_arg:
            return
        kwargs: dict[str, Any] = {
            "dest": self.name,
            "default": None,
            "help": self.prompt or self.name,
        }
        if self.options:
            kwargs["choices"] = list(self.options.keys())
        else:
            kwargs["metavar"] = self.name.upper()
        parser.add_argument(f"--{self.name.replace('_', '-')}", **kwargs)

    async def aconfigure(
        self,
        get_value: GetValueFn | None = None,
        args: argparse.Namespace | None = None,
        non_interactive: bool = False,
    ) -> None:
        resolved = self._try_resolve_without_prompt(args)
        if resolved is not None:
            options = await self._resolve_options(get_value)
            if not options or resolved in options:
                self.value = resolved
                self._on_answer()
                return
        if non_interactive:
            raise ValueError(f"Missing required argument --{self.name.replace('_', '-')} (non-interactive mode)")
        options = await self._resolve_options(get_value)
        if len(options) == 1:
            self.value = next(iter(options))
        else:
            self.value = await self._build_question(options).ask_async()
        self._on_answer()


class ChainConfiguration(_Configuration):
    type: Literal["chain"] = "chain"
    chain: list["Configuration"]

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        for configuration in self.chain:
            configuration.add_to_parser(parser)

    async def aconfigure(
        self,
        get_value: GetValueFn | None = None,
        args: argparse.Namespace | None = None,
        non_interactive: bool = False,
    ) -> None:
        for configuration in self.chain:
            await configuration.aconfigure(get_value or self.get_value, args, non_interactive=non_interactive)

    def get_value(self, name: str) -> str | None:
        for configuration in self.chain:
            try:
                return configuration.get_value(name)
            except KeyError:
                continue
        raise KeyError(f"Configuration '{name}' not found.")

    def set_value(self, name: str, value: str) -> None:
        for configuration in self.chain:
            try:
                configuration.set_value(name, value)
                return
            except KeyError:
                continue
        raise KeyError(f"Configuration '{name}' not found.")


class BranchConfiguration(_Configuration):
    type: Literal["branch"] = "branch"
    head_configuration: OptionsConfiguration
    branches: dict[str, "Configuration | None"]

    def _resolve_branch(self) -> "Configuration | None":
        branch_key = self.head_configuration.value
        if branch_key not in self.branches:
            raise RuntimeError(f"Invalid choice: {branch_key}. Valid choices are: {list(self.branches.keys())}.")
        return self.branches[branch_key]

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        self.head_configuration.add_to_parser(parser)
        for branch in self.branches.values():
            if branch is not None:
                branch.add_to_parser(parser)

    async def aconfigure(
        self,
        get_value: GetValueFn | None = None,
        args: argparse.Namespace | None = None,
        non_interactive: bool = False,
    ) -> None:
        await self.head_configuration.aconfigure(get_value, args, non_interactive=non_interactive)
        branch = self._resolve_branch()
        if branch is not None:
            await branch.aconfigure(get_value, args, non_interactive=non_interactive)

    def get_value(self, name: str) -> str | None:
        try:
            return self.head_configuration.get_value(name)
        except KeyError:
            pass
        for branch in self.branches.values():
            if branch is not None:
                try:
                    return branch.get_value(name)
                except KeyError:
                    continue
        raise KeyError(f"Configuration '{name}' not found.")

    def set_value(self, name: str, value: str) -> None:
        try:
            self.head_configuration.set_value(name, value)
            return
        except KeyError:
            pass
        for branch in self.branches.values():
            if branch is not None:
                try:
                    branch.set_value(name, value)
                    return
                except KeyError:
                    continue
        raise KeyError(f"Configuration '{name}' not found.")


Configuration = Annotated[
    TextConfiguration | PasswordConfiguration | OptionsConfiguration | ChainConfiguration | BranchConfiguration,
    Field(discriminator="type"),
]

ChainConfiguration.model_rebuild()
BranchConfiguration.model_rebuild()
