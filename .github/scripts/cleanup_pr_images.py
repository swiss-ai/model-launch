#!/usr/bin/env python3

import asyncio
import os
import sys

import firecrest as f7t
from build_image import _CAPSTOR_IMAGES, _CHANNEL_RE, _RELEASE_CHANNEL

from swiss_ai_model_launch.launchers.firecrest_auth import build_client_from_env


async def _rm(client: f7t.v2.AsyncFirecrest, system_name: str, account: str, path: str) -> None:
    print(f"Removing {path}")
    try:
        await client.rm(system_name=system_name, path=path, account=account, blocking=True)
    except Exception as e:  # noqa: BLE001
        # The directory is absent whenever the PR touched no image, which is
        # the common case. Cleanup must never fail the workflow.
        print(f"  Skipped: {e}")


async def main(channel: str) -> int:
    firecrest_url = os.environ["SML_FIRECREST_URL"]
    system_name = os.environ["SML_SYSTEM"]

    client = build_client_from_env(firecrest_url)

    user_info = await client.userinfo(system_name)
    username = user_info["user"]["name"]
    account = user_info["group"]["name"]

    # capstor is shared across both build clusters, so one endpoint suffices.
    await _rm(client, system_name, account, f"{_CAPSTOR_IMAGES}/{channel}")

    # Build contexts are per-cluster, but home is shared with the same layout;
    # remove whatever this endpoint can see.
    builds_dir = f"/users/{username}/.sml/image-builds"
    try:
        entries = await client.list_files(system_name=system_name, path=builds_dir, account=account)
    except Exception as e:  # noqa: BLE001
        print(f"Could not list {builds_dir}: {e}")
        return 0

    for entry in entries:
        name = entry.get("name", "")
        if name.endswith(f"-{channel}"):
            await _rm(client, system_name, account, f"{builds_dir}/{name}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <channel>", file=sys.stderr)
        sys.exit(1)
    channel_arg = sys.argv[1]
    if not _CHANNEL_RE.match(channel_arg) or channel_arg == _RELEASE_CHANNEL:
        # Guard hard: this script deletes directories by path.
        print(f"Refusing to clean up channel '{channel_arg}' (expected 'pr-<number>')", file=sys.stderr)
        sys.exit(1)
    sys.exit(asyncio.run(main(channel_arg)))
