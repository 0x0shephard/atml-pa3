#!/usr/bin/env python3
"""Isolate Kaggle OAuth credentials, including refresh writes, by profile."""
import os
from pathlib import Path
from kagglesdk.kaggle_creds import KaggleCredentials

profile = os.environ.get("PA3_KAGGLE_PROFILE")
if profile:
    path = str(Path(profile).expanduser().resolve() / "credentials.json")
    KaggleCredentials.DEFAULT_CREDENTIALS_FILE = path
    # SDK default arguments are bound at definition time, separately from the constant.
    KaggleCredentials.save.__defaults__ = (path,)
    KaggleCredentials.delete.__defaults__ = (path,)
    os.environ["KAGGLE_CONFIG_DIR"] = profile

from kaggle.cli import main
if __name__ == "__main__":
    main()
