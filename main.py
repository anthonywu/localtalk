#!/usr/bin/env python3
"""
Main entry point for Local Talk App.

Run this file to start the voice assistant.

If using this for the first time, a system popup box
will come up to let you affirm the usage of the microphone.
"""

import tqdm

from localtalk.cli import main

tqdm.tqdm.disable = True

if __name__ == "__main__":
    main()
