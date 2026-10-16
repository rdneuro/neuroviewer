#! /bin/bash
# AppImage entry point (python-appimage template; {{ ... }} is filled at build time).
# The fallback libraries (libxcb-cursor and friends, needed by Qt >= 6.5 on X11)
# go first in the search path only for this process; -I isolates the bundled
# Python from the user's site-packages and PYTHONPATH (no clash with a Spyder
# or conda Qt installation).
export LD_LIBRARY_PATH="${APPDIR}/usr/lib/neuroviewer-extra${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
{{ python-executable }} -I -m neuroviewer "$@"
