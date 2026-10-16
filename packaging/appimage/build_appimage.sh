#!/usr/bin/env bash
# Build Neuroviewer-<version>-x86_64.AppImage with python-appimage.
#
#   bash packaging/appimage/build_appimage.sh            # from the local source tree
#   bash packaging/appimage/build_appimage.sh --pypi     # from the published PyPI release
#
# Environment:
#   PY_VERSION   Python inside the AppImage (default 3.12)
#
# Needs: python3 with pip, internet access (GitHub for the python-appimage base
# image and appimagetool, PyPI for the wheels). Build on the OLDEST distro you
# want to support (e.g. Ubuntu 22.04): the bundled X11 fallback libraries are
# copied from the build machine. The result needs glibc >= 2.28 on the target
# (PySide6 wheels are manylinux_2_28) and the host's OpenGL driver.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
PY_VERSION="${PY_VERSION:-3.12}"
FROM_PYPI=0
[[ "${1:-}" == "--pypi" ]] && FROM_PYPI=1

VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "${ROOT}/src/neuroviewer/__init__.py")"
BUILD="${HERE}/build"
RECIPE="${BUILD}/neuroviewer"
echo "[appimage] Neuroviewer ${VERSION} | Python ${PY_VERSION} | source: $([[ ${FROM_PYPI} == 1 ]] && echo PyPI || echo local tree)"

rm -rf "${BUILD}"
mkdir -p "${RECIPE}"

python3 -m pip install --quiet --upgrade "python-appimage>=1.4" build

# ------------------------------------------------------------------ recipe
if [[ ${FROM_PYPI} == 1 ]]; then
    echo "neuroviewer==${VERSION}" > "${RECIPE}/requirements.txt"
else
    python3 -m build --wheel --outdir "${BUILD}/dist" "${ROOT}"
    WHEEL="$(ls "${BUILD}"/dist/neuroviewer-"${VERSION}"-*.whl)"
    echo "${WHEEL}" > "${RECIPE}/requirements.txt"
fi
cp "${HERE}/entrypoint.sh" "${RECIPE}/entrypoint.sh"
cp "${ROOT}/src/neuroviewer/resources/neuroviewer.desktop" "${RECIPE}/neuroviewer.desktop"
cp "${ROOT}/src/neuroviewer/resources/neuroviewer.png" "${RECIPE}/neuroviewer.png"
sed -e "s/@VERSION@/${VERSION}/" -e "s/@DATE@/$(date +%Y-%m-%d)/" \
    "${HERE}/neuroviewer.appdata.xml" > "${RECIPE}/neuroviewer.appdata.xml"

# ------------------------------------------------------------------ AppDir
cd "${BUILD}"
python3 -m python_appimage build app --no-packaging -p "${PY_VERSION}" "${RECIPE}"
APPDIR="${BUILD}/Neuroviewer-$(uname -m)"
[[ -d "${APPDIR}" ]] || { echo "[appimage] AppDir not found: ${APPDIR}"; exit 1; }

# X11 fallback libraries for Qt >= 6.5 (often missing on minimal installs).
EXTRA="${APPDIR}/usr/lib/neuroviewer-extra"
mkdir -p "${EXTRA}"
for lib in libxcb-cursor.so.0 libxcb-image.so.0 libxcb-render-util.so.0 libxcb-util.so.1; do
    src="$(ldconfig -p | awk -v l="${lib}" '$1 == l && tolower($0) ~ /x86-64|aarch64/ {print $NF; exit}')"
    if [[ -n "${src}" ]]; then
        cp -L "${src}" "${EXTRA}/"
        echo "[appimage] bundled ${lib}"
    else
        echo "[appimage] WARNING: ${lib} not found on the build machine (apt install libxcb-cursor0); not bundled"
    fi
done

# Trim what a viewer never needs (tests, caches, Qt translations stay).
find "${APPDIR}/opt" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${APPDIR}/opt" -type d \( -name "tests" -o -name "test" \) -path "*site-packages/*" -prune -exec rm -rf {} + || true

# ------------------------------------------------------------------ pack
OUT="Neuroviewer-${VERSION}-$(uname -m).AppImage"
python3 - "${APPDIR}" "${OUT}" <<'EOF'
import sys
from python_appimage.appimage import build_appimage
build_appimage(sys.argv[1], destination=sys.argv[2])
EOF
mv "${OUT}" "${ROOT}/${OUT}"
chmod +x "${ROOT}/${OUT}"
echo "[appimage] done: ${ROOT}/${OUT} ($(du -h "${ROOT}/${OUT}" | cut -f1))"
