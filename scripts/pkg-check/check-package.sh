#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="all"
WORK_DIR="$REPO_DIR/build/pkg-check"
PACKAGE_PREFIX="$WORK_DIR/install"
ARCHIVE_PATH="$WORK_DIR/visioncpp-package.tgz"
SOURCE_INPUT=""

usage() {
  cat <<EOF
Usage: $0 [all|build|check] [options]

Modes:
  all     Run build and check (default)
  build   Build and install package with static ggml
  check   Build and run consumer project against an installed package

Options:
  --work-dir <dir>        Working directory (default: $WORK_DIR)
  --prefix <dir>          Install/extract prefix (default: <work-dir>/install)
  --archive <file>        Archive path (default: <work-dir>/visioncpp-package.tgz)
  --from-prefix <dir>     Use an existing installed prefix for check step
  --from-archive <file>   Extract and use an existing archive for check step
  -h, --help              Show this help

Examples:
  $0 all
  $0 build --archive /tmp/visioncpp.tgz
  $0 check --from-archive /tmp/visioncpp.tgz --work-dir /tmp/pkg-check
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    all|build|check)
      MODE="$1"
      shift
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --prefix)
      PACKAGE_PREFIX="$2"
      shift 2
      ;;
    --archive)
      ARCHIVE_PATH="$2"
      shift 2
      ;;
    --from-prefix)
      SOURCE_INPUT="prefix:$2"
      shift 2
      ;;
    --from-archive)
      SOURCE_INPUT="archive:$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

SRC_BUILD_DIR="$WORK_DIR/source-build"
PKG_BUILD_DIR="$WORK_DIR/consumer-build"
CONSUMER_SRC_DIR="$REPO_DIR/scripts/pkg-check"
MODEL_REL_PATH="share/visioncpp/Depth-Anything-V2-Small-F16.gguf"

step1() {
  echo "[pkg-check] Build: configure and build package producer"
  rm -rf "$SRC_BUILD_DIR"
  cmake -S "$REPO_DIR" -B "$SRC_BUILD_DIR" \
    -D CMAKE_BUILD_TYPE=Release \
    -D VISP_VULKAN=OFF \
    -D VISP_TESTS=OFF \
    -D VISP_STATIC_GGML=ON \
    -D VISP_INSTALL_MODELS=ON

  cmake --build "$SRC_BUILD_DIR"

  echo "[pkg-check] Build: install to $PACKAGE_PREFIX"
  rm -rf "$PACKAGE_PREFIX"
  cmake --install "$SRC_BUILD_DIR" --prefix "$PACKAGE_PREFIX"

  echo "[pkg-check] Build: archive package at $ARCHIVE_PATH"
  mkdir -p "$(dirname "$ARCHIVE_PATH")"
  tar -C "$PACKAGE_PREFIX" -czf "$ARCHIVE_PATH" .
}

prepare_step2_prefix() {
  mkdir -p "$WORK_DIR"

  if [[ -n "$SOURCE_INPUT" ]]; then
    case "$SOURCE_INPUT" in
      prefix:*)
        PACKAGE_PREFIX="${SOURCE_INPUT#prefix:}"
        ;;
      archive:*)
        local src_archive="${SOURCE_INPUT#archive:}"
        rm -rf "$PACKAGE_PREFIX"
        mkdir -p "$PACKAGE_PREFIX"
        tar -C "$PACKAGE_PREFIX" -xzf "$src_archive"
        ;;
      *)
        echo "Invalid SOURCE_INPUT: $SOURCE_INPUT" >&2
        exit 2
        ;;
    esac
    return
  fi

  if [[ "$MODE" == "check" ]]; then
    if [[ -f "$ARCHIVE_PATH" ]]; then
      rm -rf "$PACKAGE_PREFIX"
      mkdir -p "$PACKAGE_PREFIX"
      tar -C "$PACKAGE_PREFIX" -xzf "$ARCHIVE_PATH"
      return
    fi
    if [[ ! -d "$PACKAGE_PREFIX" ]]; then
      echo "Check needs an installed package. Run build first or pass --from-prefix/--from-archive." >&2
      exit 2
    fi
  fi
}

step2() {
  prepare_step2_prefix

  local model_path="$PACKAGE_PREFIX/$MODEL_REL_PATH"
  if [[ ! -f "$model_path" ]]; then
    echo "Model not found: $model_path" >&2
    exit 1
  fi

  echo "[pkg-check] Check: run vision-cli"
  LD_LIBRARY_PATH="$PACKAGE_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$PACKAGE_PREFIX/bin/vision-cli" depthany -m "$model_path" \
    -i "$REPO_DIR/tests/input/wardrobe.jpg" \
    -o "$WORK_DIR/cli_output.png"
  if [[ ! -f "$WORK_DIR/cli_output.png" ]]; then
    echo "vision-cli failed to produce output image" >&2
    exit 1
  fi

  echo "[pkg-check] Check: configure consumer project"
  cmake -S "$CONSUMER_SRC_DIR" -B "$PKG_BUILD_DIR" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_PREFIX_PATH="$PACKAGE_PREFIX"

  echo "[pkg-check] Check: build consumer project"
  cmake --build "$PKG_BUILD_DIR"

  echo "[pkg-check] Check: run consumer inference"
  LD_LIBRARY_PATH="$PACKAGE_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$PKG_BUILD_DIR/pkg-check" "$model_path"
}

mkdir -p "$WORK_DIR"

case "$MODE" in
  all)
    step1
    step2
    ;;
  build)
    step1
    ;;
  check)
    step2
    ;;
  *)
    echo "Invalid mode: $MODE" >&2
    exit 2
    ;;
esac

echo "[pkg-check] done"
