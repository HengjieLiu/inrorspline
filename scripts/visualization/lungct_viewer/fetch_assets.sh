#!/usr/bin/env bash
set -euo pipefail
target="${1:?Usage: fetch_assets.sh EXTERNAL_BUNDLE_ROOT}"
mkdir -p "$target/deps"
download() {
  local name="$1" url="$2" expected="$3"
  if [[ ! -f "$target/deps/$name" ]]; then
    curl --fail --location --retry 2 "$url" -o "$target/deps/$name"
  fi
  local actual
  actual="$(sha256sum "$target/deps/$name")"
  [[ "${actual%% *}" == "$expected" ]] || { echo "Dependency checksum mismatch: $name" >&2; exit 1; }
}
download three.module.js https://unpkg.com/three@0.180.0/build/three.module.js c8211c69345d2e9949dc7a8ac969380497aa0600a5a8ac6a459c8cd02dd9cb8a
download three.core.js https://unpkg.com/three@0.180.0/build/three.core.js eb077d2417f61d3e6d9264c317cabc4ea35769ed6b0ab533067292a550784c20
download OrbitControls.js https://unpkg.com/three@0.180.0/examples/jsm/controls/OrbitControls.js b97879c748170baadeb3fb84cea1ffdf4674e283dc06042f34e2acb95a76042c
download LICENSE https://unpkg.com/three@0.180.0/LICENSE bfe119ea4fd413f5f7ca3fcd63adb0c4a073ed39daa2fe7d3e6b769e21272601
echo 'Three.js0.180.0 assets verified; runtime requires no CDN.'
