The test suite now passes on macOS hosts: the runtime-preflight contract test
compares the resolved interpreter path (crossing the `/var` -> `/private/var`
symlink), and the loopback-address network-guard test skips when `127.0.0.2`
is not bound on `lo0` (macOS default) instead of erroring during collection.
