Remote buildx / registry-exporter guidance

Why use a remote buildx or registry-exporter?

- Large image builds that download many heavy native wheels (torch, pypdfium2, opencv) can exhaust GitHub Actions runner disk and network time.
- Using a registry-exporter (push directly to GHCR) or a remote buildkit host delegates heavy layer storage and export to a more suitable place and avoids creating large local tar archives on the runner.

Options

1) Registry exporter (recommended for hosted CI)

- The existing `docker/build-push-action@v5` with `push: true` already uses the registry exporter: buildx writes layers directly to the registry (GHCR) instead of producing a local image tar and loading it into Docker on the runner.
- Use short-lived tags for validation (e.g. `pr-<run_id>`) and remove them after validation with the GHCR API.

Example (already used in this repo):

uses: docker/build-push-action@v5
with:
  context: .
  platforms: linux/amd64
  push: true
  tags: ghcr.io/${{ github.repository }}:pr-${{ github.run_id }}
  build-args: |
    PDF_PARSER=marker
    SKIP_PARSER_VENVS=false
  cache-from: type=gha
  cache-to: type=gha,mode=max

2) Remote buildkitd (recommended when you want to avoid pushing to GHCR during validation)

- Provision a remote BuildKit host (VM or managed service) with enough disk and CPU, and expose it over TCP or SSH.
- Secure it: only allow connections from CI IPs or use SSH keys stored in GitHub Secrets.
- Use `docker/setup-buildx-action@v3` and configure the remote builder. Example approach (SSH):

  - Store a deploy SSH key in `secrets.BUILDKIT_SSH_KEY` and the remote host `user@host:port` in `secrets.BUILDKIT_SSH_HOST`.
  - In a workflow step, add the SSH key to an ssh-agent and configure buildx to create a builder that connects via SSH. The exact setup depends on your buildkitd configuration.

Example snippet (conceptual):

- name: Setup SSH for remote buildkit
  run: |
    mkdir -p ~/.ssh
    echo "${{ secrets.BUILDKIT_SSH_KEY }}" > ~/.ssh/id_rsa
    chmod 600 ~/.ssh/id_rsa
    ssh-keyscan -H ${{ secrets.BUILDKIT_SSH_HOST }} >> ~/.ssh/known_hosts

- name: Create remote buildx builder
  uses: docker/setup-buildx-action@v3
  with:
    driver: remote
    driver-opts: |
      network=host
    # note: you may need to pass an endpoint/host depending on your buildx configuration

- name: Build on remote builder and push to registry
  uses: docker/build-push-action@v5
  with:
    builder: my-remote-builder
    context: .
    push: true
    tags: ghcr.io/${{ github.repository }}:pr-${{ github.run_id }}

Caveats: exact configuration varies by BuildKit deployment. Consult your cloud/VM provider docs for setting up buildkitd and exposing it securely.

3) Self-hosted runners

- Another approach is to use self-hosted runners with larger disk and memory specifically for heavy builds.
- Pros: full control over disk, cache, and network.
- Cons: management overhead, security considerations, and potential cost.

Recommendations

- For most GitHub-hosted CI setups, prefer the registry-exporter approach (push: true) with short-lived tags and cleanup via the GHCR API.
- For extremely heavy or frequent builds, provision a remote buildkitd and connect via SSH (store SSH key in secrets). Use BuildKit's GC and cache features.

If you want I can:
- add a sample workflow that demonstrates an SSH-driven remote buildx configuration (you'll need to provide a remote host and an SSH key stored in repo/organization secrets), or
- add a README section with exact commands to provision a small AWS/GCP VM with buildkitd and systemd unit to run it.
