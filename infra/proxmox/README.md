Proxmox helper scripts

This folder contains a small helper script to create a cloud-init based VM in Proxmox that you can use as a GitHub Actions self-hosted runner.

create-runner-vm.sh

- Run this on the Proxmox host (requires sudo/root) and the `qm` and `pvesm` tools.
- Examples:

  # Create VM with default minimal snippet
  sudo ./create-runner-vm.sh 210 github-runner-01 local https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img

  # Create VM and render a cloud-init template with vars
  sudo ./create-runner-vm.sh 211 github-runner-02 local https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img \
    /path/to/self-hosted-runner-userdata.yaml /path/to/runner.vars

The second form will run `infra/proxmox/render-cloudinit.sh` to substitute placeholders of the form {{VAR}} in the template using variables from the provided vars file (KEY=VALUE lines). The rendered user-data will be uploaded to the storage snippets and attached to the VM.
Notes & next steps
- The script uploads a minimal `user-data` snippet by default; for production use the templates in `infra/cloud-init/`.
- To render templates with variables, prepare a vars file with KEY=VALUE lines (e.g. OWNER=your-org, REPO=your-repo, RUNNER_NAME=runner-01, RUNNER_TOKEN=<token>, RUNNER_REMOVE_PAT=<PAT>). Then call the script with the template path and vars file as the 5th and 6th arguments.
- For secure operation, prefer the `self-hosted-runner-userdata-fetch-token.yaml` template that obtains a short-lived registration token from an internal token service instead of embedding a token in the vars file.
- The script assumes storage name (e.g. `local`) supports `snippets` namespace. Adjust for your Proxmox storage setup.
Proxmox helper scripts

This folder contains a small helper script to create a cloud-init based VM in Proxmox that you can use as a GitHub Actions self-hosted runner.

create-runner-vm.sh

- Run this on the Proxmox host (requires sudo/root) and the `qm` and `pvesm` tools.
- Example:

  sudo ./create-runner-vm.sh 210 github-runner-01 local https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img

What the script does
- Downloads an Ubuntu Jammy cloud image into `/var/lib/vz/template/iso` (if missing).
- Creates a VM (ID and name provided).
- Imports the image as a disk into the chosen storage.
- Uploads a minimal cloud-init `user-data` snippet to `<storage>:snippets/user-data-<VMID>.yaml` and configures the VM to use it.
- Starts the VM.

Notes & next steps
- The script uploads a minimal `user-data` snippet; after VM creation you should update the cloud-init snippet or use the `infra/cloud-init/` templates in this repo.
- For production runners, replace the minimal snippet with `infra/cloud-init/self-hosted-runner-userdata-fetch-token.yaml` (upload it to storage and point `--cicustom` to it) and ensure the token endpoint and other placeholders are provided.
- The script assumes storage name (e.g. `local`) supports `snippets` namespace. Adjust for your Proxmox storage setup.

Security
- Use internal network access for token endpoints and do not expose a token service publicly.
- Consider using ephemeral runners and deregistering on shutdown for security and scalability.
