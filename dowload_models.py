from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="username/arc3-models",
    local_dir=".",
    local_dir_use_symlinks=False
)