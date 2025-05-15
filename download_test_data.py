from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="DMindAI/DMind_Benchmark",
    repo_type="dataset",
    allow_patterns="test_data/**",
    local_dir="."
) 