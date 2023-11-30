from huggingface_hub import hf_hub_download

filename = "__assets__/poses_skeleton_gifs/dance1_corr.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(
    repo_type="space",
    repo_id=repo_id,
    filename=filename,
    local_dir=".",
    local_dir_use_symlinks=False)