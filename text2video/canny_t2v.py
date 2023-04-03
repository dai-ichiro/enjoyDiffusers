from mymodel import Model

model = Model()

model.process_controlnet_canny(
    prompt='a beautiful girl running',
    video_path = 'run.mp4', 
    low_threshold=50,
    high_threshold=50,
    save_path = 'canny_result.mp4',
    chunk_size = 2)