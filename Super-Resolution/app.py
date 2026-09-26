import gradio as gr
from image_gen_aux import UpscaleWithModel
from image_gen_aux.utils import load_image

MODELS = {
    "UltraSharp (x4)": "models/UltraSharp/4x-UltraSharp.safetensors",
    "DAT X2": "models/DAT/DAT_x2.safetensors",
    "DAT X3": "models/DAT/DAT_x3.safetensors",
    "DAT X4": "models/DAT/DAT_x4.safetensors",
    "RealPLKSR (x4)": "models/RealPLKSR/4xNomosWebPhoto_RealPLKSR.safetensors",
    "DAT-2 RealWebPhoto (x4)": "models/RealWebPhoto/4xRealWebPhoto_v4_dat2.safetensors",
    "4xRemacri": "models/4xRemacri/4x_foolhardy_Remacri.safetensors",
    "RealESRGAN general v3 (x4)": "models/RealESRGAN/general_x4v3.safetensors",
    "RealESRGAN anime v3 (x4)": "models/RealESRGAN/anime_x4v3.safetensors",
}

'''
UltraSharp (x4): Kim2091/UltraSharp
DAT X2: OzzyGT/DAT_X2
DAT X3: OzzyGT/DAT_X3
DAT X4: OzzyGT/DAT_X4
RealPLKSR (x4): OzzyGT/4xNomosWebPhoto_RealPLKSR
DAT-2 RealWebPhoto (x4): Phips/4xRealWebPhoto_v4_dat2
4xRemacri: OzzyGT/4xRemacri
RealESRGAN general v3 (x4): OzzyGT/RealESRGAN_general_x4v3
RealESRGAN anime v3 (x4): OzzyGT/RealESRGAN_anime_v3
'''


def upscale_image(image, model_selection):
    original = load_image(image)

    upscaler = UpscaleWithModel.from_pretrained(MODELS[model_selection]).to("cuda")
    image = upscaler(original, tiling=True, tile_width=1024, tile_height=1024)

    return original, image


def clear_result():
    return gr.update(value=None)


title = """<h2 align="center">Image Upscaler</h2>"""

with gr.Blocks() as demo:
    gr.HTML(title)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")

            model_selection = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="UltraSharp (x4)",
                label="Model",
            )

            run_button = gr.Button("Upscale")
        with gr.Column():
            result = gr.ImageSlider(
                interactive=False,
                label="Generated Image",
                format="png",
            )

    run_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=upscale_image,
        inputs=[input_image, model_selection],
        outputs=result,
    )

demo.launch(share=False, inbrowser=True)
