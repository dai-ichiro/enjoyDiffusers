from diffusers.utils import load_image, make_image_grid
from pathlib import Path
from typer import Typer, Option

app = Typer()

@app.command()
def grid(
    folder: str=Option(..., help="folder path"),
    jpg: bool=Option(default=False, help="png or jpg"),
    rows: int=Option(default=1, help="rows"),
    resize: int=Option(default=None, help="size(width and height)")
):

    image_list = [load_image(x.as_posix()) for x in Path(folder).glob("*")]

    nrow = rows
    ncol = (len(image_list) -1) // nrow + 1

    result = make_image_grid(
        images=image_list,
        rows=nrow,
        cols=ncol,
        resize=resize
    )

    if jpg:
        result.save("stack.jpg")
    else:
        result.save("stack.png")

if __name__=="__main__":
    app()
