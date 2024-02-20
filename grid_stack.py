from diffusers.utils import load_image, make_image_grid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--list',
    required=True,
    nargs="*",
    type=str,
    help='image list')

parser.add_argument(
    '--jpg',
    action='store_true',
    help='if true, use jpeg format')

parser.add_argument(
    '--rows',
    type=int,
    default=1,
    help='rows')

parser.add_argument(
    "--resize",
    type=int,
    help="size(width and height)"
)
args = parser.parse_args()

image_list =[load_image(x) for x in args.list]

nrow = args.rows
ncol = (len(image_list) -1) // nrow + 1

result = make_image_grid(
    images=image_list,
    rows=nrow,
    cols=ncol,
    resize=args.resize
)

if args.jpg:
    result.save("stack.jpg")
else:
    result.save("stack.png")