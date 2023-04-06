from PIL import Image

a = Image.open('0.png')

b = Image.open('sample.jpg')
b.putalpha(alpha=255)

out = Image.alpha_composite(a, b)