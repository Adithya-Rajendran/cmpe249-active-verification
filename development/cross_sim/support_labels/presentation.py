"""Create a reference-box-only viewing aid while preserving both full RGB inputs."""
from PIL import Image, ImageDraw, ImageFont


def card(reference, query, box, destination):
    width, height = 1200, 675
    panels = Image.new("RGB", (1280, 756), "white")
    draw = ImageDraw.Draw(panels)
    fonts = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
    title = ImageFont.truetype(fonts + "-Bold.ttf", 20)
    label = ImageFont.truetype(fonts + ".ttf", 15)
    tick = ImageFont.truetype(fonts + ".ttf", 13)
    x0, y0, x1, y1 = box
    crop_w = min(width, max(420, x1-x0+280))
    crop_h = min(height, max(round(crop_w*.58), y1-y0+190))
    left = min(width-crop_w, max(0, (x0+x1-crop_w)//2))
    top = min(height-crop_h, max(0, (y0+y1-crop_h)//2))
    crop = (left, top, left+crop_w, top+crop_h)
    for index, (path, name) in enumerate(((reference, "Reference"), (query, "Revisit"))):
        with Image.open(path) as src:
            im = src.convert("RGB")
        if im.size != (width, height):
            raise ValueError("Expected a 1200 by 675 input")
        origin = 10 if index == 0 else 650
        draw.text((origin, 3), name, fill="#15293d", font=title)
        if index == 0:
            ImageDraw.Draw(im).rectangle((x0, y0, x1-1, y1-1), outline="#ffc900", width=3)
        panels.paste(im.resize((618, 348), Image.Resampling.LANCZOS), (origin, 32))
        scale = min(573/crop_w, 283/crop_h)
        sw, sh = round(crop_w*scale), round(crop_h*scale)
        px, py = origin+38, 440
        draw.text((origin, 394), name + " detail in full-image pixel coordinates", font=label, fill="#354658")
        panels.paste(im.crop(crop).resize((sw, sh), Image.Resampling.NEAREST), (px, py))
        for x in range(((left+49)//50)*50, left+crop_w, 50):
            xx = px + (x-left+.5)*sw/crop_w
            draw.line((xx,py-5,xx,py-1), fill="#364657")
            draw.text((xx, py-7), str(x), font=tick, anchor="mb", fill="#293e4f")
        for y in range(((top+49)//50)*50, top+crop_h, 50):
            yy = py + (y-top+.5)*sh/crop_h
            draw.line((px-5,yy,px-1,yy), fill="#364657")
            draw.text((px-7,yy), str(y), font=tick, anchor="rm", fill="#293e4f")
    panels.save(destination)
    return list(crop)
