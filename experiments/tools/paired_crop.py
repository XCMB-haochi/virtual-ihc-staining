"""Training-only integer crop, shared by source, label and teacher target."""


def draw_crop(generator, height, width, size=240, probability=.5):
    if not (0<size<=min(height,width) and size%16==0):
        raise ValueError('Crop must fit the image and the model spatial stride')
    if not 0<=probability<=1: raise ValueError('Invalid crop probability')
    if generator.random()>=probability: return None
    top=generator.randrange(height-size+1)
    left=generator.randrange(width-size+1)
    return top,left,size


def apply_crop(image, box):
    if image is None or box is None: return image
    top,left,size=box
    height,width=image.shape[-2:]
    if top<0 or left<0 or top+size>height or left+size>width:
        raise ValueError('Crop must remain inside the original image')
    return image[...,top:top+size,left:left+size].contiguous()
