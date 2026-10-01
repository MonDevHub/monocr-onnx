"""Reading an image the way it is displayed.

Shared by the predictor and the segmenter, so every greyscale conversion in
this package composites transparency onto white and every file it opens has
its EXIF orientation applied.
"""

from pathlib import Path

from PIL import Image, ImageOps


def load_image(source):
    """Open ``source`` the right way up.

    A path is opened and its EXIF Orientation tag applied, so a phone photo
    stored sideways is read as it is displayed. Pillow does not do this on its
    own: ``Image.open`` returns the stored pixels, and for tags 2-8 those are
    mirrored or rotated. An ``Image`` the caller already holds is returned as
    given -- whoever decoded it owns its orientation, and transposing it again
    here could undo a rotation they already applied.

    The other three bindings make the same split: they orient what they decode
    from a file and leave a decoded image alone.
    """
    if isinstance(source, (str, Path)):
        return ImageOps.exif_transpose(Image.open(source))
    return source


def to_grey(img):
    """Return ``img`` as 8-bit greyscale, composited onto white first if it is transparent.

    Converting an RGBA image straight to ``L`` drops the alpha channel, so a
    transparent background stored as (0, 0, 0, 0) -- the common encoding --
    becomes black, and dark text on it disappears into the background. The web
    and mobile apps flatten onto white before reading; this does the same.

    Only images with a pixel that is not fully opaque are composited. Anything
    else takes the conversion it always took, so opaque input is byte-identical
    to before.
    """
    if img.has_transparency_data:
        rgba = img.convert("RGBA")
        if rgba.getchannel("A").getextrema()[0] < 255:
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(white, rgba).convert("L")
    if img.mode != "L":
        return img.convert("L")
    return img
