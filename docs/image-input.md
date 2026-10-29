# How each binding prepares an input image

Orientation, transparency and bit depth are handled before segmentation. The
bindings mostly agree; the exceptions are listed here.

## Orientation

A file the binding opens itself, and in JavaScript an encoded `Buffer`, is turned
the way it is displayed, from its EXIF Orientation tag. An image you pass in
already decoded (a PIL `Image`, a sharp instance, an `image.Image`) is not
re-oriented.

Go and Rust read the tag only from JPEG and from a PNG eXIf chunk. Python and
JavaScript also read it from the other formats their imaging libraries decode,
such as TIFF and WebP. Python alone also honours an eXIf chunk placed after the
image data, an eXIf chunk with a stray `Exif\0\0` prefix, and an ImageMagick
"Raw profile type exif" text chunk.

## Transparency

Transparency is composited onto white before the conversion to grey for every
input, opened by the binding or decoded by you, so a transparent background is
not read as black. The one exception is a NumPy array handed straight to
Python's `LineSegmenter.segment`, which is converted as given. Compositing a
partly transparent pixel can differ by one grey level between bindings.

## 16-bit greyscale

Python clips a 16-bit greyscale PNG when it converts it to 8 bits, so a page
with a background of 60000 and ink of 6000 reads as all white in Python and
keeps its contrast in the other three.
