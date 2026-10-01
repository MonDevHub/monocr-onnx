// Package imageio reads an image the way it is displayed: EXIF orientation
// applied, transparency composited onto white. The predictor, the segmenter and
// the page paths all go through it, so every greyscale conversion in this
// module sees the same pixels.
package imageio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	// Load decodes what the bindings have always decoded; registering the two
	// decoders here keeps that true for a caller who imports only this package.
	_ "image/jpeg"
	_ "image/png"
	"os"

	"golang.org/x/image/draw"
)

// Load decodes the image file at path the right way up.
//
// Go's image/jpeg and image/png return the stored pixels and ignore the EXIF
// Orientation tag, so a phone photo stored sideways was read sideways: for tags
// 2-8 the model saw a mirrored or rotated picture. The tag is read here from a
// JPEG APP1 segment or a PNG eXIf chunk and applied.
//
// Only a file this binding decodes itself is oriented. An image.Image handed
// to Predictor.Predict was decoded by the caller, who owns its orientation; the other
// three bindings make the same split.
func Load(path string) (image.Image, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %v", err)
	}
	return ApplyOrientation(img, ExifOrientation(data)), nil
}

// ExifOrientation returns the EXIF Orientation tag (1-8) of an encoded JPEG or
// PNG, or 1 when there is none or it cannot be read. A malformed tag is treated
// as absent rather than as an error, so a file that decoded before still does.
func ExifOrientation(data []byte) int {
	var tiff []byte
	switch {
	case len(data) >= 2 && data[0] == 0xFF && data[1] == 0xD8:
		tiff = jpegExif(data)
	case len(data) >= 8 && bytes.Equal(data[:8], []byte("\x89PNG\r\n\x1a\n")):
		tiff = pngExif(data)
	}
	if o := tiffOrientation(tiff); o >= 1 && o <= 8 {
		return o
	}
	return 1
}

// jpegExif returns the TIFF structure inside the first APP1 "Exif" segment.
func jpegExif(data []byte) []byte {
	i := 2
	for i+4 <= len(data) {
		if data[i] != 0xFF {
			return nil
		}
		marker := data[i+1]
		switch {
		case marker == 0xFF: // fill byte
			i++
			continue
		case marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7): // no length
			i += 2
			continue
		case marker == 0xDA || marker == 0xD9: // start of scan, end of image
			return nil
		}
		n := int(binary.BigEndian.Uint16(data[i+2 : i+4]))
		if n < 2 || i+2+n > len(data) {
			return nil
		}
		seg := data[i+4 : i+2+n]
		if marker == 0xE1 && len(seg) >= 6 && bytes.Equal(seg[:6], []byte("Exif\x00\x00")) {
			return seg[6:]
		}
		i += 2 + n
	}
	return nil
}

// pngExif returns the contents of the eXIf chunk, which is a bare TIFF
// structure with no "Exif\0\0" prefix.
func pngExif(data []byte) []byte {
	i := 8
	for i+12 <= len(data) {
		// Compared as uint64 before converting, so a length near 2^32 cannot
		// wrap a 32-bit int negative and slip past the bounds check.
		length := binary.BigEndian.Uint32(data[i : i+4])
		if uint64(length) > uint64(len(data)-i-12) {
			return nil
		}
		n := int(length)
		typ := string(data[i+4 : i+8])
		if typ == "eXIf" {
			return data[i+8 : i+8+n]
		}
		if typ == "IDAT" || typ == "IEND" {
			// eXIf must precede IDAT; the PNG spec allows nothing later.
			return nil
		}
		i += 12 + n
	}
	return nil
}

// tiffOrientation reads tag 0x0112 from IFD0, or returns 0.
func tiffOrientation(t []byte) int {
	if len(t) < 8 {
		return 0
	}
	var bo binary.ByteOrder
	switch string(t[:2]) {
	case "II":
		bo = binary.LittleEndian
	case "MM":
		bo = binary.BigEndian
	default:
		return 0
	}
	if bo.Uint16(t[2:4]) != 42 {
		return 0
	}
	off := bo.Uint32(t[4:8])
	if off < 8 || uint64(off)+2 > uint64(len(t)) {
		return 0
	}
	ifd := int(off)
	count := int(bo.Uint16(t[ifd : ifd+2]))
	for k := 0; k < count; k++ {
		e := ifd + 2 + 12*k
		if e+12 > len(t) {
			return 0
		}
		if bo.Uint16(t[e:e+2]) != 0x0112 {
			continue
		}
		// The standard type is SHORT (3). Some writers use LONG (4), and
		// Pillow and libvips both accept it, so this does too.
		switch bo.Uint16(t[e+2 : e+4]) {
		case 3:
			return int(bo.Uint16(t[e+8 : e+10]))
		case 4:
			if v := bo.Uint32(t[e+8 : e+12]); v <= 8 {
				return int(v)
			}
		}
		return 0
	}
	return 0
}

// ApplyOrientation returns img transformed the way EXIF Orientation tag
// orientation says to display it. Tag 1, or any value outside 2-8, returns img
// itself, untouched.
//
// Pixels are copied, never resampled. A greyscale source stays *image.Gray;
// anything else is copied into *image.RGBA64, which holds every value a
// colour's RGBA() can return, so the later conversion to grey sees exactly the
// colours it would have seen in the source.
func ApplyOrientation(img image.Image, orientation int) image.Image {
	if orientation < 2 || orientation > 8 {
		return img
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	dw, dh := w, h
	if orientation >= 5 {
		dw, dh = h, w
	}
	// src maps a destination pixel to the source pixel that lands there.
	src := func(x, y int) (int, int) {
		switch orientation {
		case 2: // mirrored horizontally
			return w - 1 - x, y
		case 3: // rotated 180
			return w - 1 - x, h - 1 - y
		case 4: // mirrored vertically
			return x, h - 1 - y
		case 5: // transposed
			return y, x
		case 6: // stored rotated 90 counter-clockwise; display turns it clockwise
			return y, h - 1 - x
		case 7: // transversed
			return w - 1 - y, h - 1 - x
		default: // 8: stored rotated 90 clockwise; display turns it counter-clockwise
			return w - 1 - y, x
		}
	}
	rect := image.Rect(0, 0, dw, dh)
	if g, ok := img.(*image.Gray); ok {
		out := image.NewGray(rect)
		for y := 0; y < dh; y++ {
			for x := 0; x < dw; x++ {
				sx, sy := src(x, y)
				out.SetGray(x, y, g.GrayAt(b.Min.X+sx, b.Min.Y+sy))
			}
		}
		return out
	}
	out := image.NewRGBA64(rect)
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			sx, sy := src(x, y)
			out.Set(x, y, color.RGBA64Model.Convert(img.At(b.Min.X+sx, b.Min.Y+sy)))
		}
	}
	return out
}

// FlattenOnWhite composites img onto a white background when any of its pixels
// is not fully opaque, and returns img itself otherwise.
//
// Converting to grey reads colour through RGBA(), which is alpha-premultiplied,
// so a transparent pixel stored as (0, 0, 0, 0) -- the common encoding -- read
// as black, and dark text on a transparent background vanished into it. The
// web and mobile apps flatten onto white before reading; this does the same.
// An opaque image takes the path it always took, so its bytes are unchanged.
func FlattenOnWhite(img image.Image) image.Image {
	if o, ok := img.(interface{ Opaque() bool }); ok && o.Opaque() {
		return img
	}
	b := img.Bounds()
	out := image.NewRGBA64(b)
	draw.Draw(out, b, image.White, image.Point{}, draw.Src)
	draw.Draw(out, b, img, b.Min, draw.Over)
	return out
}
