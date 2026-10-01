package segmenter

import (
	"image"
	"image/color"
	"reflect"
	"testing"
)

// Two lines of blobs; on a (0, 0, 0, 0) background when transparent.
func blobPage(transparent bool) image.Image {
	const w, h = 600, 200
	ink := func(x, y int) bool {
		for _, top := range []int{30, 110} {
			if y >= top && y < top+40 && x >= 60 && x < 60+20*24 && (x-60)%24 < 14 {
				return true
			}
		}
		return false
	}
	if !transparent {
		img := image.NewGray(image.Rect(0, 0, w, h))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				v := uint8(255)
				if ink(x, y) {
					v = 0
				}
				img.SetGray(x, y, color.Gray{Y: v})
			}
		}
		return img
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if ink(x, y) {
				img.SetNRGBA(x, y, color.NRGBA{A: 255})
			}
		}
	}
	return img
}

// A transparent background stored as (0, 0, 0, 0) read as black, which the
// `< 128` threshold calls ink, so the page was one solid band.
func TestATransparentPageSegmentsLikeItsOpaqueTwin(t *testing.T) {
	s := NewLineSegmenter(10, 3)
	want, err := s.Segment(blobPage(false))
	if err != nil || len(want) != 2 {
		t.Fatalf("opaque page: %d lines, %v", len(want), err)
	}
	got, err := s.Segment(blobPage(true))
	if err != nil {
		t.Fatal(err)
	}
	var wb, gb []image.Rectangle
	for _, l := range want {
		wb = append(wb, l.BBox)
	}
	for _, l := range got {
		gb = append(gb, l.BBox)
	}
	if !reflect.DeepEqual(gb, wb) {
		t.Fatalf("transparent page bboxes %v, want %v", gb, wb)
	}
}
