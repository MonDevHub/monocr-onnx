package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/MonDevHub/monocr-onnx/go"
	"github.com/MonDevHub/monocr-onnx/go/pkg/model"
	"github.com/MonDevHub/monocr-onnx/go/pkg/predictor"
	"github.com/spf13/cobra"
)

func main() {
	var rootCmd = &cobra.Command{
		Use:   "monocr",
		Short: "Mon language OCR",
		Long:  `MonOCR is a tool for recognizing Mon language text from images and PDFs using ONNX Runtime.`,
	}

	var imageCmd = &cobra.Command{
		Use:   "image [path]",
		Short: "Recognize text from an image file",
		Args:  cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			text, err := monocr.ReadImage(args[0])
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			fmt.Println(text)
		},
	}

	var pdfCmd = &cobra.Command{
		Use:   "pdf [path]",
		Short: "Recognize text from a PDF file",
		Args:  cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			pages, err := monocr.ReadPDF(args[0])
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			for i, page := range pages {
				fmt.Printf("--- Page %d ---\n", i+1)
				fmt.Println(page)
				fmt.Println()
			}
		},
	}

	var downloadCmd = &cobra.Command{
		Use:   "download",
		Short: "Download model to local cache",
		Run: func(cmd *cobra.Command, args []string) {
			manager, err := model.NewManager()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			if err := manager.DownloadModel(); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
		},
	}

	var batchCmd = &cobra.Command{
		Use:   "batch [directory]",
		Short: "Process all images in a directory",
		Args:  cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			failed, err := runBatch(args[0], monocr.ReadImage, os.Stdout, os.Stderr)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error reading directory: %v\n", err)
				os.Exit(1)
			}
			if failed > 0 {
				os.Exit(1)
			}
		},
	}

	var runtimeCmd = &cobra.Command{
		Use:   "runtime",
		Short: "Report the ONNX Runtime version this binding loaded",
		Long: "Load the ONNX Runtime shared library and print its version.\n\n" +
			"The runtime is supplied by the host, not by go.mod, so this is the only\n" +
			"way to know which one a result came from. Set " + predictor.SharedLibraryPathEnv +
			"\nto an absolute path to choose a specific library.",
		Run: func(cmd *cobra.Command, args []string) {
			version, err := monocr.RuntimeVersion()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("onnxruntime %s (tested against %s, requires >= %s)\n",
				version, predictor.TestedORTVersion, predictor.MinORTVersion)
		},
	}

	rootCmd.AddCommand(imageCmd, pdfCmd, downloadCmd, batchCmd, runtimeCmd)

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// isImage reports whether name has an extension batch reads, in any case: a
// camera's IMG_0001.JPG was skipped when the match was case-sensitive.
func isImage(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".jpg", ".jpeg", ".png":
		return true
	}
	return false
}

// runBatch reads every image in dir and returns how many failed, so the
// command can exit non-zero: it used to exit 0 even when every file failed.
func runBatch(dir string, read func(string) (string, error), stdout, stderr io.Writer) (int, error) {
	files, err := os.ReadDir(dir)
	if err != nil {
		return 0, err
	}
	failed, total := 0, 0
	for _, file := range files {
		if file.IsDir() || !isImage(file.Name()) {
			continue
		}
		total++
		fmt.Fprintf(stderr, "Processing %s...\n", file.Name())
		text, err := read(filepath.Join(dir, file.Name()))
		if err != nil {
			failed++
			fmt.Fprintf(stderr, "Failed to process %s: %v\n", file.Name(), err)
			continue
		}
		fmt.Fprintf(stdout, "--- %s ---\n%s\n\n", file.Name(), text)
	}
	if failed > 0 {
		fmt.Fprintf(stderr, "%d of %d files failed.\n", failed, total)
	}
	return failed, nil
}
