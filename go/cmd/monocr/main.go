package main

import (
	"fmt"
	"os"
	"path/filepath"

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
			dir := args[0]
			files, err := os.ReadDir(dir)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error reading directory: %v\n", err)
				os.Exit(1)
			}

			for _, file := range files {
				ext := filepath.Ext(file.Name())
				if ext == ".jpg" || ext == ".png" || ext == ".jpeg" {
					path := filepath.Join(dir, file.Name())
					fmt.Fprintf(os.Stderr, "Processing %s...\n", file.Name())
					text, err := monocr.ReadImage(path)
					if err != nil {
						fmt.Fprintf(os.Stderr, "Failed to process %s: %v\n", file.Name(), err)
					} else {
						fmt.Printf("--- %s ---\n%s\n\n", file.Name(), text)
					}
				}
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
