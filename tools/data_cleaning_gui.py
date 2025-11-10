"""
Data Cleaning GUI Tool
Allows manual classification of images as good, bad font, or bad others.
Saves results to JSON and copies images to appropriate directories.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading


class DataCleaningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Cleaning Tool")
        self.root.geometry("1000x700")

        # Data directories
        self.data_dir = Path("/home/xplus/cs4243/data")
        self.raw_dir = self.data_dir / "raw"
        self.cleaned_dir = self.data_dir / "cleaned"
        self.good_dir = self.cleaned_dir / "good"
        self.bad_dir = self.cleaned_dir / "bad"

        # Ensure directories exist
        self.good_dir.mkdir(parents=True, exist_ok=True)
        self.bad_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.classifications = {}  # {filename: {reason, label}}
        self.image_files = []
        self.current_index = 0
        self.current_dataset = None  # 'train' or 'test'

        # Load existing classifications
        self.json_path = self.cleaned_dir / "classifications.json"
        self.load_classifications()

        # Setup UI
        self.setup_ui()
        self.bind_keys()

    def setup_ui(self):
        """Setup the GUI layout"""
        # Top frame with dataset and progress info
        top_frame = tk.Frame(self.root, bg="lightgray", height=50)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(top_frame, text="Dataset:", bg="lightgray", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.dataset_var = tk.StringVar()
        tk.Radiobutton(top_frame, text="Train", variable=self.dataset_var, value="train",
                      command=self.load_dataset, bg="lightgray").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(top_frame, text="Test", variable=self.dataset_var, value="test",
                      command=self.load_dataset, bg="lightgray").pack(side=tk.LEFT, padx=5)

        tk.Label(top_frame, text="Progress:", bg="lightgray", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=20)
        self.progress_label = tk.Label(top_frame, text="", bg="lightgray", font=("Arial", 10))
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # Image display frame
        image_frame = tk.Frame(self.root, bg="white", width=600, height=600)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.image_label = tk.Label(image_frame, bg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Right panel with controls
        control_frame = tk.Frame(self.root, bg="white", width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

        # File info
        self.filename_label = tk.Label(control_frame, text="", font=("Arial", 10, "bold"), wraplength=250, justify=tk.LEFT)
        self.filename_label.pack(pady=10)

        # Control buttons
        tk.Label(control_frame, text="Classification:", font=("Arial", 11, "bold")).pack(pady=(20, 10))

        button_frame = tk.Frame(control_frame)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="1: GOOD", width=15, height=3, bg="green", fg="white",
                 command=lambda: self.classify("good"), font=("Arial", 10, "bold")).pack(pady=5)
        tk.Button(button_frame, text="2: BAD FONT", width=15, height=3, bg="orange", fg="white",
                 command=lambda: self.classify("bad_font"), font=("Arial", 10, "bold")).pack(pady=5)
        tk.Button(button_frame, text="3: BAD OTHERS", width=15, height=3, bg="red", fg="white",
                 command=lambda: self.classify("bad_others"), font=("Arial", 10, "bold")).pack(pady=5)

        # Navigation
        nav_frame = tk.Frame(control_frame)
        nav_frame.pack(pady=20)

        tk.Button(nav_frame, text="← Previous", width=12, command=self.prev_image).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next →", width=12, command=self.next_image).pack(side=tk.LEFT, padx=5)

        # Status
        self.status_label = tk.Label(control_frame, text="", font=("Arial", 9), wraplength=250, justify=tk.LEFT)
        self.status_label.pack(pady=20)

        # Save button
        tk.Button(control_frame, text="Save & Copy Files", width=20, height=2, bg="blue", fg="white",
                 command=self.save_and_copy, font=("Arial", 10, "bold")).pack(pady=10)

        # Instructions
        instr_frame = tk.Frame(control_frame, bg="lightyellow")
        instr_frame.pack(pady=20, padx=5, fill=tk.BOTH, expand=True)

        instructions = """KEYBOARD SHORTCUTS:
1 = GOOD
2 = BAD FONT
3 = BAD OTHERS
← → = Navigate
S = Save & Copy"""

        tk.Label(instr_frame, text=instructions, font=("Arial", 8), justify=tk.LEFT, bg="lightyellow").pack(padx=10, pady=10)

    def bind_keys(self):
        """Bind keyboard shortcuts"""
        self.root.bind('1', lambda e: self.classify("good"))
        self.root.bind('2', lambda e: self.classify("bad_font"))
        self.root.bind('3', lambda e: self.classify("bad_others"))
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('s', lambda e: self.save_and_copy())
        self.root.bind('S', lambda e: self.save_and_copy())

    def load_dataset(self):
        """Load dataset when selected"""
        dataset = self.dataset_var.get()
        if not dataset:
            messagebox.showwarning("Warning", "Please select a dataset")
            return

        self.current_dataset = dataset
        dataset_path = self.raw_dir / dataset

        if not dataset_path.exists():
            messagebox.showerror("Error", f"Dataset path not found: {dataset_path}")
            return

        # Get all PNG images, sort by name
        self.image_files = sorted([f for f in dataset_path.glob("*.png")])
        self.current_index = 0

        if not self.image_files:
            messagebox.showwarning("Warning", f"No images found in {dataset_path}")
            return

        self.display_image()

    def display_image(self):
        """Display current image"""
        if not self.image_files or self.current_index >= len(self.image_files):
            messagebox.showinfo("Info", "No more images to classify")
            return

        image_path = self.image_files[self.current_index]
        filename = image_path.name

        # Update filename label
        self.filename_label.config(text=f"File: {filename}\nDataset: {self.current_dataset}")

        # Load and display image
        try:
            img = Image.open(image_path)
            # Resize to fit display
            max_size = (600, 600)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            return

        # Update status
        if filename in self.classifications:
            status = f"Already classified as: {self.classifications[filename]['reason']}"
        else:
            status = "Not yet classified"

        progress = f"{self.current_index + 1}/{len(self.image_files)}"
        self.progress_label.config(text=progress)
        self.status_label.config(text=status)

    def classify(self, label: str):
        """Classify current image"""
        if not self.image_files or self.current_index >= len(self.image_files):
            messagebox.showwarning("Warning", "No image to classify")
            return

        image_path = self.image_files[self.current_index]
        filename = image_path.name

        # Map label to reason
        reason_map = {
            "good": "Good image",
            "bad_font": "Bad font",
            "bad_others": "Bad others"
        }

        self.classifications[filename] = {
            "reason": reason_map[label],
            "label": label,
            "dataset": self.current_dataset
        }

        # Move to next
        self.next_image()

    def next_image(self):
        """Go to next image"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_image()
        else:
            messagebox.showinfo("Info", "Reached end of dataset")

    def prev_image(self):
        """Go to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()

    def load_classifications(self):
        """Load existing classifications from JSON"""
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    self.classifications = json.load(f)
            except Exception as e:
                print(f"Failed to load classifications: {e}")

    def save_and_copy(self):
        """Save classifications to JSON and copy files to appropriate directories"""
        if not self.classifications:
            messagebox.showwarning("Warning", "No classifications to save")
            return

        try:
            # Save to JSON
            with open(self.json_path, 'w') as f:
                json.dump(self.classifications, f, indent=2)

            # Copy files to appropriate directories
            for filename, info in self.classifications.items():
                label = info.get('label')
                dataset = info.get('dataset')

                # Find source file
                source_path = None
                if dataset:
                    potential_path = self.raw_dir / dataset / filename
                    if potential_path.exists():
                        source_path = potential_path

                if not source_path:
                    # Search all datasets
                    for dataset_name in ['train', 'test']:
                        potential_path = self.raw_dir / dataset_name / filename
                        if potential_path.exists():
                            source_path = potential_path
                            dataset = dataset_name
                            break

                if not source_path:
                    print(f"Source file not found: {filename}")
                    continue

                # Determine destination with dataset subfolder
                if label == "good":
                    dest_dir = self.good_dir / dataset
                else:
                    dest_dir = self.bad_dir / dataset

                # Create dataset subdirectory if it doesn't exist
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / filename

                # Copy if not already copied
                if not dest_path.exists():
                    shutil.copy2(source_path, dest_path)

            messagebox.showinfo("Success", f"Saved {len(self.classifications)} classifications and copied files")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save and copy: {e}")


def main():
    root = tk.Tk()
    gui = DataCleaningGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
