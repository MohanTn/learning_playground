import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageTk

CUSTOM_CONFIG = r'--oem 3 --psm 6'
MAX_DISPLAY_WIDTH = 900


def process_image(image_path: str) -> dict:
    # 1. Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Could not open or find the image.")

    # 2. Invert colors: text must be white, background black for angle detection
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # 3. Grab all coordinates where pixels are white (the text)
    coords = np.column_stack(np.where(thresh > 0))

    # 4. Find the minimum bounding box around all text pixels
    angle = cv2.minAreaRect(coords)[-1]

    # 5. Correct the angle calculation based on OpenCV box orientation rules
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # 6. Rotate the original image to straighten it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 7. Run OCR on the straightened image
    clean_text = pytesseract.image_to_string(rotated_img, config=CUSTOM_CONFIG).strip()

    # 8. Run word-level OCR to get bounding boxes for highlighting
    data = pytesseract.image_to_data(rotated_img, config=CUSTOM_CONFIG, output_type=pytesseract.Output.DICT)
    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        try:
            conf = int(float(data["conf"][i]))
        except (ValueError, TypeError):
            conf = -1
        if text and conf > 0:
            words.append({
                "text": text,
                "conf": conf,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            })

    # 9. Draw highlight boxes around every detected word
    highlighted_img = cv2.cvtColor(rotated_img, cv2.COLOR_GRAY2BGR)
    for word in words:
        x, y, bw, bh = word["left"], word["top"], word["width"], word["height"]
        cv2.rectangle(highlighted_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    return {
        "detected_skew_angle": angle,
        "clean_text": clean_text,
        "words": words,
        "highlighted_img": highlighted_img,
    }


class OcrDeskewApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OCR Deskew Viewer")
        self.root.geometry("1000x800")
        self.tk_image = None  # keep a reference so it isn't garbage collected

        toolbar = tk.Frame(root)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(toolbar, text="Open Image...", command=self.open_image).pack(side=tk.LEFT, padx=6, pady=6)
        self.angle_label = tk.Label(toolbar, text="Skew angle: -")
        self.angle_label.pack(side=tk.LEFT, padx=6)

        paned = tk.PanedWindow(root, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True)

        # Top pane: scrollable canvas showing the deskewed image with highlights
        image_frame = tk.Frame(paned, bg="gray20")
        h_scroll = tk.Scrollbar(image_frame, orient=tk.HORIZONTAL)
        v_scroll = tk.Scrollbar(image_frame, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(image_frame, bg="gray20",
                                 xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        paned.add(image_frame, height=500)

        # Bottom pane: extracted text and per-word values
        text_frame = tk.Frame(paned)
        text_scroll = tk.Scrollbar(text_frame)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=text_scroll.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.config(command=self.text.yview)
        paned.add(text_frame, height=250)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            result = process_image(path)
        except Exception as exc:
            messagebox.showerror("OCR failed", str(exc))
            return
        self.show_result(result)

    def show_result(self, result: dict):
        self.angle_label.config(text=f"Skew angle: {result['detected_skew_angle']:.2f} deg")

        img_rgb = cv2.cvtColor(result["highlighted_img"], cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        if pil_img.width > MAX_DISPLAY_WIDTH:
            scale = MAX_DISPLAY_WIDTH / pil_img.width
            pil_img = pil_img.resize(
                (MAX_DISPLAY_WIDTH, int(pil_img.height * scale)), Image.LANCZOS
            )
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, pil_img.width, pil_img.height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, f"Detected skew angle: {result['detected_skew_angle']:.2f} deg\n\n")
        self.text.insert(tk.END, "Extracted text:\n")
        self.text.insert(tk.END, result["clean_text"] + "\n\n")
        self.text.insert(tk.END, "Detected values (green boxes above):\n")
        for word in result["words"]:
            self.text.insert(
                tk.END,
                f"  - \"{word['text']}\" (conf {word['conf']}%) at "
                f"({word['left']}, {word['top']})\n",
            )


def main():
    root = tk.Tk()
    app = OcrDeskewApp(root)
    app.open_image()  # prompt for an image as soon as the app launches
    root.mainloop()


if __name__ == "__main__":
    main()
