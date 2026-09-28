"""Local desktop interface for viewing MRI slices and running a 2D U-Net."""

import json
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import nibabel as nib
import numpy as np
from PIL import Image, ImageTk
import torch

from inference.predict import count_lesions
from models.unet import UNet


def read_slice(path):
    if path.name.lower().endswith((".nii", ".nii.gz")):
        volume = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
        if volume.ndim == 4:
            volume = volume[..., 0]
        if volume.ndim != 3:
            raise ValueError("Ожидается трёхмерный NIfTI-файл")
        axis = int(np.argmax(volume.shape))
        gray = np.take(volume, volume.shape[axis] // 2, axis=axis)
    else:
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    finite = np.isfinite(gray)
    if not finite.any():
        raise ValueError("В изображении нет числовых данных")
    low, high = np.min(gray[finite]), np.max(gray[finite])
    gray = np.nan_to_num(gray, nan=low, posinf=high, neginf=low)
    return (gray - low) / (high - low) if high > low else np.zeros_like(gray)


class DesktopApp:
    BG = "#06111b"
    PANEL = "#0d1d2d"
    SIDE = "#07131f"
    CYAN = "#39e8ff"
    TEXT = "#edf7ff"
    MUTED = "#8aa8ba"

    def __init__(self, root):
        self.root = root
        self.root.title("МРТешечка — AI-анализ МРТ")
        self.root.geometry("1400x850")
        self.root.minsize(1050, 650)
        self.root.configure(bg=self.BG)
        self.gray = None
        self.file_path = None
        self.photo = None
        self.mask = None
        self.overlay = None
        self.history = []
        self.page = "Главная"
        self.view = "Оригинал"
        self.weights = tk.StringVar()
        self.min_size = tk.IntVar(value=20)
        self.status = tk.StringVar(value="Откройте снимок для просмотра. Для анализа нужны веса 2D U-Net.")
        self.count = tk.StringVar(value="—")
        self.sizes = tk.StringVar(value="Загрузите снимок для подсчёта.")
        self.confidence = tk.StringVar(value="—")

        self.sidebar = tk.Frame(root, bg=self.SIDE, width=245, padx=18, pady=22)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self.label(self.sidebar, "🧠 МРТешечка", 21, self.CYAN, True).pack(anchor="w", pady=(0, 6))
        self.label(self.sidebar, "AI-анализ МРТ", 11, self.MUTED).pack(anchor="w", pady=(0, 28))
        self.label(self.sidebar, "НАВИГАЦИЯ", 10, self.MUTED, True).pack(anchor="w", pady=(0, 9))
        for page in ("Главная", "Анализ", "История", "О проекте"):
            self.button(self.sidebar, page, lambda p=page: self.navigate(p), False).pack(fill="x", pady=3)
        self.label(self.sidebar, "ПАРАМЕТРЫ", 10, self.MUTED, True).pack(anchor="w", pady=(30, 10))
        self.label(self.sidebar, "Минимальный размер очага, px", 10).pack(anchor="w")
        tk.Scale(self.sidebar, from_=1, to=500, orient="horizontal", variable=self.min_size,
                 bg=self.SIDE, fg=self.TEXT, troughcolor=self.PANEL, highlightthickness=0).pack(fill="x")
        self.button(self.sidebar, "Выбрать веса U-Net", self.choose_weights, False).pack(fill="x", pady=(18, 6))
        tk.Label(self.sidebar, textvariable=self.weights, bg=self.SIDE, fg=self.MUTED,
                 wraplength=205, justify="left").pack(anchor="w")
        self.label(self.sidebar, "Без весов доступен просмотр снимка.\nРезультаты анализа появятся после загрузки модели.",
                   10, self.MUTED).pack(anchor="w", pady=(24, 0))

        self.main = tk.Frame(root, bg=self.BG, padx=24, pady=20)
        self.main.pack(side="left", fill="both", expand=True)
        self.render_page()

    def label(self, parent, text, size=11, color=None, bold=False):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color or self.TEXT,
                        font=("Segoe UI", size, "bold" if bold else "normal"),
                        justify="left", anchor="w", wraplength=700)

    def button(self, parent, text, command, primary=True):
        return tk.Button(parent, text=text, command=command, relief="flat", cursor="hand2",
                         bg=self.CYAN if primary else self.PANEL,
                         fg=self.BG if primary else self.TEXT,
                         activebackground="#42c8ff", activeforeground=self.BG,
                         font=("Segoe UI", 11, "bold"), padx=12, pady=9)

    def card(self, parent, width=None):
        frame = tk.Frame(parent, bg=self.PANEL, padx=18, pady=18,
                         highlightbackground="#1b394b", highlightthickness=1)
        if width:
            frame.configure(width=width)
            frame.pack_propagate(False)
        return frame

    def navigate(self, page):
        self.page = page
        self.render_page()

    def render_page(self):
        for child in self.main.winfo_children():
            child.destroy()
        self.label(self.main, "🧠 МРТешечка", 28, self.TEXT, True).pack(anchor="w")
        self.label(self.main, "AI-анализ МРТ", 12, self.MUTED).pack(anchor="w", pady=(0, 18))
        if self.page == "История":
            card = self.card(self.main)
            card.pack(fill="both", expand=True)
            self.label(card, "История анализов", 20, self.CYAN, True).pack(anchor="w", pady=(0, 12))
            if not self.history:
                self.label(card, "Пока анализов нет. Откройте снимок на главной странице.", 12, self.MUTED).pack(anchor="w")
            for item in reversed(self.history):
                self.label(card, f"{item['file']} — очагов: {item['lesion_count']}", 12).pack(anchor="w", pady=5)
            return
        if self.page == "О проекте":
            card = self.card(self.main)
            card.pack(fill="both", expand=True)
            self.label(card, "Что умеет МРТешечка?", 20, self.CYAN, True).pack(anchor="w", pady=(0, 12))
            self.label(card, "Локальный просмотр МРТ-срезов PNG, JPG и NIfTI. При наличии весов 2D U-Net приложение показывает маску, число очагов и их размеры.", 13).pack(anchor="w")
            self.label(card, "Исследовательский прототип. Результат модели не является медицинским заключением.", 11, self.MUTED).pack(anchor="w", pady=24)
            return

        columns = tk.Frame(self.main, bg=self.BG)
        columns.pack(fill="both", expand=True)
        columns.grid_columnconfigure(0, weight=1, uniform="cols")
        columns.grid_columnconfigure(1, weight=2, uniform="cols")
        columns.grid_columnconfigure(2, weight=1, uniform="cols")
        columns.grid_rowconfigure(0, weight=1)
        left = self.card(columns)
        center = self.card(columns)
        right = self.card(columns)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        center.grid(row=0, column=1, sticky="nsew", padx=8)
        right.grid(row=0, column=2, sticky="nsew", padx=(8, 0))

        self.label(left, "ЗАГРУЗИТЬ ИЗОБРАЖЕНИЕ", 10, self.CYAN, True).pack(anchor="w")
        self.label(left, "Кидай МРТ сюда 👇", 17, self.TEXT, True).pack(anchor="w", pady=(8, 16))
        self.button(left, "Открыть снимок", self.open_image).pack(fill="x")
        self.label(left, "PNG, JPG или NIfTI (.nii / .nii.gz). Для NIfTI берётся центральный срез.",
                   10, self.MUTED).pack(anchor="w", pady=(12, 24))
        self.label(left, "БЫСТРЫЙ СТАРТ", 10, self.CYAN, True).pack(anchor="w")
        self.label(left, "Откройте файл, чтобы посмотреть снимок. Затем подключите обученные веса модели для анализа.",
                   11, self.MUTED).pack(anchor="w", pady=8)

        self.label(center, "ИЗОБРАЖЕНИЕ", 10, self.CYAN, True).pack(anchor="w")
        self.label(center, "Срез МРТ + сегментация", 17, self.TEXT, True).pack(anchor="w", pady=(8, 12))
        tabs = tk.Frame(center, bg=self.PANEL)
        tabs.pack(fill="x")
        for name in ("Оригинал", "Маска", "Наложение"):
            self.button(tabs, name, lambda n=name: self.select_view(n), False).pack(side="left", padx=(0, 5))
        self.image_label = tk.Label(center, bg=self.PANEL, fg=self.MUTED,
                                    text="🧠\n\nЗагрузите МРТ", font=("Segoe UI", 18, "bold"))
        self.image_label.pack(fill="both", expand=True, pady=14)
        self.show_current_view()

        self.label(right, "РЕЗУЛЬТАТЫ АНАЛИЗА", 10, self.CYAN, True).pack(anchor="w")
        self.label(right, "✨ Что нашли", 17, self.TEXT, True).pack(anchor="w", pady=(8, 16))
        self.label(right, "Найдено очагов", 11, self.CYAN).pack(anchor="w")
        tk.Label(right, textvariable=self.count, bg=self.PANEL, fg=self.TEXT,
                 font=("Segoe UI", 40, "bold")).pack(anchor="w", pady=(0, 22))
        self.label(right, "Размеры очагов", 12, self.TEXT, True).pack(anchor="w")
        tk.Label(right, textvariable=self.sizes, bg=self.PANEL, fg=self.MUTED,
                 justify="left", wraplength=220).pack(anchor="w", pady=(7, 20))
        self.label(right, "Оценка модели", 12, self.TEXT, True).pack(anchor="w")
        tk.Label(right, textvariable=self.confidence, bg=self.PANEL, fg=self.CYAN).pack(anchor="w", pady=(7, 22))
        self.button(right, "▶ Анализировать", self.analyze).pack(fill="x", pady=4)
        self.button(right, "↓ Скачать результаты", self.save_report, False).pack(fill="x", pady=4)

        tk.Label(self.main, textvariable=self.status, bg=self.BG, fg=self.MUTED,
                 anchor="w", wraplength=900).pack(fill="x", pady=(12, 0))
        self.label(self.main, "Исследовательский прототип. Результат модели не является медицинским заключением.",
                   10, self.MUTED).pack(anchor="w", pady=(8, 0))

    def show_image(self, image):
        preview = image.copy()
        preview.thumbnail((540, 490), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.photo, text="")

    def select_view(self, name):
        self.view = name
        self.show_current_view()

    def show_current_view(self):
        if self.gray is None:
            return
        base = Image.fromarray((self.gray * 255).astype(np.uint8)).convert("RGB")
        if self.view == "Оригинал":
            image = base
        elif self.view == "Маска" and self.mask is not None:
            image = self.mask
        elif self.view == "Наложение" and self.overlay is not None:
            image = self.overlay
        else:
            self.image_label.configure(image="", text="Маска появится после анализа")
            self.photo = None
            return
        self.show_image(image)

    def open_image(self):
        name = filedialog.askopenfilename(
            filetypes=[("МРТ и изображения", "*.nii *.nii.gz *.png *.jpg *.jpeg"), ("Все файлы", "*.*")]
        )
        if not name:
            return
        try:
            self.gray = read_slice(Path(name))
            self.file_path = Path(name)
            self.mask = self.overlay = None
            self.view = "Оригинал"
            self.count.set("—")
            self.sizes.set("Данные появятся после анализа.")
            self.confidence.set("—")
            self.navigate("Анализ")
            self.status.set(f"Открыт: {name}")
        except Exception as exc:
            messagebox.showerror("Не удалось открыть снимок", str(exc))

    def choose_weights(self):
        name = filedialog.askopenfilename(filetypes=[("Веса PyTorch", "*.pth *.pt"), ("Все файлы", "*.*")])
        if name:
            self.weights.set(name)

    def analyze(self):
        if self.gray is None:
            messagebox.showinfo("Нет снимка", "Сначала откройте снимок.")
            return
        path = Path(self.weights.get())
        if not path.is_file():
            messagebox.showinfo("Нет весов модели", "Выберите файл весов 2D U-Net для анализа.")
            return
        self.status.set("Выполняется анализ…")
        threading.Thread(target=self._run_analysis, args=(path, self.gray.copy(), self.min_size.get()), daemon=True).start()

    def _run_analysis(self, path, gray, min_size):
        try:
            model = UNet()
            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.eval()
            resized = Image.fromarray((gray * 255).astype(np.uint8)).resize((256, 256))
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            x = torch.from_numpy(np.stack([arr] * 3)).unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(model(x)).squeeze().numpy()
            mask = prob > 0.5
            count, labels, sizes = count_lesions(mask, min_size=min_size)
            lab = Image.fromarray(labels.astype(np.int32)).resize((gray.shape[1], gray.shape[0]), Image.Resampling.NEAREST)
            rgb = np.stack([gray * 255] * 3, axis=-1).astype(np.uint8)
            rgb[np.asarray(lab) > 0] = [255, 80, 80]
            mask_image = Image.fromarray((mask.astype(np.uint8) * 255)).resize((gray.shape[1], gray.shape[0]), Image.Resampling.NEAREST)
            confidence = float(prob[mask].mean() if mask.any() else prob.max())
            self.root.after(0, lambda: self._show_result(Image.fromarray(rgb), mask_image, count, sizes, confidence))
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): messagebox.showerror("Ошибка анализа", error))
            self.root.after(0, lambda: self.status.set("Анализ не выполнен."))

    def _show_result(self, image, mask_image, count, sizes, confidence):
        self.overlay = image
        self.mask = mask_image
        self.view = "Наложение"
        self.show_current_view()
        self.count.set(str(count))
        self.sizes.set("\n".join(f"Очаг {i}: {size} px" for i, size in enumerate(sizes, 1)) or "Очаги не найдены")
        self.confidence.set(f"Уверенность: {confidence:.0%}")
        self.history.append({"file": self.file_path.name, "lesion_count": count,
                             "lesion_sizes_px": sizes, "confidence": round(confidence, 4)})
        self.status.set(f"Найдено очагов: {count}. Красным показана маска модели.")

    def save_report(self):
        if not self.history:
            messagebox.showinfo("Нет результатов", "Сначала выполните анализ.")
            return
        name = filedialog.asksaveasfilename(defaultextension=".json", initialfile="results.json",
                                            filetypes=[("JSON", "*.json")])
        if name:
            Path(name).write_text(json.dumps(self.history[-1], ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    window = tk.Tk()
    DesktopApp(window)
    window.mainloop()
