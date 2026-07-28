import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from collections import Counter

# ----------------------------
# 1. 加载模型（首次运行会自动下载 yolov8n.pt，之后不需要网络）
# ----------------------------
model = YOLO('yolov8n.pt')

# ----------------------------
# 2. 检测函数：返回标注后的图像（RGB）和摘要文本
# ----------------------------
def detect_and_summarize(image_path):
    # 用 OpenCV 读取图片（BGR）
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, "无法读取图片，请检查路径。"
    # YOLOv8 推理
    results = model(img_bgr)[0]
    # 绘制标注框（plot 返回 BGR 图像）
    annotated_bgr = results.plot()
    # 转 RGB 用于显示
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    # 统计信息
    if results.boxes is None:
        summary = "未检测到任何物体。"
    else:
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[cid] for cid in class_ids]
        counter = Counter(class_names)
        lines = [f"- {name}: {count} 个" for name, count in counter.items()]
        summary = "检测结果：\n" + "\n".join(lines)
        summary += f"\n\n共检测到 {len(class_ids)} 个物体。"
    return annotated_rgb, summary

# ----------------------------
# 3. GUI 类
# ----------------------------
class YOLODemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 目标检测 Demo")
        self.root.geometry("1000x500")

        # 选择图片按钮
        self.btn_select = tk.Button(root, text="选择图片并检测", command=self.select_image, font=("Arial", 14))
        self.btn_select.pack(pady=10)

        # 两个图片显示区域（左：原图，右：检测结果）
        self.frame_images = tk.Frame(root)
        self.frame_images.pack()

        self.lbl_original = tk.Label(self.frame_images, text="原图")
        self.lbl_original.grid(row=0, column=0, padx=10)
        self.lbl_result = tk.Label(self.frame_images, text="检测结果")
        self.lbl_result.grid(row=0, column=1, padx=10)

        self.canvas_original = tk.Label(self.frame_images)
        self.canvas_original.grid(row=1, column=0, padx=10)
        self.canvas_result = tk.Label(self.frame_images)
        self.canvas_result.grid(row=1, column=1, padx=10)

        # 检测统计文本框
        self.text_summary = tk.Text(root, height=10, width=80, font=("Arial", 12))
        self.text_summary.pack(pady=10)

        # 用来保存当前显示的图像对象，防止被垃圾回收
        self.img_tk_original = None
        self.img_tk_result = None

    def select_image(self):
        # 弹出文件选择对话框
        file_path = filedialog.askopenfilename(
            title="选择一张图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        # 读取并显示原图
        original_img = Image.open(file_path)
        original_img.thumbnail((400, 300))  # 缩放到合适大小
        self.img_tk_original = ImageTk.PhotoImage(original_img)
        self.canvas_original.configure(image=self.img_tk_original)

        # 执行检测
        annotated_rgb, summary = detect_and_summarize(file_path)
        if annotated_rgb is None:
            messagebox.showerror("错误", summary)
            return

        # 显示检测结果图
        result_img = Image.fromarray(annotated_rgb)
        result_img.thumbnail((400, 300))
        self.img_tk_result = ImageTk.PhotoImage(result_img)
        self.canvas_result.configure(image=self.img_tk_result)

        # 显示统计文本
        self.text_summary.delete("1.0", tk.END)
        self.text_summary.insert(tk.END, summary)

# ----------------------------
# 4. 启动 UI
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODemoApp(root)
    root.mainloop()