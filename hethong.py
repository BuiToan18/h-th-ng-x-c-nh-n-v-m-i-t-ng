import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

# Khởi tạo mô hình YOLO
model = YOLO("yolov5s.pt")

def select_image():
    global panel_original, panel_processed, img_path

    # Chọn ảnh từ máy tính
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not img_path:
        return

    # Hiển thị ảnh gốc
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    img_display = ImageTk.PhotoImage(image_pil)

    panel_original.config(image=img_display)
    panel_original.image = img_display

    process_image(image)

def process_image(image):
    global panel_processed, result_label

    # Nhận diện đối tượng
    results = model(image)
    result_image = results[0].plot()  # Vẽ bounding box lên ảnh

    # Đếm số lượng đối tượng
    objects = results[0].names
    counts = {}
    for obj in results[0].boxes.cls:
        obj_name = objects[int(obj)]
        counts[obj_name] = counts.get(obj_name, 0) + 1

    # Hiển thị kết quả số lượng đối tượng
    result_text = "\n".join([f"{obj}: {count}" for obj, count in counts.items()])
    result_label.config(text=f"Các đối tượng đếm được:\n{result_text}")

    # Hiển thị ảnh đã xử lý
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_image)
    img_display = ImageTk.PhotoImage(result_pil)

    panel_processed.config(image=img_display)
    panel_processed.image = img_display

def save_results():
    if img_path:
        # Đọc ảnh gốc
        image = cv2.imread(img_path)
        
        # Nhận diện đối tượng và vẽ bounding boxes lên ảnh
        results = model(image)
        result_image = results[0].plot()  # Vẽ bounding box lên ảnh

        # Lưu ảnh đã xử lý
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if save_path:
            cv2.imwrite(save_path, result_image)  # Lưu ảnh với bounding boxes
            messagebox.showinfo("Save Image", "Image saved successfully!")


# Giao diện chính
root = tk.Tk()
root.title("Xác nhận đối tượng và đếm đối tượng")
root.geometry("1650x1000")  # Tăng kích thước cửa sổ
root.configure(bg="#f0f0f0")  # Màu nền giao diện

# Frame cho phần điều khiển
control_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
control_frame.place(relx=0.5, rely=0.05, anchor="n", width=335, height=60)

# Nút chọn ảnh
btn_select = tk.Button(control_frame, text="Chọn ảnh", command=select_image, bg="#4CAF50", fg="white", font=("Arial", 12), width=15)
btn_select.grid(row=0, column=0, padx=10, pady=10)

# Nút lưu kết quả
btn_save = tk.Button(control_frame, text="lưu kết quả", command=save_results, bg="#008CBA", fg="white", font=("Arial", 12), width=15)
btn_save.grid(row=0, column=1, padx=10, pady=10)

# Frame cho ảnh gốc
original_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
original_frame.place(relx=0.25, rely=0.15, anchor="n", width=800, height=600)

original_label = tk.Label(original_frame, text="Ảnh gốc", bg="#ffffff", font=("Arial", 14, "bold"))
original_label.pack(pady=10)

panel_original = tk.Label(original_frame, bg="#e0e0e0")
panel_original.pack(fill="both", expand=True, padx=10, pady=10)

# Frame cho ảnh đã xử lý
processed_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
processed_frame.place(relx=0.75, rely=0.15, anchor="n", width=800, height=600)

processed_label = tk.Label(processed_frame, text="Ảnh đã xử lý", bg="#ffffff", font=("Arial", 14, "bold"))
processed_label.pack(pady=10)

panel_processed = tk.Label(processed_frame, bg="#e0e0e0")
panel_processed.pack(fill="both", expand=True, padx=10, pady=10)

# Frame cho kết quả với Canvas và thanh cuộn
result_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
result_frame.place(relx=0.5, rely=0.78, anchor="n", width=800, height=200)  # Tăng chiều cao khung kết quả

# Canvas cho kết quả và thanh cuộn
result_canvas = tk.Canvas(result_frame)
scrollbar = tk.Scrollbar(result_frame, orient="vertical", command=result_canvas.yview)
result_canvas.configure(yscrollcommand=scrollbar.set)

# Gắn các widget vào Canvas
result_canvas.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")

# Thiết lập cấu trúc của grid
result_frame.grid_rowconfigure(0, weight=1)
result_frame.grid_columnconfigure(0, weight=1)

result_label = tk.Label(result_canvas, text="Các đối tượng đếm được: ", bg="#ffffff", font=("Arial", 12), justify="left", anchor="w")
result_canvas.create_window(10, 10, anchor="nw", window=result_label)

root.mainloop()