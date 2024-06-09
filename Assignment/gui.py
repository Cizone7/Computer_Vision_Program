import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from image_retrieval_process import (load_codebook, encode_features,
                                     knn_search, load_model,
                                     extract_features_from_image, evaluate)
import time
import image_retrieval_evaluation
import threading
import matplotlib.pyplot as plt 
import tensorflow as tf


class ImageRetrievalGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Retrieval GUI")
        self.root.geometry("1200x900")

        # 用于存储评估结果的内容
        self.evaluation_result_text = ""

        # 创建顶层框架
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(pady=10)  # 减小上边距

        # 创建按钮框架
        self.button_frame = tk.Frame(self.top_frame)
        self.button_frame.pack(pady=5)

        # 选择特征提取方法
        self.feature_method = tk.StringVar(value="ORB")  # 默认为ORB
        self.feature_method_label = tk.Label(self.button_frame, text="选择特征提取方法:")
        self.feature_method_label.grid(row=0, column=0, padx=5, pady=5)
        self.feature_method_selection = tk.OptionMenu(self.button_frame, self.feature_method, "ORB", "SIFT")
        self.feature_method_selection.grid(row=0, column=1, padx=5, pady=5)

        # 选择图片按钮
        self.btn_select_image = tk.Button(self.button_frame, text="选择图片", command=self.select_image)
        self.btn_select_image.grid(row=0, column=2, padx=5, pady=5)

        # 添加一个按钮用于触发评估事件
        self.btn_evaluate = tk.Button(self.button_frame, text="评估测试数据", command=self.evaluate_test_data)
        self.btn_evaluate.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

        # 添加一个按钮用于显示评估结果
        self.btn_show_results = tk.Button(self.button_frame, text="显示评估结果", command=self.load_evaluation_results)
        self.btn_show_results.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

        # 创建下部分框架
        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(pady=10, fill="both", expand=True)

        # 创建左右部分框架
        self.left_frame = tk.Frame(self.bottom_frame)
        self.left_frame.pack(side="left", padx=10, fill="both", expand=True)

        self.right_frame = tk.Frame(self.bottom_frame)
        self.right_frame.pack(side="right", padx=10, fill="both", expand=True)

        # 创建图片显示框架
        self.image_frame = tk.Frame(self.left_frame)
        self.image_frame.pack(pady=5, fill="both", expand=True)

        self.image_canvas = tk.Canvas(self.image_frame, width=500, height=600)
        self.image_canvas.pack(fill="both", expand=True)

        # 创建待匹配图像框架
        self.query_image_frame = tk.Frame(self.image_canvas)
        self.query_image_frame.place(relx=0.5, rely=0.5, anchor="center")

        # 创建匹配图像框架
        self.matched_images_frame = tk.Frame(self.left_frame)
        self.matched_images_frame.pack(pady=5, fill="both", expand=True)

        # 创建输出文本框架
        self.output1_frame = tk.Frame(self.left_frame)
        self.output1_frame.pack(pady=5, fill="both", expand=True)

        # 创建数值输出标签
        self.output1_label = tk.Label(self.output1_frame, text="", justify="left", wraplength=500)
        self.output1_label.pack(pady=10)

        # 创建第二个输出文本框架
        self.output2_frame = tk.Frame(self.right_frame)
        self.output2_frame.pack(pady=5, fill="both", expand=True)

        # 创建第二个数值输出标签
        self.output2_label = tk.Label(self.output2_frame, text="", justify="left", wraplength=500)
        self.output2_label.pack(pady=10)

        # 创建PR曲线显示框架
        self.pr_curve_frame = tk.Frame(self.right_frame)
        self.pr_curve_frame.pack(pady=5, fill="both", expand=True)

        # 尝试加载之前保存的评估结果
        # self.load_evaluation_results()

    # 添加事件处理函数，用于评估测试数据并显示结果
    def evaluate_test_data(self):
        # 创建一个新线程来执行评估任务
        evaluation_thread = threading.Thread(target=self.perform_evaluation)
        evaluation_thread.start()

    # 在新线程中执行评估任务
    def perform_evaluation(self):
        test_folder = "../test"
        feature_method = self.feature_method.get()  # 获取选定的特征提取方法
        model_path = f"encoded/{feature_method.lower()}_encoded_features_1000.pkl"
        codebook_path = f"features/{feature_method.lower()}_codebook_1000.pkl"
        tfidf_model_path = f"model/{feature_method.lower()}_tfidf_1000.pkl"
        cnn_path = "CNN/CNN_20240528-193503.h5"
        model = load_model(model_path)
        codebook = load_codebook(codebook_path)
        tfidf_model = load_model(tfidf_model_path)
        cnn_model = tf.keras.models.load_model(cnn_path)
        num_clusters = codebook_path.split('.')[0].split('_')[-1]
        average_precision, average_recall, average_response_time, ap_score = image_retrieval_evaluation.evaluate_test_dataset(
            test_folder, feature_method, model, codebook, tfidf_model, num_clusters, cnn_model)

        # 转换字典为列表
        precision_values = [average_precision[k] for k in range(1, 11)]
        recall_values = [average_recall[k] for k in range(1, 11)]

        map_score = {k: 0 for k in range(0, 11)}
        for k in range(1, 11):
            map_score[k] = (map_score[k-1] + ap_score[k])

        # 绘制 PR 曲线
        self.plot_pr_curve(precision_values, recall_values)

        output_text = "Precision\t\tRecall\t\tMAP\n"
        for k in range(1, 11):
            output_text += f"{average_precision[k]:.4f}\t\t{average_recall[k]:.4f}\t\t{map_score[k]/k:.4f}\n"
        output_text += f"Average Response Time: {average_response_time:.4f} seconds"

        self.evaluation_result_text = output_text  # 将结果保存到类变量中

        # 保存评估结果到文本文件
        self.save_evaluation_results(output_text)

        self.root.after(0, lambda: self.update_output_label(output_text, target_label="output2"))

    # 更新评估结果的标签
    def update_output_label(self, text, target_label="output1"):
        if target_label == "output1":
            self.output1_label.config(text=text)  # 修正这里的错误
        elif target_label == "output2":
            self.output2_label.config(text=text)

    # 显示评估结果
    def show_evaluation_results(self):
        self.update_output_label(self.evaluation_result_text, target_label="output2")

    # 绘制PR曲线
    def plot_pr_curve(self, precision, recall):
        # 清除先前的 PR 曲线
        for widget in self.pr_curve_frame.winfo_children():
            widget.destroy()

        # 调整图形大小并绘制 PR 曲线
        plt.figure(figsize=(4.5, 3.5))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.tight_layout()

        # 保存 PR 曲线图像
        pr_curve_img = plt.gcf()
        pr_curve_img.savefig("pr_curve.png")

        # 打开保存的 PR 曲线图像并转换为 Tkinter 图像
        pr_curve_tk_img = Image.open("pr_curve.png")
        pr_curve_tk_img = ImageTk.PhotoImage(pr_curve_tk_img)

        # 创建画布并在画布上显示 PR 曲线图像
        pr_curve_canvas = tk.Canvas(self.pr_curve_frame, width=500, height=400)
        pr_curve_canvas.create_image(0, 0, anchor="nw", image=pr_curve_tk_img)
        pr_curve_canvas.image = pr_curve_tk_img
        pr_curve_canvas.pack()

    # 保存评估结果到文本文件
    def save_evaluation_results(self, text):
        with open("evaluation_results.txt", "w") as file:
            file.write(text)

    # 加载评估结果从文本文件
    def load_evaluation_results(self):
        if os.path.exists("evaluation_results.txt"):
            with open("evaluation_results.txt", "r") as file:
                self.evaluation_result_text = file.read()
            self.update_output_label(self.evaluation_result_text, target_label="output2")
            # 打开保存的 PR 曲线图像并转换为 Tkinter 图像
            pr_curve_tk_img = Image.open("pr_curve.png")
            pr_curve_tk_img = ImageTk.PhotoImage(pr_curve_tk_img)

            # 创建画布并在画布上显示 PR 曲线图像
            pr_curve_canvas = tk.Canvas(self.pr_curve_frame, width=500, height=400)
            pr_curve_canvas.create_image(0, 0, anchor="nw", image=pr_curve_tk_img)
            pr_curve_canvas.image = pr_curve_tk_img
            pr_curve_canvas.pack()

    def run(self):
        self.root.mainloop()

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def resize_image(self, image):
        return cv2.resize(image, (125, 125), interpolation=cv2.INTER_AREA)

    def process_image(self, file_path):
        feature_method = self.feature_method.get()
        cnn_path = "CNN/CNN_20240528-193503.h5"
        if feature_method == "SIFT":
            codebook_file = "features/sift_codebook_1000.pkl"
            encoded_features_file = "encoded/sift_encoded_features_1000.pkl"
            model_file = "model/sift_tfidf_1000.pkl"
        else:
            codebook_file = "features/orb_codebook_1000.pkl"
            encoded_features_file = "encoded/orb_encoded_features_1000.pkl"
            model_file = "model/orb_tfidf_1000.pkl"

        model = load_model(encoded_features_file)
        codebook = load_codebook(codebook_file)
        tfidf_model = load_model(model_file)
        cnn_model = tf.keras.models.load_model(cnn_path)
        num_clusters = codebook_file.split('.')[0].split('_')[-1]
        start_time = time.time()
        # 提取图像特征
        keypoints, descriptors = extract_features_from_image(file_path, feature_method)
        # 对特征进行编码
        encoded_features = encode_features([descriptors], codebook, tfidf_model, num_clusters, 946)
        # 使用 KNN 搜索

        k = 10
        knn_result = knn_search(encoded_features[0], model, cnn_model, k=k)
        average_precision_10, average_recall_10, map_score_10 = evaluate(file_path.split('/')[-1], knn_result)
        end_time = time.time()
        response_time = end_time - start_time
        # 更新 GUI 上的文本标签
        output_text = f"Precision@10: {average_precision_10:.4f}\n"
        output_text += f"Recall@10: {average_recall_10:.4f}\n"
        output_text += f"AP@10: {map_score_10:.4f}\n"
        output_text += f"Response Time: {response_time:.4f} seconds"
        self.update_output_label(output_text, target_label="output1")

        self.display_image(file_path, (100, 100))  # 将图像稍微上移

        self.display_retrieved_images(knn_result)  # 显示检索到的图片

    def display_image(self, file_path, position):
        image = Image.open(file_path)
        image = ImageTk.PhotoImage(image.resize((250, 250)))  # 调整图片大小
        # 调整画布大小
        self.image_canvas.config(width=600, height=280)
        self.image_canvas.create_image(300, 140, anchor="center", image=image)  # 将图像放置在画布中央
        self.image_canvas.image = image  # 防止垃圾回收

    def display_retrieved_images(self, distances):
        # 分成上下两行
        rows = 2
        cols = 5

        for row in range(rows):
            for col in range(cols):
                index = row * cols + col
                if index < len(distances):
                    name, _ = distances[index]
                    similar_image_path = os.path.join('../image', name.split('_')[0], name)
                    similar_image = cv2.imread(similar_image_path)
                    similar_image_resized = self.resize_image(similar_image)
                    tk_img = cv2.cvtColor(similar_image_resized, cv2.COLOR_BGR2RGB)
                    tk_img = Image.fromarray(tk_img)
                    tk_img = ImageTk.PhotoImage(tk_img)
                    frame = self.matched_images_frame  # 匹配图像框架中显示
                    retrieved_image_canvas = tk.Canvas(frame, width=125, height=125)
                    retrieved_image_canvas.grid(row=row, column=col, padx=2, pady=2)  # 减小与后面图像的间距
                    retrieved_image_canvas.create_image(0, 0, anchor="nw", image=tk_img)
                    retrieved_image_canvas.image = tk_img  # 防止垃圾回收


if __name__ == "__main__":
    image_retrieval_gui = ImageRetrievalGUI()
    image_retrieval_gui.run()
