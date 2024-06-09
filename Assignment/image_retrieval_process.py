# image_retrieval_process.py
import numpy as np
import cv2
import os
import logging
import pickle
from bof_encoding import calculate_weights, logger
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def extract_sift_features(image_path):
    sift = cv2.SIFT_create(nfeatures=1000)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    # print("Descriptors shape:", descriptors.shape)
    return keypoints, descriptors


def extract_orb_features(image_path):
    orb = cv2.ORB_create(nfeatures=1000)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    # print("Descriptors shape:", descriptors.shape)
    return keypoints, descriptors


def extract_features_from_image(image_path, method):
    if method == "SIFT":
        return extract_sift_features(image_path)
    elif method == "ORB":
        return extract_orb_features(image_path)
    else:
        raise ValueError("Invalid feature extraction method")


def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None


def load_codebook(codebook_path):
    try:
        with open(codebook_path, 'rb') as f:
            codebook = pickle.load(f)
        return codebook
    except Exception as e:
        logger.error(f"Failed to load codebook from {codebook_path}: {e}")
        return None


def encode_features(features, codebook, df, num_clusters, total_images):
    encoded_features = []
    # print(f'features:{features}')
    for i, descriptors in enumerate(features):
        if descriptors is not None:
            # 创建一个与码本长度相同的全零数组，用于存储编码后的特征
            encoded_feature = np.zeros(len(codebook))
            # 计算每个描述符与码本中心的距离（欧氏距离），并找到最近的聚类中心的索引
            distances = np.linalg.norm(codebook - descriptors[:, np.newaxis], axis=-1)
            nearest_cluster_indices = np.argmin(distances, axis=1)
            weights = calculate_weights(nearest_cluster_indices, df, num_clusters, total_images)
            # 归一化权重
            normalized_weights = weights / np.sum(weights)

            for index, weight in zip(nearest_cluster_indices, normalized_weights):
                encoded_feature[index] = max(encoded_feature[index], weight)

            encoded_features.append(encoded_feature)
    return encoded_features


def knn_search(encoded_feature, encoded_features, model, k=10):
    distances = []
    # 计算每个样本与查询特征的距离
    for index, feature in enumerate(encoded_features):
        file_name = feature[0]
        feature_data = np.array(feature[1:], dtype=float)  # 确保特征数据为数值类型
        distance = calculate_distance(np.array(encoded_feature, dtype=float), feature_data)
        distances.append((file_name, distance, feature_data))

    # 按距离排序
    sorted_distances = sorted(distances, key=lambda x: x[1])
    k_nearest_neighbors = sorted_distances[:40]

    # 对 k 个最近邻的样本进行预测，并根据预测标签对它们进行排序
    predictions = []
    for item in k_nearest_neighbors:
        file_name = item[0]
        image_path = os.path.join("../image", file_name.split('_')[0], file_name)  # 构建图像路径

        # 加载图像并预处理
        image = img_to_array(load_img(image_path, target_size=(224, 224)))
        image = image / 255.0  # 数据标准化

        # 使用模型进行预测
        predicted_label = np.argmax(model.predict(np.array([image])))
        predictions.append((file_name, predicted_label, item[1]))  # 存储图像路径及距离
    # 按预测标签排序，同时保持第一个匹配图像的顺序
    first_label = predictions[0][1]  # 获取第一个匹配图像的标签
    # print(f'first_label: {first_label}')
    sorted_predictions = [predictions[0]]  # 将第一个匹配图像放在第一个位置
    i = 1
    for item in predictions[1:]:
        if item[1] == first_label:  # 如果当前样本的预测标签与第一个匹配图像的标签相同
            sorted_predictions.insert(i, item)
            i += 1
        else:
            sorted_predictions.append(item)
    # print(f'sorted_predictions: {sorted_predictions}')
    true_predictions = [(item[0], item[2]) for item in sorted_predictions[:i]]
    # print(f'true_predictions: {true_predictions}')
    # 将 sorted_distances 转换为字典
    distance_dict = {item[0]: item[2] for item in sorted_distances[:40]}

    # 从 sorted_predictions 中选取前 n 张图片的名字
    top_n_names = [item[0] for item in sorted_predictions[:i]]
    # print(f'top_n_names: {top_n_names}')

    # 找到 top_n_names 对应的特征
    selected_features = [distance_dict[name] for name in top_n_names if name in distance_dict]
    # 定义权重
    weights = np.array([j for j in range(i, 0, -1)], dtype=float)

    # 计算加权平均特征
    weighted_sum = np.sum([weight * feature for weight, feature in zip(weights, selected_features)], axis=0)
    average_feature = weighted_sum / np.sum(weights)

    # 使用加权平均特征进行查询
    distances = []

    for index, feature in enumerate(encoded_features):
        file_name = feature[0]
        feature_data = feature[1:]
        distance = calculate_distance(average_feature, feature_data)
        distances.append((file_name, distance))

    # 按距离排序
    sorted_distances = sorted(distances, key=lambda x: x[1])
    # 遍历排序后的距离列表
    for sort in sorted_distances:
        # 检查是否已经存在相同文件名的项
        if sort[0] not in [item[0] for item in true_predictions]:
            # 如果不重复，则添加到 true_predictions 中
            true_predictions.append((sort[0], sort[1]))

    return true_predictions[:k]


def calculate_distance(feature1, feature2):
    feature2_float = feature2.astype(float)
    # 计算欧氏距离
    distance = np.sqrt(np.sum((feature1 - feature2_float)**2))
    return distance


def is_image_file(file_path):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return any(file_path.lower().endswith(ext) for ext in img_extensions)


def count_total_positive_samples(data_folder):
    total_positive_samples = 0
    files_in_folder = os.listdir(data_folder)
    for file_name in files_in_folder:
        file_path = os.path.join(data_folder, file_name)
        if os.path.isfile(file_path) and is_image_file(file_name):
            total_positive_samples += 1
    print(f'total_positive_samples：{total_positive_samples}')
    return total_positive_samples


def calculate_precision_recall(actual_label, retrieved_labels, total_positive_samples):
    # 统计真正例：检索结果中与实际标签相符的样本数
    true_positives = sum(1 for label in retrieved_labels if label == actual_label)
    # 统计假正例：检索结果中非实际标签的样本数
    false_positives = len(retrieved_labels) - true_positives
    # 统计假负例：未能检索到的实际标签的样本数
    false_negatives = total_positive_samples - true_positives
    # print(f'false_negatives: {false_negatives}')
    # 计算精确率和召回率，防止除零错误
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall


def evaluate(test_image, knn_result):
    total_precision = {k: 0 for k in range(1, 11)}
    total_recall = {k: 0 for k in range(1, 11)}
    actual_label = test_image.split('_')[0]
    retrieved_labels = [retrieved_image_name.split('_')[0] for retrieved_image_name, _ in knn_result]

    data_folder = f'../image/{actual_label}'
    total_positive_samples = count_total_positive_samples(data_folder)
    for k in range(1, 11):
        top_k_retrieved_labels = retrieved_labels[:k]
        precision, recall = calculate_precision_recall(actual_label, top_k_retrieved_labels, total_positive_samples)
        total_precision[k] += precision
        total_recall[k] += recall

    print(f'total_precision: {total_precision}')
    print(f'total_recall: {total_recall}')
    ap_sum = 0
    for k in range(1, 11):
        ap_sum += total_precision[k]
    average_ap_10 = ap_sum/10
    average_precision_10 = total_precision[10]
    average_recall_10 = total_recall[10]

    return average_precision_10, average_recall_10, average_ap_10


# def main():
#     # 设置日志记录器
#     logging.basicConfig(level=logging.INFO)
#
#     # 图像路径和特征提取方法
#     image_path = "../test3/A0C573/A0C573_20151103073308_3029240562.jpg"
#     method = "ORB"
#
#     # 加载模型和代码本
#     model_path = "encoded/orb_encoded_features_200.pkl"
#     codebook_path = "features/orb_codebook_200.pkl"
#     tfidf_model_path = "model/tfidf_model_orb.pkl"
#     model = load_model(model_path)
#     codebook = load_codebook(codebook_path)
#     tfidf_model = load_model(tfidf_model_path)
#
#     if model is None or codebook is None or tfidf_model is None:
#         print("Failed to load model, codebook, or TF-IDF model.")
#         return
#
#     # 提取图像特征
#     keypoints, descriptors = extract_features_from_image(image_path, method)
#
#
#     # 对特征进行编码
#     encoded_features = encode_features([descriptors], codebook, tfidf_model)
#     # 使用 KNN 搜索
#     k = 5
#     knn_result = knn_search(encoded_features[0], model, k=k)
#
#     # 输出 KNN 返回结果
#     print(f"Top {k} similar images:")
#     for i, (name, distance) in enumerate(knn_result):
#         print(f"{i + 1}. {name} - Distance: {distance}")
#
#
# if __name__ == "__main__":
#     main()

