import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import logging
import pickle

# 设置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_image_file(file_path):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return any(file_path.lower().endswith(ext) for ext in img_extensions)


def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
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


def extract_features(image_folder, feature_extractor):
    features = []
    file_names = []
    total_image = 0# 添加文件名列表
    for subdir, dirs, files in os.walk(image_folder):
        for file in files:
            image_path = os.path.join(subdir, file)
            if is_image_file(image_path):
                total_image += 1
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    keypoints, descriptors = feature_extractor.detectAndCompute(image, None)
                    if descriptors is not None:
                        features.append(descriptors)
                        file_names.append(file)  # 添加文件名到列表中
    return features, file_names, total_image  # 返回特征和文件名列表


def calculate_weights(nearest_cluster_indices, df, num_clusters, num_images):
    raw_tf = np.bincount(nearest_cluster_indices, minlength=int(num_clusters))
    # 对数转换消除文档长度对 TF 的影响
    tf = (1 + np.log(raw_tf + 1)) / (1 + np.log(len(nearest_cluster_indices))) + 0.5
    # 调整IDF平滑和常数,IDF作为权重时可以保持一定的影响力
    idf = np.log((num_images + 0.1) / (df + 0.1)) + 1.6

    weights = tf * idf

    return weights


def encode_features(features, codebook, df, file_paths, num_clusters, total_image):
    encoded_features = []
    for i, descriptors in enumerate(features):
        if descriptors is not None:
            # print("Descriptors shape:", descriptors.shape)
            # print("Codebook shape:", codebook.shape)
            # 创建一个与码本长度相同的全零数组，用于存储编码后的特征
            encoded_feature = np.zeros(len(codebook))
            # 计算每个描述符与码本中心的距离（欧氏距离），并找到最近的聚类中心的索引
            distances = np.linalg.norm(codebook - descriptors[:, np.newaxis], axis=-1)
            nearest_cluster_indices = np.argmin(distances, axis=1)

            # print(np.unique(nearest_cluster_indices))
            weights = calculate_weights(nearest_cluster_indices, df, num_clusters, total_image)
            # print(f'weights:{weights}')
            # 归一化权重
            normalized_weights = weights / np.sum(weights)

            for index, weight in zip(nearest_cluster_indices, normalized_weights):
                encoded_feature[index] = max(encoded_feature[index], weight)

            # 将文件路径和编码后的特征合并（假设file_paths[i]是一个字符串）
            encoded_feature_with_path = np.append(file_paths[i], encoded_feature)

            # 将编码后的特征添加到列表中
            encoded_features.append(encoded_feature_with_path)

    return encoded_features


def save_encoded_features(encoded_features, file_path):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(encoded_features, f)
        logger.info(f"Encoded features saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save encoded features to {file_path}: {e}")


def main1(image_folder, orb_codebook_path, orb_encoded_features_file):
    # ORB 特征处理
    orb = cv2.ORB_create(nfeatures=1000)
    orb_codebook = load_codebook(orb_codebook_path)
    if orb_codebook is not None:
        # 加载 TF-IDF 模型
        tfidf_model_path = "model/orb_tfidf_1000.pkl"
        num_clusters = tfidf_model_path.split('.')[0].split('_')[-1]
        tfidf_model = load_model(tfidf_model_path)
        print(tfidf_model)
        if tfidf_model is not None:
            orb_features, file_names, total_image = extract_features(image_folder, orb)  # 获取特征和文件名列表
            orb_encoded_features = encode_features(orb_features, orb_codebook, tfidf_model, file_names, num_clusters, total_image)  # 传递文件名列表
            save_encoded_features(orb_encoded_features, orb_encoded_features_file)


def main2(image_folder, sift_codebook_path, sift_encoded_features_file):
    sift = cv2.SIFT_create(nfeatures=1000)
    sift_codebook = load_codebook(sift_codebook_path)
    if sift_codebook is not None:
        # 加载 TF-IDF 模型
        tfidf_model_path = "model/sift_tfidf_1000.pkl"
        num_clusters = tfidf_model_path.split('.')[0].split('_')[-1]
        tfidf_model = load_model(tfidf_model_path)
        print(tfidf_model)
        if tfidf_model is not None:
            sift_features, file_names, total_image = extract_features(image_folder, sift)  # 获取特征和文件名列表
            sift_encoded_features = encode_features(sift_features, sift_codebook, tfidf_model, file_names, num_clusters,total_image)  # 传递文件名列表
            save_encoded_features(sift_encoded_features, sift_encoded_features_file)


if __name__ == "__main__":
    image_folder = "../image"

    # ORB参数
    orb_codebook_path = "features/orb_codebook_1000.pkl"
    orb_encoded_features_file = "encoded/orb_encoded_features_1000_N.pkl"

    # SIFT参数
    sift_codebook_path = "features/sift_codebook_1000.pkl"
    sift_encoded_features_file = "encoded/sift_encoded_features_1000_N.pkl"

    main1(image_folder, orb_codebook_path, orb_encoded_features_file)
    main2(image_folder, sift_codebook_path, sift_encoded_features_file)
