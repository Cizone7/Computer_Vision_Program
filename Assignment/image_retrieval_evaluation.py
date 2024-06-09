import os
import time
from image_retrieval_process import (load_codebook, encode_features,
                                     knn_search, load_model,
                                     extract_features_from_image, is_image_file)
from sklearn.cluster import KMeans


def count_total_positive_samples(data_folder):
    total_positive_samples = 0
    files_in_folder = os.listdir(data_folder)
    for file_name in files_in_folder:
        file_path = os.path.join(data_folder, file_name)
        if os.path.isfile(file_path) and is_image_file(file_name):
            total_positive_samples += 1
    return total_positive_samples


def calculate_precision_recall(actual_label, retrieved_labels, total_positive_samples):
    # 统计真正例：检索结果中与实际标签相符的样本数
    true_positives = sum(1 for label in retrieved_labels if label == actual_label)
    # 统计假正例：检索结果中非实际标签的样本数
    false_positives = len(retrieved_labels) - true_positives
    # 统计假负例：未能检索到的实际标签的样本数
    false_negatives = total_positive_samples - true_positives

    # 计算精确率和召回率，防止除零错误
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall


def calculate_response_speed(start_time):
    return time.time() - start_time


def evaluate_test_dataset(test_folder, method, model, codebook, tfidf_model, num_clusters, cnn):
    total_precision = {k: 0 for k in range(1, 11)}
    total_recall = {k: 0 for k in range(1, 11)}
    total_response_time = 0
    total_images = 0
    ap_sum = {k: 0 for k in range(1, 11)}

    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_path = os.path.join(root, file)
                actual_label = file.split('_')[0]
                start_time = time.time()
                _, descriptors = extract_features_from_image(image_path, method)
                encoded_features = encode_features([descriptors], codebook, tfidf_model, num_clusters, 946)
                knn_result = knn_search(encoded_features[0], model, cnn)
                response_time = calculate_response_speed(start_time)
                retrieved_labels = [retrieved_image_name.split('_')[0] for retrieved_image_name, _ in knn_result]

                data_folder = f'../image/{actual_label}'
                total_positive_samples = count_total_positive_samples(data_folder)

                for k in range(1, 11):
                    top_k_retrieved_labels = retrieved_labels[:k]
                    precision, recall = calculate_precision_recall(actual_label, top_k_retrieved_labels, total_positive_samples)
                    total_precision[k] += precision
                    total_recall[k] += recall
                    ap_sum[k] += precision

                total_response_time += response_time
                total_images += 1

    ap_score = {k: ap_sum[k] / total_images for k in range(1, 11)}

    average_precision = {k: total_precision[k] / total_images for k in range(1, 11)}
    average_recall = {k: total_recall[k] / total_images for k in range(1, 11)}
    average_response_time = total_response_time / total_images

    return average_precision, average_recall, average_response_time, ap_score


# # Evaluation
# test_folder = "../test3"
# method = "ORB"  # or "SIFT"
# model_path = "encoded/orb_encoded_features_200.pkl"
# codebook_path = "features/orb_codebook_200.pkl"
# tfidf_model_path = "model/tfidf_model_orb.pkl"
#
# model = load_model(model_path)
# codebook = load_codebook(codebook_path)
# tfidf_model = load_model(tfidf_model_path)
#
# if model is None or codebook is None or tfidf_model is None:
#     print("Failed to load model, codebook, or TF-IDF model.")
# else:
#     average_precision, average_recall, average_response_time,map = evaluate_test_dataset(test_folder, method, model,
#                                                                                      codebook, tfidf_model)
#     print(f"Average Precision: {average_precision}")
#     print(f"Average Recall: {average_recall}")
#     print(f"Average Response Time: {average_response_time} seconds")
