import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle


def save_codebook(codebook_model, model_name):
    with open(model_name, 'wb') as f:
        pickle.dump(codebook_model.cluster_centers_, f)


def save_kmeans(codebook_model, model_name):
    with open(model_name, 'wb') as f:
        pickle.dump(codebook_model, f)


def save_tfidf_model(tfidf_model, model_name):
    with open(model_name, 'wb') as f:
        pickle.dump(tfidf_model, f)


def load_codebook(codebook_path):
    try:
        with open(codebook_path, 'rb') as f:
            codebook = pickle.load(f)
        return codebook
    except Exception as e:
        logger.error(f"Failed to load codebook from {codebook_path}: {e}")
        return None


def is_image_file(file_path):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return any(file_path.lower().endswith(ext) for ext in img_extensions)


def extract_orb_features(image_folder):
    orb = cv2.ORB_create(nfeatures=1000)
    features = []
    for subdir, dirs, files in os.walk(image_folder):
        for file in files:
            image_path = os.path.join(subdir, file)
            if is_image_file(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    keypoints, descriptors = orb.detectAndCompute(image, None)
                    if descriptors is not None:
                        features.extend(descriptors)
    return features


def extract_sift_features(image_folder):
    sift = cv2.SIFT_create(nfeatures=1000)
    features = []
    for subdir, dirs, files in os.walk(image_folder):
        for file in files:
            image_path = os.path.join(subdir, file)
            if is_image_file(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    keypoints, descriptors = sift.detectAndCompute(image, None)
                    if descriptors is not None:
                        features.extend(descriptors)
    return features


def kmeans_clustering(feature, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(feature)
    return kmeans


def calculate_df(image_folder, kmeans_model, num_clusters, method):
    df = np.zeros(num_clusters)
    if method == "SIFT":
        extractor = cv2.SIFT_create(nfeatures=1000)
    else:
        extractor = cv2.ORB_create(nfeatures=1000)

    for subdir, dirs, files in os.walk(image_folder):
        for file in files:
            image_path = os.path.join(subdir, file)
            if is_image_file(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    keypoints, descriptors = extractor.detectAndCompute(image, None)
                    if descriptors is not None:
                        distances = np.linalg.norm(kmeans_model - descriptors[:, np.newaxis], axis=-1)
                        nearest_cluster_indices = np.argmin(distances, axis=1)
                        unique_indices = set(nearest_cluster_indices)
                        for index in unique_indices:
                            df[index] += 1
    return df




def main():
    image_folder = "../image"
    num_clusters = 1000

    print('extract features begin...')
    features_orb = extract_orb_features(image_folder)
    features_sift = extract_sift_features(image_folder)
    print('extract features end...')

    print('kmeans begin...')
    kmeans_orb = kmeans_clustering(np.array(features_orb), num_clusters)
    kmeans_sift = kmeans_clustering(np.array(features_sift), num_clusters)
    print('kmeans end...')

    save_codebook(kmeans_orb, f"features/orb_codebook_{num_clusters}.pkl")
    save_codebook(kmeans_sift, f"features/sift_codebook_{num_clusters}.pkl")
    save_kmeans(kmeans_orb, f"kmeans/orb_codebook_{num_clusters}.pkl")
    save_kmeans(kmeans_sift, f"kmeans/sift_codebook_{num_clusters}.pkl")
    # kmeans_orb = load_codebook("kmeans/orb_codebook_100.pkl")
    # kmeans_sift = load_codebook("kmeans/sift_codebook_100.pkl")

    df_orb = calculate_df(image_folder, kmeans_orb.cluster_centers_, num_clusters, "ORB")
    df_sift = calculate_df(image_folder, kmeans_sift.cluster_centers_, num_clusters, "SIFT")
    save_tfidf_model(df_orb, f"model/orb_tfidf_{num_clusters}.pkl")
    save_tfidf_model(df_sift, f"model/sift_tfidf_{num_clusters}.pkl")


if __name__ == "__main__":
    main()

