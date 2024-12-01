import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import multimodal_model


def load_data(dataset_path, task, bad_tweets_path):
    dataframe = pd.read_csv(dataset_path)
    bad_tweets = pd.read_csv(bad_tweets_path)["tweet_id"]
    dataframe = dataframe.merge(
        bad_tweets, how="left", indicator=True
    ).query('_merge == "left_only"').drop("_merge", axis=1)

    if task == "AS":
        y_true = np.array(dataframe.stance)
    else:
        y_true = np.array(dataframe.persuasiveness)

    return dataframe, y_true


def filter_data_with_existing_images(dataframe, images_path):
    """Filter out rows where images are missing."""
    dataframe["image_exists"] = dataframe["tweet_id"].apply(
        lambda x: os.path.exists(os.path.join(images_path, f"{x}.jpg"))
    )
    filtered_dataframe = dataframe[dataframe["image_exists"]].drop(
        "image_exists", axis=1
    )
    print(f"Filtered out {len(dataframe) - len(filtered_dataframe)} rows without images.")
    return filtered_dataframe


def handle_approach(train_dataset_path, dev_dataset_path, test_dataset_path,
                    train_images, test_images, bad_tweets_path, classifier, task="AS"):
    # Load datasets
    train_df, y_train = load_data(train_dataset_path, task, bad_tweets_path)
    dev_df, y_dev_true = load_data(dev_dataset_path, task, bad_tweets_path)
    test_df, y_test_true = load_data(test_dataset_path, task, bad_tweets_path)

    # Filter datasets
    train_df = filter_data_with_existing_images(train_df, train_images)
    dev_df = filter_data_with_existing_images(dev_df, train_images)
    test_df = filter_data_with_existing_images(test_df, test_images)

    # Extract features
    X_train = multimodal_model.clip32(train_df, train_images)
    if X_train.size == 0:
        print("No valid training features extracted. Skipping this approach.")
        return

    X_dev = multimodal_model.clip32(dev_df, train_images)
    if X_dev.size == 0:
        print("No valid dev features extracted. Skipping this approach.")
        return

    X_test = multimodal_model.clip32(test_df, test_images)
    if X_test.size == 0:
        print("No valid test features extracted. Skipping this approach.")
        return

    # Train classifier
    classifier.fit(X_train, y_train)

    # Predict
    y_dev_pred = classifier.predict(X_dev)
    y_test_pred = classifier.predict(X_test)

    # Evaluate
    pos_label = "support" if task == "AS" else "yes"
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label), 4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label), 4)
    print(f"Dev F1 Score: {dev_f1}")
    print(f"Test F1 Score: {test_f1}")


if __name__ == "__main__":
    # Paths
    GC_train_path = "data/gun_control_train.csv"
    A_train_path = "data/abortion_train_flan.csv"
    GC_dev_path = "data/gun_control_dev.csv"
    A_dev_path = "data/abortion_dev.csv"
    GC_test_path = "test/data/gun_control_test.csv"
    A_test_path = "test/data/abortion_test.csv"

    bad_tweets_path = "data/bad_tweets.csv"

    train_images = "data/images/image/"
    test_images = "test/data/images/image/"

    # Define classifiers
    svc_poly = SVC(kernel="poly", degree=3, C=1.0, coef0=0.02, shrinking=False, probability=True)
    svc_rbf = SVC(kernel="rbf", C=1.0, shrinking=False, probability=True)
    knn = KNeighborsClassifier(7)

    # Handle approaches
    print("Approach 1 (Poly SVC):")
    handle_approach(
        train_dataset_path=GC_train_path,
        dev_dataset_path=GC_dev_path,
        test_dataset_path=GC_test_path,
        train_images=train_images,
        test_images=test_images,
        bad_tweets_path=bad_tweets_path,
        classifier=svc_poly,
        task="IP"
    )

    print("\nApproach 2 (RBF SVC):")
    handle_approach(
        train_dataset_path=GC_train_path,
        dev_dataset_path=GC_dev_path,
        test_dataset_path=GC_test_path,
        train_images=train_images,
        test_images=test_images,
        bad_tweets_path=bad_tweets_path,
        classifier=svc_rbf,
        task="IP"
    )

    print("\nApproach 3 (KNN):")
    handle_approach(
        train_dataset_path=GC_train_path,
        dev_dataset_path=GC_dev_path,
        test_dataset_path=GC_test_path,
        train_images=train_images,
        test_images=test_images,
        bad_tweets_path=bad_tweets_path,
        classifier=knn,
        task="IP"
    )
