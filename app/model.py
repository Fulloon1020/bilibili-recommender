"""
Author: Fulloon
Discription: Wide&Deep + NLP + Sequential(GRU) Recommender
"""

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from app.getHistoryData import get_history_data

save_dir = "saved_model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

import tensorflow as tf
import keras
import numpy as np
import json
import jieba
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import class_weight

# --- 超参数定义 ---
MAX_TITLE_LEN = 20    # 标题最大长度
MAX_HISTORY_LEN = 10  # 回溯历史长度
VOCAB_SIZE = 5000     # 词表最大容量

class ResizableEmbedding(keras.layers.Layer):
    """
    可扩展的Embedding层
    """
    def __init__(self, initial_num_items, embedding_dim):
        super(ResizableEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.initial_num_items = initial_num_items
        self.embedding_matrix = self.add_weight(
            shape=(initial_num_items, embedding_dim),
            initializer="random_normal",
            trainable=True,
            name="embedding_matrix",
        )

    def expand(self, new_num_items):
        current_size = self.embedding_matrix.shape[0]
        if new_num_items <= current_size:
            return
        additional_embeddings = tf.random.normal(
            [new_num_items - current_size, self.embedding_dim]
        )
        new_embedding_matrix = tf.concat(
            [self.embedding_matrix, additional_embeddings], axis=0
        )
        self.embedding_matrix.assign(new_embedding_matrix)

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embedding_matrix, inputs)


class AttentionLayer(keras.layers.Layer):
    """
    注意力层 (用于处理当前视频的 Tag 列表)
    """
    def __init__(self, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.attention_dim),
            initializer="random_normal",
            trainable=True,
            name="attention_w",
        )
        self.V = self.add_weight(
            shape=(self.attention_dim, 1),
            initializer="random_normal",
            trainable=True,
            name="attention_v",
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, num_tags, embedding_dim)
        uit = tf.tensordot(inputs, self.W, axes=1)
        uit = tf.nn.tanh(uit)
        ait = tf.tensordot(uit, self.V, axes=1)
        attention_weights = tf.nn.softmax(ait, axis=1)
        weighted_input = attention_weights * inputs
        output = tf.reduce_sum(weighted_input, axis=1)
        return output


class FeatureProcessor:
    """
    特征处理器：处理 Tag, Author, Title, History
    """
    def __init__(self):
        self.tag2idx = defaultdict(lambda: len(self.tag2idx))
        # 0 reserved for padding/unknown
        self.tag2idx['<PAD>'] = 0
        self.author2idx = defaultdict(lambda: len(self.author2idx))
        self.word_index = {} # 词汇表

    def build_vocab(self, titles):
        """
        基于所有标题构建词表
        """
        word_counts = {}
        for title in titles:
            if not title: continue
            words = jieba.cut(title)
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1

        # 按频率排序，取前 VOCAB_SIZE 个
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:VOCAB_SIZE]
        # 0 is padding, start from 1
        self.word_index = {w: i+1 for i, (w, _) in enumerate(sorted_words)}

    def text_to_sequence(self, text):
        """
        将文本转换为 ID 序列
        """
        if not text:
            return [0] * MAX_TITLE_LEN

        words = jieba.cut(text)
        seq = [self.word_index.get(w, 0) for w in words][:MAX_TITLE_LEN]
        # Padding
        seq += [0] * (MAX_TITLE_LEN - len(seq))
        return seq

    def calculate_quality_score(self, views, likes, favs):
        log_views = np.log(views + 1)
        alpha, beta, gamma = 0.4, 0.3, 0.3
        like_ratio = likes / log_views if log_views > 0 else 0
        fav_ratio = favs / log_views if log_views > 0 else 0
        score = alpha * like_ratio + beta * fav_ratio + gamma * log_views / 20
        return np.clip(score, 0, 1)

    def calculate_interest_score(self, progress, duration, is_liked, is_faved):
        progress_ratio = progress / duration if duration > 0 else 0
        interaction_score = float(is_liked) + float(is_faved)
        interest_score = max(interaction_score / 2, progress_ratio)
        return np.clip(interest_score, 0, 1)

    def process_video_features(self, video_data):
        """
        处理单个视频的基础特征 (不包含 History)
        """
        tags = [self.tag2idx[tag] for tag in video_data.get("tag", [])]
        author_idx = self.author2idx[video_data.get("author", "")]

        quality_score = self.calculate_quality_score(
            video_data.get("view", 0), video_data.get("like", 0), video_data.get("favorite", 0)
        )

        interest_score = self.calculate_interest_score(
            video_data.get("progress", 0),
            video_data.get("duration", 1),
            video_data.get("isliked", 0),
            video_data.get("isfaved", 0),
        )

        title_seq = self.text_to_sequence(video_data.get("title", ""))

        threshold = 0.5
        label = 1 if interest_score > threshold else 0

        return {
            "tags": tags,
            "author": author_idx,
            "quality_score": quality_score,
            "title": title_seq,
            "label": label,
        }


class VideoRecommender(keras.Model):
    """
    升级版视频推荐器: Wide&Deep + NLP + GRU
    """
    def __init__(self, num_tags, num_authors, embedding_dim, vocab_size=5000, max_title_len=20):
        super(VideoRecommender, self).__init__()
        self.embedding_dim = embedding_dim

        # 1. 基础特征 Embeddings
        self.tag_embedding = ResizableEmbedding(num_tags, embedding_dim)
        self.author_embedding = ResizableEmbedding(num_authors, embedding_dim)

        # 2. NLP Title Embedding
        self.word_embedding = keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, mask_zero=True)
        self.title_pool = keras.layers.GlobalAveragePooling1D() # 简单高效提取句子特征

        # 3. Attention (用于处理当前视频的多个 Tags)
        self.attention = AttentionLayer(attention_dim=embedding_dim)

        # 4. Sequence Modeling (GRU) - 处理历史 Tag 序列
        self.history_gru = keras.layers.GRU(32, return_sequences=False)

        # 5. Layers
        self.wide = keras.layers.Dense(1, name="wide_dense")

        self.deep_layer1 = keras.layers.Dense(128, activation="relu")
        self.deep_layer2 = keras.layers.Dense(64, activation="relu")
        self.deep_layer3 = keras.layers.Dense(32, activation="relu")

        self.final_dense = keras.layers.Dense(1, activation="sigmoid", name="final")

    def call(self, inputs):
        # inputs 顺序: [tags, author, quality, title, history]
        tags, author, quality_score, title, history = inputs

        # --- Part A: 当前视频特征处理 ---
        # 1. Tags
        tag_embeddings = self.tag_embedding(tags) # (batch, num_tags, emb)

        # 2. Author
        author_embedding = self.author_embedding(author)
        author_embedding = tf.expand_dims(author_embedding, axis=1) # (batch, 1, emb)

        # 3. NLP Title
        title_emb = self.word_embedding(title) # (batch, len, emb)
        title_feature = self.title_pool(title_emb) # (batch, emb)

        # 4. Attention Merge for Tags + Author
        combined_embedded = tf.concat([tag_embeddings, author_embedding], axis=1)
        content_features = self.attention(combined_embedded) # (batch, emb)

        # --- Part B: 用户历史序列处理 (Sequential) ---
        # history shape: (batch, history_len, 1) -> 需要先把 tag id 转为 embedding
        # 这里的 history 输入的是 tag id 序列
        history_emb = self.tag_embedding(history) # (batch, history_len, emb)
        # 通过 GRU 提取兴趣演变
        user_interest_vector = self.history_gru(history_emb) # (batch, 32)

        # --- Part C: Wide & Deep 合并 ---
        quality_score = tf.expand_dims(tf.cast(quality_score, tf.float32), -1)
        wide_output = self.wide(quality_score)

        # Deep Input: Content + Author + Quality + Title + History
        deep_features = tf.concat(
            [content_features, author_embedding[:, 0, :], quality_score, title_feature, user_interest_vector], axis=1
        )

        deep_features = self.deep_layer1(deep_features)
        deep_features = self.deep_layer2(deep_features)
        deep_features = self.deep_layer3(deep_features)

        combined = tf.concat([wide_output, deep_features], axis=1)

        return self.final_dense(combined)

    def save_model(self, filepath):
        if not filepath.endswith(".weights.h5"):
            filepath = filepath + ".weights.h5"
        self.save_weights(filepath)

    def load_model_weights(self, filepath):
        if not filepath.endswith(".weights.h5"):
            filepath = filepath + ".weights.h5"
        self.load_weights(filepath)


def load_and_process_data(file_path, processor, cookies=None):
    """
    加载并处理数据 (包含序列构建)
    """
    if file_path is not None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            data = []
    else:
        data = get_history_data(cookies, 100)

    if not data:
        print("警告: 没有数据用于训练")
        return {}, np.array([]), 10

    # 1. 构建词表 (NLP)
    all_titles = [v.get("title", "") for v in data]
    processor.build_vocab(all_titles)

    # 2. 准备容器
    all_tags = []
    all_authors = []
    all_quality_scores = []
    all_labels = []
    all_titles_seq = []
    all_histories = []

    max_tags = 0
    for v in data:
        max_tags = max(max_tags, len(v.get("tag", [])))
    if max_tags == 0: max_tags = 1 # 防止全空报错

    # 3. 遍历数据构建特征 (含滑动窗口历史)
    # data 应该是按时间排序的 (在 getHistoryData 做了)
    for i, video in enumerate(data):
        processed = processor.process_video_features(video)

        # Pad Tags
        tags = processed["tags"]
        tags = tags + [0] * (max_tags - len(tags))

        # 构建 History Sequence
        # 取过去 MAX_HISTORY_LEN 个视频的第一个 Tag 作为历史特征
        history_seq = []
        start_idx = max(0, i - MAX_HISTORY_LEN)
        # 获取过去 N 个视频的数据
        past_videos = data[start_idx:i]

        for pv in past_videos:
            ptags = pv.get("tag", [])
            # 如果有tag，取第一个tag的idx；否则填0
            if ptags and ptags[0] in processor.tag2idx:
                history_seq.append(processor.tag2idx[ptags[0]])
            else:
                history_seq.append(0)

        # Padding History
        history_seq = history_seq + [0] * (MAX_HISTORY_LEN - len(history_seq))
        # 截断 (理论上不需要，但保险起见)
        history_seq = history_seq[-MAX_HISTORY_LEN:]

        all_tags.append(tags)
        all_authors.append(processed["author"])
        all_quality_scores.append(processed["quality_score"])
        all_titles_seq.append(processed["title"])
        all_histories.append(history_seq)
        all_labels.append(processed["label"])

    processed_data = {
        "tags": np.array(all_tags, dtype=np.int32),
        "author": np.array(all_authors, dtype=np.int32),
        "quality_score": np.array(all_quality_scores, dtype=np.float32),
        "title": np.array(all_titles_seq, dtype=np.int32),
        "history": np.array(all_histories, dtype=np.int32)
    }
    labels = np.array(all_labels, dtype=np.float32)

    # 返回增加: vocab, word_index 以便 Recommender 使用
    return processed_data, labels, max_tags, processor.word_index


def save_model_and_processor(model, processor, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.save_model(os.path.join(save_dir, "best_model.weights.h5"))
    with open(os.path.join(save_dir, "feature_processor.pkl"), "wb") as f:
        pickle.dump(
            {
                "tag2idx": dict(processor.tag2idx),
                "author2idx": dict(processor.author2idx),
                "word_index": dict(processor.word_index) # 保存 NLP 词表
            },
            f,
        )

def load_model_and_processor(save_dir):
    try:
        with open(os.path.join(save_dir, "feature_processor.pkl"), "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("未找到保存的模型处理器")
        return None, None

    processor = FeatureProcessor()
    processor.tag2idx = defaultdict(lambda: len(processor.tag2idx), data["tag2idx"])
    processor.author2idx = defaultdict(lambda: len(processor.author2idx), data["author2idx"])
    processor.word_index = data.get("word_index", {})

    num_tags = len(processor.tag2idx)
    num_authors = len(processor.author2idx)
    vocab_size = len(processor.word_index) + 1
    embedding_dim = 32

    model = VideoRecommender(num_tags, num_authors, embedding_dim, vocab_size=vocab_size)

    # 建立 Dummy Input 以初始化权重
    dummy_tags = tf.zeros((1, 10), dtype=tf.int32)
    dummy_author = tf.zeros((1,), dtype=tf.int32)
    dummy_qual = tf.zeros((1,), dtype=tf.float32)
    dummy_title = tf.zeros((1, MAX_TITLE_LEN), dtype=tf.int32)
    dummy_history = tf.zeros((1, MAX_HISTORY_LEN), dtype=tf.int32)

    model([dummy_tags, dummy_author, dummy_qual, dummy_title, dummy_history])

    try:
        model.load_model_weights(os.path.join(save_dir, "best_model.weights.h5"))
    except:
        print("权重加载失败，将使用随机初始化权重")

    return model, processor


def train_model(model, data, labels, optimizer, class_weights):
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "tags": data["tags"],
                "author": data["author"],
                "quality_score": data["quality_score"],
                "title": data["title"],
                "history": data["history"]
            },
            labels,
        )
    )

    dataset = dataset.shuffle(buffer_size=1000).batch(32)
    batch_losses = []

    for features, batch_labels in dataset:
        with tf.GradientTape() as tape:
            # 输入顺序要对应 model.call
            predictions = model(
                [
                    features["tags"],
                    features["author"],
                    features["quality_score"],
                    features["title"],
                    features["history"]
                ]
            )

            loss = keras.losses.binary_crossentropy(
                batch_labels, tf.squeeze(predictions)
            )

            # 安全处理 class_weight key
            # 将 label 转为 int 作为 key
            batch_labels_int = tf.cast(batch_labels, tf.int32)
            # 使用 tf.gather 来应用权重，避免 key error
            weights = tf.gather([class_weights.get(0, 1.0), class_weights.get(1, 1.0)], batch_labels_int)

            loss = loss * tf.cast(weights, tf.float32)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        batch_losses.append(float(loss))

    return np.mean(batch_losses)


def evaluate_model(model, data, labels):
    predictions = (
        model([
            data["tags"],
            data["author"],
            data["quality_score"],
            data["title"],
            data["history"]
        ]).numpy().flatten()
    )

    labels = labels.flatten()

    try:
        auc = roc_auc_score(labels, predictions)
        average_precision = average_precision_score(labels, predictions)
    except ValueError:
        auc = 0.5
        average_precision = 0

    k = min(10, len(predictions))
    if k > 0:
        indices = np.argsort(predictions)[-k:]
        top_k_labels = labels[indices]
        precision_at_k = np.sum(top_k_labels) / k
        recall_at_k = np.sum(top_k_labels) / np.sum(labels) if np.sum(labels) > 0 else 0
    else:
        precision_at_k = 0
        recall_at_k = 0

    return {
        "AUC-ROC": auc,
        "Average Precision": average_precision,
        "Precision@k": precision_at_k,
        "Recall@k": recall_at_k,
    }


def predict_interests(model, processor, video_data):
    all_tags = []
    all_authors = []
    all_quality_scores = []
    all_titles = []
    all_histories = []
    video_info = []

    # 预测时，我们需要一个假设的"History"状态
    # 理想情况下应该传入用户真实的近期观看历史
    # 这里为了演示，生成一个全0的 history，或者应该从 Recommender 传入
    dummy_history = [0] * MAX_HISTORY_LEN

    processed_videos = []
    for video in video_data:
        valid_tags = [tag for tag in video.get("tag", []) if tag in processor.tag2idx]
        # 即使没有 Tag，也可以通过 Title 预测，所以放宽条件
        if not valid_tags and not video.get("title"):
            continue
        processed_videos.append({**video, "tag": valid_tags})

    if not processed_videos:
        return []

    print(f"找到 {len(processed_videos)} 个有效候选视频")
    max_tags = 10 # 预测时固定一个长度即可

    for video in processed_videos:
        tags = [processor.tag2idx[tag] for tag in video["tag"]]
        tags = tags + [0] * (max_tags - len(tags))
        tags = tags[:max_tags]

        author_idx = processor.author2idx.get(video.get("author"), 0)

        quality_score = processor.calculate_quality_score(
            video.get("view", 0), video.get("like", 0), video.get("favorite", 0)
        )

        title_seq = processor.text_to_sequence(video.get("title", ""))

        all_tags.append(tags)
        all_authors.append(author_idx)
        all_quality_scores.append(quality_score)
        all_titles.append(title_seq)
        all_histories.append(dummy_history) # 广播

        video_info.append(
            {
                "bvid": video["bvid"],
                "title": video.get("title", ""),
                "author": video.get("author", ""),
                "original_tags": video["tag"],
                "quality_score": quality_score,
            }
        )

    processed_data = {
        "tags": np.array(all_tags, dtype=np.int32),
        "author": np.array(all_authors, dtype=np.int32),
        "quality_score": np.array(all_quality_scores, dtype=np.float32),
        "title": np.array(all_titles, dtype=np.int32),
        "history": np.array(all_histories, dtype=np.int32)
    }

    print("模型预测中...")
    predictions = (
        model(
            [
                processed_data["tags"],
                processed_data["author"],
                processed_data["quality_score"],
                processed_data["title"],
                processed_data["history"]
            ]
        )
        .numpy()
        .flatten()
    )

    results = []
    for i, pred in enumerate(predictions):
        results.append({**video_info[i], "interest_score": float(pred)})

    results.sort(key=lambda x: x["interest_score"], reverse=True)
    return results

if __name__ == "__main__":
    # 简单的本地测试逻辑
    pass