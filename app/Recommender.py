"""
Author: Fulloon
Discription: Recommender system (Optimized: Fast Load & Agg Backend & NaN Fix)
"""

import os
# [优化1] 设置 Matplotlib 后端为 'Agg' (非交互模式)
# 必须在 import pyplot 之前设置，解决 UserWarning 和卡顿问题
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jieba
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight

from app.getHistoryData import get_history_data
from app.getHotData import get_hot_data
from app.getRecommandData import get_recommand_data
from app.model import *

MAX_TITLE_LEN = 20
MAX_HISTORY_LEN = 10

class Recommender:
    def __init__(self, cookies):
        self.cookies = cookies
        self.video_pool_hot = []
        self.video_pool_recommend = []

        # 1. 初始化处理器
        self.processor = FeatureProcessor()

        # 2. 加载数据
        print("正在加载历史数据...")
        data_tuple = load_and_process_data("historyVideo.json", self.processor)

        if len(data_tuple) == 4:
            self.processed_data, self.labels, self.max_tags, self.word_index = data_tuple
            self.processor.word_index = self.word_index
        elif len(data_tuple) == 3:
            self.processed_data, self.labels, self.max_tags = data_tuple
            self.word_index = {}
            print("警告: 检测到旧版数据格式。")
        else:
            raise ValueError(f"load_and_process_data 返回异常")

        # 3. 计算权重
        if len(self.labels) > 0:
            self.class_weights_array = class_weight.compute_class_weight(
                class_weight="balanced", classes=np.unique(self.labels), y=self.labels
            )
            self.class_weights_dict = {i: w for i, w in enumerate(self.class_weights_array)}
        else:
            self.class_weights_dict = {0: 1.0, 1: 1.0}

        # 4. 初始化模型结构
        self.num_tags = len(self.processor.tag2idx)
        self.num_authors = len(self.processor.author2idx)
        self.vocab_size = len(self.word_index) + 1 if self.word_index else 5000
        self.embedding_dim = 32

        self.model = VideoRecommender(
            self.num_tags,
            self.num_authors,
            self.embedding_dim,
            vocab_size=self.vocab_size,
            max_title_len=MAX_TITLE_LEN
        )
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)

        # [优化2] 尝试加载已有模型，避免重复训练
        if self._try_load_model():
            print(">>> 模型加载成功，跳过训练步骤")
        else:
            print(">>> 未找到可用模型或结构变化，开始重新训练...")
            self._train_loop()

    def _try_load_model(self):
        """
        尝试加载保存的权重
        """
        weight_path = os.path.join(save_dir, "best_model.weights.h5")
        if os.path.exists(weight_path):
            try:
                # 必须先 Build 模型（跑一次假数据）才能加载权重
                self._build_dummy_graph()
                self.model.load_model_weights(weight_path)
                return True
            except Exception as e:
                print(f"加载旧模型失败 ({e})，将重新训练。")
                return False
        return False

    def _build_dummy_graph(self):
        """
        构建虚拟计算图，用于初始化权重形状
        """
        dummy_tags = tf.zeros((1, 10), dtype=tf.int32)
        dummy_author = tf.zeros((1,), dtype=tf.int32)
        dummy_qual = tf.zeros((1,), dtype=tf.float32)
        dummy_title = tf.zeros((1, MAX_TITLE_LEN), dtype=tf.int32)
        dummy_history = tf.zeros((1, MAX_HISTORY_LEN), dtype=tf.int32)
        # 调用一次 call
        self.model([dummy_tags, dummy_author, dummy_qual, dummy_title, dummy_history])

    def _train_loop(self):
        # [优化3] 减少演示用的训练轮数，从 30 降到 5，大幅提速
        num_epochs = 5
        best_loss = float("inf")
        losses = []
        AUCROC = []

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not self.processed_data or len(self.processed_data.get('tags', [])) == 0:
            print("无数据，跳过训练。")
            return

        for epoch in range(num_epochs):
            # print(f"Epoch {epoch + 1}/{num_epochs}") # 减少刷屏
            loss = train_model(
                self.model,
                self.processed_data,
                self.labels,
                self.optimizer,
                self.class_weights_dict,
            )
            losses.append(loss)

            # 这里的评估也可以减少频率，比如每5轮评一次
            metrics = evaluate_model(self.model, self.processed_data, self.labels)
            AUCROC.append(metrics["AUC-ROC"])

            if loss < best_loss:
                best_loss = loss
                save_model_and_processor(self.model, self.processor, save_dir)

            print(f"Epoch {epoch+1}: Loss={loss:.4f}, AUC={metrics['AUC-ROC']:.4f}")

        # 训练结束后统一画图（由于设置了 Agg，不会弹窗阻塞）
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(losses) + 1), losses, "b-", label="Loss")
            plt.plot(range(1, len(AUCROC) + 1), AUCROC, "r-", label="AUC")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "training_loss.png"))
            plt.close() # 必须关闭，释放内存
        except Exception as e:
            print(f"绘图忽略: {e}")

    def expand_pool(self, pool_type, video_cnt):
        if pool_type == "hot":
            data = get_hot_data(self.cookies, video_cnt)
            if data: self.video_pool_hot += data
        elif pool_type == "recommend":
            data = get_recommand_data(self.cookies, video_cnt)
            if data: self.video_pool_recommend += data

    def recommend(self, pool_type, video_cnt=1):
        pool = self.video_pool_hot if pool_type == "hot" else self.video_pool_recommend

        # 如果池子空了，扩充
        if video_cnt > len(pool):
            self.expand_pool(pool_type, video_cnt * 3 - len(pool))

        processed_videos = []
        for video in pool:
            valid_tags = [tag for tag in video.get("tag", []) if tag in self.processor.tag2idx]
            has_title = "title" in video and video["title"]
            # 只要有标题或者有有效Tag，都保留，防止过滤太狠
            if not valid_tags and not has_title:
                continue
            processed_videos.append({**video, "tag": valid_tags})

        print(f"找到 {len(processed_videos)} 个有效候选视频")
        if not processed_videos:
            return []

        all_tags, all_authors, all_quality_scores, all_titles, all_histories = [], [], [], [], []
        mock_history = [0] * MAX_HISTORY_LEN

        for video in processed_videos:
            tags = [self.processor.tag2idx[tag] for tag in video["tag"]]
            tags = tags + [0] * (self.max_tags - len(tags))
            tags = tags[:self.max_tags]

            author_idx = self.processor.author2idx.get(video.get("author"), 0)
            quality_score = self.processor.calculate_quality_score(
                video.get("view", 0), video.get("like", 0), video.get("favorite", 0)
            )

            title_seq = [0] * MAX_TITLE_LEN
            if "title" in video and self.word_index:
                words = jieba.cut(video["title"])
                t_seq = [self.word_index.get(w, 0) for w in words][:MAX_TITLE_LEN]
                title_seq[:len(t_seq)] = t_seq

            all_tags.append(tags)
            all_authors.append(author_idx)
            all_quality_scores.append(quality_score)
            all_titles.append(title_seq)
            all_histories.append(mock_history)

        processed_input = {
            "tags": np.array(all_tags, dtype=np.int32),
            "author": np.array(all_authors, dtype=np.int32),
            "quality_score": np.array(all_quality_scores, dtype=np.float32),
            "title": np.array(all_titles, dtype=np.int32),
            "history": np.array(all_histories, dtype=np.int32)
        }

        # 预测
        try:
            predictions = (
                self.model(
                    [
                        processed_input["tags"],
                        processed_input["author"],
                        processed_input["quality_score"],
                        processed_input["title"],
                        processed_input["history"]
                    ]
                )
                .numpy()
                .flatten()
            )
        except Exception as e:
            print(f"模型预测出错: {e}")
            # 出错时返回未排序的列表，防止前端白屏
            return processed_videos[:video_cnt]

        results = []
        for i, pred in enumerate(predictions):
            score = float(pred)
            # [Fix] 关键修复: 处理 NaN (非数字) 和 Inf (无穷大)
            # 如果不处理，前端 JSON 解析会报错 "Unexpected token 'N'"
            if np.isnan(score) or np.isinf(score):
                score = 0.0

            results.append({**processed_videos[i], "rating": score})

        results.sort(key=lambda x: x["rating"], reverse=True)
        return results[:video_cnt]

if __name__ == "__main__":
    cookie_path = os.path.join("user_data", "cookie.txt")
    if os.path.exists(cookie_path):
        with open(cookie_path, "r", encoding="utf-8") as file:
            debug_cookies = file.read().strip()
        recommender = Recommender(debug_cookies)
        print(recommender.recommend("hot", 5))