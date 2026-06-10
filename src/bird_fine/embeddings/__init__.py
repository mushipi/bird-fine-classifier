"""Foundation Model 埋め込み抽出（Perch 2.0 / BirdAVES）。

凍結バックボーンから埋め込みを取り出し data/embeddings/{model}/{split}.npz にキャッシュする。
プローブ学習・評価は training.train_probe / training.eval_probe が npz を読んで行う。
"""
