"""AST (Audio Spectrogram Transformer) 分類器のロード。

事前学習モデルの分類ヘッドを num_labels クラスに置き換えてfine-tune用に返す。
"""
from __future__ import annotations

from transformers import ASTForAudioClassification


def build_ast_classifier(
    pretrained: str,
    num_labels: int,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
) -> ASTForAudioClassification:
    """事前学習済みASTをロード、分類ヘッドを置換。

    Args:
        pretrained: HuggingFaceモデルID (例: MIT/ast-finetuned-audioset-10-10-0.4593)
        num_labels: 分類クラス数
        id2label: ラベルID → 種名 (任意、save時にconfigへ保存される)
        label2id: 種名 → ラベルID (任意)

    Returns:
        ASTForAudioClassification: ヘッドを置換したモデル
    """
    model = ASTForAudioClassification.from_pretrained(
        pretrained,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # ヘッドの形が変わるため必須
    )
    return model
