import tensorflow as tf
import sys

from transformer_model import TransformerModel
from data_loader import load_vectorization_from_disk

# Tải vectorization
source_vectorization = load_vectorization_from_disk('source_vectorization_layer.pkl')
target_vectorization = load_vectorization_from_disk('target_vectorization_layer.pkl')

# Tải model đã huấn luyện
transformer = TransformerModel(
    source_vectorization=source_vectorization,
    target_vectorization=target_vectorization,
    model_path='restore_diacritic.keras'
)

def restore_diacritics(text):
    # Dự đoán và khôi phục dấu cho văn bản không dấu
    restored_text = transformer.predict(text)
    return restored_text

# Ví dụ sử dụng
# input_texts = [
    # "toi di hoc",
    # "em con nho hay em da quen",
    # "ha noi mua nay vang nhung con mua",
    # "dat nuoc toi thon tha giot dan bau",
    # "cam on ban",
    # "toi yeu viet nam",
    # "doi tuyen all-star viet nam tham du kespa cup 2024 bao gom 6 nguoi", 
#     "chao co a",
# ]

# for text in input_texts:
#     output = restore_diacritics(text)
#     print(f"Input: {text} -> Output: {output}")
